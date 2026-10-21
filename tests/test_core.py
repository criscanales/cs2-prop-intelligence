import json
import numpy as np
import pandas as pd
import pytest

from cs2model import odds
from cs2model.features.build import build_features
from cs2model.inference import distribution as D
from cs2model.inference.simulate import MatchSpec, PlayerSpec, leg_totals, simulate_match
from cs2model.names import norm_player, team_similarity
from cs2model.parlay import engine as E
from cs2model.scraper import http, lines
from cs2model.tracking import store


def test_american_decimal_roundtrip():
    assert odds.american_to_decimal(-110) == pytest.approx(1.9091, 1e-3)
    assert odds.american_to_decimal(150) == 2.5
    assert odds.decimal_to_american(2.5) == 150
    assert odds.decimal_to_american(1.5) == -200


def test_devig_sums_to_one():
    p = odds.devig(1.8, 2.1)
    assert sum(p) == pytest.approx(1)
    assert p[0] > p[1]


def test_series_to_map_prob():
    p = odds.map_prob_from_series(0.648, 3)
    assert p == pytest.approx(0.6, abs=0.01)
    assert odds.map_prob_from_series(0.5, 3) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("text,expected", [
    ("Kills on Maps 1+2", ("kills", (1, 2))),
    ("MAPS 1-3 Headshots", ("headshots", (1, 2, 3))),
    ("MAP 3 Kills", ("kills", (3,))),
    ("Assists", None),
])
def test_parse_stat(text, expected):
    assert lines._parse_stat(text) == expected


def test_name_matching():
    assert norm_player("KSCERATO") == norm_player("kscerato")
    assert norm_player("ZywOo") == "zywoo"
    assert team_similarity("Team Vitality", "Vitality") == 1.0
    assert team_similarity("FURIA Esports", "FURIA") == 1.0
    assert team_similarity("MOUZ", "NAVI") < 0.5


def _hist(n=30):
    rows = []
    t0 = pd.Timestamp("2026-07-01", tz="UTC")
    for g in range(n):
        for side, (tid, opp) in enumerate(((1, 2), (2, 1))):
            for p in range(5):
                pid = tid * 10 + p
                rows.append(dict(date=t0 + pd.Timedelta(hours=g), match_id=g, game_id=g, map_number=1,
                                 map_name="de_mirage", bo_type=3, tier="a", tournament_id=1, event_type="lan",
                                 team_id=tid, team=f"T{tid}", opp_id=opp, opponent=f"T{opp}", player_id=pid,
                                 player_slug=f"p{pid}", player=f"p{pid}", won=int(side == g % 2), rounds=20,
                                 team_rounds=13 if side == g % 2 else 7, opp_rounds=7 if side == g % 2 else 13,
                                 kills=10 + g + p, deaths=12, assists=3, headshots=5, first_kills=2, first_deaths=2,
                                 kast=0.7, rating=6.0, adr=70.0, team_odds=None, opp_odds=None))
    return pd.DataFrame(rows)


def test_features_have_no_lookahead():
    h = _hist()
    f = build_features(h)
    r = f[(f.player_id == 10)].reset_index(drop=True)
    # kills for player 10 on game g are 10+g; last-5 KPR on game 7 = mean(kills g2..g6)/20
    assert r.loc[7, "kpr_5"] == pytest.approx(np.mean([12, 13, 14, 15, 16]) / 20)
    assert np.isnan(r.loc[0, "kpr_5"])
    assert r.loc[7, "n_maps"] == 7
    # changing a future outcome must not change past features
    h2 = h.copy()
    h2.loc[h2.game_id == 20, "kills"] = 99
    f2 = build_features(h2)
    cols = ["kpr_5", "kpr_ewm", "opp_dpr", "team_elo"]
    a = f[f.game_id <= 20].sort_values(["game_id", "player_id"])[cols].values
    b = f2[f2.game_id <= 20].sort_values(["game_id", "player_id"])[cols].values
    np.testing.assert_allclose(a, b)


def test_future_rows_see_only_history():
    h = _hist()
    fut = h[h.game_id == 0].copy()
    fut[["kills", "headshots", "won", "rounds"]] = np.nan
    fut["game_id"] = -1
    f = build_features(h, fut)
    r = f[f.future & (f.player_id == 10)].iloc[0]
    assert r.n_maps == 30
    assert r.kpr_5 == pytest.approx(np.mean([35, 36, 37, 38, 39]) / 20)


def test_distribution_probabilities():
    sup, pr = np.array([20.0, 24.0, 30.0]), np.array([0.3, 0.5, 0.2])
    pmf = D.map_kills_pmf(0.7, 0.02, sup, pr, 0.55, 1.3)
    assert pmf.sum() == pytest.approx(1)
    exp_mean = 0.7 * (sup @ pr)
    assert D.mean(pmf) == pytest.approx(exp_mean, rel=0.01)
    assert D.p_over(pmf, 15.5) + D.p_under(pmf, 15.5) == pytest.approx(1)
    h = D.hs_pmf(pmf, 0.5, 0.03)
    assert D.mean(h) == pytest.approx(exp_mean * 0.5, rel=0.02)


def test_rounds_tilt_and_ot():
    tab = {0: {"20": 0.4, "24": 0.4, "28": 0.2}, 1: {"20": 0.4, "24": 0.4, "28": 0.2},
           2: {"20": 0.4, "24": 0.4, "28": 0.2}, 3: {"20": 0.4, "24": 0.4, "28": 0.2}}
    sup, pr = D.rounds_dist(tab, 0.9, ladder=[(21.5, 0.7)], p_ot=0.15)
    assert pr[sup > 24].sum() == pytest.approx(0.15)
    assert pr.sum() == pytest.approx(1)


def test_win_multipliers_unbiased():
    mw, ml = D.win_multipliers(0.6, 1.4)
    assert 0.6 * mw + 0.4 * ml == pytest.approx(1)
    assert mw / ml == pytest.approx(1.4)


def _spec(n_players=2):
    r = (np.array([18.0, 22, 26, 30]), np.array([0.25, 0.35, 0.3, 0.1]))
    ps = [PlayerSpec(f"p{i}", 1 if i % 2 == 0 else 2, {"a": 0.7, "b": 0.75}, {"a": 0.5, "b": 0.5}, 0.02, 0.03)
          for i in range(n_players)]
    return MatchSpec("m", 3, ["a", "b"], [0.5, 0.5], {1: 0.5, 2: 0.5, 3: 0.5}, {1: r, 2: r, 3: r}, 1.3, ps)


def test_simulation_correlation_and_map3():
    sim = simulate_match(_spec(), 50000, np.random.default_rng(0))
    assert sim["_played"][:, 2].mean() == pytest.approx(0.5, abs=0.02)
    k = leg_totals(sim, "p0", "kills", (1, 2))
    h = leg_totals(sim, "p0", "headshots", (1, 2))
    assert np.corrcoef(k, h)[0, 1] > 0.5           # kills vs HS same player
    k1 = leg_totals(sim, "p1", "kills", (1, 2))
    assert abs(np.corrcoef(k, k1)[0, 1]) < 0.3     # opponents: pace (+) vs winner (-) offset
    assert k.mean() == pytest.approx(0.7 * 2 * 23.2, rel=0.03)


def _cfg():
    return {"power_2": {"legs": 2, "type": "power", "payout": 3.0},
            "flex_5": {"legs": 5, "type": "flex", "payouts": {5: 10.0, 4: 2.0, 3: 0.4}},
            "power_6": {"legs": 6, "type": "power", "payout": 37.5}}


def test_payout_and_breakeven():
    c = _cfg()
    assert E.breakeven_per_leg(c["power_2"]) == pytest.approx(3 ** -0.5, 1e-6)
    assert E.breakeven_per_leg(c["power_6"]) == pytest.approx(37.5 ** (-1 / 6), 1e-6)
    v = E.payout_vector(c["flex_5"])
    assert list(v) == [0, 0, 0, 0.4, 2.0, 10.0]


def test_flex_ev_uses_exact_hit_distribution():
    c = _cfg()["flex_5"]
    d = E.iid_hit_dist(0.6, 5)
    assert d.sum() == pytest.approx(1)
    manual = 10 * 0.6**5 + 2 * 5 * 0.6**4 * 0.4 + 0.4 * 10 * 0.6**3 * 0.4**2 - 1
    assert E.ev_from_hits(d, c) == pytest.approx(manual)


def test_hit_counts_from_matrix_and_correlation_ratio():
    rng = np.random.default_rng(1)
    base = rng.random(50000) < 0.6
    H = np.column_stack([base, base, rng.random(50000) < 0.6])
    r = E.evaluate(H, (0, 1), _cfg()["power_2"])
    assert r["p_all"] == pytest.approx(0.6, abs=0.01)
    assert r["corr_ratio"] > 1.5
    r2 = E.evaluate(H, (0, 2), _cfg()["power_2"])
    assert r2["corr_ratio"] == pytest.approx(1, abs=0.05)
    assert r2["hit_dist"].sum() == pytest.approx(1)


def test_kelly_power():
    play = _cfg()["power_2"]
    p = 0.36  # P(all hit)
    f = E.kelly_fraction(np.array([0.64 * 0.5, 0.64 * 0.5, p]), play)
    assert f == pytest.approx((p * 3 - 1) / 2, abs=0.002)
    assert E.kelly_fraction(np.array([0.4, 0.4, 0.2]), play) == 0


def test_optimizer_prefers_uncorrelated_and_respects_match_cap():
    rng = np.random.default_rng(2)
    base = rng.random(50000) < 0.62
    H = np.column_stack([base, base, rng.random(50000) < 0.62, rng.random(50000) < 0.62])
    legs = [{"match": "m1", "player": "a", "stat": "kills"}, {"match": "m1", "player": "a", "stat": "kills"},
            {"match": "m2", "player": "b", "stat": "kills"}, {"match": "m3", "player": "c", "stat": "kills"}]
    res = E.optimize(H, legs, _cfg()["power_2"], top_n=3, max_per_match=1)
    assert all(not ({0, 1} <= set(r["legs"])) for r in res)


def test_source_failure_is_reported_not_raised(monkeypatch):
    def boom(*a, **k):
        raise http.AccessDenied("403 from example")
    monkeypatch.setattr(lines, "fetch_json", boom)
    got, errs = lines.fetch_all_lines()
    assert got == [] and len(errs) == 2 and "403" in errs[0]


def test_http_access_denied(monkeypatch):
    class R:
        status_code = 403
        headers = {}
    monkeypatch.setattr(http._session, "request", lambda *a, **k: R())
    with pytest.raises(http.AccessDenied):
        http.fetch_json("https://example.invalid/x", min_interval=0)


def test_platt_calibration_smooth_and_monotone():
    from cs2model.training.calibration import Platt
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 4000)
    y = rng.random(4000) < (0.5 + 0.8 * (p - 0.5))  # overconfident raw probabilities
    c = Platt().fit(p, y)
    grid = np.linspace(0.1, 0.9, 50)
    out = c.predict(grid)
    assert np.all(np.diff(out) > 0)  # strictly monotone, no isotonic steps
    assert out[-1] < 0.9 and out[0] > 0.1  # shrinks overconfidence
    assert len(np.unique(out.round(4))) == 50


def test_optimizer_one_leg_per_player():
    rng = np.random.default_rng(1)
    base = rng.random(20000)
    H = np.column_stack([base < 0.7, base < 0.68, rng.random(20000) < 0.6])  # legs 0/1 same player
    legs = [{"match": "m1", "player": "p1", "stat": "kills"}, {"match": "m1", "player": "p1", "stat": "headshots"},
            {"match": "m2", "player": "p2", "stat": "kills"}]
    for r in E.optimize(H, legs, _cfg()["power_2"], top_n=5):
        assert not {0, 1} <= set(r["legs"])


def test_push_conditional_side_probability():
    # whole-number line: P(over | no push) must exceed raw P(over) when pushes are possible
    tot = np.array([13] * 30 + [14] * 20 + [15] * 50)
    p_over, p_push = (tot > 14).mean(), (tot == 14).mean()
    assert p_over == 0.5 and p_push == 0.2
    assert abs(p_over / (1 - p_push) - 0.625) < 1e-12


def test_line_board_diff():
    from cs2model.pipeline import diff_sig
    old = {("underdog", "a", "kills", (1, 2), "standard"): 30.5, ("underdog", "b", "kills", (1, 2), "standard"): 28.5}
    new = {("underdog", "a", "kills", (1, 2), "standard"): 31.5, ("prizepicks", "c", "kills", (1,), "standard"): 15.0}
    d = diff_sig(old, new)
    assert len(d["added"]) == 1 and len(d["moved"]) == 1 and len(d["removed"]) == 1


def test_skip_record_and_reconcile():
    import pandas as pd
    from cs2model.inference.predict import SKIP_COLS, skip_record
    from cs2model.pipeline import reconcile
    from cs2model.types import PropLine
    ln = PropLine("prizepicks", "x", "A", "B", "kills", (4,), 15.0)
    sk = pd.DataFrame([skip_record(ln, "map 4/5 lines not modelled"),
                       skip_record(ln, "not enough tier-1/2 maps: B")], columns=SKIP_COLS)
    assert {"platform", "stat", "line", "maps"} <= set(sk.columns) and sk.loc[0, "maps_label"] == "Map 4"
    slate = pd.DataFrame({"source": ["underdog"], "match": ["1-2"]})
    r = reconcile(3, slate, sk)
    assert r["balanced"] and r["by_reason"]["not enough tier-1/2 maps"] == 1
    assert r["per_platform"]["underdog"] == {"lines": 1, "priced_props": 1, "priced_matches": 1}


def _toy_spec(bo=3):
    from cs2model.inference.simulate import MatchSpec, PlayerSpec
    maps = ["de_ancient", "de_anubis", "de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_train"]
    r = np.random.default_rng(0)
    sup = np.arange(16, 31).astype(float)
    pr = r.random(len(sup))
    pr /= pr.sum()
    ps = [PlayerSpec(f"p{i}", 1 + i % 2, dict(zip(maps, r.uniform(.55, .85, 7))), dict(zip(maps, r.uniform(.35, .6, 7))), .024, .014)
          for i in range(2)]
    return MatchSpec("x", bo, maps, list(r.dirichlet(np.ones(7) * 3)), {1: .6, 2: .55, 3: .58}, {1: (sup, pr)}, 1.27, ps)


def test_analytic_leg_matches_monte_carlo():
    from cs2model.inference import analytic as A, distribution as D
    from cs2model.inference.simulate import leg_totals, simulate_match
    spec = _toy_spec()
    sim = simulate_match(spec, 200000, np.random.default_rng(1))
    for stat, mp in (("kills", (1, 2)), ("headshots", (1,)), ("kills", (1, 2, 3)), ("kills", (3,))):
        tot = leg_totals(sim, "p0", stat, mp)
        pmf = A.leg_pmf(spec.players[0], spec, stat, mp)
        assert abs(tot.mean() - D.mean(pmf)) < 0.1
        for line in (np.floor(tot.mean()) - 2.5, np.floor(tot.mean()) + 0.5, np.floor(tot.mean()) + 3.5):
            assert abs((tot > line).mean() - D.p_over(pmf, line)) < 0.006


def test_sim_seed_and_append_topup():
    from cs2model.inference.predict import _append_sim, _sim_job
    spec = _toy_spec()
    a = _sim_job({"spec": spec, "n": 2000, "seed": 11, "state": None})
    b = _sim_job({"spec": spec, "n": 2000, "seed": 11, "state": None})
    assert np.array_equal(a["p0"]["kills"], b["p0"]["kills"])  # fixed per-match seed
    top = _append_sim(a, _sim_job({"spec": spec, "n": 3000, "seed": 11, "state": a["_state"]}))
    assert top["_n"] == 5000 and top["p0"]["kills"].shape[0] == 5000
    assert np.array_equal(top["p0"]["kills"][:2000], a["p0"]["kills"])  # earlier draws kept, not redrawn
    assert not np.array_equal(top["p0"]["kills"][2000:4000], a["p0"]["kills"])  # new draws continue the stream


def test_portrait_frame_falls_back_without_face():
    from cs2model.scraper.portraits import DEFAULT, frame
    assert frame(np.zeros((455, 354, 4), np.uint8)) == DEFAULT
    blob = np.zeros((455, 354, 4), np.uint8)
    blob[40:300, 120:240] = 200
    assert frame(blob) == DEFAULT


def test_alt_line_mode():
    from cs2model.inference.predict import alt_line_mode
    from cs2model.types import PropLine
    pp = {"prizepicks": {"alt_lines": {"demon": "standard", "goblin": "exclude"}}}
    ln = PropLine(source="prizepicks", player="x", team="a", opponent="b", stat="kills", maps=(1,), line=14.5,
                  odds_type="demon", over_only=True)
    assert alt_line_mode(ln, pp) == "standard"
    assert alt_line_mode(PropLine(**{**ln.__dict__, "odds_type": "goblin"}), pp) == "exclude"
    assert alt_line_mode(PropLine(**{**ln.__dict__, "odds_type": "goblin"}), {"prizepicks": {"alt_lines": {"goblin": 1.6}}}) == 1.6
    assert alt_line_mode(ln, {"prizepicks": {"exclude_odds_types": ["demon"]}}) == "exclude"
    assert alt_line_mode(ln, {"prizepicks": {}}) == "exclude"
    assert alt_line_mode(PropLine(**{**ln.__dict__, "odds_type": "standard"}), pp) == "standard"


def test_map_pool_helpers():
    from cs2model.maps import pool as P
    for bo in (1, 3, 5):
        assert P.map_p_from_series(P.series_p(0.62, bo), bo) == pytest.approx(0.62, abs=1e-6)
    w, dd = np.array([0.4, 0.35, 0.25]), np.array([0.6, -0.2, 0.1])
    p = P.anchored(0.57, dd, w)
    assert np.dot(w, p) == pytest.approx(0.57) and p[0] > p[2] > p[1]   # re-centred on the base price
    assert np.allclose(P.anchored(0.57, np.zeros(3), w), 0.57)
    r = {"pool": ["a", "b", "c"], "s1": np.array([.6, .4, 0]), "s2": np.array([.5, .5, 0]), "R1": np.zeros(3),
         "N1": np.zeros(3), "R2": np.zeros(3), "N2": np.zeros(3), "f1": np.array([.6, .4, 0]), "f2": np.array([.5, .5, 0])}
    wt = P.pick_weights(r, [1.0, 0, 0, 0], 16.0)
    assert wt.sum() == pytest.approx(1) and wt[2] < wt[1] < wt[0]  # a map neither team plays is rarely picked
    assert P.p_played12(wt, 3).sum() == pytest.approx(2)


def test_map_state_is_leak_free():
    from cs2model.maps import pool as P
    rows = []
    for i, (d, mp, w) in enumerate([("2026-07-01", "de_nuke", 1), ("2026-07-02", "de_mirage", 0), ("2026-07-03", "de_nuke", 1)]):
        for tid, opp, won in ((1, 2, w), (2, 1, 1 - w)):
            rows.append({"date": d, "match_id": i, "game_id": i, "map_number": 1, "map_name": mp, "bo_type": 1,
                         "team_id": tid, "opp_id": opp, "won": won, "team_odds": np.nan, "opp_odds": np.nan})
    st = P.MapState(pd.DataFrame(rows), 30.0)
    r0, r2 = st.records[0], st.records[2]
    assert r0["N1"].sum() == 0 and r0["p_elo"] == 0.5          # first match sees no history
    assert r2["f1"][r2["pool"].index("de_nuke")] == pytest.approx(0.5)  # third sees exactly the two before it
    fast = P.MapState(pd.DataFrame(rows), 30.0, records=False)
    assert fast.elo == st.elo


def test_per_map_win_probs_analytic_matches_monte_carlo():
    from cs2model.inference import analytic as A, distribution as D
    from cs2model.inference.simulate import leg_totals, simulate_match
    spec = _toy_spec()
    base = A.leg_pmf(spec.players[0], spec, "kills", (1, 2, 3))
    flat = dataclasses_replace(spec, p_win_map={j: [spec.p_map_team1[j]] * 7 for j in (1, 2, 3)})
    assert np.allclose(A.leg_pmf(flat.players[0], flat, "kills", (1, 2, 3)), base)  # constant per-map = old behaviour
    pw = {j: list(np.linspace(0.3, 0.8, 7)) for j in (1, 2, 3)}
    spec = dataclasses_replace(spec, p_win_map=pw)
    sim = simulate_match(spec, 200000, np.random.default_rng(3))
    assert sim["_played"][:, 2].mean() == pytest.approx(A.p_map3(spec), abs=0.005)
    for stat, mp in (("kills", (1, 2)), ("kills", (1, 2, 3)), ("headshots", (3,))):
        tot = leg_totals(sim, "p1", stat, mp)
        pmf = A.leg_pmf(spec.players[1], spec, stat, mp)
        assert abs(tot.mean() - D.mean(pmf)) < 0.1
        line = np.floor(tot.mean()) + 0.5
        assert abs((tot > line).mean() - D.p_over(pmf, line)) < 0.006


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def test_preds_log_grade_and_bias_report(tmp_path):
    from cs2model.tracking import store
    con = store.connect(tmp_path / "t.db")
    st = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)
    slate = pd.DataFrame([{"source": "underdog", "player": "x", "team": "A", "opponent": "B", "stat": "kills", "maps": [1, 2],
                           "start_time": st, "odds_type": "standard", "line": 30.5, "model_mean": 33.0, "p_over": 0.6}])
    store.log_preds(slate, con)
    store.log_preds(slate.assign(line=31.5), con)  # re-pricing keeps only the latest pre-start price
    assert con.execute("SELECT count(*), max(line) FROM preds").fetchone() == (1, 31.5)
    con.execute("UPDATE preds SET actual=29")
    br = store.bias_report(con)
    assert br.loc[0, "scope"] == "Maps 1+2" and br.loc[0, "model_minus_actual"] == pytest.approx(4.0)


def test_polymarket_event_parsing():
    from cs2model.scraper.books import polymarket_match

    def mk(kind, q, outs, bid, ask):
        return {"sportsMarketType": kind, "question": q, "outcomes": json.dumps(outs), "bestBid": bid, "bestAsk": ask}

    e = {"title": "Counter-Strike: A vs B (BO3) - Cup", "startTime": "2026-10-19T18:00:00Z", "markets": [
        mk("moneyline", "A vs B", ["A", "B"], 0.61, 0.62),
        mk("child_moneyline", "A vs B - Map 1 Winner", ["B", "A"], 0.40, 0.44),
        mk("totals", "Games Total: O/U 2.5", ["Over", "Under"], 0.43, 0.50),
        mk("round_over_under_game_1", "Map 1 Total Rounds: Over/Under 21.5", ["Over", "Under"], 0.08, 0.92),
        mk("round_over_under_game_2", "Map 2 Total Rounds: Over/Under 21.5", ["Over", "Under"], 0.36, 0.40),
    ]}
    mo = polymarket_match(e)
    assert (mo.source, mo.team1, mo.team2, mo.best_of) == ("polymarket", "A", "B", 3)
    assert mo.p_series_team1 == pytest.approx(0.615)
    assert mo.p_map_team1[1] == pytest.approx(0.58)
    assert 3 in mo.p_map_team1
    assert mo.p_map3 == pytest.approx(0.465)
    assert mo.rounds_ladder == {2: [(21.5, pytest.approx(0.38))]}
    e["markets"][0]["bestAsk"] = 0.9
    assert polymarket_match({**e, "markets": e["markets"][:1]}) is None


def test_find_odds_book_priority():
    from cs2model.inference.predict import _find_odds
    from cs2model.types import MatchOdds
    bk = [MatchOdds(src, "Team Liquid", "fnatic", None, 3) for src in ("thunderpick", "polymarket", "pinnacle")]
    assert _find_odds(bk, "Liquid", "fnatic").source == "pinnacle"
    assert _find_odds(bk[:2], "fnatic", "Liquid").source == "polymarket"
    assert _find_odds(bk[:1], "Liquid", "fnatic").source == "thunderpick"
    assert _find_odds(bk, "NAVI", "fnatic") is None


def test_line_probs_push_and_edge():
    import numpy as np
    from cs2model.inference.predict import line_probs
    pmf = np.zeros(41)
    pmf[18:23] = 0.2  # uniform on 18..22
    a = line_probs(pmf, 20.5, be_over=0.5, be_under=0.5)
    assert abs(a["p_over"] - 0.4) < 1e-9 and abs(a["p_under"] - 0.6) < 1e-9 and a["p_push"] == 0
    b = line_probs(pmf, 20, be_over=0.5, be_under=0.5)  # whole-number line: push voids the leg
    assert abs(b["p_push"] - 0.2) < 1e-9 and abs(b["edge_over"]) < 1e-9 and abs(b["edge_under"]) < 1e-9
    assert line_probs(pmf, 25.5)["p_over"] == 0


def test_series_win():
    from cs2model.inference.predict import series_win
    assert abs(series_win({1: 0.5, 2: 0.5, 3: 0.5}, 3) - 0.5) < 1e-12
    assert abs(series_win({1: 0.6}, 1) - 0.6) < 1e-12
    assert abs(series_win({1: 0.6, 2: 0.6, 3: 0.6}, 3) - 0.648) < 1e-12
    assert series_win({1: 0.7, 2: 0.7, 3: 0.7}, 5) > 0.7


def test_add_match_meta_unknown_and_mapping():
    import pandas as pd
    from datetime import datetime, timezone
    from cs2model.pipeline import add_match_meta
    s = pd.DataFrame({"team_id": [1, 2, 5], "opp_id": [2, 1, 6], "odds_source": ["polymarket", "polymarket", "elo"]})
    meta = {frozenset((1, 2)): {"event": "Major", "match_tier": "s", "event_type": "lan"}}
    t = datetime(2026, 10, 19, tzinfo=timezone.utc)
    out = add_match_meta(s, t, meta)
    assert list(out.event) == ["Major", "Major", None] and list(out.event_type) == ["lan", "lan", None]
    assert out.odds_at[0] == t and pd.isna(out.odds_at[2])


def test_awp_share_features_leak_free():
    import numpy as np
    import pandas as pd
    from cs2model.features.build import attach_awp, _awp_features
    h = pd.DataFrame({"game_id": [1, 2, 3, 4], "player_id": [7, 7, 7, 8]})
    awp = pd.DataFrame({"game_id": [1, 2, 4], "player_id": [7, 7, 8], "weapon_kills": [20, 10, 15],
                        "awp_kills": [10, 5, 0], "awp_hs": [1, 0, 0]})
    out = _awp_features(attach_awp(h, awp))
    assert np.isnan(out.awp_share[0]) and np.isnan(out.awp_share[3])
    assert abs(out.awp_share[1] - (10 + 30 * 0.12) / (20 + 30)) < 1e-12
    assert abs(out.awp_share[2] - (15 + 30 * 0.12) / (30 + 30)) < 1e-12
    assert out.awp_maps[2] == 2 and np.isnan(out.awp_kills[2])


def test_game_weapons_parse(monkeypatch):
    from cs2model.scraper import bo3_history as b
    rows = [{"steam_profile": {"player": {"id": 5}, "player_weapons_stats": [
        {"weapon_slug": "awp", "kills_sum": 7, "headshots_sum": 1},
        {"weapon_slug": "ak-47", "kills_sum": 9, "headshots_sum": 6}]}},
        {"steam_profile": {"player": {"id": 6}}, "player_weapons_stats": [
            {"weapon_slug": "m4a1", "kills_sum": 4, "headshots_sum": 2}]}]
    monkeypatch.setattr(b, "fetch_json", lambda *a, **k: rows)
    assert b.game_weapons(1) == {5: (16, 7, 1), 6: (4, 0, 0)}


def test_awp_role_limited_samples():
    from cs2model.features.build import awp_role
    assert awp_role(40, 100, 4) == (None, 0.4)
    assert awp_role(0, 0, 10) == (None, 0.0)
    assert awp_role(30, 100, 5)[0] == "AWPer"
    assert awp_role(29, 100, 20)[0] == "Hybrid"
    assert awp_role(10, 100, 20)[0] == "Hybrid"
    assert awp_role(9, 100, 20)[0] == "Rifler"


def test_crawl_reports_progress(monkeypatch):
    from datetime import date
    from cs2model.scraper import bo3_history as b
    monkeypatch.setattr(b, "finished_matches", lambda day: ([], {}))
    seen = []
    b.crawl(2, end=date(2026, 1, 3), write=False, progress=lambda d, n, r: seen.append((d, n, r)))
    assert seen == [(1, 3, 0), (2, 3, 0), (3, 3, 0)]


def test_hit_matrix_mixed_sim_counts():
    import numpy as np
    from cs2model.pipeline import hit_matrix
    d = {1: {"hit": np.ones(50, bool)}, 2: {"hit": np.zeros(20, bool)}, 3: {"hit": np.arange(50) % 2 == 0}}
    H = hit_matrix(d, [1, 2, 3])
    assert H.shape == (20, 3)
    assert (H[:, 2] == (np.arange(20) % 2 == 0)).all()


def test_pending_picks_status(tmp_path):
    import pandas as pd
    from cs2model.tracking import store
    con = store.connect(tmp_path / "t.db")
    now = pd.Timestamp.now(tz="UTC")
    base = dict(source="prizepicks", team="A", opponent="B", stat="kills", maps="1+2", line=30.5, side="over", p_side=0.6,
                breakeven=0.55, edge=0.05, ev=0.1, tier="B", model_mean=31.0)
    store.log_picks(pd.DataFrame([dict(base, player="x", start_time=str(now + pd.Timedelta(hours=2))),
                                  dict(base, player="y", start_time=str(now - pd.Timedelta(hours=5)))]), con=con)
    p = store.pending(con)
    assert dict(zip(p.player, p.status)) == {"x": "upcoming", "y": "awaiting results"}


def test_team_map_counts_follow_a_remade_roster():
    from cs2model.inference.predict import team_map_counts
    h = _hist(20)
    new = h[(h.game_id == 19) & (h.team_id == 1)].assign(game_id=99, match_id=99, team_id=3, team="T3",
                                                         date=h.date.max() + pd.Timedelta(hours=1))
    fresh = new.assign(game_id=98, match_id=98, team_id=4, team="T4", player_id=new.player_id + 100)
    c = team_map_counts(pd.concat([h, new, fresh]), ("s", "a", "b"), 92)
    assert c[2] == 20
    assert c[3] == 21  # its players' 20 maps on T1 + 1 as T3, not just the team's own 1
    assert c[4] == 1   # new players with no history keep the team's own count


def _rule_row(**kw):
    r = dict(source="underdog", player="x", start_time=None, stat="kills", maps=[1, 2], line=30.5, side="over",
             model_mean=33.0, edge=0.08, p_side=0.61, breakeven=0.53, n_maps=60, tenure=40, role_shift=0.0,
             odds_source="pinnacle", kpr_10=0.8, hs_pct_10=0.5, opp_dpr=0.7, p_map_win=0.5, p_ot=0.1, p_map3=0.0,
             team_maps=30, t3=False, t3_validated=False)
    r.update(kw)
    return r


def test_tier3_rules_cap_confidence_and_mark_unvalidated_as_experimental():
    from cs2model.inference.predict import apply_rules
    st = {"min_maps": 15, "standin_tenure": 3, "role_shift_flag": 0.18, "max_line_gap": 0.35, "line_move_flag": 1.5,
          "min_edge": 0.03, "min_team_maps": 12, "tier3_min_edge": 0.06, "tier3_max_conf": "B"}
    s = apply_rules(pd.DataFrame([_rule_row(), _rule_row(t3=True, t3_validated=True, team_maps=0),
                                  _rule_row(t3=True, t3_validated=True, team_maps=0, edge=0.05),
                                  _rule_row(t3=True, t3_validated=False, team_maps=0, edge=0.2)]), st)
    assert list(s.tier) == ["A", "B", "-", "-"]
    assert s.why[1].startswith("T3 · ") and "capped at B" in s.why[1]
    assert "No bet" in s.why[2] and "6%" in s.why[2]
    assert s.why[3].startswith("T3 experimental") and not s.plus_ev[3]


def test_t3_picks_are_tracked_separately(tmp_path):
    from cs2model.tracking import store
    con = store.connect(tmp_path / "t.db")
    base = dict(source="prizepicks", team="A", opponent="B", stat="kills", maps="1+2", line=30.5, side="over", p_side=0.6,
                breakeven=0.5, edge=0.1, ev=0.2, tier="B", model_mean=31.0, start_time="2026-01-01T00:00:00+00:00")
    store.log_picks(pd.DataFrame([dict(base, player="x"), dict(base, player="y", t3=True), dict(base, player="z", t3=True)]), con)
    con.execute("UPDATE picks SET outcome = CASE player WHEN 'z' THEN 'loss' ELSE 'win' END")
    g = {r["group"]: r for r in store.performance(con)[0]["by_group"]}
    assert g["Tier 1/2"]["picks"] == 1 and g["Tier 3"]["picks"] == 2 and g["Tier 3"]["flat_roi"] == pytest.approx(0.0)
    con.close()
    assert store.connect(tmp_path / "t.db").execute("SELECT count(*) FROM picks WHERE t3=1").fetchone()[0] == 2  # reopen keeps column


def test_tier3_calibrator_only_validates_when_it_beats_the_current_one():
    from cs2model.training import tier3
    from cs2model.training.calibration import Platt
    rng = np.random.default_rng(0)
    n = 4000
    p_raw = rng.uniform(0.2, 0.8, n)
    y = (rng.random(n) < 0.5 + 0.4 * (p_raw - 0.5)).astype(int)  # raw probabilities are overconfident
    groups = np.arange(n) // 2
    good = Platt().fit(p_raw[:2000], y[:2000])
    ident = Platt().fit(p_raw, (rng.random(n) < p_raw).astype(int))  # stand-in "current" calibrator ~ identity
    share = tier3._boot_share(ident.predict(p_raw[2000:]), good.predict(p_raw[2000:]), y[2000:], groups[2000:])
    assert share > 0.9
    assert tier3.ece(good.predict(p_raw[2000:]), y[2000:]) < tier3.ece(ident.predict(p_raw[2000:]), y[2000:])
    gap, k = tier3.strong_gap(ident.predict(p_raw[2000:]), y[2000:])
    assert gap > 0.03 and k > 0


def _t3_slate(rows):
    st = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=3)
    return pd.DataFrame([dict(source="underdog", player=f"p{i}", team="a", opponent="b", stat=stat, maps=maps,
                              start_time=st, odds_type="standard", line=line, model_mean=line, p_over=p, t3=t3, p_over_raw=p)
                         for i, (stat, maps, line, p, t3) in enumerate(rows)])


def test_t3_preds_logged_with_flag_and_old_preds_table_migrates(tmp_path):
    import sqlite3
    db = tmp_path / "t.db"
    old = sqlite3.connect(db)
    old.execute("CREATE TABLE preds (source TEXT, player TEXT, team TEXT, opponent TEXT, stat TEXT, maps TEXT, start_time TEXT, "
                "ts TEXT, line REAL, model_mean REAL, p_over REAL, actual REAL, PRIMARY KEY(source, player, stat, maps, start_time))")
    old.commit()
    old.close()
    con = store.connect(db)
    store.log_preds(_t3_slate([("kills", (1, 2), 30.5, 0.66, True), ("kills", (1, 2), 30.5, 0.55, False)]), con)
    got = dict(con.execute("SELECT player, t3 FROM preds").fetchall())
    assert got == {"p0": 1, "p1": 0}


def test_t3_reliability_buckets_by_favoured_side_and_market(tmp_path):
    con = store.connect(tmp_path / "t.db")
    rows = [("kills", (1, 2), 30.5, 0.62, True)] * 4 + [("kills", (1, 2), 30.5, 0.30, True)] * 2 \
        + [("headshots", (1,), 8.5, 0.52, True), ("kills", (1, 2), 30.5, 0.70, False), ("kills", (1,), 15.5, 0.62, True)]
    store.log_preds(_t3_slate(rows), con)
    acts = [31, 31, 31, 20, 20, 40, 9, 40, 15.5]  # last one is a push (dropped)
    for i, a in enumerate(acts):
        con.execute("UPDATE preds SET actual=? WHERE player=?", (a, f"p{i}"))
    con.commit()
    rel = store.t3_reliability(con).set_index(["market", "p_range"])
    k = rel.loc[("Kills · Maps 1+2", "60–65%")]          # 4 overs at 62%: 3 hit
    assert k.lines == 4 and abs(k.avg_p - 0.62) < 1e-9 and abs(k.hit_rate - 0.75) < 1e-9 and abs(k.gap + 0.13) < 1e-9
    u = rel.loc[("Kills · Maps 1+2", "70–75%")]          # unders at P(over)=30%: 1 of 2 hit
    assert u.lines == 2 and abs(u.avg_p - 0.70) < 1e-9 and u.hit_rate == 0.5
    assert rel.loc[("Kills · Maps 1+2", "60–75% (combined)")].lines == 6
    assert rel.loc[("Headshots · Map 1", "50–55%")].hit_rate == 1.0
    assert rel.loc[("All tier-3 lines", "60–75% (combined)")].lines == 6  # tier-1/2 row and the push are excluded
    assert store.t3_reliability(store.connect(tmp_path / "empty.db")).empty


def test_bias_report_excludes_tier3_preds(tmp_path):
    con = store.connect(tmp_path / "t.db")
    store.log_preds(_t3_slate([("kills", (1,), 15.5, 0.6, True), ("kills", (1,), 15.5, 0.6, False)]), con)
    con.execute("UPDATE preds SET actual=20")
    con.commit()
    assert store.bias_report(con).lines.sum() == 1
