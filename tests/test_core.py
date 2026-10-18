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
