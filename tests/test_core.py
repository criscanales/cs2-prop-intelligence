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
