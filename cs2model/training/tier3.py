"""Tier-3 (bo3.gg tier C) probability calibration, validated chronologically before it is used.

The main models train on tier-1/2 maps only. For each market (kills/headshots x single map/Maps 1+2) this fits a
Platt calibrator on tier-3 maps in the validation window and scores it on tier-3 maps in the later test window
against the existing tier-1/2 calibrator. A market is `validated` only if the tier-3 calibrator wins on log loss
and calibration error and the log-loss gain holds in >= 90% of match-clustered bootstrap resamples.

python -m cs2model.training.tier3   # refit against the current models without retraining them
"""
import json
import logging

import numpy as np
import pandas as pd

from ..backtest import metrics as M
from ..inference import distribution as D
from .calibration import Platt

T3_TIERS = ("c",)
MARKETS = (("kills", "map"), ("kills", "series"), ("headshots", "map"), ("headshots", "series"))
MIN_ROWS = 300
BOOT_SHARE = 0.9
log = logging.getLogger(__name__)


def ece(p, y, bins=10):
    t = M.calibration_table(np.asarray(p), np.asarray(y), bins)
    return float((t.n * (t.pred - t.obs).abs()).sum() / t.n.sum()) if len(t) else float("nan")


def strong_gap(p, y, cut=0.6):
    """Model P(side) minus hit rate on the side it favours, over picks with P(side) >= cut (overconfidence > 0)."""
    p, y = np.asarray(p), np.asarray(y)
    ps, ys = np.maximum(p, 1 - p), np.where(p >= 0.5, y, 1 - y)
    m = ps >= cut
    return (float(ps[m].mean() - ys[m].mean()), int(m.sum())) if m.any() else (float("nan"), 0)


def market_rows(df, rates, art, row_pmfs, series_pairs):
    """Raw (pre-calibration) P(over) on synthetic lines, outcome and match id per market."""
    df = df.reset_index(drop=True)
    kp, hp = rates(df)
    K, H = row_pmfs(df, kp, hp, art)
    nk, nh = df.kpr_10.fillna(df.kpr_shrunk).values, df.hs_pct_10.fillna(df.hs_pct_shrunk).values
    er = df.exp_rounds.fillna(df.rounds_20).fillna(22).values
    lines = {"kills": M.synthetic_line(nk * er), "headshots": M.synthetic_line(nk * er * nh)}
    j1, j2 = series_pairs(df) if len(df) else (np.array([], int), np.array([], int))
    out = {}
    for stat, P in (("kills", K), ("headshots", H)):
        L, y = lines[stat], df[stat].values
        out[(stat, "map")] = (M.p_over_rows(P, L), (y > L).astype(int), df.match_id.values)
        if len(j1):
            P12 = np.array([D.convolve(P[a], P[b]) for a, b in zip(j1, j2)])
            L12 = np.floor(L[j1] + L[j2]) + 0.5
            out[(stat, "series")] = (M.p_over_rows(P12, L12), (y[j1] + y[j2] > L12).astype(int), df.match_id.values[j1])
        else:
            out[(stat, "series")] = (np.array([]), np.array([], int), np.array([]))
    return out


def _boot_share(p_gen, p_t3, y, groups, n=1000, seed=7):
    """Share of match-clustered resamples where the tier-3 calibrator has the lower log loss."""
    eps = 1e-6
    def ll(p):
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = pd.DataFrame({"g": groups, "d": ll(p_gen) - ll(p_t3)}).groupby("g").d.agg(["sum", "size"])
    s, c = d["sum"].values, d["size"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    return float((s[idx].sum(1) / c[idx].sum(1) > 0).mean())


def fit(feat_all, rates, art, calibrators, split_dates, breakeven, min_edge, row_pmfs, series_pairs, min_history=3):
    """Returns ({'t3_<stat>_<scope>': Platt}, report). Fit on the validation window, scored on the test window."""
    f = feat_all[feat_all.tier.isin(T3_TIERS) & (feat_all.n_maps >= min_history) & feat_all.kills.notna()
                 & feat_all.headshots.notna() & (feat_all.rounds > 0)]
    win = {s: (pd.Timestamp(a), pd.Timestamp(b)) for s, (a, b) in split_dates.items() if s in ("val", "test")}
    va = f[(f.date >= win["val"][0]) & (f.date <= win["val"][1])]
    te = f[(f.date >= win["test"][0]) & (f.date <= win["test"][1])]
    rv, rt = market_rows(va, rates, art, row_pmfs, series_pairs), market_rows(te, rates, art, row_pmfs, series_pairs)
    cals, rep = {}, {"tiers": list(T3_TIERS), "windows": {k: [str(a), str(b)] for k, (a, b) in win.items()},
                     "min_edge": min_edge, "markets": {}}
    for stat, scope in MARKETS:
        key = f"{stat}_{scope}"
        pv, yv, _ = rv[(stat, scope)]
        pt, yt, gt = rt[(stat, scope)]
        r = {"n_fit": int(len(yv)), "n_test": int(len(yt)), "validated": False}
        if len(yv) < MIN_ROWS or len(yt) < MIN_ROWS:
            r["reason"] = f"fewer than {MIN_ROWS} tier-3 rows to fit or test"
            rep["markets"][key] = r
            continue
        c = Platt().fit(pv, yv)
        gen = calibrators.get(key) or calibrators[f"{stat}_map"]
        p_gen, p_t3 = gen.predict(pt), c.predict(pt)
        for name, p in (("current", p_gen), ("tier3", p_t3)):
            gap, n_strong = strong_gap(p, yt)
            b = M.betting(p, yt, breakeven, min_edge)
            r[name] = {"brier": M.brier(p, yt), "logloss": M.logloss(p, yt), "ece": ece(p, yt), "strong_gap": gap,
                       "n_strong": n_strong, "picks": b["n_picks"], "hit_rate": b["hit_rate"], "roi": b["roi"],
                       "calibration": M.calibration_table(p, yt).to_dict("records")}
        r["params"] = c.to_params()
        r["boot_share"] = _boot_share(p_gen, p_t3, yt, gt)
        r["validated"] = bool(r["tier3"]["logloss"] < r["current"]["logloss"] and r["tier3"]["ece"] <= r["current"]["ece"]
                              and r["boot_share"] >= BOOT_SHARE)
        if not r["validated"]:
            r["reason"] = "tier-3 calibrator did not beat the current one on held-out tier-3 maps"
        cals[f"t3_{key}"] = c
        rep["markets"][key] = r
    return cals, rep


def summary(rep):
    """artifacts.json entry the predictor reads: which markets have a validated tier-3 calibrator."""
    return {"validated": {k: bool(v["validated"]) for k, v in rep["markets"].items()}, "windows": rep["windows"]}


def main():
    import joblib
    import lightgbm as lgb
    from ..features.build import build_features
    from ..parlay.engine import breakeven_per_leg, load_yaml
    from ..scraper.bo3_history import HIST_PATH
    from .train import MIN_HISTORY, MODEL_DIR, rate, row_pmfs, series_pairs, shrunk_rate

    art = json.loads((MODEL_DIR / "artifacts.json").read_text())
    bt = json.loads((MODEL_DIR / "backtest.json").read_text())
    cal = joblib.load(MODEL_DIR / "calibrators.joblib")
    km, hm = lgb.Booster(model_file=str(MODEL_DIR / "kills_lgbm.txt")), lgb.Booster(model_file=str(MODEL_DIR / "hs_lgbm.txt"))
    settings = load_yaml("settings.yaml")
    be = breakeven_per_leg(load_yaml("payouts.yaml")["prizepicks"]["plays"]["power_2"])

    def rates(df):
        return (shrunk_rate(rate(km, df), df.kpr_shrunk.values, art["shrink"]["kills"]),
                np.clip(shrunk_rate(rate(hm, df), df.hs_pct_shrunk.values, art["shrink"]["headshots"]), 0.05, 0.95))

    feat = build_features(pd.read_csv(HIST_PATH, low_memory=False))
    t3, rep = fit(feat, rates, art, cal, bt["split_dates"], be, float(settings.get("tier3_min_edge", 0.06)),
                  row_pmfs, series_pairs, MIN_HISTORY)
    cal = {k: v for k, v in cal.items() if not k.startswith("t3_")} | t3
    joblib.dump(cal, MODEL_DIR / "calibrators.joblib")
    bt["tier3"] = rep
    (MODEL_DIR / "backtest.json").write_text(json.dumps(bt, indent=1, default=float))
    art["tier3"] = summary(rep)
    (MODEL_DIR / "artifacts.json").write_text(json.dumps(art, indent=1))
    return rep


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rep = main()
    for k, r in rep["markets"].items():
        if "current" not in r:
            print(k, r)
            continue
        c, t = r["current"], r["tier3"]
        print(f"{k:18s} fit={r['n_fit']} test={r['n_test']} logloss {c['logloss']:.4f}->{t['logloss']:.4f} "
              f"brier {c['brier']:.4f}->{t['brier']:.4f} ece {c['ece']:.4f}->{t['ece']:.4f} "
              f"strong_gap {c['strong_gap']:+.3f}->{t['strong_gap']:+.3f} roi {c['roi']:+.3f}({c['picks']})->{t['roi']:+.3f}({t['picks']}) "
              f"boot={r['boot_share']:.2f} validated={r['validated']}")
