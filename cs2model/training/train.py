"""Train kill-rate / HS-rate models with a strict time split, fit dispersion + calibration,
and backtest against a negative-binomial GLM and a naive rolling-average baseline.

python -m cs2model.training.train
"""
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

from ..backtest import metrics as M
from ..features.build import FEATURES, build_features
from ..inference import distribution as D
from ..parlay.engine import breakeven_per_leg, load_yaml
from ..scraper.bo3_history import HIST_PATH
from ..scraper.http import ROOT
from .calibration import Platt

MODEL_DIR = ROOT / "models"
MIN_HISTORY = 3
PARAMS = dict(objective="poisson", learning_rate=0.03, num_leaves=31, min_data_in_leaf=80,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, seed=7)
NB_COLS = ["kpr_shrunk", "map_kpr", "opp_dpr", "elo_diff", "kpr_ewm"]
log = logging.getLogger(__name__)


def time_split(df, val_frac=0.15, test_frac=0.15):
    games = df.drop_duplicates("game_id")[["game_id", "date"]].sort_values("date")
    n = len(games)
    a, b = int(n * (1 - val_frac - test_frac)), int(n * (1 - test_frac))
    tr, va = set(games.game_id.iloc[:a]), set(games.game_id.iloc[a:b])
    return np.where(df.game_id.isin(tr), "train", np.where(df.game_id.isin(va), "val", "test"))


def _fit_lgb(tr, va, target, offset_col):
    dtr = lgb.Dataset(tr[FEATURES], tr[target], init_score=np.log(tr[offset_col]))
    dva = lgb.Dataset(va[FEATURES], va[target], init_score=np.log(va[offset_col]), reference=dtr)
    return lgb.train(PARAMS, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


def rate(model, X):
    return np.exp(model.predict(X[model.feature_name()], raw_score=True))


def shrunk_rate(r, ref, lam):
    """Pull a model rate toward the player's shrunk recent rate: ref + lam * (r - ref)."""
    r = np.asarray(r, float)
    ref = np.where(np.isfinite(np.asarray(ref, float)), ref, r)
    return np.clip(ref + lam * (r - ref), 1e-3, None)


def fit_shrink(r, ref, y, expo):
    """lam minimising Poisson deviance on held-out rows (1 = no change). The raw model over-extends when it
    moves far from a player's recent average, so this is fit on validation and applied to every prediction."""
    def nll(lam):
        mu = shrunk_rate(r, ref, lam) * expo
        return float((mu - y * np.log(mu)).sum())
    return float(optimize.minimize_scalar(nll, bounds=(0.0, 1.5), method="bounded").x)


def fit_alpha(y, mu):
    def nll(la):
        n = 1 / np.exp(la)
        return -stats.nbinom.logpmf(y, n, n / (n + mu)).sum()
    return float(np.exp(optimize.minimize_scalar(nll, bounds=(-7, 1), method="bounded").x))


def fit_rho(h, k, p):
    m = k > 0
    h, k, p = h[m], k[m], np.clip(p[m], 0.05, 0.95)

    def nll(lr):
        r = 1 / (1 + np.exp(-lr))
        return -stats.betabinom.logpmf(h, k, p * (1 - r) / r, (1 - p) * (1 - r) / r).sum()
    return float(1 / (1 + np.exp(-optimize.minimize_scalar(nll, bounds=(-9, 0), method="bounded").x)))


def rounds_table(feat):
    g = feat.drop_duplicates("game_id")
    b = np.clip(np.searchsorted(D.CLOSENESS_BINS, g.closeness, side="right") - 1, 0, len(D.CLOSENESS_BINS) - 2)
    tab = {}
    for i in range(len(D.CLOSENESS_BINS) - 1):
        r = g.rounds[b == i] if (b == i).sum() >= 50 else g.rounds
        tab[i] = {int(k): float(v) for k, v in r.value_counts(normalize=True).items()}
    return tab


def row_pmfs(df, kpr, hsr, art, alpha=None, win_ratio=None, rho=None):
    """Pregame kills & HS pmfs per row (map known, rounds/winner unknown)."""
    alpha = art["alpha"] if alpha is None else alpha
    wr = art["win_ratio"] if win_ratio is None else win_ratio
    rho = art["rho"] if rho is None else rho
    K, H = np.zeros((len(df), D.KMAX + 1)), np.zeros((len(df), D.KMAX + 1))
    cache = {}
    for i, (c, pw) in enumerate(zip(df["closeness"].values, df["p_win"].values)):
        b = round(float(c), 2)
        if b not in cache:
            cache[b] = D.rounds_dist(art["rounds_table"], b)
        sup, pr = cache[b]
        K[i] = D.map_kills_pmf(kpr[i], alpha, sup, pr, float(pw), wr)
        H[i] = D.hs_pmf(K[i], hsr[i], rho)
    return K, H


def nb_design(df):
    X = df[NB_COLS].copy()
    X["opp_dpr"] = X["opp_dpr"].fillna(D_PRIOR)
    X["kpr_ewm"] = X["kpr_ewm"].fillna(X["kpr_shrunk"])
    X["elo_diff"] = X["elo_diff"] / 400
    return sm.add_constant(X, has_constant="add")


D_PRIOR = 0.68


def series_pairs(df):
    """(match, player) with both map 1 and map 2 -> row indices."""
    d = df[df.map_number.isin([1, 2])]
    piv = d.reset_index().pivot_table(index=["match_id", "player_id"], columns="map_number", values="index", aggfunc="first")
    piv = piv.dropna()
    return piv[1].astype(int).values, piv[2].astype(int).values


def train(hist=None):
    settings = load_yaml("settings.yaml")
    payouts = load_yaml("payouts.yaml")
    be = breakeven_per_leg(payouts["prizepicks"]["plays"]["power_2"])
    hist = hist if hist is not None else pd.read_csv(HIST_PATH)
    feat = build_features(hist)
    # features use every crawled map as history; training/evaluation only on tier-1/2 maps
    feat = feat[feat.tier.isin(settings["tiers"]) & (feat.n_maps >= MIN_HISTORY) & feat.kills.notna() & feat.headshots.notna() & (feat.rounds > 0)]
    feat = feat.reset_index(drop=True)
    feat["split"] = time_split(feat)
    tr, va, te = (feat[feat.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    log.info("rows train=%d val=%d test=%d", len(tr), len(va), len(te))

    kill_model = _fit_lgb(tr, va, "kills", "rounds")
    trh, vah = tr[tr.kills > 0], va[va.kills > 0]
    hs_model = _fit_lgb(trh, vah, "headshots", "kills")

    art = {"features": FEATURES, "rounds_table": rounds_table(feat[feat.split != "test"]),
           "trained_through": str(feat.date.max()), "n_rows": int(len(feat))}
    art["shrink"] = {"kills": fit_shrink(rate(kill_model, va), va.kpr_shrunk.values, va.kills.values, va.rounds.values),
                     "headshots": fit_shrink(rate(hs_model, vah), vah.hs_pct_shrunk.values, vah.headshots.values, vah.kills.values)}

    def lgbm_rates(df, lam_k=None, lam_h=None):
        lk = art["shrink"]["kills"] if lam_k is None else lam_k
        lh = art["shrink"]["headshots"] if lam_h is None else lam_h
        return (shrunk_rate(rate(kill_model, df), df.kpr_shrunk.values, lk),
                np.clip(shrunk_rate(rate(hs_model, df), df.hs_pct_shrunk.values, lh), 0.05, 0.95))
    kv, hv = lgbm_rates(va)
    mu_v = kv * va.rounds.values
    art["alpha"] = fit_alpha(va.kills.values, mu_v)
    art["rho"] = fit_rho(va.headshots.values, va.kills.values, hv)
    resid = va.kills.values / mu_v
    art["win_ratio"] = float(resid[va.won.values == 1].mean() / resid[va.won.values == 0].mean())

    # NB GLM baseline (offset log rounds) and naive rolling average
    glm = sm.GLM(tr.kills, nb_design(tr), family=sm.families.NegativeBinomial(alpha=art["alpha"]),
                 offset=np.log(tr.rounds)).fit()
    nb_alpha = fit_alpha(va.kills.values, np.exp(nb_design(va) @ glm.params).values * va.rounds.values)

    def preds(df):
        return {
            "lgbm": (*lgbm_rates(df), {}),
            "negbin": (np.exp(nb_design(df) @ glm.params).values, df.hs_pct_shrunk.values, {"alpha": nb_alpha, "win_ratio": 1.0}),
            "naive": (df.kpr_10.fillna(df.kpr_shrunk).values, df.hs_pct_10.fillna(df.hs_pct_shrunk).values,
                      {"alpha": 1e-4, "win_ratio": 1.0, "rho": 1e-4}),
        }

    def evaluate(df):
        out = {}
        for name, (kp, hp, kw) in preds(df).items():
            K, H = row_pmfs(df, kp, hp, art, **kw)
            out[name] = (K, H)
        return out

    nv = preds(va)["naive"]
    # synthetic lines from the naive model's mean
    def lines(df, nk, nh):
        er = df.exp_rounds.fillna(df.rounds_20).fillna(22).values
        return M.synthetic_line(nk * er), M.synthetic_line(nk * er * nh)

    # --- calibration on validation (per stat, per scope)
    pv = evaluate(va)
    lk, lh = lines(va, nv[0], nv[1])
    i1, i2 = series_pairs(va)
    calibrators = {}
    for stat, P, L, y in (("kills", pv["lgbm"][0], lk, va.kills.values), ("headshots", pv["lgbm"][1], lh, va.headshots.values)):
        p = M.p_over_rows(P, L)
        calibrators[f"{stat}_map"] = Platt().fit(p, (y > L).astype(int))
        P12 = np.array([D.convolve(P[a], P[b]) for a, b in zip(i1, i2)])
        L12 = np.floor(L[i1] + L[i2]) + 0.5 if len(i1) else np.array([])
        y12 = y[i1] + y[i2]
        if len(i1) > 200:
            calibrators[f"{stat}_series"] = Platt().fit(
                M.p_over_rows(P12, L12), (y12 > L12).astype(int))
        else:
            calibrators[f"{stat}_series"] = calibrators[f"{stat}_map"]

    # --- test backtest
    pt = evaluate(te)
    nt = preds(te)["naive"]
    lk, lh = lines(te, nt[0], nt[1])
    j1, j2 = series_pairs(te)
    report = {"breakeven": be, "split_dates": {s: [str(feat[feat.split == s].date.min()), str(feat[feat.split == s].date.max())]
                                               for s in ("train", "val", "test")}, "models": {}}
    rows = te[["date", "player", "team", "opponent", "map_name", "kills", "headshots"]].copy()
    for name, (K, H) in pt.items():
        rep = {}
        for stat, P, L, y in (("kills", K, lk, te.kills.values), ("headshots", H, lh, te.headshots.values)):
            p = M.p_over_rows(P, L)
            if name == "lgbm":
                rows[f"{stat}_line"], rows[f"{stat}_p_raw"] = L, p
                p = calibrators[f"{stat}_map"].predict(p)
                rows[f"{stat}_p_over"], rows[f"{stat}_mean"] = p, (P * np.arange(P.shape[1])).sum(1)
            rep[f"{stat}_map"] = M.summarize(P, y, p, (y > L).astype(int), be, settings["min_edge"])
            if len(j1):
                P12 = np.array([D.convolve(P[a], P[b]) for a, b in zip(j1, j2)])
                L12, y12 = np.floor(L[j1] + L[j2]) + 0.5, y[j1] + y[j2]
                p12 = M.p_over_rows(P12, L12)
                if name == "lgbm":
                    p12 = calibrators[f"{stat}_series"].predict(p12)
                rep[f"{stat}_maps12"] = M.summarize(P12, y12, p12, (y12 > L12).astype(int), be, settings["min_edge"])
        report["models"][name] = rep
    for stat in ("kills", "headshots"):
        report[f"calibration_{stat}"] = M.calibration_table(rows[f"{stat}_p_over"].values,
                                                            (rows[stat] > rows[f"{stat}_line"]).astype(int).values).to_dict("records")
    report["shrink"] = {}
    for stat, i, y, expo in (("kills", 0, te.kills.values, te.rounds.values), ("headshots", 1, te.headshots.values, te.kills.values)):
        raw, adj = lgbm_rates(te, 1.0, 1.0)[i] * expo, lgbm_rates(te)[i] * expo
        report["shrink"][stat] = {"lambda": art["shrink"][stat], "test_mae_raw": float(np.abs(raw - y).mean()),
                                  "test_mae_shrunk": float(np.abs(adj - y).mean()),
                                  "test_bias_raw": float((raw - y).mean()), "test_bias_shrunk": float((adj - y).mean())}
    log.info("shrink toward recent form: %s", report["shrink"])
    imp = pd.Series(kill_model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    report["feature_importance_kills"] = (imp / imp.sum()).head(20).round(4).to_dict()

    from ..maps import pool as map_pool
    art["map_model"], report["map_model"] = map_pool.train(hist)
    log.info("map model: %s", report["map_model"])

    MODEL_DIR.mkdir(exist_ok=True)
    kill_model.save_model(str(MODEL_DIR / "kills_lgbm.txt"))
    hs_model.save_model(str(MODEL_DIR / "hs_lgbm.txt"))
    joblib.dump(calibrators, MODEL_DIR / "calibrators.joblib")
    (MODEL_DIR / "artifacts.json").write_text(json.dumps(art, indent=1))
    (MODEL_DIR / "backtest.json").write_text(json.dumps(report, indent=1, default=float))
    rows.to_csv(MODEL_DIR / "backtest_rows.csv", index=False)
    return art, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _, rep = train()
    for name, r in rep["models"].items():
        for k, v in r.items():
            b = v["betting"]
            print(f"{name:7s} {k:16s} MAE={v['mae']:.2f} pin50={v['pinball_0.5']:.2f} brier={v['brier']:.4f} "
                  f"logloss={v['logloss']:.4f} picks={b['n_picks']} hit={b['hit_rate']:.3f} roi={b['roi']:+.3f}")
