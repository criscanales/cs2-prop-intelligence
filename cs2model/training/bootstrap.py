"""Quick start and background backfill.

A fresh install downloads the last `quickstart_days` first and trains a quick-start model so the dashboard opens in
about an hour. The rest of the `history_days` window then downloads in the background at the normal request rate.
A model trained on the full history only replaces the quick-start model if it is at least as good on maps that
neither model trained or calibrated on.
"""
import json
import logging
import os
import shutil
import threading
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ..backtest import metrics as M
from ..features.build import build_features
from ..parlay.engine import load_yaml
from ..scraper.bo3_history import DATA, HIST_PATH, crawl, merge_history, save_history
from .train import MIN_HISTORY, MODEL_DIR, rate, row_pmfs, shrunk_rate, train

COVERAGE_PATH = DATA / "coverage.json"
CANDIDATE_DIR = MODEL_DIR.parent / "models_candidate"
MODEL_FILES = ("kills_lgbm.txt", "hs_lgbm.txt", "calibrators.joblib", "backtest.json", "backtest_rows.csv", "artifacts.json")
MIN_EVAL_ROWS = 500
MAX_MARKET_LOSS = 0.002  # a full-history model may not be worse than this on either market's held-out log loss
log = logging.getLogger(__name__)


def read_coverage() -> dict | None:
    try:
        return json.loads(COVERAGE_PATH.read_text())
    except (OSError, ValueError):
        return None


def write_coverage(**kw) -> dict:
    c = read_coverage() or {}
    c.update(kw, updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = COVERAGE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(c, indent=1, default=str))
    os.replace(tmp, COVERAGE_PATH)
    return c


def history_span(path=None) -> dict:
    path = HIST_PATH if path is None else path
    if not path.exists():
        return {"rows": 0, "from": None, "to": None, "days": 0}
    d = pd.to_datetime(pd.read_csv(path, usecols=["date"]).date, utc=True, format="ISO8601")
    if d.empty:
        return {"rows": 0, "from": None, "to": None, "days": 0}
    return {"rows": int(len(d)), "from": d.min(), "to": d.max(), "days": int((d.max() - d.min()).days) + 1}


def is_complete() -> bool:
    """Installs from before quick start have no coverage file and always crawled the full window."""
    c = read_coverage()
    return True if c is None else bool(c.get("complete"))


def model_history(d=None) -> dict:
    d = MODEL_DIR if d is None else d
    try:
        return json.loads((d / "artifacts.json").read_text()).get("history") or {}
    except (OSError, ValueError):
        return {}


def _tiers(settings):
    return tuple(settings.get("history_tiers", settings["tiers"]))


def quick_start(settings=None, progress=None):
    """Download the most recent quickstart_days (cached, so a rerun resumes) and train the quick-start model."""
    settings = settings or load_yaml("settings.yaml")
    days = int(settings.get("quickstart_days", 45))
    write_coverage(complete=False, target_days=int(settings["history_days"]), quickstart_days=days, phase="quick start")
    old = pd.read_csv(HIST_PATH) if HIST_PATH.exists() else None
    df = crawl(days, tiers=_tiers(settings), write=False, progress=progress, known=old)
    df = merge_history(old, df) if old is not None else df
    if df.empty:
        raise RuntimeError("bo3.gg returned no finished matches; nothing to train on yet")
    save_history(df)
    train(df, quick_start=True)
    write_coverage(phase="backfill")
    return df


def backfill(settings=None, state=None):
    """Download the rest of the history window (maps already stored are not re-fetched), then run the promotion gate."""
    settings = settings or load_yaml("settings.yaml")
    state = state if state is not None else {}
    n = int(settings["history_days"])
    old = pd.read_csv(HIST_PATH) if HIST_PATH.exists() else None
    state.update(phase="downloading", day=0, n=n + 1, rows=0)
    write_coverage(phase="backfill", backfill_started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    df = crawl(n, tiers=_tiers(settings), write=False, known=old,
               progress=lambda d, n_, r: state.update(day=d, n=n_, rows=r))
    if old is not None and len(df) < 0.9 * len(old):
        raise RuntimeError(f"only {len(df):,} rows pulled vs {len(old):,} on disk; kept the current history")
    save_history(df)
    write_coverage(complete=True, phase="evaluating", history_completed=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    state.update(phase="training")
    res = gated_retrain(df, settings)
    write_coverage(phase="done")
    return res


_gate_lock = threading.Lock()


def gated_retrain(hist=None, settings=None) -> dict | None:
    """Train on `hist` into a candidate folder and promote it only if it isn't worse on shared held-out maps.
    None (nothing done) when another check is already running in this process."""
    if not _gate_lock.acquire(blocking=False):
        return None
    try:
        return _gated_retrain(hist, settings)
    finally:
        _gate_lock.release()


def _gated_retrain(hist, settings):
    settings = settings or load_yaml("settings.yaml")
    hist = hist if hist is not None else pd.read_csv(HIST_PATH)
    shutil.rmtree(CANDIDATE_DIR, ignore_errors=True)
    train(hist, out_dir=CANDIDATE_DIR)
    res = compare(MODEL_DIR, CANDIDATE_DIR, hist, settings)
    if res["promote"]:
        for f in MODEL_FILES:  # artifacts.json last: its mtime tells the dashboard to reload
            shutil.copy2(CANDIDATE_DIR / f, MODEL_DIR / f)
    shutil.rmtree(CANDIDATE_DIR, ignore_errors=True)
    res["t"] = time.time()
    write_coverage(promotion=res)
    log.info("promotion gate: %s", res)
    return res


def _score(d, ho, lk, lh):
    km, hm = lgb.Booster(model_file=str(d / "kills_lgbm.txt")), lgb.Booster(model_file=str(d / "hs_lgbm.txt"))
    art = json.loads((d / "artifacts.json").read_text())
    cal = joblib.load(d / "calibrators.joblib")
    kp = shrunk_rate(rate(km, ho), ho.kpr_shrunk.values, art["shrink"]["kills"])
    hp = np.clip(shrunk_rate(rate(hm, ho), ho.hs_pct_shrunk.values, art["shrink"]["headshots"]), 0.05, 0.95)
    K, H = row_pmfs(ho, kp, hp, art)
    out = {}
    for stat, P, L, y in (("kills", K, lk, ho.kills.values), ("headshots", H, lh, ho.headshots.values)):
        p = cal[f"{stat}_map"].predict(M.p_over_rows(P, L))
        yo = (y > L).astype(int)
        out[stat] = {"logloss": M.logloss(p, yo), "brier": M.brier(p, yo),
                     "mae": float(np.abs((P * np.arange(P.shape[1])).sum(1) - y).mean())}
    return out


def compare(cur_dir, cand_dir, hist, settings=None) -> dict:
    """Score both models on tier-1/2 maps inside BOTH models' test windows (never used to fit or calibrate either),
    with features built from the same history and the same synthetic lines as train()."""
    settings = settings or load_yaml("settings.yaml")
    if not (cur_dir / "artifacts.json").exists():
        return {"promote": True, "reason": "no current model"}
    starts = [pd.Timestamp(json.loads((d / "backtest.json").read_text())["split_dates"]["test"][0]) for d in (cur_dir, cand_dir)]
    feat = build_features(hist)
    feat = feat[feat.tier.isin(settings["tiers"]) & (feat.n_maps >= MIN_HISTORY) & feat.kills.notna()
                & feat.headshots.notna() & (feat.rounds > 0) & (feat.date >= max(starts))].reset_index(drop=True)
    res = {"held_out_from": str(max(starts)), "held_out_rows": int(len(feat)),
           "held_out_maps": int(feat.game_id.nunique()) if len(feat) else 0}
    if len(feat) < MIN_EVAL_ROWS:
        return {**res, "promote": False, "reason": f"only {len(feat)} held-out rows (need {MIN_EVAL_ROWS}); kept the current model"}
    er = feat.exp_rounds.fillna(feat.rounds_20).fillna(22).values
    nk, nh = feat.kpr_10.fillna(feat.kpr_shrunk).values, feat.hs_pct_10.fillna(feat.hs_pct_shrunk).values
    lk, lh = M.synthetic_line(nk * er), M.synthetic_line(nk * er * nh)
    a, b = _score(cur_dir, feat, lk, lh), _score(cand_dir, feat, lk, lh)
    mean_a = (a["kills"]["logloss"] + a["headshots"]["logloss"]) / 2
    mean_b = (b["kills"]["logloss"] + b["headshots"]["logloss"]) / 2
    worst = max(b[s]["logloss"] - a[s]["logloss"] for s in ("kills", "headshots"))
    promote = bool(mean_b <= mean_a and worst <= MAX_MARKET_LOSS)
    reason = ("full-history model is at least as good on held-out maps" if promote else
              "full-history model was worse on held-out maps; kept the current model")
    return {**res, "current": a, "candidate": b, "promote": promote, "reason": reason}
