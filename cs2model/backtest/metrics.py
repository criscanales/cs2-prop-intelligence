"""Backtest metrics on held-out (time-ordered) maps."""
import numpy as np
import pandas as pd

from ..inference import distribution as D

EDGE_BUCKETS = [0.03, 0.05, 0.08, 0.12, 1.0]


def pmf_quantiles(P, q):
    return (np.cumsum(P, axis=1) < q).sum(1)


def pinball(P, y, q):
    qh = pmf_quantiles(P, q)
    d = y - qh
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(p, y, bins=10):
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return df.reset_index()


def betting(p_over, y_over, breakeven, min_edge=0.03):
    """Flat 1-unit picks on the side with edge >= min_edge, paid at the breakeven-implied price."""
    p_over = np.asarray(p_over)
    e_over, e_under = p_over - breakeven, (1 - p_over) - breakeven
    side_over = e_over >= e_under
    edge = np.where(side_over, e_over, e_under)
    hit = np.where(side_over, y_over == 1, y_over == 0)
    take = edge >= min_edge
    res = {"n_picks": int(take.sum()), "hit_rate": float(hit[take].mean()) if take.any() else np.nan,
           "roi": float((hit[take] / breakeven - 1).mean()) if take.any() else np.nan, "buckets": []}
    for lo, hi in zip(EDGE_BUCKETS[:-1], EDGE_BUCKETS[1:]):
        m = (edge >= lo) & (edge < hi)
        if m.any():
            res["buckets"].append({"edge": f"{lo:.0%}-{hi:.0%}", "n": int(m.sum()), "avg_edge": float(edge[m].mean()),
                                   "hit_rate": float(hit[m].mean()), "roi": float((hit[m] / breakeven - 1).mean())})
    return res


def summarize(P, y, p_over, y_over, breakeven, min_edge):
    out = {"n": int(len(y)), "mae": float(np.mean(np.abs((P * np.arange(P.shape[1])).sum(1) - y)))}
    for q in (0.1, 0.5, 0.9):
        out[f"pinball_{q}"] = pinball(P, y, q)
    out["brier"] = brier(p_over, y_over)
    out["logloss"] = logloss(p_over, y_over)
    out["betting"] = betting(p_over, y_over, breakeven, min_edge)
    return out


def synthetic_line(mean):
    """Stand-in for a book line when none is archived: naive mean rounded to x.5."""
    return np.floor(mean) + 0.5


def p_over_rows(P, lines):
    k = np.arange(P.shape[1])[None, :]
    return (P * (k > np.asarray(lines)[:, None])).sum(1)


__all__ = ["summarize", "calibration_table", "synthetic_line", "p_over_rows", "D"]
