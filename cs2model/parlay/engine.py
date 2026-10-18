"""Pick'em slip math: payouts from config, breakevens, EV over the exact hit distribution,
correlation-aware selection on a shared Monte Carlo hit matrix, fractional Kelly."""
from itertools import combinations
from math import comb
import os
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(os.environ.get("CS2_HOME") or Path(__file__).resolve().parents[2])


def load_yaml(name):
    return yaml.safe_load((ROOT / "config" / name).read_text())


def plays(platform, payouts=None):
    cfg = (payouts or load_yaml("payouts.yaml"))[platform]
    return {k: v for k, v in cfg["plays"].items() if v.get("enabled", True) and v["legs"] != 4}


def payout_vector(play):
    """multiplier by number of hits 0..legs."""
    v = np.zeros(play["legs"] + 1)
    if play["type"] == "power":
        v[-1] = play["payout"]
    else:
        for h, m in play["payouts"].items():
            v[int(h)] = m
    return v


def ev_from_hits(hit_dist, play):
    """hit_dist: probabilities of 0..legs hits. Returns expected return per 1 staked minus 1."""
    return float(np.dot(hit_dist, payout_vector(play)) - 1)


def iid_hit_dist(p, legs):
    return np.array([comb(legs, h) * p**h * (1 - p) ** (legs - h) for h in range(legs + 1)])


def breakeven_per_leg(play):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ev_from_hits(iid_hit_dist(mid, play["legs"]), play) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def single_breakeven(platform, decimal_price=None, payouts=None):
    if decimal_price:
        return 1.0 / decimal_price
    cfg = (payouts or load_yaml("payouts.yaml"))[platform]
    return breakeven_per_leg(cfg["plays"][cfg["single_pick_reference"]])


def kelly_fraction(hit_dist, play, grid=np.linspace(0, 0.5, 501)):
    """Full-Kelly stake fraction maximizing E[log wealth] for a multi-outcome slip."""
    r = payout_vector(play) - 1.0
    best, best_f = 0.0, 0.0
    for f in grid[1:]:
        w = 1 + f * r
        if (w <= 0).any():
            break
        g = float(np.dot(hit_dist, np.log(w)))
        if g > best:
            best, best_f = g, f
    return best_f


def evaluate(H, idx, play):
    """H: (n_sims, n_legs) bool hit matrix. Returns hit distribution, EV, independent EV, corr ratio."""
    sub = H[:, list(idx)]
    hits = sub.sum(1)
    dist = np.bincount(hits, minlength=play["legs"] + 1)[: play["legs"] + 1] / len(hits)
    p = sub.mean(0)
    indep = np.array([1.0])
    for pi in p:
        indep = np.convolve(indep, [1 - pi, pi])
    p_all, p_all_ind = dist[-1], indep[-1]
    return {"hit_dist": dist, "ev": ev_from_hits(dist, play), "ev_indep": ev_from_hits(indep, play),
            "p_all": float(p_all), "corr_ratio": float(p_all / p_all_ind) if p_all_ind > 0 else 1.0}


def sanity_check(H, idx, play, shrink=0.02):
    """6-leg (or any) +EV slip must stay +EV under independence and with every leg 2pp worse."""
    sub = H[:, list(idx)]
    p = np.clip(sub.mean(0) - shrink, 0, 1)
    d = np.array([1.0])
    for pi in p:
        d = np.convolve(d, [1 - pi, pi])
    return ev_from_hits(d, play) > 0


def optimize(H, legs, play, top_n=10, beam=300, max_per_match=2, corr_penalty=0.5):
    """Beam search over leg combinations. legs: list of dicts with match, player, stat keys.
    Score = EV - corr_penalty * max(0, |log corr_ratio| - 0.1) (prefers cross-match slips)."""
    k = play["legs"]
    n = len(legs)
    if n < k:
        return []

    def ok(combo):
        m, ps = {}, set()
        for i in combo:
            L = legs[i]
            m[L["match"]] = m.get(L["match"], 0) + 1
            if L["player"] in ps:  # PrizePicks/Underdog allow one pick per player per entry
                return False
            ps.add(L["player"])
        return max(m.values()) <= max_per_match

    def score(r):
        return r["ev"] - corr_penalty * max(0.0, abs(np.log(max(r["corr_ratio"], 1e-9))) - 0.1)

    if comb(n, k) <= 5000:
        cands = [c for c in combinations(range(n), k) if ok(c)]
    else:
        frontier = [(i,) for i in range(n)]
        for size in range(2, k + 1):
            nxt = {tuple(sorted(c + (j,))) for c in frontier for j in range(n) if j not in c}
            nxt = [c for c in nxt if ok(c)]
            sub_play = {"legs": size, "type": "power", "payout": 1.0}
            nxt.sort(key=lambda c: -evaluate(H, c, sub_play)["p_all"])
            frontier = nxt[:beam]
        cands = frontier
    res = []
    for c in cands:
        r = evaluate(H, c, play)
        r["legs"] = c
        r["score"] = score(r)
        r["heavy_corr"] = abs(np.log(max(r["corr_ratio"], 1e-9))) > 0.15
        res.append(r)
    res.sort(key=lambda r: -r["score"])
    return res[:top_n]
