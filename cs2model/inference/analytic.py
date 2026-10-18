"""Exact single-leg distributions under the same model the match simulation samples from.

Map picks follow Plackett-Luce (what Gumbel top-k draws), rounds and map winners are independent
per slot, kills ~ NB mixed over rounds and win/loss multipliers, HS | kills ~ beta-binomial.
Map 3 of a BO3 is played only when maps 1-2 split, so winner outcomes are enumerated explicitly.
"""
import itertools

import numpy as np

from . import distribution as D
from .simulate import win_vec

K = D.KMAX
KOUT = 3 * K
NF = 512  # FFT length > KOUT + 1, so convolutions of up to three maps never wrap


def pick_orders(w, L):
    """P[i1, .., iL]: probability that the first L maps picked are i1..iL, in order."""
    w = np.asarray(w, float)
    w = w / w.sum()
    M = len(w)
    if L == 1:
        return w
    P2 = w[:, None] * w[None, :] / np.clip(1 - w, 1e-12, None)[:, None]
    np.fill_diagonal(P2, 0)
    if L == 2:
        return P2
    P3 = P2[:, :, None] * w[None, None, :] / np.clip(1 - w[:, None] - w[None, :], 1e-12, None)[:, :, None]
    i = np.arange(M)
    P3[i, :, i] = 0
    P3[:, i, i] = 0
    return P3


def slot_marginal(w, j):
    """P(map i is the j-th pick) under Plackett-Luce."""
    P = pick_orders(w, j)
    return P if j == 1 else P.sum(axis=tuple(range(j - 1)))


def p_side_win(spec, ps, j):
    """The player's side's P(win map slot j), per candidate map."""
    v = win_vec(spec, j)
    return v if ps.side == 1 else 1 - v


def slot_pmfs(ps, spec, j):
    """{"kills": [won, map, k], "headshots": [won, map, h]} for map slot j (1-based)."""
    sup, pr = spec.rounds.get(j) or spec.rounds[1]
    sup, pr = np.asarray(sup, float), np.asarray(pr, float)
    m_w, m_l = D.win_multipliers(p_side_win(spec, ps, j), spec.win_ratio)
    kmean, hmean = np.mean(list(ps.kpr.values())), np.mean(list(ps.hs_rate.values()))
    kpr = np.array([ps.kpr.get(m, kmean) for m in spec.maps])
    hsr = np.array([ps.hs_rate.get(m, hmean) for m in spec.maps])
    M, R = len(kpr), len(sup)
    G = np.empty((2, M, K + 1))
    for won, mult in ((1, m_w), (0, m_l)):
        pm = D.nb_pmf((kpr[:, None] * sup[None, :] * mult[:, None]).ravel(), ps.alpha).reshape(M, R, K + 1)
        G[won] = np.einsum("r,mrk->mk", pr, pm)
    G /= G.sum(-1, keepdims=True)
    H = np.empty_like(G)
    for m in range(M):
        bb = D._bb_matrix(float(np.clip(hsr[m], 0.05, 0.95)), ps.rho, K)
        H[:, m] = G[:, m] @ bb
    return {"kills": G, "headshots": H}


def leg_pmf(ps, spec, stat, maps, cache=None):
    """pmf of the player's total `stat` over `maps` (0..KOUT)."""
    n_maps = min(spec.best_of, 3)
    want = {m for m in maps if m <= n_maps}
    out = np.zeros(KOUT + 1)
    if not want:
        out[0] = 1.0
        return out
    L = max(want)
    P = pick_orders(spec.map_weights, L)
    cache = {} if cache is None else cache
    Gf = {}
    for j in want:
        key = (ps.key, j)
        if key not in cache:
            cache[key] = {s: np.fft.rfft(v, NF, axis=-1) for s, v in slot_pmfs(ps, spec, j).items()}
        Gf[j] = cache[key][stat]
    pw = [p_side_win(spec, ps, j) for j in range(1, L + 1)]
    acc = np.zeros(NF // 2 + 1, complex)
    for wins in itertools.product((0, 1), repeat=L):
        map3_played = not (spec.best_of == 3 and L == 3 and wins[0] == wins[1])
        V = P[..., None].astype(complex)
        for j in range(L, 0, -1):
            f = (pw[j - 1] if wins[j - 1] else 1 - pw[j - 1])[:, None]  # P(this result | map), per map
            if j in want and (j != 3 or map3_played):
                V = (V * (f * Gf[j][wins[j - 1]])).sum(axis=-2)
            else:
                V = (V * f).sum(axis=-2)
        acc += V
    pmf = np.clip(np.fft.irfft(acc, NF)[: KOUT + 1], 0, None)
    return pmf / pmf.sum()


def p_map3(spec):
    if spec.best_of != 3:
        return 1.0 if spec.best_of > 3 else 0.0
    v1, v2 = win_vec(spec, 1), win_vec(spec, 2)
    if len(v1) < 2:
        p1, p2 = float(v1[0]), float(v2[0])
        return p1 * (1 - p2) + (1 - p1) * p2
    P2 = pick_orders(spec.map_weights, 2)
    return float((P2 * (v1[:, None] * (1 - v2[None, :]) + (1 - v1[:, None]) * v2[None, :])).sum())


def map_preds(spec, pkey):
    """Per map in the pool: P(played in this series) and the player's expected kills/HS on it if played."""
    players = next(v for v in vars(spec).values() if isinstance(v, list) and v and hasattr(v[0], "kpr"))
    ps = next(x for x in players if x.key == pkey)
    L = min(spec.best_of, 3)
    w = np.asarray(spec.map_weights, float)
    w = w / w.sum()
    if L == 1:
        p_play = w
    else:
        P2 = pick_orders(w, 2)
        p_play = P2.sum(1) + P2.sum(0)
        if L == 3:
            p_play = p_play + p_map3(spec) * pick_orders(w, 3).sum((0, 1))
    G, pw, k = slot_pmfs(ps, spec, 1), p_side_win(spec, ps, 1), np.arange(K + 1)
    out = []
    for i, m in enumerate(spec.maps):
        ek = pw[i] * (G["kills"][1, i] @ k) + (1 - pw[i]) * (G["kills"][0, i] @ k)
        eh = pw[i] * (G["headshots"][1, i] @ k) + (1 - pw[i]) * (G["headshots"][0, i] @ k)
        out.append({"map": m, "p_played": float(p_play[i]), "kills": float(ek), "headshots": float(eh),
                    "kpr": float(ps.kpr.get(m, np.nan)), "p_win": float(pw[i])})
    return out
