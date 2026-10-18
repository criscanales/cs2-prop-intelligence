"""Exact single-leg distributions under the same model the match simulation samples from.

Map picks follow Plackett-Luce (what Gumbel top-k draws), rounds and map winners are independent
per slot, kills ~ NB mixed over rounds and win/loss multipliers, HS | kills ~ beta-binomial.
Map 3 of a BO3 is played only when maps 1-2 split, so winner outcomes are enumerated explicitly.
"""
import itertools

import numpy as np

from . import distribution as D

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


def p_side_win(spec, ps, j):
    p1 = spec.p_map_team1.get(j, 0.5)
    return p1 if ps.side == 1 else 1 - p1


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
        pm = D.nb_pmf((kpr[:, None] * sup[None, :] * mult).ravel(), ps.alpha).reshape(M, R, K + 1)
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
        pr = float(np.prod([pw[j] if wins[j] else 1 - pw[j] for j in range(L)]))
        if pr <= 0:
            continue
        map3_played = not (spec.best_of == 3 and L == 3 and wins[0] == wins[1])
        V = P[..., None].astype(complex)
        for j in range(L, 0, -1):
            if j in want and (j != 3 or map3_played):
                V = (V * Gf[j][wins[j - 1]]).sum(axis=-2)
            else:
                V = V.sum(axis=-2)
        acc += pr * V
    pmf = np.clip(np.fft.irfft(acc, NF)[: KOUT + 1], 0, None)
    return pmf / pmf.sum()


def p_map3(spec):
    if spec.best_of != 3:
        return 1.0 if spec.best_of > 3 else 0.0
    p1, p2 = spec.p_map_team1.get(1, 0.5), spec.p_map_team1.get(2, 0.5)
    return p1 * (1 - p2) + (1 - p1) * p2
