"""Discrete predictive distributions for kills / headshots.

Per map:   rounds ~ empirical (closeness bucket, optionally tilted to market round totals / OT odds)
           win    ~ Bernoulli(p_win)  -> kill-rate multiplier (winners frag more)
           kills  ~ NegBin(mean = kpr * rounds * win_mult, dispersion alpha)
           HS | kills ~ BetaBinomial(kills, hs_rate, rho)
"""
import numpy as np
from scipy import stats

KMAX = 120
OT_ROUNDS = 24
CLOSENESS_BINS = np.array([0.0, 0.5, 0.75, 0.9, 1.01])


def nb_pmf(mu, alpha, kmax=KMAX):
    """NB2 pmf on 0..kmax for array mu (shape (m,)) -> (m, kmax+1)."""
    mu = np.atleast_1d(np.asarray(mu, float))
    n = 1.0 / alpha
    p = n / (n + mu)
    return stats.nbinom.pmf(np.arange(kmax + 1)[None, :], n, p[:, None])


def win_multipliers(p_win, win_ratio):
    """Kill-rate multipliers (win, loss) with E[mult] = 1 and win/loss = win_ratio."""
    m_l = 1.0 / (p_win * win_ratio + 1 - p_win)
    return win_ratio * m_l, m_l


def rounds_dist(table, closeness, ladder=None, p_ot=None):
    """table: {bin_index: {rounds: prob}}. Returns (support, probs)."""
    b = int(np.clip(np.searchsorted(CLOSENESS_BINS, closeness, side="right") - 1, 0, len(CLOSENESS_BINS) - 2))
    d = table.get(b) or table.get(str(b))
    sup = np.array(sorted(int(k) for k in d), float)
    pr = np.array([d[k] if k in d else d[str(int(k))] for k in sup.astype(int)], float)
    pr = pr / pr.sum()
    if ladder:
        pts, po = min(ladder, key=lambda x: abs(x[1] - 0.5))
        pr = _tilt(sup, pr, pts, po)
    if p_ot is not None and 0 < p_ot < 1:
        ot = sup > OT_ROUNDS
        if pr[ot].sum() > 0 and pr[~ot].sum() > 0:
            pr = np.where(ot, pr * p_ot / pr[ot].sum(), pr * (1 - p_ot) / pr[~ot].sum())
    return sup, pr / pr.sum()


def _tilt(sup, pr, line, p_over):
    """Exponentially tilt pr so that P(rounds > line) == p_over."""
    over = sup > line
    if not over.any() or over.all():
        return pr
    lo, hi = -3.0, 3.0
    for _ in range(60):
        lam = (lo + hi) / 2
        w = pr * np.exp(lam * (sup - sup.mean()))
        if w[over].sum() / w.sum() < p_over:
            lo = lam
        else:
            hi = lam
    w = pr * np.exp(lam * (sup - sup.mean()))
    return w / w.sum()


def map_kills_pmf(kpr, alpha, rounds_sup, rounds_p, p_win=0.5, win_ratio=1.0, kmax=KMAX):
    m_w, m_l = win_multipliers(p_win, win_ratio)
    out = np.zeros(kmax + 1)
    for mult, pw in ((m_w, p_win), (m_l, 1 - p_win)):
        if pw <= 0:
            continue
        out += pw * (rounds_p[:, None] * nb_pmf(kpr * rounds_sup * mult, alpha, kmax)).sum(0)
    return out / out.sum()


_BB_CACHE: dict = {}


def _bb_matrix(hs_rate, rho, kmax):
    key = (round(hs_rate, 3), round(rho, 4), kmax)
    if key not in _BB_CACHE:
        rho_ = max(rho, 1e-6)
        a = hs_rate * (1 - rho_) / rho_
        b = (1 - hs_rate) * (1 - rho_) / rho_
        k = np.arange(kmax + 1)
        h = np.arange(kmax + 1)
        _BB_CACHE[key] = stats.betabinom.pmf(h[None, :], k[:, None], a, b)  # [k, h]
    return _BB_CACHE[key]


def hs_pmf(kills_pmf, hs_rate, rho):
    kmax = len(kills_pmf) - 1
    return kills_pmf @ _bb_matrix(float(np.clip(hs_rate, 0.05, 0.95)), rho, kmax)


def convolve(a, b, kmax=KMAX):
    c = np.convolve(a, b)[: kmax + 1]
    return c / c.sum()


def p_over(pmf, line):
    k = np.arange(len(pmf))
    return float(pmf[k > line].sum())


def p_under(pmf, line):
    k = np.arange(len(pmf))
    return float(pmf[k < line].sum())


def mean(pmf):
    return float((np.arange(len(pmf)) * pmf).sum())


def quantile(pmf, q):
    return int(np.searchsorted(np.cumsum(pmf), q))
