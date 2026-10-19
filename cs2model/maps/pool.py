"""Team map pools from bo3.gg history: which maps a matchup is likely to play, and each team's
opponent-adjusted strength on each map. Leak-free: a match only sees matches that started before it.

Picks: Plackett-Luce over the active pool (the same model the pricing samples map orders from), with
scores from both teams' recency-weighted play shares and map strengths.
Map wins: logit P(team1 wins map m) = logit(base) + beta * (d1(m) - d2(m)), where d is a team's shrunk,
recency-weighted sum of (won - Elo expectation) on that map. When the base comes from the market it is
re-centred so the average over the likely maps still equals the market price. The per-map shift is only
fit and scored against Elo: the odds stored with bo3.gg history are last (often in-play) snapshots, so
they are not a valid pre-match base for a backtest.
"""
import json
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit, logit, logsumexp

ELO_K = 24.0
EPS = 0.02
POOL_DAYS, POOL_MIN_SHARE = 30, 0.02
BURN_IN_DAYS = 21
HALF_LIVES = (30.0, 90.0, 180.0)
DEFAULT = {"half_life": 90.0, "shrink": 16.0, "beta": 0.0, "pick": [1.0, 0.0, 0.0, 0.0],
           "use_picks": False, "use_wins": False}


def series_p(p, bo):
    """P(win a best-of-`bo`) from a per-map win probability."""
    n = bo // 2 + 1
    return sum(math.comb(n - 1 + k, k) * p ** n * (1 - p) ** k for k in range(n))


def map_p_from_series(ps, bo):
    if not 0 < ps < 1:
        return np.nan
    if bo <= 1:
        return ps
    return optimize.brentq(lambda p: series_p(p, bo) - ps, 1e-6, 1 - 1e-6)


def games_table(hist):
    """One row per map, oriented so team1 is the lower team id (consistent across a match)."""
    g = hist.drop_duplicates("game_id")[["date", "match_id", "game_id", "map_number", "map_name", "bo_type",
                                         "team_id", "opp_id", "won", "team_odds", "opp_odds"]].copy()
    g["date"] = pd.to_datetime(g["date"], utc=True, format="ISO8601")
    flip = g.team_id > g.opp_id
    g["t1"] = np.where(flip, g.opp_id, g.team_id)
    g["t2"] = np.where(flip, g.team_id, g.opp_id)
    g["won1"] = np.where(flip, 1 - g.won, g.won)
    g["o1"] = np.where(flip, g.opp_odds, g.team_odds)
    g["o2"] = np.where(flip, g.team_odds, g.opp_odds)
    g = g[g.won1.notna() & g.map_name.str.startswith("de_")]
    return g.sort_values(["date", "match_id", "map_number"]).reset_index(drop=True)


class _Decayed:
    """Per (team, map) event lists -> recency-weighted sums at a query time."""

    def __init__(self, hl):
        self.hl, self.ev = hl, defaultdict(list)

    def add(self, team, m, t, v):
        self.ev[(team, m)].append((t, v))

    def sums(self, team, m, t):
        e = self.ev.get((team, m))
        if not e:
            return 0.0, 0.0
        a = np.asarray(e)
        w = 0.5 ** ((t - a[:, 0]) / self.hl)
        return float((w * a[:, 1]).sum()), float(w.sum())


class MapState:
    """Chronological pass over history. `records` (one per match, pre-match features) feed fitting and
    backtests; after the pass the state answers queries for upcoming matches."""

    def __init__(self, hist, half_life=DEFAULT["half_life"], records=True):
        self.hl = half_life
        g = games_table(hist)
        self.elo, self.count = {}, defaultdict(float)
        self.play, self.res = _Decayed(half_life), _Decayed(half_life)
        self.records, recent = [], deque()
        self.t_last = 0.0
        g["t"] = (g.date - pd.Timestamp(0, tz="UTC")).dt.total_seconds() / 86400
        g["t"] = g.groupby("match_id", sort=False).t.transform("first")
        if not records:  # current state only: no per-match feature records
            for t, t1, t2, mp, w in zip(g.t.values, g.t1.values, g.t2.values, g.map_name.values, g.won1.values):
                self._update(int(t1), int(t2), mp, float(w), t)
            self.t_last = float(g.t.max()) if len(g) else 0.0
            return
        for mid, m in g.groupby("match_id", sort=False):
            t = m.t.iloc[0]
            while recent and recent[0][0] < t - POOL_DAYS:
                recent.popleft()
            t1, t2 = int(m.t1.iloc[0]), int(m.t2.iloc[0])
            pool = self._pool(recent)
            pool += [x for x in m.map_name if x not in pool]
            self.records.append(self._record(mid, t, t1, t2, sorted(pool), m))
            for r in m.itertuples(index=False):
                self._update(t1, t2, r.map_name, r.won1, t)
                recent.append((t, r.map_name))
            self.t_last = t

    def _update(self, t1, t2, mp, won1, t):
        e1 = self._elo_p(t1, t2)
        for tm in (t1, t2):
            self.play.add(tm, mp, t, 1.0)
            self.count[(tm, mp)] += 1
        self.res.add(t1, mp, t, won1 - e1)
        self.res.add(t2, mp, t, (1 - won1) - (1 - e1))
        ra, rb = self.elo.get(t1, 1500.0), self.elo.get(t2, 1500.0)
        self.elo[t1], self.elo[t2] = ra + ELO_K * (won1 - e1), rb - ELO_K * (won1 - e1)

    @staticmethod
    def _pool(recent):
        if not recent:
            return []
        s = pd.Series([m for _, m in recent]).value_counts(normalize=True)
        return list(s[s >= POOL_MIN_SHARE].index)

    def _elo_p(self, t1, t2):
        return 1 / (1 + 10 ** ((self.elo.get(t2, 1500.0) - self.elo.get(t1, 1500.0)) / 400))

    def features(self, t1, t2, pool, t=None):
        """Per pool map: decayed play shares, residual sums/weights for both teams, plain play shares."""
        t = self.t_last if t is None else t
        out = {}
        for k, tm in (("1", t1), ("2", t2)):
            pw = np.array([self.play.sums(tm, m, t)[0] for m in pool])
            out["s" + k] = (pw / pw.sum()) if pw.sum() > 0 else np.full(len(pool), 1 / len(pool))
            c = np.array([self.count[(tm, m)] for m in pool])
            out["f" + k] = (c / c.sum()) if c.sum() > 0 else np.full(len(pool), 1 / len(pool))
            rs = np.array([self.res.sums(tm, m, t) for m in pool])
            out["R" + k], out["N" + k] = rs[:, 0], rs[:, 1]
        return out

    def _record(self, mid, t, t1, t2, pool, m):
        idx = {x: i for i, x in enumerate(pool)}
        return {"match_id": mid, "t": t, "bo": int(m.bo_type.iloc[0]), "t1": t1, "t2": t2, "pool": pool,
                "order": [idx[x] for x in m.map_name], "won1": m.won1.astype(int).tolist(),
                "p_elo": self._elo_p(t1, t2), **self.features(t1, t2, pool, t)}


# ---------- model pieces (vectorised over a record's pool)
def strength(R, N, k):
    return R / (N + k)


def pick_x(r, k):
    l1, l2 = np.log(r["s1"] + EPS), np.log(r["s2"] + EPS)
    d1, d2 = strength(r["R1"], r["N1"], k), strength(r["R2"], r["N2"], k)
    return np.column_stack([l1 + l2, np.abs(l1 - l2), np.minimum(l1, l2), d1 + d2])


def pick_weights(r, coef, k):
    s = pick_x(r, k) @ np.asarray(coef)
    w = np.exp(s - s.max())
    return w / w.sum()


def baseline_weights(r):
    """The previous rule: blend of both teams' plain play frequencies with a uniform floor."""
    w = 0.85 * (r["f1"] + r["f2"]) / 2 + 0.15 / len(r["pool"])
    return w / w.sum()


def pl_nll(w, order):
    lw, rem, out = np.log(np.clip(w, 1e-12, None)), np.ones(len(w), bool), 0.0
    for i in order:
        out += logsumexp(lw[rem]) - lw[i]
        rem[i] = False
    return out


def p_played12(w, bo):
    """P(each map is among the first min(bo, 2) picks) under Plackett-Luce."""
    w = np.asarray(w, float) / np.sum(w)
    if bo <= 1:
        return w
    P2 = w[:, None] * w[None, :] / np.clip(1 - w, 1e-12, None)[:, None]
    np.fill_diagonal(P2, 0)
    return P2.sum(1) + P2.sum(0)


def anchored(p_base, dd, wts):
    """Shift logit(p_base) by dd per map, re-centred so sum(wts * p) == p_base."""
    lb = logit(np.clip(p_base, 1e-4, 1 - 1e-4))
    if not np.any(dd):
        return np.full(len(dd), p_base)
    f = lambda c: float(np.dot(wts, expit(lb + c + dd)) - p_base)  # noqa: E731
    return expit(lb + optimize.brentq(f, -10, 10) + dd)


def win_probs(r, beta, k):
    dd = beta * (strength(r["R1"], r["N1"], k) - strength(r["R2"], r["N2"], k))
    return expit(logit(np.clip(r["p_elo"], 1e-4, 1 - 1e-4)) + dd)


def _win_ll(recs, beta, k):
    ll, n = 0.0, 0
    for r in recs:
        p = np.clip(win_probs(r, beta, k), 1e-6, 1 - 1e-6)
        for i, y in zip(r["order"], r["won1"]):
            ll -= y * np.log(p[i]) + (1 - y) * np.log(1 - p[i])
            n += 1
    return ll, n


def fit(recs, k_grid=(4.0, 8.0, 16.0, 32.0, 64.0)):
    """Fit the per-map win shift (beta, shrink) against Elo, then the pick coefficients."""
    best = None
    for k in k_grid:
        res = optimize.minimize_scalar(lambda b: _win_ll(recs, b, k)[0], bounds=(0, 6), method="bounded")
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x, k)
    _, beta, k = best

    def nll(c):
        return sum(pl_nll(pick_weights(r, c, k), r["order"]) for r in recs)
    coef = optimize.minimize(nll, np.array([1.0, 0.0, 0.0, 0.0]), method="L-BFGS-B").x
    return {"shrink": float(k), "beta": float(beta), "pick": [float(x) for x in coef]}


def evaluate(recs, prm):
    k = prm["shrink"]
    nb, nn, bb, bn, cnt = 0.0, 0.0, 0.0, 0.0, 0
    for r in recs:
        wb, wn = baseline_weights(r), pick_weights(r, prm["pick"], k)
        nb += pl_nll(wb, r["order"])
        nn += pl_nll(wn, r["order"])
        y = np.zeros(len(r["pool"]))
        y[r["order"][: min(r["bo"], 2)]] = 1
        bb += ((p_played12(wb, r["bo"]) - y) ** 2).sum()
        bn += ((p_played12(wn, r["bo"]) - y) ** 2).sum()
        cnt += len(y)
    l0, n = _win_ll(recs, 0.0, k)
    l1, _ = _win_ll(recs, prm["beta"], k)
    return {"n_series": len(recs),
            "picks": {"nll_per_series": {"baseline": nb / len(recs), "map_pool": nn / len(recs)},
                      "brier_played_maps12": {"baseline": bb / cnt, "map_pool": bn / cnt}},
            "wins": {"maps": n, "logloss": {"same_every_map": l0 / max(n, 1), "per_map": l1 / max(n, 1)}}}


def backtest(hist, test_frac=0.3):
    """Pick the half-life on the train period, fit there, score the later test period."""
    best = None
    for hl in HALF_LIVES:
        recs = _usable(MapState(hist, hl).records)
        n_tr = int(len(recs) * (1 - test_frac))
        prm = fit(recs[:n_tr])
        sc = evaluate(recs[:n_tr], prm)
        score = sc["picks"]["nll_per_series"]["map_pool"] + sc["wins"]["logloss"]["per_map"]
        if best is None or score < best[0]:
            best = (score, hl, recs, n_tr, prm)
    _, hl, recs, n_tr, prm = best
    prm["half_life"] = hl
    test = evaluate(recs[n_tr:], prm)
    dates = [pd.Timestamp(recs[i]["t"] * 86400, unit="s", tz="UTC").date().isoformat() for i in (n_tr, -1)]
    return prm, {**test, "test_period": dates, "half_life": hl}


def _usable(recs):
    if not recs:
        return recs
    t0 = recs[0]["t"] + BURN_IN_DAYS
    return [r for r in recs if r["t"] >= t0 and len(r["pool"]) >= 3]


def train(hist):
    """Backtest, then refit on all usable history. Each part is used in pricing only if it beat
    the previous approach on the held-out period."""
    prm, rep = backtest(hist)
    full = fit(_usable(MapState(hist, prm["half_life"]).records))
    p, w = rep["picks"]["nll_per_series"], rep["wins"]
    out = {**DEFAULT, **full, "half_life": prm["half_life"],
           "use_picks": bool(p["map_pool"] < p["baseline"]),
           "use_wins": bool(w["logloss"]["per_map"] < w["logloss"]["same_every_map"])}
    return out, rep


class MapModel:
    """Current-state queries for upcoming matches."""

    def __init__(self, hist, params):
        self.p = {**DEFAULT, **(params or {})}
        self.state = MapState(hist, self.p["half_life"], records=False)

    def _rec(self, t1, t2, pool):
        return {"pool": pool, **self.state.features(t1, t2, pool)}

    def weights(self, t1, t2, pool):
        r = self._rec(t1, t2, pool)
        return pick_weights(r, self.p["pick"], self.p["shrink"]) if self.p["use_picks"] else baseline_weights(r)

    def strength_diff(self, t1, t2, pool):
        """Per pool map logit shift for team1 (all 0 when the per-map win model isn't in use)."""
        if not self.p["use_wins"]:
            return np.zeros(len(pool))
        r, k = self._rec(t1, t2, pool), self.p["shrink"]
        return self.p["beta"] * (strength(r["R1"], r["N1"], k) - strength(r["R2"], r["N2"], k))

    def strengths(self, team, pool):
        k = self.p["shrink"]
        return {m: strength(*self.state.res.sums(team, m, self.state.t_last), k) for m in pool}


if __name__ == "__main__":
    from ..scraper.bo3_history import HIST_PATH
    prm, rep = train(pd.read_csv(HIST_PATH))
    print(json.dumps({"params": prm, "report": rep}, indent=1, default=float))
