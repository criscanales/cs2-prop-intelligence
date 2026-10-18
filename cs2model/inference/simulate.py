"""Joint Monte Carlo of a match: shared map picks, rounds (pace/overtime) and map winners drive
every player's kills and headshots, so same-match / same-team / kills-vs-HS legs are correlated."""
from dataclasses import dataclass, field

import numpy as np

from .distribution import win_multipliers


@dataclass
class PlayerSpec:
    key: str                 # player id/name key
    side: int                # 1 = team1, 2 = team2
    kpr: dict                # map_name -> expected kills per round
    hs_rate: dict            # map_name -> expected HS per kill
    alpha: float             # NB dispersion
    rho: float               # beta-binomial over-dispersion for HS


@dataclass
class MatchSpec:
    key: str
    best_of: int
    maps: list               # candidate map names
    map_weights: list        # pick probabilities (sum to 1)
    p_map_team1: dict        # map number -> P(team1 wins that map)
    rounds: dict             # map number -> (support, probs)
    win_ratio: float = 1.3
    players: list = field(default_factory=list)
    p_win_map: dict = field(default_factory=dict)  # map number -> P(team1 wins) per candidate map (optional)


def win_vec(spec, j):
    """P(team1 wins map slot j) for each candidate map."""
    v = getattr(spec, "p_win_map", None) or {}
    if j in v:
        return np.asarray(v[j], float)
    return np.full(len(spec.maps), spec.p_map_team1.get(j, 0.5))


def simulate_match(spec: MatchSpec, n: int, rng: np.random.Generator):
    """Returns {player_key: {"kills": (n, n_maps), "headshots": (n, n_maps)}, "_played": (n, n_maps)}."""
    n_maps = min(spec.best_of, 3)
    w = np.asarray(spec.map_weights, float)
    w = w / w.sum()
    # map picks without replacement (Gumbel top-k)
    g = np.log(w)[None, :] - np.log(-np.log(rng.random((n, len(w)))))
    picks = np.argsort(-g, axis=1)[:, :n_maps]
    if picks.shape[1] < n_maps:  # tiny candidate pool: allow repeats for the remaining slots
        extra = rng.choice(len(w), size=(n, n_maps - picks.shape[1]), p=w)
        picks = np.hstack([picks, extra])
    rounds = np.zeros((n, n_maps))
    t1win = np.zeros((n, n_maps), bool)
    for j in range(n_maps):
        sup, pr = spec.rounds.get(j + 1) or spec.rounds[1]
        rounds[:, j] = rng.choice(sup, size=n, p=pr)
        t1win[:, j] = rng.random(n) < win_vec(spec, j + 1)[picks[:, j]]
    played = np.ones((n, n_maps), bool)
    if spec.best_of == 3 and n_maps == 3:
        played[:, 2] = t1win[:, 0] != t1win[:, 1]
    out = {"_played": played}
    for ps in spec.players:
        kpr = np.array([ps.kpr.get(m, np.mean(list(ps.kpr.values()))) for m in spec.maps])[picks]
        hsr = np.array([ps.hs_rate.get(m, np.mean(list(ps.hs_rate.values()))) for m in spec.maps])[picks]
        kills = np.zeros((n, n_maps))
        hs = np.zeros((n, n_maps))
        for j in range(n_maps):
            p1 = win_vec(spec, j + 1)[picks[:, j]]
            p_side = p1 if ps.side == 1 else 1 - p1
            m_w, m_l = win_multipliers(p_side, spec.win_ratio)
            won = t1win[:, j] if ps.side == 1 else ~t1win[:, j]
            mu = kpr[:, j] * rounds[:, j] * np.where(won, m_w, m_l)
            lam = rng.gamma(1 / ps.alpha, ps.alpha * mu)
            k = rng.poisson(lam)
            r = max(ps.rho, 1e-6)
            a = hsr[:, j] * (1 - r) / r
            b = (1 - hsr[:, j]) * (1 - r) / r
            kills[:, j] = k
            hs[:, j] = rng.binomial(k, rng.beta(a, b))
        out[ps.key] = {"kills": kills * played, "headshots": hs * played}
    return out


def leg_totals(sim, player_key, stat, maps):
    arr = sim[player_key][stat]
    idx = [m - 1 for m in maps if m - 1 < arr.shape[1]]
    return arr[:, idx].sum(1)
