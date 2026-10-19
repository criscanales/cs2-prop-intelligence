"""Leak-free features. Every feature on a row uses only maps that finished strictly before it.

Rows to be predicted (future maps) can be appended with NaN outcomes; they sort last and
only see history.
"""
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20)
EWM_HALFLIFE = 8
TIER_RANK = {"s": 0, "a": 1, "b": 2, "c": 3, "d": 4}
PRIOR_KPR, PRIOR_HS = 0.68, 0.48
OT_ROUNDS = 24  # MR12: more than 24 rounds means overtime
ELO_K = 24.0

PLAYER_FEATURES = (
    [f"{s}_{w}" for w in WINDOWS for s in ("kpr", "hs_pct", "rating", "adr", "kast", "open_rate", "fk_share")]
    + ["kpr_ewm", "hs_pct_ewm", "rating_ewm", "kpr_shrunk", "hs_pct_shrunk", "n_maps", "kills_sd_20",
       "map_kpr", "map_n", "tenure", "days_rest", "maps_7d", "role_shift", "rounds_20"]
)
TEAM_FEATURES = ["team_elo", "opp_elo", "elo_diff", "p_win_elo", "team_pace", "opp_pace", "exp_rounds",
                 "closeness", "team_ot_rate", "opp_ot_rate", "opp_dpr", "opp_map_dpr", "team_kpr",
                 "team_map_wr", "opp_map_wr", "team_new_players", "opp_new_players"]
CONTEXT_FEATURES = ["bo_type", "is_lan", "tier_rank", "map_number", "map_code"]
FEATURES = PLAYER_FEATURES + TEAM_FEATURES + CONTEXT_FEATURES
AWP_PATH = Path(__file__).resolve().parents[2] / "data" / "awp_maps.csv"
AWP_PRIOR, AWP_PRIOR_KILLS = 0.12, 30
MAPS = ["de_ancient", "de_anubis", "de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_overpass",
        "de_train", "de_vertigo"]


def _grp_roll(g, col, w, stat="sum", min_periods=1):
    """Per-group rolling stat of the previous w rows (excludes the current row); vectorised."""
    lag = g[col].shift(1)
    r = getattr(lag.groupby(g.ngroup(), sort=False).rolling(w, min_periods=min_periods), stat)()
    return r.reset_index(level=0, drop=True).reindex(lag.index)


def _grp_ewm(g, col):
    lag = g[col].shift(1)
    r = lag.groupby(g.ngroup(), sort=False).ewm(halflife=EWM_HALFLIFE, ignore_na=True).mean()
    return r.reset_index(level=0, drop=True).reindex(lag.index)


def _grp_cumsum_lag(g, col):
    return g[col].shift(1).fillna(0).groupby(g.ngroup(), sort=False).cumsum()


def _roll_sum(g, col, w):
    return _grp_roll(g, col, w, "sum")


def _player_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("kills", "deaths", "headshots", "first_kills", "first_deaths"):
        df[c] = df[c].where(df[c] >= 0)
    df["headshots"] = df["headshots"].clip(upper=df["kills"])
    df["open"] = df["first_kills"] + df["first_deaths"]
    g = df.groupby("player_id", sort=False)
    for w in WINDOWS:
        k, r, h = _roll_sum(g, "kills", w), _roll_sum(g, "rounds", w), _roll_sum(g, "headshots", w)
        fk, op = _roll_sum(g, "first_kills", w), _roll_sum(g, "open", w)
        df[f"kpr_{w}"] = k / r
        df[f"hs_pct_{w}"] = h / k.replace(0, np.nan)
        df[f"open_rate_{w}"] = op / r
        df[f"fk_share_{w}"] = fk / op.replace(0, np.nan)
        for c in ("rating", "adr", "kast"):
            df[f"{c}_{w}"] = _grp_roll(g, c, w, "mean")
    def ew(c):
        return _grp_ewm(g, c)

    kr, rr, hr = ew("kills"), ew("rounds"), ew("headshots")
    df["kpr_ewm"] = kr / rr
    df["hs_pct_ewm"] = hr / kr
    df["rating_ewm"] = ew("rating")
    df["n_maps"] = g.cumcount()
    k20, r20, h20 = _roll_sum(g, "kills", 20), _roll_sum(g, "rounds", 20), _roll_sum(g, "headshots", 20)
    df["kpr_shrunk"] = (k20.fillna(0) + 120 * PRIOR_KPR) / (r20.fillna(0) + 120)
    df["hs_pct_shrunk"] = (h20.fillna(0) + 40 * PRIOR_HS) / (k20.fillna(0) + 40)
    df["kills_sd_20"] = _grp_roll(g, "kills", 20, "std", 3)
    df["rounds_20"] = _grp_roll(g, "rounds", 20, "mean")
    df["role_shift"] = (df["fk_share_5"] - df["fk_share_20"]).abs()
    # map-specific kill rate, shrunk toward the player's overall form
    gm = df.groupby(["player_id", "map_name"], sort=False)
    mk = _grp_cumsum_lag(gm, "kills")
    mr = _grp_cumsum_lag(gm, "rounds")
    df["map_n"] = gm.cumcount()
    df["map_kpr"] = (mk + 60 * df["kpr_shrunk"]) / (mr + 60)
    df["tenure"] = df.groupby(["player_id", "team_id"], sort=False).cumcount()
    prev = g["date"].shift(1)
    df["days_rest"] = (df["date"] - prev).dt.total_seconds() / 86400
    df["maps_7d"] = _count_recent(df, 7)
    return df


AWP_MIN_MAPS = 5


def awp_role(awp_kills: int, kills: int, maps: int) -> tuple[str | None, float]:
    """Observed role from recent AWP kill share; None when there are too few maps to label."""
    share = awp_kills / kills if kills > 0 else 0.0
    if maps < AWP_MIN_MAPS or kills <= 0:
        return None, share
    return ("AWPer" if share >= 0.30 else "Hybrid" if share >= 0.10 else "Rifler"), share


def load_awp(path=AWP_PATH) -> pd.DataFrame | None:
    """Per player-map AWP kills from bo3.gg weapon stats (game_id, player_id, weapon_kills, awp_kills, awp_hs)."""
    if not Path(path).exists():
        return None
    a = pd.read_csv(path).drop_duplicates(["game_id", "player_id"], keep="last")
    return a[a.weapon_kills > 0]


def attach_awp(h: pd.DataFrame, awp: pd.DataFrame | None = None) -> pd.DataFrame:
    """AWP columns from the crawl when present, filled from the backfill file for older maps."""
    for c in ("weapon_kills", "awp_kills"):
        if c not in h:
            h[c] = np.nan
    awp = load_awp() if awp is None else awp
    if awp is None or awp.empty:
        return h
    m = h[["game_id", "player_id"]].merge(awp[["game_id", "player_id", "weapon_kills", "awp_kills"]],
                                          on=["game_id", "player_id"], how="left")
    have = (h["weapon_kills"] > 0).values
    for c in ("weapon_kills", "awp_kills"):
        h[c] = np.where(have, h[c].values, m[c].values)
    return h


def _awp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Share of a player's kills with the AWP over his last 20 maps that have weapon data, shrunk to the average."""
    g = df.groupby("player_id", sort=False)
    a20, k20 = _roll_sum(g, "awp_kills", 20), _roll_sum(g, "weapon_kills", 20)
    df["awp_maps"] = _grp_roll(g, "awp_kills", 20, "count")
    df["awp_share"] = ((a20.fillna(0) + AWP_PRIOR_KILLS * AWP_PRIOR) / (k20.fillna(0) + AWP_PRIOR_KILLS)).where(df["awp_maps"] > 0)
    return df


def _count_recent(df, days):
    out = np.zeros(len(df))
    for _, idx in df.groupby("player_id", sort=False).indices.items():
        t = df["date"].values[idx].astype("datetime64[s]").astype(np.int64)
        left = np.searchsorted(t, t - days * 86400, side="left")
        out[idx] = np.arange(len(idx)) - left
    return out


def team_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team): score, kills, deaths, lineup newcomers."""
    df["new_player"] = (df["tenure"] < 5).astype(int)
    t = df.groupby(["game_id", "team_id"], sort=False).agg(
        date=("date", "first"), opp_id=("opp_id", "first"), map_name=("map_name", "first"),
        rounds=("rounds", "first"), won=("won", "first"), team_kills=("kills", "sum"),
        team_deaths=("deaths", "sum"), new_players=("new_player", "sum"), future=("future", "first"),
    ).reset_index()
    t.loc[t["future"], ["team_kills", "team_deaths", "rounds", "won"]] = np.nan
    return t.sort_values(["date", "game_id"]).reset_index(drop=True)


def _elo(t: pd.DataFrame) -> pd.Series:
    ratings, out = {}, np.full(len(t), np.nan)
    for gid, idx in t.groupby("game_id", sort=False).indices.items():
        if len(idx) != 2:
            for i in idx:
                out[i] = ratings.get(t.at[i, "team_id"], 1500.0)
            continue
        a, b = idx
        ta, tb = t.at[a, "team_id"], t.at[b, "team_id"]
        ra, rb = ratings.get(ta, 1500.0), ratings.get(tb, 1500.0)
        out[a], out[b] = ra, rb
        w = t.at[a, "won"]
        if w == w:  # not NaN
            ea = 1 / (1 + 10 ** ((rb - ra) / 400))
            ratings[ta] = ra + ELO_K * (w - ea)
            ratings[tb] = rb - ELO_K * (w - ea)
    return pd.Series(out, index=t.index)


def _team_features(t: pd.DataFrame) -> pd.DataFrame:
    t["elo"] = _elo(t)
    g = t.groupby("team_id", sort=False)
    t["pace"] = _grp_roll(g, "rounds", 20, "mean")
    t["ot"] = (t["rounds"] > OT_ROUNDS).where(t["rounds"].notna())
    t["ot_rate"] = _grp_roll(t.assign(ot=t["ot"].astype(float)).groupby("team_id", sort=False), "ot", 20, "mean")
    t["dpr"] = _roll_sum(g, "team_deaths", 20) / _roll_sum(g, "rounds", 20) / 5
    t["kpr_team"] = _roll_sum(g, "team_kills", 20) / _roll_sum(g, "rounds", 20) / 5
    gm = t.groupby(["team_id", "map_name"], sort=False)
    wins = _grp_cumsum_lag(gm, "won")
    n = gm.cumcount()
    t["map_wr"] = (wins + 2.5) / (n + 5)
    md = _grp_cumsum_lag(gm, "team_deaths")
    mr = _grp_cumsum_lag(gm, "rounds")
    t["map_dpr"] = (md / 5 + 100 * t["dpr"].fillna(PRIOR_KPR)) / (mr + 100)
    return t


def build_features(hist: pd.DataFrame, future: pd.DataFrame | None = None) -> pd.DataFrame:
    """hist: rows from bo3_history. future: same columns (outcomes may be NaN)."""
    h = hist.copy()
    h["future"] = False
    if future is not None and len(future):
        f = future.copy()
        f["future"] = True
        f["date"] = pd.Timestamp("2100-01-01", tz="UTC")
        h = pd.concat([h, f], ignore_index=True)
    h["date"] = pd.to_datetime(h["date"], utc=True, format="ISO8601")
    h = h.sort_values(["future", "date", "game_id", "player_id"], kind="stable").reset_index(drop=True)
    h = _awp_features(_player_features(attach_awp(h)))
    t = _team_features(team_table(h))
    tf = t.set_index(["game_id", "team_id"])
    key_team = pd.MultiIndex.from_arrays([h["game_id"], h["team_id"]])
    key_opp = pd.MultiIndex.from_arrays([h["game_id"], h["opp_id"]])
    for src, dst in [("elo", "team_elo"), ("pace", "team_pace"), ("ot_rate", "team_ot_rate"),
                     ("kpr_team", "team_kpr"), ("map_wr", "team_map_wr"), ("new_players", "team_new_players")]:
        h[dst] = tf[src].reindex(key_team).values
    for src, dst in [("elo", "opp_elo"), ("pace", "opp_pace"), ("ot_rate", "opp_ot_rate"), ("dpr", "opp_dpr"),
                     ("map_dpr", "opp_map_dpr"), ("map_wr", "opp_map_wr"), ("new_players", "opp_new_players")]:
        h[dst] = tf[src].reindex(key_opp).values
    h["opp_elo"] = h["opp_elo"].fillna(1500.0)
    h["elo_diff"] = h["team_elo"] - h["opp_elo"]
    h["p_win_elo"] = 1 / (1 + 10 ** (-h["elo_diff"] / 400))
    if "p_win_market" in h:
        h["p_win"] = h["p_win_market"].fillna(h["p_win_elo"])
    else:
        h["p_win"] = h["p_win_elo"]
    h["closeness"] = 1 - (2 * h["p_win"] - 1).abs()
    h["exp_rounds"] = h[["team_pace", "opp_pace"]].mean(axis=1)
    h["is_lan"] = (h["event_type"] == "lan").astype(float).where(h["event_type"].notna())
    h["tier_rank"] = h["tier"].map(TIER_RANK)
    h["map_code"] = h["map_name"].map({m: i for i, m in enumerate(MAPS)}).fillna(-1)
    return h
