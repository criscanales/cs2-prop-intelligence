"""Filters for the Player panel's recent-series chart and its L5/L10/L15/H2H cells. Display only: never used for pricing."""
import html

import numpy as np
import pandas as pd

LIST_KEYS = ("opp", "team", "event", "bo", "tier", "maps")


def series_table(h, maps):
    """One row per series in the player's history: the context the chart can be filtered on."""
    g = h[h.map_number.isin(maps)]
    sm = g.groupby("match_id").agg(opp=("opponent", "first"), team=("team", "first"), event=("tournament_id", "first"),
                                   bo=("bo_type", "first"), tier=("tier", "first"), lan=("event_type", "first"),
                                   rounds=("rounds", "sum"), map_names=("map_name", lambda s: tuple(s.dropna())))
    w = h[h.match_id.isin(sm.index)].groupby("match_id").won.agg(["sum", "size"]).reindex(sm.index)
    wins, losses = w["sum"], w["size"] - w["sum"]
    sm["res"] = np.where(wins > losses, "Won", np.where(losses > wins, "Lost", None))
    return sm


def apply(sm, f):
    """match_ids passing every active filter (empty list / "Any" = not filtering)."""
    m = pd.Series(True, index=sm.index)
    for k in ("opp", "team", "event", "bo", "tier"):
        if f.get(k):
            m &= sm[k].isin(f[k])
    if f.get("result") in ("Won", "Lost"):
        m &= sm.res == f["result"]
    if f.get("lan") in ("LAN", "Online"):
        m &= sm.lan == f["lan"].lower()
    if f.get("rounds"):
        lo, hi = f["rounds"]
        m &= sm.rounds.between(lo, hi)
    if f.get("maps"):
        sel = set(f["maps"])
        m &= sm.map_names.map(lambda t: bool(sel & set(t)))
    return sm.index[m.values]


def n_active(f, full_rounds):
    return (sum(bool(f.get(k)) for k in LIST_KEYS) + (f.get("result") in ("Won", "Lost")) + (f.get("lan") in ("LAN", "Online"))
            + (f.get("rounds") is not None and tuple(f["rounds"]) != tuple(full_rounds)))


def widget_prefix(state, pre):
    """Key prefix for the filter widgets; a reset bumps the nonce so Streamlit mounts fresh widgets at their defaults."""
    return f"{pre}{state.get(pre + 'nonce', 0)}_"


def reset_state(state, pre):
    """Drop every filter value under `pre` and move the widgets to new keys (clears applied and displayed selections)."""
    n = state.get(pre + "nonce", 0)
    for k in [k for k in state if str(k).startswith(pre)]:
        del state[k]
    state[pre + "nonce"] = n + 1


def kpi_value_html(v):
    """KPI value that wraps between " / " parts (e.g. career K / D / A) instead of clipping in a narrow tile."""
    parts = [html.escape(p) for p in str(v).split(" / ")]
    if len(parts) == 1:
        return f'<span style="white-space:nowrap">{parts[0]}</span>'
    return " / ".join(f'<span style="white-space:nowrap">{p}</span>' for p in parts)
