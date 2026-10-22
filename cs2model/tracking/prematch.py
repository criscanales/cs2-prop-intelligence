"""Verified pre-match status of bo3.gg matches. Records are tied to a bo3.gg match id (never just a team pair) and are
loggable only while bo3.gg lists that match as upcoming with no map begun; DFS scheduled start times are never trusted
on their own, and a match still listed as upcoming after its own scheduled start is only "uncertain"."""
from datetime import datetime, timedelta, timezone

import pandas as pd

UNAVAILABLE = None  # status could not be fetched: nothing is treated as pre-match
AMBIGUOUS_GAP = timedelta(hours=2)


def _key(a, b):
    return frozenset((int(a), int(b)))


def _ts(x):
    if x is None:
        return None
    try:
        t = pd.Timestamp(x)
    except (TypeError, ValueError):
        return None
    if pd.isna(t):
        return None
    return t.tz_localize("UTC") if t.tzinfo is None else t


def fetch_match_status(ttl=60, days_ahead=1):
    """{frozenset(team ids): [{"match_id", "status", "started", "scheduled"}, ...]}: one entry per bo3.gg match id from
    the upcoming, live and finished feeds (a match seen as started in any feed is started). Raises when a feed fails,
    so callers can treat the status as unknown."""
    from ..scraper.books import BO3_API
    from ..scraper.http import fetch_json
    today = datetime.now(timezone.utc).date()
    by_id = {}

    def add(m, started):
        if not (m.get("id") and m.get("team1_id") and m.get("team2_id")):
            return
        games = m.get("games") or []
        started = started or m.get("status") != "upcoming" or any((g or {}).get("status") not in (None, "upcoming") for g in games)
        e = by_id.setdefault(int(m["id"]), {"match_id": int(m["id"]), "teams": _key(m["team1_id"], m["team2_id"]),
                                           "status": m.get("status"), "started": False, "scheduled": m.get("start_date")})
        if started and not e["started"]:
            e.update(started=True, status=m.get("status"))

    def matches(d):
        data = d.get("data")
        if isinstance(data, list):
            return data
        return [m for t in ((data or {}).get("tiers") or {}).values() for m in t.get("matches", [])]

    params = {"utc_offset": 0, "filter[discipline_id][eq]": 1}
    for i in range(days_ahead + 1):
        for m in matches(fetch_json(f"{BO3_API}/v2/matches/upcoming", ttl=ttl, params={**params, "date": (today + timedelta(days=i)).isoformat()})):
            add(m, False)
    for feed in ("finished", "live"):
        for m in matches(fetch_json(f"{BO3_API}/v2/matches/{feed}", ttl=ttl, params={**params, "date": today.isoformat()})):
            add(m, True)
    out = {}
    for e in by_id.values():
        out.setdefault(e.pop("teams"), []).append(e)
    return out


def safe_status(fetch=fetch_match_status):
    try:
        return fetch()
    except Exception:  # noqa: BLE001
        return UNAVAILABLE


def classify(team_id, opp_id, status, start_time=None, now=None):
    """(level, bo3 match id, reason). level "scheduled": bo3.gg lists the match as upcoming, no map begun, and its own
    scheduled start is still ahead. "uncertain": still listed upcoming but past its scheduled start (a delay, or a stale
    feed), so the record is held out of reports until the history verifies it was logged before Map 1. None: not
    loggable; reason is status_unavailable, no_verified_match, ambiguous_match or started. When the same teams have
    several bo3.gg matches, the one scheduled nearest the record's start time is used, unless two are within
    AMBIGUOUS_GAP of each other."""
    if status is UNAVAILABLE:
        return None, None, "status_unavailable"
    try:
        cands = status.get(_key(team_id, opp_id))
    except (TypeError, ValueError):
        return None, None, "no_verified_match"
    if isinstance(cands, dict):
        cands = [cands]
    if not cands:
        return None, None, "no_verified_match"
    if len(cands) > 1:
        st = _ts(start_time)
        if st is None:
            return None, None, "ambiguous_match"
        far = timedelta(days=3650)
        d = sorted((abs(_ts(c.get("scheduled")) - st) if _ts(c.get("scheduled")) else far, i) for i, c in enumerate(cands))
        if d[1][0] - d[0][0] < AMBIGUOUS_GAP:
            return None, None, "ambiguous_match"
        cands = [cands[d[0][1]]]
    c = cands[0]
    if c["started"]:
        return None, c["match_id"], "started"
    sched = _ts(c.get("scheduled"))
    if sched is None or (now or pd.Timestamp.now(tz="UTC")) >= sched:
        return "uncertain", c["match_id"], "past_scheduled_start"
    return "scheduled", c["match_id"], None
