"""Verified pre-match status of bo3.gg matches. A record may be logged as pre-match only while bo3.gg still lists the
match as upcoming with no map begun; sportsbook/DFS scheduled start times are never trusted on their own."""
from datetime import datetime, timedelta, timezone

UNAVAILABLE = None  # status could not be fetched: nothing is treated as pre-match


def _key(a, b):
    return frozenset((int(a), int(b)))


def fetch_match_status(ttl=60, days_ahead=1):
    """{frozenset(team ids): {"match_id", "status", "started", "scheduled"}} from bo3.gg upcoming, live and finished
    feeds. Raises when any feed fails, so callers can treat the status as unknown."""
    from ..scraper.books import BO3_API
    from ..scraper.http import fetch_json
    today = datetime.now(timezone.utc).date()
    out = {}

    def add(m, started):
        if not (m.get("team1_id") and m.get("team2_id")):
            return
        games = m.get("games") or []
        started = started or m.get("status") != "upcoming" or any((g or {}).get("status") not in (None, "upcoming") for g in games)
        k = _key(m["team1_id"], m["team2_id"])
        prev = out.get(k)
        if prev and prev["started"] and not started:
            return  # an earlier meeting of the same teams already started today; keep the conservative entry
        out[k] = {"match_id": int(m["id"]), "status": m.get("status"), "started": bool(started), "scheduled": m.get("start_date")}

    def tiers(d):
        data = d.get("data")
        if isinstance(data, list):
            return data
        return [m for t in ((data or {}).get("tiers") or {}).values() for m in t.get("matches", [])]

    params = {"utc_offset": 0, "filter[discipline_id][eq]": 1}
    for i in range(days_ahead + 1):
        for m in tiers(fetch_json(f"{BO3_API}/v2/matches/upcoming", ttl=ttl, params={**params, "date": (today + timedelta(days=i)).isoformat()})):
            add(m, False)
    for m in tiers(fetch_json(f"{BO3_API}/v2/matches/finished", ttl=ttl, params={**params, "date": today.isoformat()})):
        add(m, True)
    for m in tiers(fetch_json(f"{BO3_API}/v2/matches/live", ttl=ttl, params={**params, "date": today.isoformat()})):
        add(m, True)
    return out


def safe_status(fetch=fetch_match_status):
    try:
        return fetch()
    except Exception:  # noqa: BLE001
        return UNAVAILABLE


def classify(team_id, opp_id, status):
    """(eligible, bo3 match id, reason). Unknown status, unknown match or a started match is never eligible."""
    if status is UNAVAILABLE:
        return False, None, "status_unavailable"
    try:
        info = status.get(_key(team_id, opp_id))
    except (TypeError, ValueError):
        return False, None, "no_verified_match"
    if not info:
        return False, None, "no_verified_match"
    if info["started"]:
        return False, info["match_id"], "started"
    return True, info["match_id"], None
