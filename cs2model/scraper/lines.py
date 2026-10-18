"""Player prop lines from Underdog Fantasy and PrizePicks public endpoints."""
import re
from datetime import datetime

from ..types import PropLine
from .http import fetch_json

UNDERDOG_URL = "https://api.underdogfantasy.com/v1/over_under_lines"
PRIZEPICKS_URL = "https://partner-api.prizepicks.com/projections"
PRIZEPICKS_CS2_LEAGUE = 265


def _parse_time(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_stat(text: str):
    """'Kills on Maps 1+2' / 'MAPS 1-3 Headshots' / 'MAP 3 Kills' -> ('kills', (1, 2))."""
    t = text.lower()
    if "headshot" in t:
        stat = "headshots"
    elif "kill" in t:
        stat = "kills"
    else:
        return None
    m = re.search(r"maps?\s*([\d+\-\s]+)", t)
    if not m:
        return None
    nums = [int(x) for x in re.findall(r"\d", m.group(1))]
    if "-" in m.group(1) and len(nums) == 2:
        maps = tuple(range(nums[0], nums[1] + 1))
    else:
        maps = tuple(nums)
    if not maps:
        return None
    return stat, maps


def fetch_underdog(ttl=120) -> list[PropLine]:
    d = fetch_json(UNDERDOG_URL, ttl=ttl)
    games = {g["id"]: g for g in d.get("games", []) if g.get("sport_id") == "CS"}
    players = {p["id"]: p for p in d.get("players", []) if p.get("sport_id") == "CS"}
    apps = {a["id"]: a for a in d.get("appearances", [])}
    out = []
    for ln in d.get("over_under_lines", []):
        if ln.get("status") not in (None, "active"):
            continue
        ou = ln.get("over_under") or {}
        st = ou.get("appearance_stat") or {}
        app = apps.get(st.get("appearance_id"))
        if not app or app.get("player_id") not in players:
            continue
        game = games.get(app.get("match_id"))
        if not game:
            continue
        parsed = _parse_stat(st.get("display_stat", ""))
        if not parsed:
            continue
        p = players[app["player_id"]]
        name = f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip()
        title = game.get("full_team_names_title") or game.get("title") or ""
        teams = [t.strip() for t in re.split(r"\s+(?:vs\.?|@)\s+", title)]
        if len(teams) != 2:
            continue
        # Underdog titles list the away team first ("away vs home" / "away @ home").
        away, home = teams
        if app.get("team_id") == game.get("home_team_id"):
            team, opp = home, away
        else:
            team, opp = away, home
        over = under = None
        for opt in ln.get("options", []):
            dec = opt.get("decimal_price") or ((opt.get("odds") or {}).get("fantasy") or {}).get("decimal")
            if opt.get("choice") == "higher":
                over = float(dec) if dec else None
            elif opt.get("choice") == "lower":
                under = float(dec) if dec else None
        out.append(PropLine(
            source="underdog", player=name, team=team, opponent=opp, stat=parsed[0], maps=parsed[1],
            line=float(ln["stat_value"]), start_time=_parse_time(game.get("scheduled_at")),
            over_decimal=over, under_decimal=under, over_only=under is None,
        ))
    return out


def fetch_prizepicks(ttl=120) -> list[PropLine]:
    d = fetch_json(PRIZEPICKS_URL, params={"league_id": PRIZEPICKS_CS2_LEAGUE, "per_page": 500}, ttl=ttl)
    inc = {(i["type"], i["id"]): i for i in d.get("included", [])}
    out = []
    for proj in d.get("data", []):
        a = proj["attributes"]
        if a.get("event_type") == "team" and a.get("projection_type") != "Single Stat":
            continue
        parsed = _parse_stat(a.get("stat_type", ""))
        if not parsed:
            continue
        pid = ((proj.get("relationships") or {}).get("new_player") or {}).get("data") or {}
        pl = inc.get(("new_player", pid.get("id")))
        if not pl:
            continue
        pa = pl["attributes"]
        if pa.get("combo"):
            continue
        opp = re.sub(r"\s*MAPS?\s*[\d\-+ ]+$", "", a.get("description") or "", flags=re.I).strip()
        out.append(PropLine(
            source="prizepicks", player=pa.get("name") or pa.get("display_name"), team=pa.get("team_name") or pa.get("team") or "",
            opponent=opp, stat=parsed[0], maps=parsed[1], line=float(a["line_score"]),
            start_time=_parse_time(a.get("start_time")), odds_type=a.get("odds_type") or "standard",
            over_only=(a.get("odds_type") in ("goblin", "demon")),
        ))
    return out


def fetch_all_lines(sources=("underdog", "prizepicks")) -> tuple[list[PropLine], list[str]]:
    lines, errors = [], []
    fns = {"underdog": fetch_underdog, "prizepicks": fetch_prizepicks}
    for s in sources:
        try:
            lines.extend(fns[s]())
        except Exception as e:  # noqa: BLE001 - one dead source should not kill the run
            errors.append(f"{s}: {e}")
    return lines, errors
