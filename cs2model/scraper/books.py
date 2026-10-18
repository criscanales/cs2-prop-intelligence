"""Match / map moneylines, round totals and overtime odds.

- Pinnacle guest API (same public endpoints pinnacle.com calls when logged out).
- Thunderpick moneylines as relayed by bo3.gg's public upcoming-matches feed.
  thunderpick.io itself answers 403 to non-browser clients, so it is never called directly.
"""
import re
from datetime import datetime, timedelta, timezone

from ..types import MatchOdds
from .http import fetch_json
from ..odds import american_to_decimal, devig, map_prob_from_series

PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
# Public key the pinnacle.com web app sends for guest (logged-out) requests.
PINNACLE_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"
PINNACLE_ESPORTS = 12

BO3_API = "https://api.bo3.gg/api"


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def fetch_pinnacle(ttl=300) -> list[MatchOdds]:
    h = {"X-API-Key": PINNACLE_KEY, "Referer": "https://www.pinnacle.com/"}
    matchups = fetch_json(f"{PINNACLE_BASE}/sports/{PINNACLE_ESPORTS}/matchups", headers=h, ttl=ttl)
    markets = fetch_json(f"{PINNACLE_BASE}/sports/{PINNACLE_ESPORTS}/markets/straight", headers=h, ttl=ttl)
    mk_by = {}
    for q in markets:
        mk_by.setdefault(q["matchupId"], []).append(q)

    out = {}
    for m in matchups:
        if not m["league"]["name"].startswith("CS2") or m.get("parentId") or m.get("type") != "matchup":
            continue
        home = next(p for p in m["participants"] if p["alignment"] == "home")["name"]
        away = next(p for p in m["participants"] if p["alignment"] == "away")["name"]
        mo = MatchOdds("pinnacle", home, away, _t(m.get("startTime")), int(m.get("bestOfX") or 3))
        for q in mk_by.get(m["id"], []):
            if q.get("status", "open") != "open":
                continue
            per = q["period"]
            prices = {p.get("designation"): american_to_decimal(p["price"]) for p in q["prices"] if "designation" in p}
            if q["type"] == "moneyline" and not q.get("isAlternate") and {"home", "away"} <= prices.keys():
                ph, _ = devig(prices["home"], prices["away"])
                if per == 0:
                    mo.p_series_team1 = ph
                else:
                    mo.p_map_team1[per] = ph
            elif q["type"] == "total" and {"over", "under"} <= prices.keys():
                pts = next(p["points"] for p in q["prices"] if p.get("designation") == "over")
                po, _ = devig(prices["over"], prices["under"])
                if per == 0:
                    if abs(pts - 2.5) < 1e-9 and mo.best_of == 3:
                        mo.p_map3 = po
                else:
                    mo.rounds_ladder.setdefault(per, []).append((pts, po))
        out[m["id"]] = mo

    # Overtime yes/no specials hang off the parent matchup.
    for m in matchups:
        parent = out.get(m.get("parentId"))
        desc = ((m.get("special") or {}).get("description") or "")
        mm = re.search(r"\(Map (\d)\)", desc)
        if not parent or m.get("units") != "Overtime" or not mm:
            continue
        mapno = int(mm.group(1))
        ids = {p["name"]: p["id"] for p in m["participants"]}
        for q in mk_by.get(m["id"], []):
            pr = {p["participantId"]: american_to_decimal(p["price"]) for p in q["prices"] if "participantId" in p}
            if ids.get("Yes") in pr and ids.get("No") in pr:
                parent.p_overtime[mapno] = devig(pr[ids["Yes"]], pr[ids["No"]])[0]
    for mo in out.values():
        for k in mo.rounds_ladder:
            mo.rounds_ladder[k].sort()
        _fill_series(mo)
    return list(out.values())


def fetch_thunderpick_via_bo3(ttl=300, days_ahead=2) -> list[MatchOdds]:
    """Pre-match Thunderpick decimal odds embedded in bo3.gg upcoming matches (`bet_updates`)."""
    out = []
    today = datetime.now(timezone.utc).date()
    for i in range(days_ahead + 1):
        d = fetch_json(f"{BO3_API}/v2/matches/upcoming", ttl=ttl,
                       params={"date": (today + timedelta(days=i)).isoformat(), "utc_offset": 0,
                               "filter[discipline_id][eq]": 1})
        teams = (d.get("included") or {}).get("teams") or {}
        for tier in ((d.get("data") or {}).get("tiers") or {}).values():
            for m in tier.get("matches", []):
                bu = m.get("bet_updates") or {}
                t1, t2 = bu.get("team_1") or {}, bu.get("team_2") or {}
                if not (t1.get("coeff") and t2.get("coeff") and t1.get("active", True) and t2.get("active", True)):
                    continue
                n1 = (teams.get(str(m["team1_id"])) or {}).get("name") or t1.get("name")
                n2 = (teams.get(str(m["team2_id"])) or {}).get("name") or t2.get("name")
                c1, c2 = (t1, t2) if t1.get("team_id") == m["team1_id"] else (t2, t1)
                mo = MatchOdds("thunderpick", n1, n2, _t(m.get("start_date")), int(m.get("bo_type") or 3))
                mo.p_series_team1 = devig(float(c1["coeff"]), float(c2["coeff"]))[0]
                _fill_series(mo)
                out.append(mo)
    return out


def _fill_series(mo: MatchOdds) -> None:
    """Fill in whichever of series / per-map win probabilities is missing."""
    if mo.p_series_team1 is not None and mo.best_of > 1:
        p = map_prob_from_series(mo.p_series_team1, mo.best_of)
        for k in range(1, mo.best_of + 1):
            mo.p_map_team1.setdefault(k, p)
    elif mo.p_series_team1 is not None:
        mo.p_map_team1.setdefault(1, mo.p_series_team1)


def fetch_all_books(sources=("pinnacle", "thunderpick")) -> tuple[list[MatchOdds], list[str]]:
    res, errors = [], []
    fns = {"pinnacle": fetch_pinnacle, "thunderpick": fetch_thunderpick_via_bo3}
    for s in sources:
        try:
            res.extend(fns[s]())
        except Exception as e:  # noqa: BLE001
            errors.append(f"{s}: {e}")
    return res, errors
