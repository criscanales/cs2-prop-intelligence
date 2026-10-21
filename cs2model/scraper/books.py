"""Match / map moneylines, round totals and overtime odds.

- Pinnacle guest API (same public endpoints pinnacle.com calls when logged out).
- Thunderpick moneylines as relayed by bo3.gg's public upcoming-matches feed.
- Polymarket CS2 match markets (public Gamma API): series / per-map winner, maps total and map round totals.
  thunderpick.io itself answers 403 to non-browser clients, so it is never called directly.
"""
import json
import re
from datetime import datetime, timedelta, timezone

from ..types import MatchOdds
from .http import (ROOT, AccessDenied, CircuitBreaker, CircuitOpen, UpstreamUnavailable, _cache_key, cache_get, cached_at,
                   fetch_json)
from ..odds import american_to_decimal, devig, map_prob_from_series

PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
# Public key the pinnacle.com web app sends for guest (logged-out) requests.
PINNACLE_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"
PINNACLE_ESPORTS = 12

BO3_API = "https://api.bo3.gg/api"
POLYMARKET_API = "https://gamma-api.polymarket.com"
POLYMARKET_MAX_SPREAD = 0.10


def _t(s):
    if not s:
        return None
    s = re.sub(r"(\.\d{1,6})\d*", lambda m: m.group(1).ljust(7, "0"), s.replace("Z", "+00:00"))
    return datetime.fromisoformat(s)


PINNACLE_CIRCUIT = CircuitBreaker(threshold=2, cooldown=300.0, max_cooldown=1800.0)
PINNACLE_LAST_GOOD = ROOT / "data" / "pinnacle_last_good.json"


def _save_last_good(matchups, markets, fetched_at: datetime):
    """Keep the last successful Pinnacle response with its original timestamp (display only, never priced)."""
    PINNACLE_LAST_GOOD.parent.mkdir(parents=True, exist_ok=True)
    tmp = PINNACLE_LAST_GOOD.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fetched_at": fetched_at.isoformat(), "matchups": matchups, "markets": markets}))
    tmp.replace(PINNACLE_LAST_GOOD)


def pinnacle_last_good() -> dict | None:
    """{'fetched_at': datetime, 'n_matches': CS2 matches} for the last successful Pinnacle pull, or None."""
    try:
        if PINNACLE_LAST_GOOD.exists():
            d = json.loads(PINNACLE_LAST_GOOD.read_text())
            at, mu, mk = datetime.fromisoformat(d["fetched_at"]), d["matchups"], d["markets"]
        else:  # installs from before the snapshot file existed: fall back to the expired HTTP cache entries
            urls = [f"{PINNACLE_BASE}/sports/{PINNACLE_ESPORTS}/matchups", f"{PINNACLE_BASE}/sports/{PINNACLE_ESPORTS}/markets/straight"]
            ts = [cached_at(u) for u in urls]
            if not all(ts):
                return None
            mu, mk = (cache_get(_cache_key("GET", u, None, None), None) for u in urls)
            at = datetime.fromtimestamp(min(ts), timezone.utc)
        return {"fetched_at": at, "n_matches": len(parse_pinnacle(mu, mk))}
    except Exception:  # noqa: BLE001
        return None


def fetch_pinnacle(ttl=300) -> list[MatchOdds]:
    h = {"X-API-Key": PINNACLE_KEY, "Referer": "https://www.pinnacle.com/"}
    kw = {"headers": h, "ttl": ttl, "retries": 2, "max_backoff": 4.0, "circuit": PINNACLE_CIRCUIT}
    urls = [f"{PINNACLE_BASE}/sports/{PINNACLE_ESPORTS}/matchups", f"{PINNACLE_BASE}/sports/{PINNACLE_ESPORTS}/markets/straight"]
    matchups, markets = (fetch_json(u, **kw) for u in urls)
    ts = [t for t in (cached_at(u) for u in urls) if t]
    fetched = datetime.fromtimestamp(min(ts), timezone.utc) if ts else datetime.now(timezone.utc)
    _save_last_good(matchups, markets, fetched)
    return parse_pinnacle(matchups, markets)


def parse_pinnacle(matchups, markets) -> list[MatchOdds]:
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
                    mo.series_dec = (prices["home"], prices["away"])
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
                mo.series_dec = (float(c1["coeff"]), float(c2["coeff"]))
                _fill_series(mo)
                out.append(mo)
    return out


def _pm_prob(m):
    """Mid price of the market's first outcome, or None when the book is too thin to trust."""
    try:
        bid, ask = float(m["bestBid"]), float(m["bestAsk"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 < bid <= ask < 1 or ask - bid > POLYMARKET_MAX_SPREAD:
        return None
    return (bid + ask) / 2


def _pm_outcomes(m):
    try:
        return json.loads(m.get("outcomes") or "[]")
    except ValueError:
        return []


def polymarket_match(e) -> MatchOdds | None:
    mks = e.get("markets") or []
    ml = next((m for m in mks if m.get("sportsMarketType") == "moneyline" and not m.get("closed")), None)
    outs = _pm_outcomes(ml) if ml else []
    if len(outs) != 2:
        return None
    bo = re.search(r"\(BO(\d)\)", e.get("title") or "")
    mo = MatchOdds("polymarket", outs[0], outs[1], _t(e.get("startTime")), int(bo.group(1)) if bo else 3)
    mo.p_series_team1 = _pm_prob(ml)
    for m in mks:
        if m.get("closed"):
            continue
        p, kind, q = _pm_prob(m), m.get("sportsMarketType") or "", m.get("question") or ""
        if p is None:
            continue
        if kind == "child_moneyline" and (mm := re.search(r"Map (\d) Winner", q)):
            o = _pm_outcomes(m)
            if len(o) == 2:
                mo.p_map_team1[int(mm.group(1))] = p if o[0] == outs[0] else 1 - p
        elif kind == "totals" and mo.best_of == 3 and re.search(r"O/U 2\.5", q) and _pm_outcomes(m)[:1] == ["Over"]:
            mo.p_map3 = p
        elif (mm := re.fullmatch(r"round_over_under_game_(\d)", kind)) and (ln := re.search(r"(\d+\.5)", q)) \
                and _pm_outcomes(m)[:1] == ["Over"]:
            mo.rounds_ladder.setdefault(int(mm.group(1)), []).append((float(ln.group(1)), p))
    if mo.p_series_team1 is None and not mo.p_map_team1:
        return None
    for k in mo.rounds_ladder:
        mo.rounds_ladder[k].sort()
    _fill_series(mo)
    return mo


def fetch_polymarket(ttl=300, days_ahead=3, pages=4) -> list[MatchOdds]:
    out, now = [], datetime.now(timezone.utc)
    for i in range(pages):
        evs = fetch_json(f"{POLYMARKET_API}/events", ttl=ttl,
                         params={"closed": "false", "tag_slug": "counter-strike-2", "limit": 100, "offset": 100 * i})
        for e in evs or []:
            st = _t(e.get("startTime"))
            if st is None or not (now - timedelta(hours=6) <= st <= now + timedelta(days=days_ahead)):
                continue
            if (mo := polymarket_match(e)) is not None:
                out.append(mo)
        if len(evs or []) < 100:
            break
    return out


def _fill_series(mo: MatchOdds) -> None:
    """Fill in whichever of series / per-map win probabilities is missing."""
    if mo.p_series_team1 is not None and mo.best_of > 1:
        p = map_prob_from_series(mo.p_series_team1, mo.best_of)
        for k in range(1, mo.best_of + 1):
            mo.p_map_team1.setdefault(k, p)
    elif mo.p_series_team1 is not None:
        mo.p_map_team1.setdefault(1, mo.p_series_team1)


def fetch_all_books(sources=("pinnacle", "polymarket", "thunderpick")) -> tuple[list[MatchOdds], list[str]]:
    res, errors = [], []
    fns = {"pinnacle": fetch_pinnacle, "polymarket": fetch_polymarket, "thunderpick": fetch_thunderpick_via_bo3}
    for s in sources:
        try:
            res.extend(fns[s]())
            continue
        except AccessDenied as e:
            msg = f"{s}: access denied ({e}). Not retried; this is an access/region restriction, not an outage"
        except CircuitOpen as e:
            msg = f"{s}: temporarily unavailable, paused to avoid repeat requests ({e})"
        except UpstreamUnavailable as e:
            msg = f"{s}: temporarily unavailable ({e}), retrying with backoff"
        except Exception as e:  # noqa: BLE001
            msg = f"{s}: {e}"
        lg = pinnacle_last_good() if s == "pinnacle" else None
        if lg:
            msg += (f". Last good odds {lg['fetched_at']:%Y-%m-%d %H:%M} UTC ({lg['n_matches']} matches) are stale "
                    "and not used for pricing")
        errors.append(msg)
    return res, errors


def fetch_bo3_match_meta(ttl=300, days_ahead=2) -> dict:
    """Event name, tier and LAN/online per upcoming bo3.gg match, keyed by frozenset of bo3.gg team ids.
    Fields bo3.gg does not provide are left as None (shown as Unknown)."""
    from .bo3_history import tournament
    out, tours = {}, {}
    today = datetime.now(timezone.utc).date()
    for i in range(days_ahead + 1):
        d = fetch_json(f"{BO3_API}/v2/matches/upcoming", ttl=ttl,
                       params={"date": (today + timedelta(days=i)).isoformat(), "utc_offset": 0,
                               "filter[discipline_id][eq]": 1})
        inc = (d.get("included") or {}).get("tournaments") or {}
        for tier in ((d.get("data") or {}).get("tiers") or {}).values():
            for m in tier.get("matches", []):
                if not (m.get("team1_id") and m.get("team2_id")):
                    continue
                tid = m.get("tournament")
                t = inc.get(str(tid)) or {}
                if tid and tid not in tours:
                    tours[tid] = (tournament(tid) or {}).get("event_type")
                et = tours.get(tid)
                out[frozenset((int(m["team1_id"]), int(m["team2_id"])))] = {
                    "event": t.get("name"), "match_tier": m.get("tier") or t.get("tier"),
                    "event_type": et if et in ("lan", "online") else None}
    return out
