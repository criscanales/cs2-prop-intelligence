"""Crawl finished CS2 matches from bo3.gg's public JSON API into one row per player-map.

Endpoints (all public, discovered from the bo3.gg web app):
  /api/v2/matches/finished?date=YYYY-MM-DD      matches finished that day (+ Thunderpick odds snapshot)
  /api/v1/games?filter[games.match_id][eq]=ID   maps of a match (map name, score, rounds)
  /api/v1/games/{game_id}/players_stats         per-player stats for one map
  /api/v1/tournaments?filter[tournaments.id][eq]=ID  event type (lan/online), tier
"""
import argparse
import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from .http import ROOT, fetch_json

API = "https://api.bo3.gg/api"
DATA = ROOT / "data"
HIST_PATH = DATA / "player_maps.csv"
log = logging.getLogger(__name__)

PAST = None  # finished matches/maps never change -> cache forever


def finished_matches(day: date):
    d = fetch_json(f"{API}/v2/matches/finished",
                   params={"date": day.isoformat(), "utc_offset": 0, "filter[discipline_id][eq]": 1},
                   ttl=PAST if day < date.today() - timedelta(days=1) else 3600)
    inc = d.get("included") or {}
    out = []
    for tier_block in (d.get("data") or {}).get("tiers", {}).values():
        for m in tier_block.get("matches", []):
            if m.get("status") != "finished" or m.get("parsed_status") != "done":
                continue
            out.append(m)
    return out, inc


def tournament(tid):
    try:
        res = fetch_json(f"{API}/v1/tournaments", params={"filter[tournaments.id][eq]": tid}, ttl=PAST)
        return (res.get("results") or [{}])[0]
    except Exception:  # noqa: BLE001
        return {}


def match_games(match_id):
    d = fetch_json(f"{API}/v1/games", params={"sort": "number", "filter[games.match_id][eq]": match_id,
                                               "with": "winner_team_clan,loser_team_clan"}, ttl=PAST)
    return d.get("results", [])


def game_players(game_id):
    return fetch_json(f"{API}/v1/games/{game_id}/players_stats", ttl=PAST)


def _nn(x):
    """bo3 uses -1 for 'not parsed'; treat negatives as missing."""
    return None if x is None or x < 0 else x


def _odds(m, team_id):
    bu = m.get("bet_updates") or {}
    for k in ("team_1", "team_2"):
        t = bu.get(k) or {}
        if t.get("team_id") == team_id and t.get("coeff"):
            return float(t["coeff"])
    return None


def rows_for_match(m, teams_inc, tour):
    rows = []
    t1, t2 = m["team1_id"], m["team2_id"]
    names = {t1: (teams_inc.get(str(t1)) or {}).get("name"), t2: (teams_inc.get(str(t2)) or {}).get("name")}
    for g in match_games(m["id"]):
        if g.get("status") != "finished" or not g.get("rounds_count"):
            continue
        win_tid = ((g.get("winner_team_clan") or {}).get("team_id"))
        try:
            players = game_players(g["id"])
        except Exception as e:  # noqa: BLE001
            log.warning("game %s: %s", g["id"], e)
            continue
        for p in players:
            sp = p.get("steam_profile") or {}
            pl = sp.get("player") or {}
            tid = (p.get("team_clan") or {}).get("team_id")
            if not pl.get("id") or tid not in (t1, t2):
                continue
            opp = t2 if tid == t1 else t1
            won = int(tid == win_tid)
            rows.append({
                "date": g.get("begin_at") or m["start_date"], "match_id": m["id"], "game_id": g["id"],
                "map_number": g.get("number"), "map_name": g.get("map_name"), "bo_type": m.get("bo_type"),
                "tier": m.get("tier"), "tournament_id": m.get("tournament"), "event_type": tour.get("event_type"),
                "team_id": tid, "team": names.get(tid) or p.get("clan_name"), "opp_id": opp,
                "opponent": names.get(opp) or p.get("enemy_clan_name"),
                "player_id": pl["id"], "player_slug": pl.get("slug"), "player": pl.get("nickname") or sp.get("nickname"),
                "won": won, "rounds": g["rounds_count"],
                "team_rounds": g["winner_clan_score"] if won else g["loser_clan_score"],
                "opp_rounds": g["loser_clan_score"] if won else g["winner_clan_score"],
                "kills": _nn(p.get("kills")), "deaths": _nn(p.get("death")), "assists": _nn(p.get("assists")),
                "headshots": _nn(p.get("headshots")), "first_kills": _nn(p.get("first_kills")),
                "first_deaths": _nn(p.get("first_death")), "kast": p.get("kast"), "rating": p.get("player_rating"),
                "adr": p.get("adr"), "team_odds": _odds(m, tid), "opp_odds": _odds(m, opp),
            })
    return rows


def crawl(days=92, end: date | None = None, tiers=("s", "a", "b")):
    end = end or datetime.now(timezone.utc).date()
    rows, tours = [], {}
    for i in range(days, -1, -1):
        day = end - timedelta(days=i)
        try:
            matches, inc = finished_matches(day)
        except Exception as e:  # noqa: BLE001
            log.warning("day %s: %s", day, e)
            continue
        teams_inc = inc.get("teams") or {}
        n0 = len(rows)
        for m in matches:
            if m.get("tier") not in tiers:
                continue
            tid = m.get("tournament")
            if tid not in tours:
                tours[tid] = tournament(tid)
            try:
                rows.extend(rows_for_match(m, teams_inc, tours[tid]))
            except Exception as e:  # noqa: BLE001
                log.warning("match %s: %s", m.get("slug"), e)
        log.info("%s: %d matches, +%d rows (total %d)", day, len(matches), len(rows) - n0, len(rows))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.drop_duplicates(["game_id", "player_id"]).sort_values(["date", "game_id"])
    DATA.mkdir(exist_ok=True)
    df.to_csv(HIST_PATH, index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=92)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = crawl(a.days)
    print(f"wrote {len(df)} rows to {HIST_PATH}")
