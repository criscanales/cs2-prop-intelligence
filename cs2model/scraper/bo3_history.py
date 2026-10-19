"""Crawl finished CS2 matches from bo3.gg's public JSON API into one row per player-map.

Endpoints (all public, discovered from the bo3.gg web app):
  /api/v2/matches/finished?date=YYYY-MM-DD      matches finished that day (+ Thunderpick odds snapshot)
  /api/v1/games?filter[games.match_id][eq]=ID   maps of a match (map name, score, rounds)
  /api/v1/games/{game_id}/players_stats         per-player stats for one map
  /api/v1/tournaments?filter[tournaments.id][eq]=ID  event type (lan/online), tier
  /api/v1/players/{slug}/general_stats          career totals (all tiers and dates)
"""
import argparse
import os
import json
import logging
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import pandas as pd

from .http import CACHE_DIR, ROOT, fetch_json

API = "https://api.bo3.gg/api"
DATA = ROOT / "data"
HIST_PATH = DATA / "player_maps.csv"
log = logging.getLogger(__name__)

PAST = None  # finished matches/maps never change -> cache forever


def finished_matches(day: date):
    d = fetch_json(f"{API}/v2/matches/finished",
                   params={"date": day.isoformat(), "utc_offset": 0, "filter[discipline_id][eq]": 1},
                   ttl=PAST if day < date.today() - timedelta(days=1) else 900)
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


def game_weapons(game_id):
    """AWP kills per bo3.gg player id for one map: {player_id: (weapon_kills, awp_kills, awp_hs)}."""
    out = {}
    for p in fetch_json(f"{API}/v1/games/{game_id}/weapons_stats", ttl=PAST) or []:
        sp = p.get("steam_profile") or {}
        pid = (sp.get("player") or {}).get("id")
        ws = p.get("player_weapons_stats") or sp.get("player_weapons_stats") or []
        if pid and ws:
            aw = [x for x in ws if x.get("weapon_slug") == "awp"]
            out[pid] = (sum(x.get("kills_sum") or 0 for x in ws), sum(x.get("kills_sum") or 0 for x in aw),
                        sum(x.get("headshots_sum") or 0 for x in aw))
    return out


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
        try:
            weapons = game_weapons(g["id"])
        except Exception as e:  # noqa: BLE001
            log.warning("game %s weapons: %s", g["id"], e)
            weapons = {}
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
                "weapon_kills": (weapons.get(pl["id"]) or (None,) * 3)[0],
                "awp_kills": (weapons.get(pl["id"]) or (None,) * 3)[1],
                "awp_hs": (weapons.get(pl["id"]) or (None,) * 3)[2],
            })
    return rows


def known_matches(hist: pd.DataFrame | None) -> dict:
    """{match_id: [row dicts]} for matches already in the history with kills on every map row. The crawler reuses
    these instead of re-fetching their maps (finished maps never change), so an imported bundle or an existing history
    only costs one request per day plus the new matches."""
    if hist is None or hist.empty:
        return {}
    h = hist.copy()
    h["date"] = h["date"].astype(str)
    ok = h.kills.notna().groupby(h.match_id).transform("all")
    return {mid: g.to_dict("records") for mid, g in h[ok].groupby("match_id")}


def crawl(days=92, end: date | None = None, tiers=("s", "a", "b"), write=True, progress=None, known=None,
          keep_days=None):
    """progress(done_days, total_days, rows) is called after each day. Responses are cached, so a rerun after an
    interruption skips everything already downloaded. `known` (existing history) skips matches already stored.
    write=True merges the crawled days into the history on disk (never shrinks it) and returns the merged history;
    keep_days drops rows older than that many days before `end`."""
    end = end or datetime.now(timezone.utc).date()
    rows, tours = [], {}
    known = known_matches(known) if isinstance(known, pd.DataFrame) else (known or {})
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
            if m.get("id") in known:
                rows.extend(known[m["id"]])
                continue
            tid = m.get("tournament")
            if tid not in tours:
                tours[tid] = tournament(tid)
            try:
                rows.extend(rows_for_match(m, teams_inc, tours[tid]))
            except Exception as e:  # noqa: BLE001
                log.warning("match %s: %s", m.get("slug"), e)
        log.info("%s: %d matches, +%d rows (total %d)", day, len(matches), len(rows) - n0, len(rows))
        if progress:
            progress(days + 1 - i, days + 1, len(rows))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
        df = df.drop_duplicates(["game_id", "player_id"]).sort_values(["date", "game_id"])
    if write:
        df = merge_history(pd.read_csv(HIST_PATH, low_memory=False) if HIST_PATH.exists() else None, df)
        if keep_days is not None and not df.empty:
            df = df[df["date"] >= pd.Timestamp(end - timedelta(days=keep_days), tz="UTC")].reset_index(drop=True)
        save_history(df)
    return df


CAREER_KEYS = ("matches_count", "matches_won_count", "matches_lost_count", "games_count", "games_won_count",
               "games_lost_count", "events_count", "events_won_count", "kills_sum", "deaths_sum", "assists_sum",
               "damage_sum", "rounds_count")


def career_stats(slug: str) -> dict | None:
    """bo3.gg career totals for one player (all tiers, all dates; no headshots in the source). None if unavailable."""
    if not slug or not isinstance(slug, str):
        return None
    d = fetch_json(f"{API}/v1/players/{quote(slug, safe='')}/general_stats", ttl=86400)
    if not isinstance(d, dict) or not d.get("games_count"):
        return None
    return {k: d.get(k) for k in CAREER_KEYS}


def merge_history(*frames: pd.DataFrame) -> pd.DataFrame:
    """Union of histories, one row per (game_id, player_id); when both have a row the more complete one wins."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat([f.assign(date=pd.to_datetime(f["date"], utc=True, format="ISO8601")) for f in frames], ignore_index=True)
    df = df.assign(_nn=df.notna().sum(1)).sort_values("_nn", kind="stable")
    df = df.drop_duplicates(["game_id", "player_id"], keep="last").drop(columns="_nn")
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def save_history(df):
    """Write via a temp file + rename so readers never see a half-written CSV."""
    DATA.mkdir(exist_ok=True)
    tmp = HIST_PATH.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, HIST_PATH)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=92)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = crawl(a.days)
    print(f"wrote {len(df)} rows to {HIST_PATH}")


def team_logos(write=True):
    """{team_id: image_url} gathered from bo3 responses already in the disk cache (no extra requests)."""
    import glob
    out = {}

    def walk(o):
        if isinstance(o, dict):
            url = o.get("image_url")
            if isinstance(url, str) and "/uploads/team/" in url and o.get("id") is not None:
                out[int(o["id"])] = url
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for f in glob.glob(str(CACHE_DIR / "*" / "*.json")):
        try:
            txt = open(f, encoding="utf-8").read()
            if "/uploads/team/" in txt:
                walk(json.loads(txt))
        except (OSError, ValueError):
            continue
    if write:
        (DATA / "team_logos.json").write_text(json.dumps(out))
    return out


def player_photos(write=True):
    """{player_id: image_url} gathered from bo3 responses already in the disk cache (no extra requests)."""
    import glob
    out = {}

    def walk(o):
        if isinstance(o, dict):
            url = o.get("image_url")
            if isinstance(url, str) and "/uploads/player/" in url and o.get("id") is not None and "nickname" in o:
                out[int(o["id"])] = url
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for f in glob.glob(str(CACHE_DIR / "*" / "*.json")):
        try:
            txt = open(f, encoding="utf-8").read()
            if "/uploads/player/" in txt:
                walk(json.loads(txt))
        except (OSError, ValueError):
            continue
    if write:
        (DATA / "player_photos.json").write_text(json.dumps(out))
    return out
