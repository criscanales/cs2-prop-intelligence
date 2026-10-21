"""SQLite log of line snapshots, recommended picks/slips and their graded results (ROI, CLV)."""
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ..scraper.http import ROOT

DB_PATH = ROOT / "data" / "tracking.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS line_snapshots (
  ts TEXT, source TEXT, player TEXT, team TEXT, opponent TEXT, stat TEXT, maps TEXT,
  line REAL, odds_type TEXT, start_time TEXT);
CREATE INDEX IF NOT EXISTS ix_snap ON line_snapshots(source, player, stat, maps, start_time);
CREATE TABLE IF NOT EXISTS picks (
  id INTEGER PRIMARY KEY, created TEXT, source TEXT, player TEXT, team TEXT, opponent TEXT,
  stat TEXT, maps TEXT, line REAL, side TEXT, p_model REAL, breakeven REAL, edge REAL, ev REAL,
  tier TEXT, model_mean REAL, start_time TEXT, closing_line REAL, actual REAL, outcome TEXT,
  UNIQUE(source, player, stat, maps, start_time, side, line));
CREATE TABLE IF NOT EXISTS preds (
  source TEXT, player TEXT, team TEXT, opponent TEXT, stat TEXT, maps TEXT, start_time TEXT, ts TEXT,
  line REAL, model_mean REAL, p_over REAL, actual REAL,
  PRIMARY KEY(source, player, stat, maps, start_time));
CREATE TABLE IF NOT EXISTS slips (
  id INTEGER PRIMARY KEY, created TEXT, platform TEXT, play TEXT, legs TEXT, p_all REAL,
  ev REAL, stake REAL, hit_dist TEXT, hits INTEGER, payout REAL, profit REAL);
CREATE TABLE IF NOT EXISTS map_odds_log (
  id INTEGER PRIMARY KEY, snapshot_ts TEXT, pred_ts TEXT, source TEXT, match_key TEXT, team_a_id INTEGER,
  team_b_id INTEGER, team_a TEXT, team_b TEXT, start_time TEXT, best_of INTEGER, t3 INTEGER, p_series_a REAL,
  p_map1_quoted REAL, p_map2_quoted REAL, p_map3_quoted REAL, p_map1_used REAL, p_map2_used REAL, p_map3_used REAL,
  src_map1 TEXT, src_map2 TEXT, src_map3 TEXT, p_map3_played REAL, p_map3_played_book REAL, model_version TEXT,
  inputs_hash TEXT, UNIQUE(source, match_key, start_time, snapshot_ts));
CREATE TRIGGER IF NOT EXISTS map_odds_log_no_update BEFORE UPDATE ON map_odds_log
  BEGIN SELECT RAISE(ABORT, 'map_odds_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS map_odds_log_no_delete BEFORE DELETE ON map_odds_log
  BEGIN SELECT RAISE(ABORT, 'map_odds_log is append-only'); END;
CREATE TABLE IF NOT EXISTS match_results (
  match_key TEXT, start_time TEXT, bo3_match_id INTEGER, best_of INTEGER, maps_played INTEGER, map3_played INTEGER,
  a_won_map1 INTEGER, a_won_map2 INTEGER, a_won_map3 INTEGER, a_won_series INTEGER, graded_ts TEXT,
  PRIMARY KEY(match_key, start_time));
CREATE TABLE IF NOT EXISTS data_gaps (
  id INTEGER PRIMARY KEY, recorded_ts TEXT, gap_start TEXT, gap_end TEXT, scope TEXT, reason TEXT,
  UNIQUE(gap_start, gap_end, scope));
CREATE TRIGGER IF NOT EXISTS data_gaps_no_update BEFORE UPDATE ON data_gaps
  BEGIN SELECT RAISE(ABORT, 'data_gaps is append-only'); END;
CREATE TRIGGER IF NOT EXISTS data_gaps_no_delete BEFORE DELETE ON data_gaps
  BEGIN SELECT RAISE(ABORT, 'data_gaps is append-only'); END;
"""

MAP_ODDS_FIELDS = ["pred_ts", "source", "match_key", "team_a_id", "team_b_id", "team_a", "team_b", "start_time", "best_of",
                   "t3", "p_series_a", "p_map1_quoted", "p_map2_quoted", "p_map3_quoted", "p_map1_used", "p_map2_used",
                   "p_map3_used", "src_map1", "src_map2", "src_map3", "p_map3_played", "p_map3_played_book", "model_version"]


def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    if "t3" not in {r[1] for r in con.execute("PRAGMA table_info(picks)")}:
        con.execute("ALTER TABLE picks ADD COLUMN t3 INTEGER DEFAULT 0")  # tier-3 picks, tracked separately
        con.commit()
    if "t3" not in {r[1] for r in con.execute("PRAGMA table_info(preds)")}:
        con.execute("ALTER TABLE preds ADD COLUMN t3 INTEGER DEFAULT 0")  # tier-3 lines, for the tier-3 reliability check
        con.execute("ALTER TABLE preds ADD COLUMN p_over_raw REAL")
        con.commit()
    if "void" not in {r[1] for r in con.execute("PRAGMA table_info(preds)")}:
        con.execute("ALTER TABLE preds ADD COLUMN void INTEGER DEFAULT 0")  # a requested map was never played
        con.commit()
    return con



def record_gap(gap_start, gap_end, scope, reason, con=None):
    """Permanently mark a period with no tracking records (append-only; re-recording is a no-op)."""
    con = con or connect()
    con.execute("INSERT OR IGNORE INTO data_gaps(recorded_ts, gap_start, gap_end, scope, reason) VALUES (?,?,?,?,?)",
                (_now(), gap_start, gap_end, scope, reason))
    con.commit()


def first_observation(con):
    """Earliest timestamp of any collected record (line snapshot, prediction, map-odds snapshot), or None."""
    ts = [r[0] for q in ("SELECT min(ts) FROM line_snapshots", "SELECT min(ts) FROM preds", "SELECT min(snapshot_ts) FROM map_odds_log")
          for r in con.execute(q).fetchall() if r[0]]
    return min(ts, key=lambda t: pd.to_datetime(t, utc=True)) if ts else None


def data_gaps(con=None):
    con = con or connect()
    return pd.read_sql("SELECT * FROM data_gaps ORDER BY gap_start", con)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _maps(m):
    return "+".join(str(x) for x in m) if isinstance(m, (list, tuple)) else str(m)


def snapshot_lines(lines, con=None):
    con = con or connect()
    ts = _now()
    con.executemany("INSERT INTO line_snapshots VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (ts, ln.source, ln.player, ln.team, ln.opponent, ln.stat, _maps(ln.maps), ln.line, ln.odds_type,
         ln.start_time.isoformat() if ln.start_time else None) for ln in lines])
    con.commit()


def log_preds(slate, con=None):
    """Keep the latest pre-start model price of every standard line (graded later for the bias check)."""
    con = con or connect()
    if slate is None or slate.empty:
        return
    now = pd.Timestamp.now(tz="UTC")
    s = slate[slate.odds_type.eq("standard") & slate.start_time.notna()]
    rows = []
    for r in s.itertuples():
        st = pd.Timestamp(r.start_time)
        st = st.tz_localize("UTC") if st.tzinfo is None else st
        if st > now:
            t3, raw = getattr(r, "t3", False), getattr(r, "p_over_raw", None)
            rows.append((r.source, r.player, r.team, r.opponent, r.stat, _maps(r.maps), st.isoformat(), _now(),
                         float(r.line), float(r.model_mean), float(r.p_over), int(t3 is True or t3 == 1),
                         None if raw is None or raw != raw else float(raw)))
    con.executemany("INSERT OR REPLACE INTO preds (source, player, team, opponent, stat, maps, start_time, ts, line, model_mean, "
                    "p_over, t3, p_over_raw) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()


def _inputs_hash(r):
    import hashlib
    keys = [k for k in MAP_ODDS_FIELDS if k not in ("pred_ts", "team_a", "team_b")]
    return hashlib.sha1(json.dumps([r.get(k) if not isinstance(r.get(k), float) else round(r[k], 6) for k in keys]).encode()).hexdigest()


def log_map_odds(records, snapshot_ts, con=None):
    """Append one map-odds provenance row per (source, match, snapshot). Pre-match snapshots only; a snapshot whose inputs
    are identical to that source/match's latest row (same book prices, same model version) is not appended again."""
    con = con or connect()
    snap = pd.Timestamp(snapshot_ts)
    snap = snap.tz_localize("UTC") if snap.tzinfo is None else snap.tz_convert("UTC")
    n = 0
    for r in records:
        if not r.get("start_time") or pd.Timestamp(r["start_time"]) <= snap:
            continue
        h = _inputs_hash(r)
        last = con.execute("SELECT inputs_hash FROM map_odds_log WHERE source=? AND match_key=? AND start_time=? "
                           "ORDER BY snapshot_ts DESC, id DESC LIMIT 1", (r["source"], r["match_key"], r["start_time"])).fetchone()
        if last and last[0] == h:
            continue
        cur = con.execute(f"INSERT OR IGNORE INTO map_odds_log (snapshot_ts, {', '.join(MAP_ODDS_FIELDS)}, inputs_hash) "
                          f"VALUES ({', '.join('?' * (len(MAP_ODDS_FIELDS) + 2))})",
                          [snap.isoformat(timespec="seconds")] + [r.get(k) for k in MAP_ODDS_FIELDS] + [h])
        n += cur.rowcount
    con.commit()
    return n


def grade_match_results(hist, con=None):
    """Record the map-by-map result of every logged match once its series is decided in `hist` (never before)."""
    con = con or connect()
    todo = con.execute("SELECT DISTINCT m.match_key, m.start_time, m.team_a_id, m.team_b_id, m.best_of FROM map_odds_log m "
                       "LEFT JOIN match_results r ON r.match_key = m.match_key AND r.start_time = m.start_time "
                       "WHERE r.match_key IS NULL").fetchall()
    now, n = pd.Timestamp.now(tz="UTC"), 0
    for key, st, a, b, bo in todo:
        st = pd.Timestamp(st)
        if st > now:
            continue
        h = hist[(hist.team_id == a) & (hist.opp_id == b) & (hist.date >= st - pd.Timedelta(hours=3))
                 & (hist.date <= st + pd.Timedelta(hours=12))]
        if h.empty:
            continue
        h = h[h.match_id == h.match_id.mode().iloc[0]].drop_duplicates("map_number").sort_values("map_number")
        if h.won.isna().any() or list(h.map_number) != list(range(1, len(h) + 1)):
            continue
        need, wins = int(bo) // 2 + 1, int(h.won.sum())
        if max(wins, len(h) - wins) < need:
            continue  # series not finished (or later maps not in history yet)
        won = {int(m): int(w) for m, w in zip(h.map_number, h.won)}
        con.execute("INSERT OR IGNORE INTO match_results VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (key, st.isoformat(), int(h.match_id.iloc[0]), int(bo), len(h), int(len(h) >= 3) if int(bo) == 3 else None,
                     won.get(1), won.get(2), won.get(3), int(wins >= need), _now()))
        n += 1
    con.commit()
    return n


def map3_report(con=None, bands=(0.5, 0.65, 0.8, 1.0)):
    """Predicted vs actual P(Map 3 played) for finished tier-1/2 BO3s, from each source's last pre-match snapshot,
    by series-favourite strength and by whether maps 1-2 were quoted or inferred."""
    con = con or connect()
    d = pd.read_sql("SELECT m.*, r.map3_played FROM map_odds_log m JOIN match_results r ON r.match_key = m.match_key "
                    "AND r.start_time = m.start_time WHERE m.best_of = 3 AND COALESCE(m.t3, 0) = 0 AND r.map3_played IS NOT NULL", con)
    cols = ["source", "inputs", "fav_band", "matches", "pred_map3", "actual_map3"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d.sort_values(["snapshot_ts", "id"]).groupby(["source", "match_key", "start_time"]).tail(1)
    p = d.p_series_a.fillna(d[["p_map1_used", "p_map2_used"]].mean(axis=1))
    d["fav_band"] = pd.cut(p.where(p >= 0.5, 1 - p), list(bands), include_lowest=True).astype(str)
    d["inputs"] = ["maps 1-2 quoted" if a == b == "quoted" else "maps 1-2 inferred" if a == b == "inferred" else "mixed"
                   for a, b in zip(d.src_map1, d.src_map2)]
    g = d.groupby(["source", "inputs", "fav_band"]).agg(matches=("map3_played", "size"), pred_map3=("p_map3_played", "mean"),
                                                         actual_map3=("map3_played", "mean")).reset_index()
    return g[cols]


def _settle(hist, resolve_player, player, team, opponent, stat, maps, start_time):
    """("graded", total) when every requested map has the player's stats, ("void", None) when a requested map
    was never played because the series was already decided (e.g. Map 3 after a 2-0), None while unresolved
    (not started, match/player not in history yet, or a played map's data is missing)."""
    from ..names import team_similarity
    st = pd.Timestamp(start_time)
    st = st.tz_localize("UTC") if st.tzinfo is None else st
    if st > pd.Timestamp.now(tz="UTC"):
        return None
    pid = resolve_player(player, team)
    if pid is None:
        return None
    h = hist[(hist.player_id == pid) & (hist.date >= st - pd.Timedelta(hours=3)) & (hist.date <= st + pd.Timedelta(hours=12))]
    h = h[h.opponent.map(lambda o: team_similarity(o, opponent) >= 0.75)]
    if h.empty:
        return None
    h = h[h.match_id == h.match_id.mode().iloc[0]].drop_duplicates("map_number")
    mapnos = [int(x) for x in str(maps).split("+")]
    missing = [m for m in mapnos if m not in set(h.map_number)]
    if not missing:
        return "graded", float(h[h.map_number.isin(mapnos)][stat].sum())
    if set(missing) & set(hist.loc[hist.match_id == h.match_id.iloc[0], "map_number"]):
        return None  # the map was played but this player's row is missing
    first = min(missing)
    prior = h[h.map_number < first]
    if len(prior) != first - 1 or prior.won.isna().any():
        return None
    bo = h["bo_type"].dropna() if "bo_type" in h else pd.Series(dtype=float)
    need = int(bo.iloc[0]) // 2 + 1 if len(bo) else 2
    wins = int(prior.won.sum())
    return ("void", None) if max(wins, len(prior) - wins) >= need else None


def _actual(hist, resolve_player, player, team, opponent, stat, maps, start_time):
    r = _settle(hist, resolve_player, player, team, opponent, stat, maps, start_time)
    return r[1] if r else None


def _outcome(actual, line, side):
    return "push" if actual == line else ("win" if (actual > line) == (side == "over") else "loss")


def grade_preds(hist, resolve_player, con=None):
    con = con or connect()
    p = pd.read_sql("SELECT rowid AS rid, * FROM preds WHERE actual IS NULL AND COALESCE(void, 0) = 0", con)
    n = 0
    for r in p.itertuples():
        res = _settle(hist, resolve_player, r.player, r.team, r.opponent, r.stat, r.maps, r.start_time)
        if res is not None:
            con.execute("UPDATE preds SET actual=?, void=? WHERE rowid=?", (res[1], int(res[0] == "void"), int(r.rid)))
            n += 1
    con.commit()
    return n


def resettle(hist, resolve_player, con=None, dry_run=False):
    """Re-check already-settled preds and picks against the history and fix any whose settlement differs.
    Only settlement fields (actual, void, outcome) change; rows that can't be re-verified are left as they are.
    Idempotent: a second run finds nothing to change. Returns one row per change."""
    con = con or connect()
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"], utc=True, format="ISO8601")
    changes = []
    for r in pd.read_sql("SELECT rowid AS rid, * FROM preds WHERE actual IS NOT NULL OR COALESCE(void, 0) = 1", con).itertuples():
        res = _settle(hist, resolve_player, r.player, r.team, r.opponent, r.stat, r.maps, r.start_time)
        if res is None:
            continue
        old = ("void", None) if r.void == 1 else ("graded", r.actual)
        if old != res:
            changes.append(dict(table="preds", source=r.source, player=r.player, stat=r.stat, maps=r.maps, start_time=r.start_time,
                                line=r.line, t3=r.t3, old_actual=old[1], old_status=old[0], new_actual=res[1], new_status=res[0]))
            if not dry_run:
                con.execute("UPDATE preds SET actual=?, void=? WHERE rowid=?", (res[1], int(res[0] == "void"), int(r.rid)))
    for r in pd.read_sql("SELECT * FROM picks WHERE outcome IS NOT NULL", con).itertuples():
        res = _settle(hist, resolve_player, r.player, r.team, r.opponent, r.stat, r.maps, r.start_time)
        if res is None:
            continue
        new = "void" if res[0] == "void" else _outcome(res[1], r.line, r.side)
        if (r.outcome, None if r.outcome == "void" else r.actual) != (new, res[1]):
            changes.append(dict(table="picks", source=r.source, player=r.player, stat=r.stat, maps=r.maps, start_time=r.start_time,
                                line=r.line, t3=r.t3, side=r.side, old_actual=r.actual, old_status=r.outcome,
                                new_actual=res[1], new_status=new))
            if not dry_run:
                con.execute("UPDATE picks SET actual=?, outcome=? WHERE id=?", (res[1], new, int(r.id)))
    con.commit()
    if not dry_run and any(c["table"] == "picks" for c in changes):
        con.execute("UPDATE slips SET hits=NULL, payout=NULL, profit=NULL WHERE hits IS NOT NULL")
        con.commit()
        _grade_slips(con)
    return pd.DataFrame(changes)


def bias_report(con=None):
    """Model vs results on every graded line (not just picks): positive model - actual = model runs high."""
    con = con or connect()
    p = pd.read_sql("SELECT * FROM preds WHERE actual IS NOT NULL AND COALESCE(t3, 0) = 0", con)
    if p.empty:
        return p
    p["scope"] = p.maps.map(lambda m: "Map " + m if "+" not in m else "Maps " + m)
    g = p.groupby(["stat", "scope"])[["model_mean", "actual", "line", "p_over"]]
    out = g.apply(lambda x: pd.Series({
        "lines": len(x), "model_minus_actual": (x.model_mean - x.actual).mean(), "line_minus_actual": (x.line - x.actual).mean(),
        "model_minus_line": (x.model_mean - x.line).mean(), "avg_p_over": x.p_over.mean(), "over_rate": (x.actual > x.line).mean()}))
    return out.reset_index()


T3_BINS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1.0001)


def t3_reliability(con=None):
    """Graded tier-3 lines: predicted P of the side the model favours vs how often that side hit, by probability band
    and market (pushes dropped). Gap > 0 = the model was overconfident in that band."""
    con = con or connect()
    p = pd.read_sql("SELECT stat, maps, line, p_over, actual FROM preds WHERE t3 = 1 AND actual IS NOT NULL", con)
    p = p[p.actual != p.line]
    if p.empty:
        return pd.DataFrame(columns=["market", "p_range", "lines", "avg_p", "hit_rate", "gap"])
    over = p.p_over >= 0.5
    p = p.assign(p_side=p.p_over.where(over, 1 - p.p_over), hit=((p.actual > p.line) == over).astype(float),
                 market=p.stat.str.title() + " · " + p.maps.map(lambda m: "Maps " + m if "+" in m else "Map " + m))
    p = pd.concat([p, p.assign(market="All tier-3 lines")])
    labels = [f"{a * 100:.0f}–{b:.0%}" if b < 1 else f"{a:.0%}+" for a, b in zip(T3_BINS, T3_BINS[1:])]
    p["p_range"] = pd.cut(p.p_side, list(T3_BINS), labels=labels, right=False)
    band = p[(p.p_side >= 0.6) & (p.p_side < 0.75)].assign(p_range="60–75% (combined)")
    out = pd.concat([p, band]).groupby(["market", "p_range"], observed=True).agg(
        lines=("hit", "size"), avg_p=("p_side", "mean"), hit_rate=("hit", "mean")).reset_index()
    out["gap"] = out.avg_p - out.hit_rate
    return out


def first_line(con, source, player, stat, maps, start_time):
    r = con.execute("SELECT line FROM line_snapshots WHERE source=? AND player=? AND stat=? AND maps=? "
                    "AND start_time IS ? ORDER BY ts LIMIT 1", (source, player, stat, _maps(maps), start_time)).fetchone()
    return r[0] if r else None


def log_picks(df: pd.DataFrame, con=None):
    con = con or connect()
    cols = ["source", "player", "team", "opponent", "stat", "maps", "line", "side", "p_side", "breakeven",
            "edge", "ev", "tier", "model_mean", "start_time"]
    rows = [(_now(), *[(_maps(r[c]) if c == "maps" else (None if pd.isna(r[c]) else r[c])) for c in cols],
             int(r.get("t3") is True or r.get("t3") == 1)) for _, r in df.iterrows()]
    con.executemany("INSERT OR IGNORE INTO picks (created, source, player, team, opponent, stat, maps, line, side, "
                    "p_model, breakeven, edge, ev, tier, model_mean, start_time, t3) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [tuple(str(x) if isinstance(x, pd.Timestamp) else x for x in r) for r in rows])
    con.commit()


def log_slip(platform, play, legs, p_all, ev, stake, hit_dist, con=None):
    con = con or connect()
    con.execute("INSERT INTO slips (created, platform, play, legs, p_all, ev, stake, hit_dist) VALUES (?,?,?,?,?,?,?,?)",
                (_now(), platform, play, json.dumps(legs), p_all, ev, stake, json.dumps(list(map(float, hit_dist)))))
    con.commit()


def grade(hist: pd.DataFrame, resolve_player, con=None, payouts=None):
    """Fill actual results, closing lines and outcomes (win/loss/push/void) for picks whose matches are in `hist`."""
    con = con or connect()
    picks = pd.read_sql("SELECT * FROM picks WHERE outcome IS NULL AND start_time IS NOT NULL", con)
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"], utc=True, format="ISO8601")
    grade_preds(hist, resolve_player, con)
    grade_match_results(hist, con)
    n = 0
    for _, p in picks.iterrows():
        res = _settle(hist, resolve_player, p.player, p.team, p.opponent, p.stat, p.maps, p.start_time)
        if res is None:
            continue
        outcome = "void" if res[0] == "void" else _outcome(res[1], p.line, p.side)
        close = con.execute("SELECT line FROM line_snapshots WHERE source=? AND player=? AND stat=? AND maps=? "
                            "AND start_time=? ORDER BY ts DESC LIMIT 1",
                            (p.source, p.player, p.stat, p.maps, p.start_time)).fetchone()
        con.execute("UPDATE picks SET actual=?, outcome=?, closing_line=? WHERE id=?",
                    (res[1], outcome, close[0] if close else None, int(p.id)))
        n += 1
    con.commit()
    _grade_slips(con, payouts)
    return n


def _grade_slips(con, payouts=None):
    from ..parlay.engine import load_yaml, payout_vector
    payouts = payouts or load_yaml("payouts.yaml")
    out = {r[0]: r[1] for r in con.execute("SELECT id, outcome FROM picks")}
    for sid, platform, play, legs, stake in con.execute(
            "SELECT id, platform, play, legs, stake FROM slips WHERE hits IS NULL").fetchall():
        ids = json.loads(legs)
        res = [out.get(i) for i in ids]
        if any(r is None for r in res):
            continue
        hits = sum(r == "win" for r in res)
        cfg = payouts[platform]["plays"][play]
        mult = payout_vector(cfg)[hits]
        con.execute("UPDATE slips SET hits=?, payout=?, profit=? WHERE id=?", (hits, mult * stake, mult * stake - stake, sid))
    con.commit()


def pending(con=None):
    """Logged picks that aren't graded yet, with `status` upcoming / awaiting results."""
    con = con or connect()
    p = pd.read_sql("SELECT * FROM picks WHERE outcome IS NULL ORDER BY start_time", con)
    st = pd.to_datetime(p.start_time, utc=True, format="ISO8601", errors="coerce")
    p["status"] = (st > pd.Timestamp.now(tz="UTC")).map({True: "upcoming", False: "awaiting results"})
    return p


def performance(con=None):
    """ROI, hit rate, CLV and calibration of graded picks."""
    con = con or connect()
    p = pd.read_sql("SELECT * FROM picks WHERE outcome IN ('win','loss')", con)
    s = pd.read_sql("SELECT * FROM slips WHERE hits IS NOT NULL", con)
    other = dict(con.execute("SELECT outcome, count(*) FROM picks WHERE outcome IN ('push','void') GROUP BY outcome").fetchall())
    if p.empty:
        return {"picks": 0, "pushes": other.get("push", 0), "voids": other.get("void", 0)}, p, s
    p["win"] = (p.outcome == "win").astype(int)
    p["roi"] = p.win / p.breakeven - 1
    sign = p.side.map({"over": 1, "under": -1})
    p["clv"] = (p.closing_line - p.line) * sign  # positive = line moved in our favour
    summ = {"picks": len(p), "pushes": other.get("push", 0), "voids": other.get("void", 0), "hit_rate": p.win.mean(), "flat_roi": p.roi.mean(), "avg_edge": p.edge.mean(),
            "avg_clv": p.clv.mean(), "pct_beat_close": (p.clv > 0).mean(),
            "slips": len(s), "slip_profit": float(s.profit.sum()) if len(s) else 0.0,
            "slip_roi": float(s.profit.sum() / s.stake.sum()) if len(s) and s.stake.sum() else None}
    p["group"] = p.t3.fillna(0).astype(int).map({0: "Tier 1/2", 1: "Tier 3"})
    summ["by_group"] = p.groupby("group").agg(picks=("win", "size"), hit_rate=("win", "mean"), flat_roi=("roi", "mean"),
                                              avg_edge=("edge", "mean"), avg_clv=("clv", "mean")).reset_index().to_dict("records")
    return summ, p, s
