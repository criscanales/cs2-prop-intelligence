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
CREATE TABLE IF NOT EXISTS quarantine (
  id INTEGER PRIMARY KEY, recorded_ts TEXT, tbl TEXT, reason TEXT, match_id INTEGER, payload TEXT);
CREATE TRIGGER IF NOT EXISTS quarantine_no_update BEFORE UPDATE ON quarantine
  BEGIN SELECT RAISE(ABORT, 'quarantine is append-only'); END;
CREATE TRIGGER IF NOT EXISTS quarantine_no_delete BEFORE DELETE ON quarantine
  BEGIN SELECT RAISE(ABORT, 'quarantine is append-only'); END;
CREATE TABLE IF NOT EXISTS exclusions (
  id INTEGER PRIMARY KEY, recorded_ts TEXT, tbl TEXT, row_key TEXT, reason TEXT, match_id INTEGER, UNIQUE(tbl, row_key));
CREATE TRIGGER IF NOT EXISTS exclusions_no_update BEFORE UPDATE ON exclusions
  BEGIN SELECT RAISE(ABORT, 'exclusions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS exclusions_no_delete BEFORE DELETE ON exclusions
  BEGIN SELECT RAISE(ABORT, 'exclusions is append-only'); END;
"""

MAP_ODDS_FIELDS = ["pred_ts", "source", "match_key", "team_a_id", "team_b_id", "team_a", "team_b", "start_time", "best_of",
                   "t3", "p_series_a", "p_map1_quoted", "p_map2_quoted", "p_map3_quoted", "p_map1_used", "p_map2_used",
                   "p_map3_used", "src_map1", "src_map2", "src_map3", "p_map3_played", "p_map3_played_book", "model_version"]


def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)  # the dashboard and the refresh loop write concurrently
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
    for t in ("preds", "picks"):
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
        if "match_id" not in cols:
            con.execute(f"ALTER TABLE {t} ADD COLUMN match_id INTEGER")  # bo3.gg match id verified at logging time
        if "prematch" not in cols:
            con.execute(f"ALTER TABLE {t} ADD COLUMN prematch TEXT")  # scheduled / uncertain / verified (NULL = legacy)
        con.commit()
    return con


_FETCH = object()


def _status(status):
    from . import prematch
    return prematch.safe_status() if status is _FETCH else status


def _quarantine(con, tbl, reason, match_id, payload):
    con.execute("INSERT INTO quarantine (recorded_ts, tbl, reason, match_id, payload) VALUES (?,?,?,?,?)",
                (_now(), tbl, reason, match_id, json.dumps(payload, default=str)))


def _gate(con, tbl, team_id, opp_id, status, payload, start_time=None, allow_uncertain=True):
    """(bo3 match id, prematch level) when the record may be logged, else quarantine it and return None."""
    from . import prematch
    level, mid, why = prematch.classify(team_id, opp_id, status, start_time)
    if level == "scheduled" or (level == "uncertain" and allow_uncertain):
        return mid, level
    _quarantine(con, tbl, why, mid, payload)
    return None



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


def log_preds(slate, con=None, status=_FETCH):
    """Keep the latest pre-start model price of every standard line (graded later for the bias check). Lines whose match
    bo3.gg does not verify as still upcoming (no map begun) are quarantined instead."""
    con = con or connect()
    if slate is None or slate.empty:
        return
    status = _status(status)
    now = pd.Timestamp.now(tz="UTC")
    s = slate[slate.odds_type.eq("standard") & slate.start_time.notna()]
    rows = []
    for r in s.itertuples():
        st = pd.Timestamp(r.start_time)
        st = st.tz_localize("UTC") if st.tzinfo is None else st
        if st > now:
            t3, raw = getattr(r, "t3", False), getattr(r, "p_over_raw", None)
            row = (r.source, r.player, r.team, r.opponent, r.stat, _maps(r.maps), st.isoformat(), _now(),
                   float(r.line), float(r.model_mean), float(r.p_over), int(t3 is True or t3 == 1),
                   None if raw is None or raw != raw else float(raw))
            g = _gate(con, "preds", getattr(r, "team_id", None), getattr(r, "opp_id", None), status, row, st)
            if g is not None:
                rows.append(row + g)
    q = ("INTO preds (source, player, team, opponent, stat, maps, start_time, ts, line, model_mean, p_over, t3, p_over_raw, "
         "match_id, prematch) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
    con.executemany("INSERT OR REPLACE " + q, [x for x in rows if x[-1] == "scheduled"])
    con.executemany("INSERT OR IGNORE " + q, [x for x in rows if x[-1] != "scheduled"])  # never overwrite a confident price
    con.commit()


def _inputs_hash(r):
    import hashlib
    keys = [k for k in MAP_ODDS_FIELDS if k not in ("pred_ts", "team_a", "team_b")]
    return hashlib.sha1(json.dumps([r.get(k) if not isinstance(r.get(k), float) else round(r[k], 6) for k in keys]).encode()).hexdigest()


def log_map_odds(records, snapshot_ts, con=None, status=_FETCH):
    """Append one map-odds provenance row per (source, match, snapshot). Pre-match snapshots only; a snapshot whose inputs
    are identical to that source/match's latest row (same book prices, same model version) is not appended again."""
    con = con or connect()
    status = _status(status)
    snap = pd.Timestamp(snapshot_ts)
    snap = snap.tz_localize("UTC") if snap.tzinfo is None else snap.tz_convert("UTC")
    n = 0
    for r in records:
        if not r.get("start_time") or pd.Timestamp(r["start_time"]) <= snap:
            continue
        if _gate(con, "map_odds_log", r.get("team_a_id"), r.get("team_b_id"), status, {"snapshot_ts": str(snap), **r},
                 r.get("start_time"), allow_uncertain=False) is None:  # append-only table: only confident snapshots
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
        if h.match_id.nunique() > 1:
            continue  # two series between these teams in the window (e.g. a rematch): can't tell which one was logged
        h = h.drop_duplicates("map_number").sort_values("map_number")
        if h.won.isna().any() or list(h.map_number) != list(range(1, len(h) + 1)):
            continue
        need, wins = int(bo) // 2 + 1, int(h.won.sum())
        if max(wins, len(h) - wins) < need:
            continue  # series not finished (or later maps not in history yet)
        if con.execute("SELECT 1 FROM match_results WHERE bo3_match_id=?", (int(h.match_id.iloc[0]),)).fetchone():
            continue  # already recorded under another scheduled start time
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
    d = d[~d.id.astype(str).isin(excluded(con, "map_odds_log"))
          & ~(d.match_key + "|" + d.start_time).isin(excluded(con, "match_results"))]
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


def _player_match_rows(hist, pid, opponent, st, match_id=None):
    """The player's maps in the logged series: by the verified bo3.gg match id when the record has one, else the only
    series against that opponent in the time window (None if there is none, or several, e.g. a same-day rematch)."""
    from ..names import team_similarity
    if match_id is not None and match_id == match_id:
        h = hist[(hist.player_id == pid) & (hist.match_id == int(match_id))]
        return h if len(h) else None
    h = hist[(hist.player_id == pid) & (hist.date >= st - pd.Timedelta(hours=3)) & (hist.date <= st + pd.Timedelta(hours=12))]
    h = h[h.opponent.map(lambda o: team_similarity(o, opponent) >= 0.75)]
    if h.empty or h.match_id.nunique() > 1:
        return None
    return h


def _settle(hist, resolve_player, player, team, opponent, stat, maps, start_time, match_id=None):
    """("graded", total) when every requested map has the player's stats, ("void", None) when a requested map
    was never played because the series was already decided (e.g. Map 3 after a 2-0), None while unresolved
    (not started, match/player not in history yet, or a played map's data is missing)."""
    st = pd.Timestamp(start_time)
    st = st.tz_localize("UTC") if st.tzinfo is None else st
    if st > pd.Timestamp.now(tz="UTC"):
        return None
    pid = resolve_player(player, team)
    if pid is None:
        return None
    h = _player_match_rows(hist, pid, opponent, st, match_id)
    if h is None:
        return None
    h = h.drop_duplicates("map_number")
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


def _match_of(hist, resolve_player, player, team, opponent, start_time, match_id=None):
    """(bo3 match id, verified Map 1 begin) of the series a logged prop belongs to, or (None, None) if not in `hist`."""
    st = pd.Timestamp(start_time)
    st = st.tz_localize("UTC") if st.tzinfo is None else st
    pid = resolve_player(player, team)
    if pid is None:
        return None, None
    h = _player_match_rows(hist, pid, opponent, st, match_id)
    if h is None:
        return None, None
    mid = int(h.match_id.iloc[0])
    return mid, hist.loc[hist.match_id == mid, "date"].min()


def pred_key(source, player, stat, maps, start_time):
    return f"{source}|{player}|{stat}|{maps}|{start_time}"


def excluded(con, tbl):
    return {r[0] for r in con.execute("SELECT row_key FROM exclusions WHERE tbl=?", (tbl,))}


def _ts(x):
    t = pd.to_datetime(x, utc=True, errors="coerce", format="ISO8601")
    return t


def audit(hist, resolve_player, con=None, dry_run=False):
    """Mark (never delete) settled records that must not count in reports: rows logged at or after the verified Map 1
    begin (post_start), older scheduled-start copies of the same selection (superseded_reschedule: the latest eligible
    copy is kept), duplicate match results of one bo3 match (duplicate_match), map-odds snapshots taken after Map 1
    began, and slips with a post-start leg or repeating an identical earlier slip. Only matches already in `hist` are
    audited, so every pre-match copy exists by then. Idempotent. Returns the new exclusions."""
    con = con or connect()
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"], utc=True, format="ISO8601")
    begin = hist.groupby("match_id").date.min()
    new, verified = [], []

    def mark(tbl, key, reason, mid):
        new.append(dict(tbl=tbl, row_key=str(key), reason=reason, match_id=None if mid is None else int(mid)))

    P = pd.read_sql("SELECT * FROM preds WHERE actual IS NOT NULL OR COALESCE(void, 0) = 1", con)
    if len(P):
        P["key"] = [pred_key(*r) for r in zip(P.source, P.player, P.stat, P.maps, P.start_time)]
        P = P[~P.key.isin(excluded(con, "preds"))]
    K = pd.read_sql("SELECT * FROM picks WHERE outcome IS NOT NULL", con)
    K = K[~K.id.astype(str).isin(excluded(con, "picks"))]
    for D, tcol, tbl, keycol, group in ((P, "ts", "preds", "key", ["source", "player", "stat", "maps"]),
                                        (K, "created", "picks", "id", ["source", "player", "stat", "maps", "side", "line"])):
        if D.empty:
            continue
        mb = [_match_of(hist, resolve_player, r.player, r.team, r.opponent, r.start_time, getattr(r, "match_id", None))
              for r in D.itertuples()]
        D = D.assign(mid=[m for m, _ in mb], begin=[b for _, b in mb], st=_ts(D.start_time), t=_ts(D[tcol]))
        D = D[D.mid.notna()]
        late = D[D.t.isna() | (D.t >= D.begin)]
        for r in late.itertuples():
            mark(tbl, getattr(r, keycol), "unverified_time" if pd.isna(r.t) else "post_start", r.mid)
        ok = D.drop(late.index).sort_values(["st", "t"])
        for r in ok[ok.duplicated(group + ["mid"], keep="last")].itertuples():
            mark(tbl, getattr(r, keycol), "superseded_reschedule", r.mid)
        held = ok[ok.prematch.eq("uncertain")] if "prematch" in ok else ok.iloc[:0]
        verified.extend((tbl, r) for r in held.itertuples())
    R = pd.read_sql("SELECT * FROM match_results", con)
    if len(R):
        R["key"] = R.match_key + "|" + R.start_time
        R = R[~R.key.isin(excluded(con, "match_results"))].assign(st=lambda d: _ts(d.start_time)).sort_values("st")
        for r in R[R.duplicated("bo3_match_id", keep="last")].itertuples():
            mark("match_results", r.key, "duplicate_match", r.bo3_match_id)
        M = pd.read_sql("SELECT id, match_key, snapshot_ts FROM map_odds_log", con)
        M = M[~M.id.astype(str).isin(excluded(con, "map_odds_log"))]
        mb = R.drop_duplicates("match_key", keep="last").set_index("match_key").bo3_match_id
        M = M[M.match_key.isin(mb.index)].assign(mid=lambda d: d.match_key.map(mb))
        M = M[_ts(M.snapshot_ts) >= M.mid.map(begin)]
        for r in M.itertuples():
            mark("map_odds_log", r.id, "post_start", r.mid)
    late_picks = {int(x["row_key"]) for x in new if x["tbl"] == "picks" and x["reason"] == "post_start"} | \
        {int(k) for k, why in con.execute("SELECT row_key, reason FROM exclusions WHERE tbl='picks' AND reason='post_start'")}
    S = pd.read_sql("SELECT * FROM slips WHERE hits IS NOT NULL", con)
    S = S[~S.id.astype(str).isin(excluded(con, "slips"))]
    if len(S):
        ident = {r[0]: r[1:] for r in con.execute("SELECT id, match_id, source, player, stat, maps, side, line FROM picks")}
        S["legs_l"] = S.legs.map(json.loads)
        S["sig"] = [(r.platform, r.play, tuple(sorted(map(str, (ident.get(i) for i in r.legs_l))))) for r in S.itertuples()]
        bad = S.legs_l.map(lambda ls: any(i in late_picks for i in ls))
        for r in S[bad].itertuples():
            mark("slips", r.id, "post_start_leg", None)
        ok = S[~bad].sort_values("created")
        for r in ok[ok.duplicated("sig", keep="last")].itertuples():
            mark("slips", r.id, "duplicate_slip", None)
    if not dry_run:  # held "uncertain" rows proven to be logged before Map 1 began
        for tbl, r in verified:
            if tbl == "preds":
                con.execute("UPDATE preds SET prematch='verified' WHERE source=? AND player=? AND stat=? AND maps=? AND start_time=?",
                            (r.source, r.player, r.stat, r.maps, r.start_time))
            else:
                con.execute("UPDATE picks SET prematch='verified' WHERE id=?", (int(r.id),))
        con.commit()
    if not dry_run and new:
        con.executemany("INSERT OR IGNORE INTO exclusions (recorded_ts, tbl, row_key, reason, match_id) VALUES (?,?,?,?,?)",
                        [(_now(), x["tbl"], x["row_key"], x["reason"], x["match_id"]) for x in new])
        con.commit()
    return pd.DataFrame(new, columns=["tbl", "row_key", "reason", "match_id"])


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
        res = _settle(hist, resolve_player, r.player, r.team, r.opponent, r.stat, r.maps, r.start_time, getattr(r, "match_id", None))
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
        res = _settle(hist, resolve_player, r.player, r.team, r.opponent, r.stat, r.maps, r.start_time, getattr(r, "match_id", None))
        if res is None:
            continue
        old = ("void", None) if r.void == 1 else ("graded", r.actual)
        if old != res:
            changes.append(dict(table="preds", source=r.source, player=r.player, stat=r.stat, maps=r.maps, start_time=r.start_time,
                                line=r.line, t3=r.t3, old_actual=old[1], old_status=old[0], new_actual=res[1], new_status=res[0]))
            if not dry_run:
                con.execute("UPDATE preds SET actual=?, void=? WHERE rowid=?", (res[1], int(res[0] == "void"), int(r.rid)))
    for r in pd.read_sql("SELECT * FROM picks WHERE outcome IS NOT NULL", con).itertuples():
        res = _settle(hist, resolve_player, r.player, r.team, r.opponent, r.stat, r.maps, r.start_time, getattr(r, "match_id", None))
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
    p = pd.read_sql("SELECT * FROM preds WHERE actual IS NOT NULL AND COALESCE(t3, 0) = 0 AND COALESCE(prematch, '') != 'uncertain'", con)
    p = p[~pd.Series([pred_key(*r) for r in zip(p.source, p.player, p.stat, p.maps, p.start_time)], index=p.index, dtype=object)
          .isin(excluded(con, "preds"))]
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
    p = pd.read_sql("SELECT source, player, start_time, stat, maps, line, p_over, actual FROM preds WHERE t3 = 1 AND actual IS NOT NULL AND COALESCE(prematch, '') != 'uncertain'", con)
    p = p[~pd.Series([pred_key(*r) for r in zip(p.source, p.player, p.stat, p.maps, p.start_time)], index=p.index, dtype=object)
          .isin(excluded(con, "preds"))]
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


def log_picks(df: pd.DataFrame, con=None, status=_FETCH):
    """Record picks whose match bo3.gg verifies as still upcoming (no map begun); quarantine the rest."""
    con = con or connect()
    status = _status(status)
    cols = ["source", "player", "team", "opponent", "stat", "maps", "line", "side", "p_side", "breakeven",
            "edge", "ev", "tier", "model_mean", "start_time"]
    rows = []
    for _, r in df.iterrows():
        row = tuple(str(x) if isinstance(x, pd.Timestamp) else x for x in (
            _now(), *[(_maps(r[c]) if c == "maps" else (None if pd.isna(r[c]) else r[c])) for c in cols],
            int(r.get("t3") is True or r.get("t3") == 1)))
        g = _gate(con, "picks", r.get("team_id"), r.get("opp_id"), status, row, r.get("start_time"))
        if g is not None:
            rows.append(row + g)
    con.executemany("INSERT OR IGNORE INTO picks (created, source, player, team, opponent, stat, maps, line, side, "
                    "p_model, breakeven, edge, ev, tier, model_mean, start_time, t3, match_id, prematch) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()


def log_slip(platform, play, legs, p_all, ev, stake, hit_dist, con=None):
    con = con or connect()
    if any(i is None for i in legs):  # a leg was not logged (quarantined), so the slip is not a verified pre-match slip
        _quarantine(con, "slips", "leg_not_logged", None, {"platform": platform, "play": play, "legs": legs})
        con.commit()
        return
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
        res = _settle(hist, resolve_player, p.player, p.team, p.opponent, p.stat, p.maps, p.start_time, p.get("match_id"))
        if res is None:
            continue
        outcome = "void" if res[0] == "void" else _outcome(res[1], p.line, p.side)
        _, b = _match_of(hist, resolve_player, p.player, p.team, p.opponent, p.start_time, p.get("match_id"))
        close = con.execute("SELECT line FROM line_snapshots WHERE source=? AND player=? AND stat=? AND maps=? "
                            "AND start_time=? AND (? IS NULL OR ts < ?) ORDER BY ts DESC LIMIT 1",
                            (p.source, p.player, p.stat, p.maps, p.start_time, *([None if b is None else b.isoformat()] * 2))).fetchone()
        con.execute("UPDATE picks SET actual=?, outcome=?, closing_line=? WHERE id=?",
                    (res[1], outcome, close[0] if close else None, int(p.id)))
        n += 1
    con.commit()
    _grade_slips(con, payouts)
    audit(hist, resolve_player, con)
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
    allp = pd.read_sql("SELECT * FROM picks WHERE outcome IS NOT NULL AND COALESCE(prematch, '') != 'uncertain'", con)
    allp = allp[~allp.id.astype(str).isin(excluded(con, "picks"))]
    p = allp[allp.outcome.isin(["win", "loss"])].copy()
    s = pd.read_sql("SELECT * FROM slips WHERE hits IS NOT NULL", con)
    held = {r[0] for r in con.execute("SELECT id FROM picks WHERE prematch = 'uncertain'")}
    s = s[~s.id.astype(str).isin(excluded(con, "slips")) & ~s.legs.map(lambda ls: bool(held & set(json.loads(ls))))]
    other = allp.outcome.value_counts().to_dict()
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
