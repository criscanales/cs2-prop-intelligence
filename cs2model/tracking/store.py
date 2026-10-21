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
"""


def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    if "t3" not in {r[1] for r in con.execute("PRAGMA table_info(picks)")}:
        con.execute("ALTER TABLE picks ADD COLUMN t3 INTEGER DEFAULT 0")  # tier-3 picks, tracked separately
        con.commit()
    return con


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
            rows.append((r.source, r.player, r.team, r.opponent, r.stat, _maps(r.maps), st.isoformat(), _now(),
                         float(r.line), float(r.model_mean), float(r.p_over)))
    con.executemany("INSERT OR REPLACE INTO preds VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)", rows)
    con.commit()


def _actual(hist, resolve_player, player, team, opponent, stat, maps, start_time):
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
    mapnos = [int(x) for x in str(maps).split("+")]
    if not set(h.map_number).issuperset(m for m in mapnos if m <= 2):
        return None
    return float(h[h.map_number.isin(mapnos)][stat].sum())


def grade_preds(hist, resolve_player, con=None):
    con = con or connect()
    p = pd.read_sql("SELECT rowid AS rid, * FROM preds WHERE actual IS NULL", con)
    n = 0
    for r in p.itertuples():
        a = _actual(hist, resolve_player, r.player, r.team, r.opponent, r.stat, r.maps, r.start_time)
        if a is not None:
            con.execute("UPDATE preds SET actual=? WHERE rowid=?", (a, int(r.rid)))
            n += 1
    con.commit()
    return n


def bias_report(con=None):
    """Model vs results on every graded line (not just picks): positive model - actual = model runs high."""
    con = con or connect()
    p = pd.read_sql("SELECT * FROM preds WHERE actual IS NOT NULL", con)
    if p.empty:
        return p
    p["scope"] = p.maps.map(lambda m: "Map " + m if "+" not in m else "Maps " + m)
    g = p.groupby(["stat", "scope"])[["model_mean", "actual", "line", "p_over"]]
    out = g.apply(lambda x: pd.Series({
        "lines": len(x), "model_minus_actual": (x.model_mean - x.actual).mean(), "line_minus_actual": (x.line - x.actual).mean(),
        "model_minus_line": (x.model_mean - x.line).mean(), "avg_p_over": x.p_over.mean(), "over_rate": (x.actual > x.line).mean()}))
    return out.reset_index()


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
    """Fill actual results, closing lines and outcomes for picks whose matches are in `hist`."""
    from ..names import team_similarity
    con = con or connect()
    picks = pd.read_sql("SELECT * FROM picks WHERE outcome IS NULL AND start_time IS NOT NULL", con)
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"], utc=True, format="ISO8601")
    grade_preds(hist, resolve_player, con)
    n = 0
    for _, p in picks.iterrows():
        st = pd.Timestamp(p.start_time)
        st = st.tz_localize("UTC") if st.tzinfo is None else st
        if st > pd.Timestamp.now(tz="UTC"):
            continue
        pid = resolve_player(p.player, p.team)
        if pid is None:
            continue
        h = hist[(hist.player_id == pid) & (hist.date >= st - pd.Timedelta(hours=3)) & (hist.date <= st + pd.Timedelta(hours=12))]
        h = h[h.opponent.map(lambda o: team_similarity(o, p.opponent) >= 0.75)]
        if h.empty:
            continue
        mapnos = [int(x) for x in str(p.maps).split("+")]
        if not set(h.map_number).issuperset(m for m in mapnos if m <= 2):
            continue
        actual = float(h[h.map_number.isin(mapnos)][p.stat].sum())
        outcome = "push" if actual == p.line else ("win" if (actual > p.line) == (p.side == "over") else "loss")
        close = con.execute("SELECT line FROM line_snapshots WHERE source=? AND player=? AND stat=? AND maps=? "
                            "AND start_time=? ORDER BY ts DESC LIMIT 1",
                            (p.source, p.player, p.stat, p.maps, p.start_time)).fetchone()
        con.execute("UPDATE picks SET actual=?, outcome=?, closing_line=? WHERE id=?",
                    (actual, outcome, close[0] if close else None, int(p.id)))
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
    if p.empty:
        return {"picks": 0}, p, s
    p["win"] = (p.outcome == "win").astype(int)
    p["roi"] = p.win / p.breakeven - 1
    sign = p.side.map({"over": 1, "under": -1})
    p["clv"] = (p.closing_line - p.line) * sign  # positive = line moved in our favour
    summ = {"picks": len(p), "hit_rate": p.win.mean(), "flat_roi": p.roi.mean(), "avg_edge": p.edge.mean(),
            "avg_clv": p.clv.mean(), "pct_beat_close": (p.clv > 0).mean(),
            "slips": len(s), "slip_profit": float(s.profit.sum()) if len(s) else 0.0,
            "slip_roi": float(s.profit.sum() / s.stake.sum()) if len(s) and s.stake.sum() else None}
    p["group"] = p.t3.fillna(0).astype(int).map({0: "Tier 1/2", 1: "Tier 3"})
    summ["by_group"] = p.groupby("group").agg(picks=("win", "size"), hit_rate=("win", "mean"), flat_roi=("roi", "mean"),
                                              avg_edge=("edge", "mean"), avg_clv=("clv", "mean")).reset_index().to_dict("records")
    return summ, p, s
