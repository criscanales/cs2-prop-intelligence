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
CREATE TABLE IF NOT EXISTS slips (
  id INTEGER PRIMARY KEY, created TEXT, platform TEXT, play TEXT, legs TEXT, p_all REAL,
  ev REAL, stake REAL, hit_dist TEXT, hits INTEGER, payout REAL, profit REAL);
"""


def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
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


def first_line(con, source, player, stat, maps, start_time):
    r = con.execute("SELECT line FROM line_snapshots WHERE source=? AND player=? AND stat=? AND maps=? "
                    "AND start_time IS ? ORDER BY ts LIMIT 1", (source, player, stat, _maps(maps), start_time)).fetchone()
    return r[0] if r else None


def log_picks(df: pd.DataFrame, con=None):
    con = con or connect()
    cols = ["source", "player", "team", "opponent", "stat", "maps", "line", "side", "p_side", "breakeven",
            "edge", "ev", "tier", "model_mean", "start_time"]
    rows = [(_now(), *[(_maps(r[c]) if c == "maps" else (None if pd.isna(r[c]) else r[c])) for c in cols])
            for _, r in df.iterrows()]
    con.executemany("INSERT OR IGNORE INTO picks (created, source, player, team, opponent, stat, maps, line, side, "
                    "p_model, breakeven, edge, ev, tier, model_mean, start_time) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
    hist["date"] = pd.to_datetime(hist["date"], utc=True)
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
    return summ, p, s
