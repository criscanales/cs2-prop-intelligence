import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone

import pytest

from cs2model.tracking import backup as BK
from cs2model.tracking import store

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(BK, "STATUS_PATH", tmp_path / "status.json")
    p = tmp_path / "tracking.db"
    con = store.connect(p)
    con.execute("INSERT INTO preds(ts, source, player, stat, maps, line, p_over, actual, t3) "
                "VALUES ('2026-10-20', 'prizepicks', 'donk', 'kills', '1+2', 38.5, 0.55, 41, 0)")
    con.execute("INSERT INTO preds(ts, source, player, stat, maps, line, p_over, t3) "
                "VALUES ('2026-10-20', 'underdog', 'zywoo', 'kills', '1+2', 36.5, 0.6, 1)")
    con.execute("INSERT INTO map_odds_log(snapshot_ts, source, match_key, start_time, p_map1_quoted, src_map1) "
                "VALUES ('2026-10-20T10:00:00+00:00', 'pinnacle', 'a|b', '2026-10-20T12:00:00+00:00', 0.6, 'quoted')")
    con.commit()
    store.record_gap("2026-10-18T15:42:00+00:00", "2026-10-21T19:00:00+00:00", "all tracking tables", "machine lost", con)
    con.close()
    return p


def _add_pred(p, n=1):
    con = sqlite3.connect(p)
    for i in range(n):
        con.execute("INSERT INTO preds(ts, player, stat, actual) VALUES ('2026-10-21', ?, 'kills', 20)", (f"p{i}",))
    con.commit()
    con.close()


def test_snapshot_restore_roundtrip_into_clean_target(db, tmp_path):
    z, m = BK.snapshot(db, tmp_path / "out", "manual", NOW)
    assert z.name.startswith("kprlab-tracking-20261021T120000Z-manual-") and BK.parse_name(z.name)
    assert m["history_counts"]["preds"] == 2 and m["history_counts"]["preds_t3"] == 1
    assert m["history_counts"]["map_odds_log"] == 1 and m["history_counts"]["data_gaps"] == 1
    with zipfile.ZipFile(z) as f:
        assert set(f.namelist()) == {"tracking.db", "manifest.json", "provenance/map_odds_log.jsonl",
                                     "provenance/match_results.jsonl", "provenance/data_gaps.jsonl"}
    rep = BK.restore(z, tmp_path / "fresh" / "tracking.db")
    assert rep["integrity"] == "ok" and rep["history_counts"] == m["history_counts"]
    assert {"map_odds_log_no_update", "map_odds_log_no_delete", "data_gaps_no_delete"} <= set(rep["triggers"])
    con = sqlite3.connect(tmp_path / "fresh" / "tracking.db")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM map_odds_log")


def test_restore_never_silently_overwrites(db, tmp_path):
    z, _ = BK.snapshot(db, tmp_path / "out", "manual", NOW)
    with pytest.raises(FileExistsError):
        BK.restore(z, db)
    BK.restore(z, db, force=True)
    assert list(tmp_path.glob("tracking.pre-restore-*.db"))


def test_tampered_backup_rejected(db, tmp_path):
    z, _ = BK.snapshot(db, tmp_path / "out", "manual", NOW)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(bad, "w") as dst:
        for n in src.namelist():
            data = src.read(n)
            dst.writestr(n, data + b"x" if n == "provenance/map_odds_log.jsonl" else data)
    with pytest.raises(ValueError, match="checksum"):
        BK.restore(bad, tmp_path / "x.db")


def test_backup_dedupes_unchanged_and_uploads_verified(db, tmp_path):
    bs = BK.DirStore(tmp_path / "remote")
    first = BK.backup(bs, "daily", db, NOW)
    assert first and set(bs.names()) == {first, first + ".sha256"}
    assert BK.backup(bs, "daily", db, NOW + timedelta(days=1)) is None
    assert BK.status()["last_result"] == "unchanged"
    _add_pred(db)
    second = BK.backup(bs, "event", db, NOW + timedelta(days=1, hours=1))
    assert second and second != first and len(bs.names()) == 4
    rep = BK.restore_latest(bs, tmp_path / "clean" / "tracking.db")
    assert rep["name"] == second and rep["history_counts"]["preds"] == 3


def test_secrets_block_backup_and_failure_is_recorded(db, tmp_path):
    con = sqlite3.connect(db)
    con.execute("INSERT INTO line_snapshots(player) VALUES (?)", ("ghp_" + "a" * 36,))
    con.commit()
    con.close()
    bs = BK.DirStore(tmp_path / "remote")
    with pytest.raises(ValueError, match="secret"):
        BK.backup(bs, "daily", db, NOW)
    s = BK.status()
    assert s["last_result"] == "failed" and s["consecutive_failures"] == 1 and bs.names() == []


def test_retention_keeps_30_plus_days_and_monthly_older():
    names = [f"kprlab-tracking-{(NOW - timedelta(days=d)):%Y%m%dT%H%M%SZ}-daily-{d:012x}.zip" for d in range(120)]
    drop = set(BK.to_prune(names, NOW))
    kept = [BK.parse_name(n) for n in names if n not in drop]
    assert all(BK.parse_name(n)["ts"] < NOW - timedelta(days=35) for n in drop)
    assert sum(k["ts"] >= NOW - timedelta(days=35) for k in kept) == 36
    old = [k for k in kept if k["ts"] < NOW - timedelta(days=35)]
    assert len({(k["ts"].year, k["ts"].month) for k in old}) == len(old)
    assert BK.to_prune(names[:1], NOW + timedelta(days=400)) == []


def test_due_daily_and_after_runs_short_configurable_interval():
    c = {"preds": 10, "preds_graded": 2, "picks_graded": 0, "match_results": 1, "map_odds_log": 3, "slips": 0}
    assert BK.due({}, c, NOW, None) == "daily"
    st = {"last_check": NOW.isoformat(), "history_counts": c}
    assert BK.due(st, c, NOW + timedelta(hours=7), NOW.isoformat()) is None
    graded = dict(c, preds_graded=5)
    assert BK.due(st, graded, NOW + timedelta(minutes=10), NOW.isoformat()) is None
    assert BK.due(st, graded, NOW + timedelta(minutes=31), NOW.isoformat()) == "event"
    assert BK.due(st, graded, NOW + timedelta(minutes=6), NOW.isoformat(), event_min=5) == "event"
    assert BK.due(st, dict(c, map_odds_log=4), NOW + timedelta(minutes=31), NOW.isoformat()) == "event"
    assert BK.due(st, dict(c, preds=40), NOW + timedelta(minutes=31), NOW.isoformat()) is None
    assert BK.due(st, c, NOW + timedelta(hours=25), NOW.isoformat()) == "daily"


def test_retention_thins_intraday_versions_after_two_days():
    names = [f"kprlab-tracking-{(NOW - timedelta(minutes=30 * i)):%Y%m%dT%H%M%SZ}-event-{i:012x}.zip" for i in range(48 * 40)]
    drop = set(BK.to_prune(names, NOW))
    kept = [BK.parse_name(n) for n in names if n not in drop]
    assert sum(k["ts"] >= NOW - timedelta(days=2) for k in kept) == 97
    mid = [k for k in kept if NOW - timedelta(days=35) <= k["ts"] < NOW - timedelta(days=2)]
    assert len({k["ts"].date() for k in mid}) == len(mid) >= 32
    assert len(kept) < 200


def test_poll_once_publishes_health_and_opens_then_closes_alert(db, tmp_path, monkeypatch):
    bs = BK.DirStore(tmp_path / "store")
    assert BK.poll_once(bs, db, NOW) is None
    h = __import__("json").loads((tmp_path / "store" / BK.HEALTH_NAME).read_text())
    assert h["last_result"] == "uploaded" and h["db_present"] and h["history_counts"]["preds"] == 2
    monkeypatch.setattr(BK, "secret_hits", lambda con: ["preds.player"])
    _add_pred(db, 60)
    assert BK.poll_once(bs, db, NOW + timedelta(hours=25)) is not None
    assert (tmp_path / "store" / "ALERT.txt").exists() and BK.status()["alert_open"]
    monkeypatch.undo()
    monkeypatch.setattr(BK, "STATUS_PATH", tmp_path / "status.json")
    assert BK.poll_once(bs, db, NOW + timedelta(hours=50)) is None
    assert not (tmp_path / "store" / "ALERT.txt").exists() and not BK.status()["alert_open"]


def test_poll_once_waits_for_missing_db_but_still_heartbeats(tmp_path, monkeypatch):
    monkeypatch.setattr(BK, "STATUS_PATH", tmp_path / "status.json")
    bs = BK.DirStore(tmp_path / "store")
    assert BK.poll_once(bs, tmp_path / "missing.db", NOW) is None
    h = __import__("json").loads((tmp_path / "store" / BK.HEALTH_NAME).read_text())
    assert h["db_present"] is False and [n for n in bs.names() if n.endswith(".zip")] == []


def test_supervisor_restarts_crashed_daemon(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(BK, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(BK, "LOG_PATH", tmp_path / "backup.log")
    n = BK.supervise([sys.executable, "-c", "raise SystemExit(3)"], max_restarts=2, backoff=(0, 0))
    assert n == 3 and BK.status()["daemon_restarts"] == 3
    assert "BACKUP DAEMON EXITED (code 3)" in (tmp_path / "backup.log").read_text()


def test_first_observation_is_earliest_collected_record(db):
    con = store.connect(db)
    assert store.first_observation(con) == "2026-10-20"
    empty = store.connect(db.with_name("empty.db"))
    assert store.first_observation(empty) is None


def test_data_gaps_append_only_and_idempotent(db):
    con = store.connect(db)
    store.record_gap("2026-10-18T15:42:00+00:00", "2026-10-21T19:00:00+00:00", "all tracking tables", "machine lost", con)
    assert len(store.data_gaps(con)) == 1
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE data_gaps SET reason='x'")


def test_never_backs_up_to_app_repo():
    with pytest.raises(ValueError):
        BK.make_store("gh:criscanales/cs2-prop-intelligence")
