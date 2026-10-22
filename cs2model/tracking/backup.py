"""Off-machine backups of data/tracking.db as release assets of a private GitHub repo.

Each backup is a zip holding a SQLite online-backup copy of tracking.db, JSONL copies of the
append-only provenance tables and a manifest with SHA-256 checksums and row counts. It is
integrity-checked and secret-scanned before upload, and skipped when the content is unchanged.
"""
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..scraper.http import ROOT
from . import store

log = logging.getLogger(__name__)

PREFIX = "kprlab-tracking-"
NAME_RE = re.compile(r"^kprlab-tracking-(\d{8}T\d{6}Z)-(daily|event|manual)-([0-9a-f]{12})\.zip$")
PROVENANCE = ("map_odds_log", "match_results", "data_gaps")
STATUS_PATH = ROOT / "data" / "backup_status.json"
LOG_PATH = ROOT / "data" / "backup.log"
APP_REPO = "criscanales/cs2-prop-intelligence"
KEEP_DAYS = 35
KEEP_ALL_DAYS = 2
EVENT_MIN_MINUTES = int(os.environ.get("KPR_BACKUP_EVENT_MIN", "30"))
HEALTH_NAME = "kprlab-health.json"
DAILY_HOURS = 24
ASSET_WARN = 900  # GitHub allows 1000 assets per release
SECRET_RE = re.compile(r"(?i)(gh[pousr]_[a-z0-9]{30,}|github_pat_[a-z0-9_]{30,}|AKIA[0-9A-Z]{16}|sk-[a-z0-9]{20,}|"
                       r"bearer\s+[a-z0-9._\-]{16,}|(api[_-]?key|access[_-]?token|secret|password)\s*[=:]\s*\S{8,})")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _tables(con):
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                      "AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def row_counts(con):
    return {t: con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in _tables(con)}


def history_counts(con):
    """Counts of the records that can't be recreated, used to verify a restore."""
    q = {"preds": "SELECT count(*) FROM preds", "preds_graded": "SELECT count(*) FROM preds WHERE actual IS NOT NULL",
         "preds_t3": "SELECT count(*) FROM preds WHERE t3=1", "picks": "SELECT count(*) FROM picks",
         "picks_graded": "SELECT count(*) FROM picks WHERE outcome IS NOT NULL", "slips": "SELECT count(*) FROM slips",
         "line_snapshots": "SELECT count(*) FROM line_snapshots", "map_odds_log": "SELECT count(*) FROM map_odds_log",
         "match_results": "SELECT count(*) FROM match_results", "data_gaps": "SELECT count(*) FROM data_gaps"}
    out = {}
    for k, s in q.items():
        try:
            out[k] = con.execute(s).fetchone()[0]
        except sqlite3.OperationalError:
            out[k] = None
    return out


def content_hash(con):
    """Hash of schema + all rows in rowid order; equal for snapshots with identical content."""
    h = hashlib.sha256()
    for (sql,) in con.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"):
        h.update(sql.encode())
    for t in _tables(con):
        h.update(t.encode())
        for row in con.execute(f'SELECT * FROM "{t}" ORDER BY rowid'):
            h.update(repr(row).encode())
    return h.hexdigest()


def secret_hits(con):
    hits = []
    for t in _tables(con):
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")') if r[2].upper() in ("TEXT", "")]
        for c in cols:
            for (v,) in con.execute(f'SELECT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL'):
                if isinstance(v, str) and SECRET_RE.search(v):
                    hits.append(f"{t}.{c}")
                    break
    return hits


def _check_db(path):
    con = sqlite3.connect(path)
    try:
        ok = con.execute("PRAGMA integrity_check").fetchone()[0]
        if ok != "ok":
            raise ValueError(f"integrity_check failed: {ok}")
        return row_counts(con), history_counts(con), content_hash(con)
    finally:
        con.close()


def snapshot(db=None, out_dir=None, reason="manual", now=None):
    """Consistent copy of a live tracking.db via the SQLite backup API, packed and verified."""
    db = Path(db or store.DB_PATH)
    if not db.exists():
        raise FileNotFoundError(db)
    now = now or datetime.now(timezone.utc)
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="kprlab-bk-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="kprlab-snap-"))
    try:
        copy = work / "tracking.db"
        src = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
        dst = sqlite3.connect(copy)
        with dst:
            src.backup(dst)
        src.close()
        hits = secret_hits(dst)
        dst.close()
        if hits:
            raise ValueError(f"refusing to back up: secret-like values in {hits}")
        counts, hist, chash = _check_db(copy)
        files = {"tracking.db": copy}
        con = sqlite3.connect(copy)
        con.row_factory = sqlite3.Row
        for t in PROVENANCE:
            if t in counts:
                p = work / f"provenance/{t}.jsonl"
                p.parent.mkdir(exist_ok=True)
                with open(p, "w") as f:
                    for r in con.execute(f'SELECT * FROM "{t}" ORDER BY rowid'):
                        f.write(json.dumps(dict(r), default=str) + "\n")
                files[f"provenance/{t}.jsonl"] = p
        con.close()
        manifest = {"format": 1, "created_utc": now.isoformat(timespec="seconds"), "reason": reason,
                    "content_hash": chash, "sqlite_version": sqlite3.sqlite_version, "integrity": "ok",
                    "row_counts": counts, "history_counts": hist,
                    "files": {k: {"sha256": _sha256(v), "bytes": v.stat().st_size} for k, v in files.items()}}
        name = f"{PREFIX}{now.strftime('%Y%m%dT%H%M%SZ')}-{reason}-{chash[:12]}.zip"
        out = out_dir / name
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for k, v in files.items():
                z.write(v, k)
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
        verify(out)
        return out, manifest
    finally:
        shutil.rmtree(work, ignore_errors=True)


def verify(zip_path, extract_to=None):
    """Check member checksums, SQLite integrity, row counts and content hash against the manifest."""
    tmp = Path(extract_to or tempfile.mkdtemp(prefix="kprlab-verify-"))
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        m = json.loads(z.read("manifest.json"))
        if names - {"manifest.json"} != set(m["files"]):
            raise ValueError(f"unexpected members: {sorted(names ^ set(m['files']) ^ {'manifest.json'})}")
        for k in m["files"]:
            if Path(k).is_absolute() or ".." in Path(k).parts:
                raise ValueError(f"unsafe member {k}")
        z.extractall(tmp)
    for k, v in m["files"].items():
        if _sha256(tmp / k) != v["sha256"]:
            raise ValueError(f"checksum mismatch: {k}")
    counts, hist, chash = _check_db(tmp / "tracking.db")
    if counts != m["row_counts"] or chash != m["content_hash"]:
        raise ValueError("row counts / content hash differ from manifest")
    if extract_to is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return m


def restore(zip_path, target=None, force=False):
    """Verify a backup and install its tracking.db at `target`. A non-empty target is kept as
    tracking.pre-restore-<ts>.db and only replaced with force=True."""
    target = Path(target or store.DB_PATH)
    tmp = Path(tempfile.mkdtemp(prefix="kprlab-restore-"))
    try:
        m = verify(zip_path, extract_to=tmp)
        if target.exists():
            con = sqlite3.connect(target)
            n = sum(row_counts(con).values())
            con.close()
            if n and not force:
                raise FileExistsError(f"{target} has {n} rows; pass force=True to replace it")
            keep = target.with_name(f"tracking.pre-restore-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.db")
            shutil.copy2(target, keep)
            log.warning("kept existing database as %s", keep)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".restoring")
        src, dst = sqlite3.connect(tmp / "tracking.db"), sqlite3.connect(part)
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
        os.replace(part, target)
        store.connect(target).close()  # schema migrations + append-only triggers
        con = sqlite3.connect(target)
        got = history_counts(con)
        ok = con.execute("PRAGMA integrity_check").fetchone()[0]
        trig = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        con.close()
        want = {k: v for k, v in m["history_counts"].items() if v is not None}
        if ok != "ok" or any(got.get(k) != v for k, v in want.items()):
            raise ValueError(f"restored database does not match backup: {ok}, {got} vs {want}")
        return {"name": Path(zip_path).name, "created_utc": m["created_utc"], "integrity": ok,
                "history_counts": got, "triggers": sorted(trig)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parse_name(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    return {"name": name, "ts": datetime.strptime(m[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc),
            "reason": m[2], "hash12": m[3]}


def to_prune(names, now=None, keep_days=KEEP_DAYS, keep_all_days=KEEP_ALL_DAYS):
    """Keep every version from the last keep_all_days, the newest per UTC day up to keep_days, the newest of
    each older month, and the newest overall. Bounds the release to roughly 100 intraday + 35 daily + monthly assets."""
    now = now or datetime.now(timezone.utc)
    items = sorted(filter(None, map(parse_name, names)), key=lambda x: x["ts"], reverse=True)
    keep, days, months = set(), set(), set()
    for i, it in enumerate(items):
        t = it["ts"]
        if i == 0 or t >= now - timedelta(days=keep_all_days):
            keep.add(it["name"])
        elif t >= now - timedelta(days=keep_days):
            if t.date() not in days:
                keep.add(it["name"])
        elif (t.year, t.month) not in months:
            keep.add(it["name"])
        days.add(t.date())
        months.add((t.year, t.month))
    return [it["name"] for it in items if it["name"] not in keep]


class DirStore:
    """Backups in a local directory (tests and an optional second copy)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def check(self):
        pass

    def names(self):
        return sorted(p.name for p in self.root.iterdir() if p.is_file())

    def put(self, path):
        shutil.copy2(path, self.root / Path(path).name)

    def get(self, name, dest):
        return Path(shutil.copy2(self.root / name, Path(dest) / name))

    def delete(self, name):
        (self.root / name).unlink(missing_ok=True)

    def alert(self, title, body):
        (self.root / "ALERT.txt").write_text(f"{title}\n{body}\n")

    def resolve(self, body):
        (self.root / "ALERT.txt").unlink(missing_ok=True)


class GitHubReleaseStore:
    """Backups as assets of one release in a private GitHub repo, via the gh CLI. Alerts are a single open
    issue labelled backup-alert (GitHub notifies the repo owner), closed again on the next success."""

    LABEL = "backup-alert"

    def __init__(self, repo, tag="tracking-backups"):
        if repo.lower() == APP_REPO:
            raise ValueError("backups must never go to the application repo")
        self.repo, self.tag = repo, tag

    def _gh(self, *args, check=True):
        r = subprocess.run(["gh", *args, "--repo", self.repo], capture_output=True, text=True, timeout=600)
        if check and r.returncode:
            raise RuntimeError(f"gh {args[0]} {args[1]} failed: {r.stderr.strip()[:300]}")
        return r

    def check(self):
        r = subprocess.run(["gh", "repo", "view", self.repo, "--json", "isPrivate"], capture_output=True, text=True, timeout=60)
        if r.returncode or not json.loads(r.stdout).get("isPrivate"):
            raise RuntimeError(f"{self.repo} is not reachable or not private; refusing to upload")
        if self._gh("release", "view", self.tag, check=False).returncode:
            self._gh("release", "create", self.tag, "--title", "tracking.db backups",
                     "--notes", "Automated KPR Lab tracking.db backups. Restore: python -m cs2model.cli restore")

    def names(self):
        r = self._gh("release", "view", self.tag, "--json", "assets")
        return sorted(a["name"] for a in json.loads(r.stdout)["assets"])

    def put(self, path):
        self._gh("release", "upload", self.tag, str(path), "--clobber")

    def get(self, name, dest):
        self._gh("release", "download", self.tag, "--pattern", name, "--dir", str(dest), "--clobber")
        return Path(dest) / name

    def delete(self, name):
        self._gh("release", "delete-asset", self.tag, name, "--yes")

    def _open_issue(self):
        r = self._gh("issue", "list", "--label", self.LABEL, "--state", "open", "--json", "number")
        n = json.loads(r.stdout)
        return n[0]["number"] if n else None

    def alert(self, title, body):
        num = self._open_issue()
        if num:
            self._gh("issue", "comment", str(num), "--body", body)
        else:
            self._gh("label", "create", self.LABEL, "--color", "d73a4a", "--force")
            self._gh("issue", "create", "--title", title, "--label", self.LABEL, "--body", body)

    def resolve(self, body):
        num = self._open_issue()
        if num:
            self._gh("issue", "close", str(num), "--comment", body)


def make_store(dest):
    kind, _, where = dest.partition(":")
    if kind == "gh":
        return GitHubReleaseStore(where)
    if kind == "dir":
        return DirStore(where)
    raise ValueError(f"unknown backup destination {dest!r} (use gh:owner/repo or dir:/path)")


def _status(**kw):
    s = json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {}
    s.update(kw)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(s, indent=2, default=str))
    return s


def status():
    return json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {}


def _upload_verified(bs, zpath):
    """Upload zip + .sha256 sidecar, then download it back and compare checksums."""
    side = zpath.with_name(zpath.name + ".sha256")
    side.write_text(f"{_sha256(zpath)}  {zpath.name}\n")
    bs.put(zpath)
    bs.put(side)
    with tempfile.TemporaryDirectory() as d:
        if _sha256(bs.get(zpath.name, d)) != side.read_text().split()[0]:
            raise RuntimeError("uploaded backup checksum mismatch")


def backup(dest, reason="manual", db=None, now=None, keep_days=KEEP_DAYS):
    """Snapshot, verify, dedupe against the newest stored backup, upload, prune. Returns the uploaded
    name or None when the content is unchanged. Failures are logged, recorded in backup_status.json and raised."""
    now = now or datetime.now(timezone.utc)
    _status(last_attempt=now.isoformat(timespec="seconds"), last_attempt_reason=reason)
    try:
        bs = make_store(dest) if isinstance(dest, str) else dest
        bs.check()
        out_dir = Path(tempfile.mkdtemp(prefix="kprlab-bk-"))
        try:
            zpath, m = snapshot(db, out_dir, reason, now)
            names = bs.names()
            stored = sorted(filter(None, map(parse_name, names)), key=lambda x: x["ts"])
            if stored and stored[-1]["hash12"] == m["content_hash"][:12]:
                log.info("backup skipped: content unchanged since %s", stored[-1]["name"])
                _status(last_check=now.isoformat(timespec="seconds"), last_result="unchanged", last_error=None,
                        consecutive_failures=0, history_counts=m["history_counts"])
                return None
            _upload_verified(bs, zpath)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
        names = bs.names()
        for n in to_prune(names, now, keep_days):
            bs.delete(n)
            bs.delete(n + ".sha256")
            log.info("pruned %s", n)
        if len(names) > ASSET_WARN:
            log.warning("backup release has %d assets (GitHub limit 1000)", len(names))
        log.info("backup uploaded: %s (%s)", zpath.name, m["history_counts"])
        _status(last_check=now.isoformat(timespec="seconds"), last_success=now.isoformat(timespec="seconds"),
                last_name=zpath.name, last_result="uploaded", last_error=None, consecutive_failures=0,
                history_counts=m["history_counts"])
        return zpath.name
    except Exception as e:
        log.error("BACKUP FAILED (%s): %s", reason, e)
        s = status()
        _status(last_result="failed", last_error=f"{now.isoformat(timespec='seconds')} {type(e).__name__}: {e}",
                consecutive_failures=s.get("consecutive_failures", 0) + 1)
        raise


def restore_latest(dest, target=None, force=False, name=None):
    bs = make_store(dest) if isinstance(dest, str) else dest
    names = bs.names()
    items = sorted(filter(None, map(parse_name, names)), key=lambda x: x["ts"])
    if not items:
        raise FileNotFoundError("no backups found")
    pick = name or items[-1]["name"]
    with tempfile.TemporaryDirectory() as d:
        z = bs.get(pick, d)
        if pick + ".sha256" in names:
            want = bs.get(pick + ".sha256", d).read_text().split()[0]
            if _sha256(z) != want:
                raise ValueError(f"{pick}: checksum differs from its .sha256 sidecar")
        return restore(z, target, force)


def due(state, counts, now, last_success, event_min=None):
    """'daily' when 24h passed since the last check; 'event' after a grading or collection run, at most every
    `event_min` minutes (KPR_BACKUP_EVENT_MIN, default 30); else None. Unchanged content is never re-uploaded."""
    event_min = EVENT_MIN_MINUTES if event_min is None else event_min
    if state.get("last_check") is None or now - datetime.fromisoformat(state["last_check"]) >= timedelta(hours=DAILY_HOURS):
        return "daily"
    last = max(filter(None, (last_success, state.get("last_check"))), default=None)
    if last and now - datetime.fromisoformat(last) < timedelta(minutes=event_min):
        return None
    base = state.get("history_counts") or {}
    grew = lambda k, n=1: (counts.get(k) or 0) - (base.get(k) or 0) >= n  # noqa: E731
    if grew("preds_graded") or grew("picks_graded") or grew("match_results") or grew("map_odds_log") \
            or grew("preds", 50) or grew("picks") or grew("slips"):
        return "event"
    return None


def health(state, counts, now, db_present):
    return {"heartbeat_utc": now.isoformat(timespec="seconds"), "db_present": db_present,
            "last_result": state.get("last_result"), "last_success": state.get("last_success"),
            "last_name": state.get("last_name"), "last_error": state.get("last_error"),
            "consecutive_failures": state.get("consecutive_failures", 0), "history_counts": counts,
            "daemon_restarts": state.get("daemon_restarts", 0), "event_min": EVENT_MIN_MINUTES}


def _publish_health(bs, h):
    d = Path(tempfile.mkdtemp(prefix="kprlab-health-"))
    try:
        (d / HEALTH_NAME).write_text(json.dumps(h, indent=1, default=str))
        bs.put(d / HEALTH_NAME)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def poll_once(dest, db=None, now=None):
    """One daemon step: back up if due, publish a heartbeat next to the backups, and open/close the external alert."""
    db = Path(db or store.DB_PATH)
    now = now or datetime.now(timezone.utc)
    bs = make_store(dest) if isinstance(dest, str) else dest
    counts, err = {}, None
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
        counts = history_counts(con)
        con.close()
        s = status()
        reason = due(s, counts, now, s.get("last_success"))
        if reason:
            try:
                backup(bs, reason, db, now)
            except Exception as e:  # logged + recorded by backup()
                err = e
    s = status()
    try:
        if err is not None:
            bs.alert("KPR Lab tracking backup failing", f"{now.isoformat(timespec='seconds')}: {type(err).__name__}: {err}\n"
                     f"consecutive failures: {s.get('consecutive_failures')}")
        elif s.get("alert_open"):
            bs.resolve(f"{now.isoformat(timespec='seconds')}: backups healthy again ({s.get('last_name')})")
        _status(alert_open=err is not None)
        bs.check()
        _publish_health(bs, health(status(), counts, now, db.exists()))
    except Exception as e:
        log.error("BACKUP HEALTH PUBLISH FAILED: %s", e)
    return err


def _log_to_file():
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
    log.setLevel(logging.INFO)


def daemon(dest, poll_min=15, db=None):
    """Waits for tracking.db to exist, then backs up daily and after grading / collection runs. Never exits on failure."""
    _log_to_file()
    log.info("backup daemon started (pid %d, dest %s, event interval %d min)", os.getpid(), dest, EVENT_MIN_MINUTES)
    while True:
        try:
            poll_once(dest, db)
        except Exception as e:
            log.error("BACKUP DAEMON ERROR: %s", e)
        time.sleep(poll_min * 60)


def supervise(cmd=None, max_restarts=None, backoff=(30, 300)):
    """Run the daemon as a child process and restart it whenever it exits."""
    _log_to_file()
    cmd = cmd or [sys.executable, "-m", "cs2model.cli", "backup-daemon", *sys.argv[2:]]
    n, delay = 0, backoff[0]
    while max_restarts is None or n <= max_restarts:
        t0 = time.time()
        rc = subprocess.call(cmd)
        n += 1
        log.error("BACKUP DAEMON EXITED (code %s); restart #%d in %ds", rc, n, delay)
        _status(daemon_restarts=status().get("daemon_restarts", 0) + 1, last_daemon_exit=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if max_restarts is not None and n > max_restarts:
            break
        time.sleep(delay)
        delay = backoff[0] if time.time() - t0 > 3600 else min(backoff[1], delay * 2)
    return n
