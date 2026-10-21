# Tracking backups and data gaps

`data/tracking.db` holds the records that cannot be recreated: pre-match predictions and real lines,
graded picks and slips, tier-3 outcomes, map-odds provenance and match results. It is gitignored and
must never be committed. Backups go to release assets of the **private** repo
`criscanales/kprlab-backups` (release `tracking-backups`), never to this repo.

## What a backup contains
`kprlab-tracking-<UTC yyyymmddThhmmssZ>-<daily|event|manual>-<content hash 12>.zip` plus a
`.sha256` sidecar:
- `tracking.db`: consistent copy taken with the SQLite online backup API (safe while the dashboard writes)
- `provenance/map_odds_log.jsonl`, `match_results.jsonl`, `data_gaps.jsonl`: plain-text copies of the append-only tables
- `manifest.json`: SHA-256 of every file, row counts, history counts, content hash, `integrity_check` result

Before upload a backup is integrity-checked, re-verified from the zip and scanned for token/key-like
values (it is refused if any are found). After upload it is downloaded again and its checksum compared.
Settings, API keys, cookies and caches are never included.

## Schedule and retention
`python -m cs2model.cli backup-supervisor` (destination from `--dest` or `$KPR_BACKUP_DEST=gh:criscanales/kprlab-backups`)
runs `backup-daemon` as a child process and restarts it whenever it exits (30 s backoff doubling to 5 min,
reset after an hour of uptime; each restart is logged as `BACKUP DAEMON EXITED` and counted in the status).
The daemon starts before `tracking.db` exists and begins backing up as soon as the dashboard or CLI creates it.
It polls every 15 min and runs:
- a **daily** backup every 24 h
- an **event** backup after a grading or collection run (new graded preds/picks, match results, slips,
  any new map-odds snapshot or 50+ new predictions), at most once every `KPR_BACKUP_EVENT_MIN` minutes (default 30)

A snapshot whose content hash equals the newest stored backup is not uploaded, so idle periods cost nothing.
Retention keeps every version from the last 2 days, the newest backup of each UTC day for 35 days, the newest
of each older month, and always the newest backup: about 100 intraday + 35 daily + 1 per month, far below
GitHub's 1000-asset release limit (the log warns above 900).

## Failure visibility (off-machine)
- Each poll uploads `kprlab-health.json` (heartbeat time, last result/error, consecutive failures, history
  counts, daemon restarts) next to the backups.
- A failed backup opens one GitHub issue labelled `backup-alert` in the private repo (later failures comment on
  it); the next successful or unchanged backup closes it. GitHub notifies the repo owner by email.
- `.github/workflows/backup-health.yml` in the backups repo (source: `ops/kprlab-backups/backup-health.yml`) runs
  on GitHub every 3 h, independent of the Devin machine. It fails and opens/updates the same alert issue when the
  newest backup is older than 26 h, the heartbeat is older than 2 h (machine down, crashed or asleep) or the last
  attempt failed. Failed scheduled runs also trigger GitHub's own failure email.
- Locally, failures are logged as `BACKUP FAILED` in `data/backup.log` and recorded in `data/backup_status.json`;
  `python -m cs2model.cli backup-status` prints it.

## Restore (fresh machine)
```bash
git clone https://github.com/criscanales/cs2-prop-intelligence && cd cs2-prop-intelligence
pip install -r requirements.txt
python -m cs2model.cli restore --dest gh:criscanales/kprlab-backups          # newest backup
python -m cs2model.cli restore --dest gh:criscanales/kprlab-backups --name kprlab-tracking-...zip
```
Restore checks the sidecar checksum, every member checksum, `integrity_check`, row counts and history
counts, then reapplies the schema so the append-only triggers exist. It refuses to replace a non-empty
`tracking.db` unless `--force`, and then keeps the old file as `tracking.pre-restore-<ts>.db`.

## Data gaps
`data_gaps` is an append-only table of periods with no tracking data. Lost records are never
recreated, and no pre-match prediction, pick or odds row is ever back-filled with information from after
a match started. Boundaries are taken from evidence, not estimated: the start is the last recoverable
observation (or, when nothing survived, the first commit that could have written a record), and the end is the
first record actually collected afterwards: `record-gap --end first-observation` reads the earliest
`line_snapshots` / `preds` / `map_odds_log` timestamp from `tracking.db`.

| gap start (UTC) | gap end (UTC) | scope | reason |
|---|---|---|---|
| 2026-10-18 15:42:00 (commit 553c2b3 introduced tracking.db; no record from the lost machine survived) | first record collected by the restored system | all tracking tables | Devin machine lost with the only copy of tracking.db; no backup existed |
