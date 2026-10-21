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
`python -m cs2model.cli backup-daemon` (destination from `--dest` or `$KPR_BACKUP_DEST=gh:criscanales/kprlab-backups`)
polls every 15 min and runs:
- a **daily** backup every 24 h
- an **event** backup after a grading or collection run (new graded preds/picks, match results, slips,
  5+ new map-odds rows or 200+ new predictions), at most once every 6 h

A snapshot whose content hash equals the newest stored backup is not uploaded. Retention keeps every
version from the last 35 days, the newest backup of each older month, and always the newest backup.
The release warns in the log above 900 assets (GitHub limit 1000 per release).

Failures are logged as `BACKUP FAILED` in `data/backup.log` and recorded in `data/backup_status.json`
(`last_result`, `last_error`, `consecutive_failures`); `python -m cs2model.cli backup-status` prints it.

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
a match started.

| gap start (UTC) | gap end (UTC) | scope | reason |
|---|---|---|---|
| 2026-10-18 15:42 (first commit) | 2026-10-21 tracking restart | all tracking tables | Devin machine lost with the only copy of tracking.db; no backup existed |
