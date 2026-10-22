#!/usr/bin/env bash
# Starts `cli backup-supervisor` unless it is already running. Idempotent; run it at session/machine start
# (needs gh auth in the environment); the supervisor itself restarts the daemon whenever it exits.
# Usage: ops/ensure_backup_supervisor.sh [app checkout with data/ (default: this repo)]
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
export CS2_HOME="${1:-${CS2_HOME:-$CODE}}"
export KPR_BACKUP_DEST="${KPR_BACKUP_DEST:-gh:criscanales/kprlab-backups}"
if pgrep -f "cs2model.cli backup-supervisor" >/dev/null; then exit 0; fi
mkdir -p "$CS2_HOME/data"
cd "$CODE"
nohup python -m cs2model.cli backup-supervisor --dest "$KPR_BACKUP_DEST" >> "$CS2_HOME/data/backup-supervisor.out" 2>&1 &
echo "started backup-supervisor pid $! (CS2_HOME=$CS2_HOME)"
