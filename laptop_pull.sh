#!/bin/bash
# laptop_pull.sh — harvest clinical data from the Jetson to the laptop.
#
# Runs on: laptop.
# Use:     after a session (or batch of sessions).  Copies data to the
#          laptop for analysis / backup.  NON-DESTRUCTIVE on the Jetson —
#          nothing there is modified or deleted.
#
# Layout: single-mirror. Every run rsyncs into pulled_data/latest/,
# which always reflects the Jetson's current state. Per-session
# filenames (session_<subject>_<ts>.json, Report_<subject>_<ts>.csv,
# etc.) guarantee dossiers and CSVs accumulate inside the mirror
# without collision. therapist.db and per-subject Q-tables are
# overwritten in place with the Jetson's latest copy. The Jetson is the
# source of truth; keep a Jetson-side backup if you need DB rollback.
#
# Pulls:
#   data/therapist.db        — the SQLite truth store
#   data/sessions/           — session dossier JSON files
#   data/logs/               — per-session CSV + NDJSON transcripts
#   data/results/            — Report_*.csv + Notes_*.csv
#   data/q_tables/           — per-subject Q-table CSV snapshots
#
# Env:
#   JETSON_HOST         required
#   JETSON_PROJECT_DIR  default ~/project
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

: "${JETSON_HOST:?JETSON_HOST is not set — add it to .env}"
REMOTE_DIR="${JETSON_PROJECT_DIR:-~/project}"
SSH_OPTS="${JETSON_SSH_OPTS:--o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new}"

ARCHIVE_ROOT="${LAPTOP_ARCHIVE_DIR:-$PROJECT_ROOT/pulled_data}"
# Single mirror: one folder that always reflects the Jetson's current
# state. The Jetson remains the source of truth — per-session filenames
# (session_<subject>_<ts>.json, Report_<subject>_<ts>.csv, etc.) guarantee
# dossiers and CSVs accumulate without collision, and therapist.db /
# q_tables are overwritten in place with the Jetson's latest copy.
DEST="$ARCHIVE_ROOT/latest"
# Migration: prior versions created `latest` as a symlink into a
# timestamped snapshot. Replace it with a real directory so the mirror
# can accumulate on re-pulls.
if [[ -L "$DEST" ]]; then
    prior_target="$(readlink -f "$DEST")"
    echo "[laptop_pull] Migrating symlink 'latest' -> real mirror directory"
    rm -f "$DEST"
    if [[ -d "$prior_target" ]]; then
        # Seed the new mirror with the prior snapshot so we don't
        # re-download everything on the first pull after the migration.
        cp -a "$prior_target/." "$DEST/"
        echo "[laptop_pull]   seeded from prior snapshot: $prior_target"
    fi
fi
mkdir -p "$DEST"

echo "[laptop_pull] Source: $JETSON_HOST:$REMOTE_DIR"
echo "[laptop_pull] Dest:   $DEST  (single-mirror mode)"

pull() {
    local remote_rel="$1"
    local friendly="$2"
    if ssh $SSH_OPTS "$JETSON_HOST" "test -e $REMOTE_DIR/$remote_rel"; then
        echo "  [pull] $friendly"
        mkdir -p "$DEST/$(dirname "$remote_rel")"
        rsync -az -e "ssh $SSH_OPTS" \
            "$JETSON_HOST:$REMOTE_DIR/$remote_rel" "$DEST/$remote_rel"
    else
        echo "  [skip] $friendly (not present on Jetson)"
    fi
}

pull "data/therapist.db"   "SQLite DB"
pull "data/sessions/"      "Session dossiers"
pull "data/logs/"          "Per-session transcripts"
pull "data/results/"       "Report + Notes CSVs"
pull "data/q_tables/"      "Q-table snapshots"

if [[ -f "$DEST/data/therapist.db" ]]; then
    db_size=$(du -h "$DEST/data/therapist.db" | awk '{print $1}')
    session_rows=$(sqlite3 "$DEST/data/therapist.db" "SELECT COUNT(*) FROM sessions" 2>/dev/null || echo "?")
    turn_rows=$(sqlite3 "$DEST/data/therapist.db" "SELECT COUNT(*) FROM turns" 2>/dev/null || echo "?")
    echo ""
    echo "[laptop_pull] Summary:"
    echo "  therapist.db   $db_size   sessions=$session_rows  turns=$turn_rows"
fi

dossier_count=$(find "$DEST/data/sessions" -name '*.json' 2>/dev/null | wc -l || echo 0)
report_count=$(find "$DEST/data/results" -name 'Report_*.csv' 2>/dev/null | wc -l || echo 0)
echo "  dossiers       $dossier_count"
echo "  reports        $report_count"

echo ""
echo "[laptop_pull] Done.  Mirror: $DEST"
