#!/bin/bash
# laptop_sync.sh — push laptop code to the Jetson.
#
# Runs on: laptop.
# Use:     whenever you change code locally and want it on the Jetson.
#          Does NOT kill processes or launch anything — pure file sync.
#
# Actions:
#   1. Load .env for JETSON_HOST + JETSON_PROJECT_DIR
#   2. rsync code, assets, scripts, entrypoint shells, config files
#   3. Preserve Jetson-side models/, data/, .venv/ (never overwritten)
#
# Env:
#   JETSON_HOST          required (e.g. user@1.2.3.4)
#   JETSON_PROJECT_DIR   remote project root   (default: ~/project)
#   DRY_RUN=1            print the rsync plan without copying
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

: "${JETSON_HOST:?JETSON_HOST is not set — add it to .env}"
REMOTE_DIR="${JETSON_PROJECT_DIR:-~/project}"
SSH_OPTS="${JETSON_SSH_OPTS:--o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new}"
DRY_FLAG=""
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_FLAG="--dry-run"
    echo "[laptop_sync] DRY_RUN — no files will actually transfer"
fi

echo "[laptop_sync] Target: $JETSON_HOST:$REMOTE_DIR"

# What we push (whitelist).  Whitelisting is safer than a big --exclude
# list — changes to the repo layout won't accidentally leak.
rsync $DRY_FLAG -az --delete -e "ssh $SSH_OPTS" \
    --filter='P /models/litert/***' \
    --filter='P /models/piper/***' \
    --filter='P /data/logs/***' \
    --filter='P /data/therapist.db' \
    --filter='P /data/sessions/***' \
    --filter='P /data/results/***' \
    --filter='P /data/q_tables/***' \
    --filter='P /backend_session.log' \
    --filter='P /.venv/***' \
    --include='/src/***' \
    --include='/assets/***' \
    --include='/data/libs/***' \
    --include='/scripts/***' \
    --include='/.env' \
    --include='/main.py' \
    --include='/LLM_therapist_Application.py' \
    --include='/config.yaml' \
    --include='/requirements.txt' \
    --include='/jetson_kill.sh' \
    --include='/jetson_setup.sh' \
    --include='/jetson_run.sh' \
    --include='/laptop_sync.sh' \
    --include='/laptop_pull.sh' \
    --include='/laptop_deploy.sh' \
    --include='/start_therapist.sh' \
    --include='/readme.md' \
    --exclude='*' \
    "$PROJECT_ROOT/" "$JETSON_HOST:$REMOTE_DIR/"

# Make the shell scripts executable on the Jetson side.
if [[ -z "$DRY_FLAG" ]]; then
    ssh $SSH_OPTS "$JETSON_HOST" "chmod +x $REMOTE_DIR/jetson_*.sh $REMOTE_DIR/laptop_*.sh $REMOTE_DIR/start_therapist.sh 2>/dev/null || true"
    echo "[laptop_sync] Done.  Files landed on Jetson; scripts chmod'd."
else
    echo "[laptop_sync] DRY_RUN complete."
fi
