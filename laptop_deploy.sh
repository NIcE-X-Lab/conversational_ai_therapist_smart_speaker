#!/bin/bash
# laptop_deploy.sh — the one-command dev loop.
#
# Runs on: laptop.
# Use:     when you want "push my code and restart CaiTI on the Jetson".
#          Composes laptop_sync → remote jetson_kill → remote jetson_run,
#          then tails the runtime log back to this terminal.
#
# Steps (each delegated to an existing script, no duplicated logic):
#   1. ./laptop_sync.sh                  (push code)
#   2. ssh jetson ./jetson_kill.sh       (clear stale processes)
#   3. ssh jetson ./jetson_run.sh        (launch main.py, streams back)
#
# Env:
#   JETSON_HOST         required
#   JETSON_PROJECT_DIR  default ~/project
#   SKIP_SYNC=1         skip step 1 (e.g. no code changed)
#   SKIP_KILL=1         skip step 2 (dangerous; only if you know the Jetson is idle)
#   LAUNCHER=start_therapist.sh   use clinician-view launcher (default: jetson_run.sh).
#                                  Accepts the bare name of any *.sh in the project root.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

: "${JETSON_HOST:?JETSON_HOST is not set — add it to .env}"
REMOTE_DIR="${JETSON_PROJECT_DIR:-~/project}"
SSH_OPTS="${JETSON_SSH_OPTS:--o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new}"

echo "========================================================================"
echo " CaiTI laptop_deploy → $JETSON_HOST:$REMOTE_DIR"
echo "========================================================================"

if [[ "${SKIP_SYNC:-0}" == "1" ]]; then
    echo "[step 1/3] SKIPPED (SKIP_SYNC=1)"
else
    echo "[step 1/3] laptop_sync.sh"
    ./laptop_sync.sh
fi

if [[ "${SKIP_KILL:-0}" == "1" ]]; then
    echo "[step 2/3] SKIPPED (SKIP_KILL=1)"
else
    echo "[step 2/3] ssh jetson_kill.sh"
    # jetson_kill returns 1 on D-state survivors — surface but don't abort;
    # the user may want to see the warning then decide to reboot.
    ssh $SSH_OPTS "$JETSON_HOST" "bash $REMOTE_DIR/jetson_kill.sh" || \
        echo "[laptop_deploy] WARN: jetson_kill reported survivors (see above)"
fi

LAUNCHER="${LAUNCHER:-jetson_run.sh}"
# Basic validation: must be a script in the project root, no path separators.
if [[ "$LAUNCHER" == *"/"* ]]; then
    echo "[laptop_deploy] ERROR: LAUNCHER must be a bare filename, got: $LAUNCHER" >&2
    exit 2
fi

echo "[step 3/3] ssh $LAUNCHER (streaming)"
echo "           Ctrl+C here to detach; the remote process keeps running."
echo ""
ssh -t $SSH_OPTS "$JETSON_HOST" "bash $REMOTE_DIR/$LAUNCHER"
