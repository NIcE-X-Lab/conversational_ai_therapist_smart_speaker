#!/bin/bash
# jetson_kill.sh — kill every CaiTI process on this machine.
#
# Runs on: Jetson (or any host CaiTI might be running on).
# Use:     as a hard stop before jetson_run.sh, or via SSH from the laptop
#          composer.  Safe to run when nothing is active — it no-ops.
#
# Actions:
#   1. Release the FastAPI ports (8000, 8001, 8080)
#   2. Kill Python processes matching any CaiTI entrypoint pattern,
#      skipping this shell's own PID chain
#   3. Flush FS caches (best-effort; requires sudo)
#   4. Report any D-state (uninterruptible sleep) survivors — the only
#      thing a reboot can fix
#
# Exit code:
#   0 — all clear (or cleaned up successfully)
#   1 — D-state processes survived; hardware reset recommended
set -uo pipefail

echo "[jetson_kill] Releasing ports 8000, 8001, 8080..."
for port in 8000 8001 8080; do
    fuser -k "$port/tcp" 2>/dev/null || true
done

PATTERNS=(
    'main\.py'
    'speech_service'
    'handler_rl'
    'LLM_therapist'
    'conversational_ai_therapist'
    'uvicorn.*main'
)

kill_matching() {
    local pat="$1"
    # Match against full cmdline, require /python/ so we never kill the
    # user's shell, and skip our own process chain.
    local pids
    pids=$(ps -eo pid=,args= | awk -v pat="$pat" '$0 ~ pat && $0 ~ /python/ {print $1}' || true)
    if [[ -z "$pids" ]]; then return 0; fi

    for pid in $pids; do
        if [[ "$pid" == "$$" || "$pid" == "$PPID" ]]; then continue; fi
        local cmd
        cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
        if [[ -n "$cmd" ]]; then
            echo "  [kill] pid=$pid  $cmd"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}

for pat in "${PATTERNS[@]}"; do
    kill_matching "$pat"
done

# Give the kernel ~1 s to reap; then drop caches best-effort.
sleep 1
sync
if command -v sudo >/dev/null 2>&1; then
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
fi

# D-state survivors can only be cleared by reboot.
d_state=$(ps aux 2>/dev/null | awk '$8 ~ /^D/ {print $2, $11}' || true)
if [[ -n "$d_state" ]]; then
    echo "[jetson_kill] WARNING: D-state processes detected (cannot be killed):"
    echo "$d_state" | while read -r line; do echo "    pid $line"; done
    echo "[jetson_kill] A reboot is required to fully clear these."
    exit 1
fi

remaining=$(pgrep -c -f 'python.*main\.py' 2>/dev/null || echo 0)
if [[ "$remaining" -gt 0 ]]; then
    echo "[jetson_kill] WARNING: $remaining CaiTI Python process(es) survived."
    exit 1
fi

echo "[jetson_kill] Done. All CaiTI processes cleared."
