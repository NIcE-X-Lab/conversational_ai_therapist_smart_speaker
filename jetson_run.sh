#!/bin/bash
# jetson_run.sh — launch CaiTI (main.py) on the Jetson.
#
# Runs on: Jetson.
# Use:     the production live-session launcher.  Assumes environment is
#          ready (run jetson_setup.sh once) and no stale processes are
#          holding ports (run jetson_kill.sh first if re-launching).
#
# Actions:
#   1. Acquire a single-instance lock so two launchers don't fight
#   2. Load .env, activate venv
#   3. Verify Piper voice + LiteRT model are present
#   4. Export CUDA + ALSA env for the Python process
#   5. Launch main.py with output tee'd to terminal + runtime log
#
# Lock:
#   /tmp/caiti_jetson_run.lock
#   Held by this script's PID; auto-released on exit.  A stale lock from
#   a crashed prior run is detected (PID not in /proc) and reclaimed.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

LOCK_DIR="/tmp/caiti_jetson_run.lock"

acquire_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "$$" > "$LOCK_DIR/pid"
        trap 'rm -rf "$LOCK_DIR" 2>/dev/null || true' EXIT
        return 0
    fi
    local prev=""
    [[ -f "$LOCK_DIR/pid" ]] && prev=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
    if [[ -n "$prev" && ! -d "/proc/$prev" ]]; then
        rm -rf "$LOCK_DIR" 2>/dev/null || true
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            echo "$$" > "$LOCK_DIR/pid"
            trap 'rm -rf "$LOCK_DIR" 2>/dev/null || true' EXIT
            return 0
        fi
    fi
    echo "[jetson_run] ERROR: another jetson_run is active${prev:+ (pid $prev)}"
    echo "             run ./jetson_kill.sh first, then try again"
    exit 1
}

load_env() {
    if [[ -f .env ]]; then
        set -a
        # shellcheck disable=SC1091
        source .env
        set +a
        echo "[jetson_run] .env loaded"
    else
        echo "[jetson_run] WARN: .env not found — using shell environment"
    fi
}

activate_venv() {
    if [[ -f .venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
        echo "[jetson_run] venv activated"
    else
        echo "[jetson_run] ERROR: .venv not found — run ./jetson_setup.sh first"
        exit 1
    fi
}

preflight_assets() {
    local voice_onnx="models/piper/en_US-amy-medium.onnx"
    local voice_json="models/piper/en_US-amy-medium.onnx.json"
    if [[ ! -s "$voice_onnx" || ! -s "$voice_json" ]]; then
        echo "[jetson_run] ERROR: Piper voice missing — run ./jetson_setup.sh"
        exit 1
    fi

    local model_found=""
    if [[ -n "${LITERT_MODEL_PATH:-}" && -s "${LITERT_MODEL_PATH}" ]]; then
        model_found="${LITERT_MODEL_PATH}"
    else
        model_found=$(find models/litert -maxdepth 1 -type f -name '*.litertlm' -size +500M 2>/dev/null | head -n 1 || true)
    fi
    if [[ -z "$model_found" ]]; then
        echo "[jetson_run] ERROR: LiteRT model missing — run ./jetson_setup.sh"
        exit 1
    fi
    echo "[jetson_run] assets OK — voice + $model_found"
}

echo "[Stage 1/4] Acquiring lock"
acquire_lock

echo "[Stage 2/4] Environment"
load_env
activate_venv
preflight_assets

echo "[Stage 3/4] Export runtime env"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ALSA_LOG_LEVEL=none
export PYTHONUNBUFFERED=1
export CONSOLE_LOG_LEVEL="${CONSOLE_LOG_LEVEL:-INFO}"

mkdir -p data/logs

echo "[Stage 4/4] Launching main.py"
echo "             logs: data/logs/latest_runtime.log + backend_session.log"
echo ""

if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL python3 -u main.py 2>&1 | tee -a data/logs/latest_runtime.log backend_session.log
else
    python3 -u main.py 2>&1 | tee -a data/logs/latest_runtime.log backend_session.log
fi
