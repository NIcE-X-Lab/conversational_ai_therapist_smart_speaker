#!/bin/bash
# Set absolute paths and ensure environment is loaded
export PATH=$PATH:/usr/local/bin:/usr/bin
set -e

# Always run from the project root regardless of caller
cd "$(dirname "$(dirname "$(realpath "$0")")")"

# Load .env early so JETSON_PASSWORD is available for sudo -S
if [ -f ".env" ]; then
    export $(grep -v '^#' ".env" | xargs)
fi

# Helper: run sudo non-interactively (works over SSH without a tty)
_sudo() {
    echo "${JETSON_PASSWORD}" | sudo -S "$@" 2>/dev/null
}

# Ensure Pins 11, 13, 15, 16, 18 are configured for GPIO in the pinmux (Final Mapping Fix)
if command -v busybox > /dev/null; then
    echo "Unlocking Hardware Pinmux (Pins 11, 13, 15, 16, 18)..."
    _sudo busybox devmem 0x2430098 w 0x55 || true  # Pin 11
    _sudo busybox devmem 0x243D030 w 0x55 || true  # Pin 13
    _sudo busybox devmem 0x2440020 w 0x55 || true  # Pin 15
    _sudo busybox devmem 0x243D020 w 0x55 || true  # Pin 16
    _sudo busybox devmem 0x243D010 w 0x5   || true  # Pin 18 (LED)
fi

# Reset run logs to avoid stale errors from previous sessions
: > ollama.log
: > backend_session.log

# ── Aggressive process sanitization ──────────────────────────────────────
echo "Stopping existing services (aggressive sanitization)..."

# Kill all Python processes associated with the project
for pat in 'LLM_therapist' 'conversational_ai_therapist' 'main\.py' 'speech_service' 'handler_rl'; do
    pids=$(pgrep -f "$pat" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  [SANITIZE] Killing processes matching '$pat': $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
done

# Kill stale llama-runner / ggml processes (but NOT ollama itself — systemd manages it)
for proc in llama-runner ggml; do
    pids=$(pgrep -f "$proc" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  [SANITIZE] Killing $proc: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
done

# Release bound ports
for port in 8000 8001 8080 11434; do
    fuser -k "$port/tcp" 2>/dev/null || true
done

# Drop filesystem caches to reclaim pagecache
sync
echo 3 | _sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

# ── Clean Sweep: shared memory & orphaned IPC segments ─────────────────
echo "  [CLEAN SWEEP] Purging POSIX shared memory..."
rm -f /dev/shm/sem.* /dev/shm/psm_* 2>/dev/null || true

echo "  [CLEAN SWEEP] Removing orphaned SysV IPC segments..."
for seg_id in $(ipcs -m 2>/dev/null | awk '/^0x/ {print $2}'); do
    ipcrm -m "$seg_id" 2>/dev/null || true
done

# ── GPU device permissions (Jetson /dev/nvhost-*) ──────────────────────
if ls /dev/nvhost-* >/dev/null 2>&1; then
    echo "  [GPU] Setting permissions on /dev/nvhost-* devices..."
    _sudo chmod 666 /dev/nvhost-* 2>/dev/null || true
fi
if [ -e /dev/nvmap ]; then
    _sudo chmod 666 /dev/nvmap 2>/dev/null || true
fi

# Nuclear kill — clear ALL stale python3 processes
echo "  [SANITIZE] pkill -9 python3 (clearing all Python)"
_sudo pkill -9 python3 2>/dev/null || true
_sudo pkill -9 python  2>/dev/null || true

sleep 2

# Detect D-state (Uninterruptible Sleep) processes — only a hard reboot can clear these
d_state=$(ps aux 2>/dev/null | awk '$8 ~ /^D/ {print $2, $11}' || true)
if [ -n "$d_state" ]; then
    echo "  [CRITICAL] D-state (Uninterruptible Sleep) processes detected:"
    echo "$d_state" | while read -r line; do echo "    PID $line"; done
    echo "  [CRITICAL] These processes CANNOT be killed. A HARD REBOOT is required: sudo reboot"
fi

remaining=$(pgrep -c python 2>/dev/null || echo 0)
if [ "$remaining" -gt 0 ]; then
    echo "  [WARN] $remaining python processes survived sanitization."
else
    echo "  [SANITIZE] Complete. All python processes cleared."
fi

# Load runtime env (already exported .env above; activate venv here)
source .venv/bin/activate
echo "Sourcing .env variables..."

# ── Auto-install audit dependencies ───────────────────────────────────────
for _dep in psutil setproctitle; do
    if ! python3 -c "import $_dep" 2>/dev/null; then
        echo "  [AUTO-INSTALL] Installing missing dependency: $_dep"
        pip install --quiet "$_dep" || echo "  [WARN] Failed to install $_dep"
    fi
done

# Start Ollama LLM Service (if not already running)
# Prefer systemd service; fall back to manual start with full path
OLLAMA_BIN=$(command -v ollama 2>/dev/null || echo "/usr/local/bin/ollama")
if ! pgrep -x ollama > /dev/null; then
    # Try systemd first
    if systemctl is-enabled ollama >/dev/null 2>&1; then
        echo "Starting Ollama via systemd..."
        _sudo systemctl start ollama || true
    else
        echo "Starting Ollama manually..."
        nohup "$OLLAMA_BIN" serve > ollama.log 2>&1 &
        echo "Ollama PID: $!"
    fi
    # Wait up to 15s for Ollama to become ready
    for i in $(seq 1 15); do
        if curl -s --max-time 1 http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama is ready."
            break
        fi
        echo "Waiting for Ollama... ($i/15)"
        sleep 1
    done
else
    echo "✅ Ollama already running."
fi

# Ensure configured model exists in Ollama registry
if [ -z "$LLM_MODEL" ]; then
    echo "❌ LLM_MODEL is not set. Please set it in .env"
    exit 1
fi

# Strip -cpu suffix — always use the base GPU-capable model to avoid 7-min deadlock
if [[ "$LLM_MODEL" == *"-cpu" ]]; then
    BASE_MODEL="${LLM_MODEL%-cpu}"
    echo "⚠️  CPU model suffix detected ($LLM_MODEL). Stripping to GPU model: $BASE_MODEL"
    export LLM_MODEL="$BASE_MODEL"
fi

OLLAMA_PULL_TIMEOUT="${OLLAMA_PULL_TIMEOUT:-300}"

# Robust model existence check — exact match on first column.
model_exists() {
    "$OLLAMA_BIN" list 2>/dev/null | awk '{print $1}' | grep -q "^${1}$"
}

if model_exists "$LLM_MODEL"; then
    echo "✅ Ollama model '$LLM_MODEL' is available (GPU execution enforced)."
else
    echo "⚠️  Model '$LLM_MODEL' not found. Attempting pull (timeout ${OLLAMA_PULL_TIMEOUT}s)..."
    if timeout "$OLLAMA_PULL_TIMEOUT" "$OLLAMA_BIN" pull "$LLM_MODEL" 2>&1; then
        echo "✅ Pulled model: $LLM_MODEL"
    else
        echo "⚠️  Pull failed/timed out. Checking for local base variant to alias..."
        BASE_TAG="${LLM_MODEL%%:*}"
        EXISTING_BASE=$("$OLLAMA_BIN" list 2>/dev/null | awk '{print $1}' \
            | grep "^${BASE_TAG}:" | head -1 || true)

        if [[ -n "$EXISTING_BASE" && "$EXISTING_BASE" != "$LLM_MODEL" ]]; then
            echo "   Found: $EXISTING_BASE. Aliasing to $LLM_MODEL..."
            if "$OLLAMA_BIN" copy "$EXISTING_BASE" "$LLM_MODEL" 2>/dev/null; then
                echo "✅ Aliased $EXISTING_BASE -> $LLM_MODEL"
            else
                echo "❌ Alias failed. Run manually: $OLLAMA_BIN pull $LLM_MODEL"
                exit 1
            fi
        else
            echo "❌ No local variant of '${BASE_TAG}:*' found."
            echo "   Run: $OLLAMA_BIN pull $LLM_MODEL"
            exit 1
        fi
    fi
fi

# CUDA GPU enforcement — ensure Maxwell/Pascal libraries are on the path
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Ollama VRAM isolation: single model, single request, zero sharing
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

# Start Backend
echo "Starting Dialogue Engine (Server)..."
export DISABLE_INTERNAL_SPEECH=1
export CONSOLE_LOG_LEVEL=DEBUG
nohup python -u main.py > backend_session.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

echo "Headless System Started!"
echo "Logs: tail -f backend_session.log ollama.log"
