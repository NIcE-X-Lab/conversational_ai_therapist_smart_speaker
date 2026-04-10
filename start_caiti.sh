#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "[Stage 1/4] Local environment preparation"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "[OK] Loaded local .env"
else
  echo "[WARN] Local .env not found. Falling back to existing shell environment."
fi

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "[OK] Local .venv activated"
else
  echo "[INFO] Local .venv not found. Continuing without local activation."
fi

JETSON_HOST_VALUE="${JETSON_HOST:-}"
if [[ -z "$JETSON_HOST_VALUE" && -n "${JETSON_USER:-}" && -n "${JETSON_IP:-}" ]]; then
  JETSON_HOST_VALUE="${JETSON_USER}@${JETSON_IP}"
fi

if [[ -z "$JETSON_HOST_VALUE" ]]; then
  echo "[ERROR] JETSON_HOST is not set. Define JETSON_HOST (or JETSON_USER and JETSON_IP) in .env."
  exit 1
fi

REMOTE_USER="${JETSON_USER:-}"
if [[ -z "$REMOTE_USER" && "$JETSON_HOST_VALUE" == *"@"* ]]; then
  REMOTE_USER="${JETSON_HOST_VALUE%@*}"
fi
if [[ -z "$REMOTE_USER" ]]; then
  REMOTE_USER="$USER"
fi

REMOTE_PROJECT_DIR="${JETSON_PROJECT_DIR:-/home/${REMOTE_USER}/project/LLM_therapist_prototype}"
SSH_OPTS="-o ConnectTimeout=10"

echo "[INFO] Target Jetson: $JETSON_HOST_VALUE"
echo "[INFO] Remote project path: $REMOTE_PROJECT_DIR"

echo "[Stage 2/4] Remote sanitization and code synchronization"
SSH_READY=0
for attempt in $(seq 1 24); do
  if ssh $SSH_OPTS "$JETSON_HOST_VALUE" "echo connected >/dev/null" 2>/dev/null; then
    SSH_READY=1
    break
  fi
  echo "[WARN] Jetson SSH not ready (attempt ${attempt}/24). Retrying in 10s..."
  sleep 10
done

if [[ "$SSH_READY" -ne 1 ]]; then
  echo "[ERROR] Jetson is unreachable via SSH after retries."
  echo "Network Check: verify Jetson power, LAN reachability, .env JETSON_HOST, and SSH access."
  exit 1
fi

echo "[INFO] Aggressive process sanitization on Jetson (clearing all CaiTI / LLM_therapist processes)"
ssh $SSH_OPTS "$JETSON_HOST_VALUE" bash -s << 'SANITIZE_EOF'
  set +e  # don't abort on individual kill failures

  # Phase 1: Kill all Python processes associated with ANY therapist project directory
  for pat in 'LLM_therapist' 'conversational_ai_therapist' 'main\.py' 'speech_service' 'handler_rl'; do
    pids=$(pgrep -f "$pat" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      echo "  [SANITIZE] Killing processes matching '$pat': $pids"
      echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
  done

  # Phase 2: Kill Ollama server and all llama-runner / ggml workers
  for proc in ollama llama-runner ggml; do
    pids=$(pgrep -f "$proc" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      echo "  [SANITIZE] Killing $proc processes: $pids"
      echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
  done

  # Phase 3: Release bound ports (API server, Flask)
  for port in 8000 8001 8080 11434; do
    fuser -k "$port/tcp" 2>/dev/null || true
  done

  # Phase 4: Drop filesystem caches to reclaim pagecache RAM on Jetson
  sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

  # Phase 5: Nuclear kill — clear ALL stale python3 processes
  echo "  [SANITIZE] Phase 5: pkill -9 python3 (clearing all Python)"
  sudo pkill -9 python3 2>/dev/null || true
  sudo pkill -9 python  2>/dev/null || true

  sleep 2

  # Phase 6: Detect D-state (Uninterruptible Sleep) processes — only a hard reboot can clear these
  d_state=$(ps aux 2>/dev/null | awk '$8 ~ /^D/ {print $2, $11}' || true)
  if [ -n "$d_state" ]; then
    echo "  [CRITICAL] D-state (Uninterruptible Sleep) processes detected:"
    echo "$d_state" | while read -r line; do echo "    PID $line"; done
    echo "  [CRITICAL] These processes CANNOT be killed. A HARD REBOOT of the Jetson is required."
    echo "  [CRITICAL] Run: sudo reboot"
  fi

  # Phase 7: Verify clean state — count surviving python processes
  remaining=$(pgrep -c python 2>/dev/null || echo 0)
  if [ "$remaining" -gt 0 ]; then
    echo "  [WARN] $remaining python processes survived sanitization."
    echo "  [WARN] They may be D-state or owned by root. Check: ps aux | grep python"
  else
    echo "  [SANITIZE] Complete. All python processes cleared."
  fi
SANITIZE_EOF
echo "[OK] Aggressive sanitization complete"

ssh $SSH_OPTS "$JETSON_HOST_VALUE" "mkdir -p '$REMOTE_PROJECT_DIR' '$REMOTE_PROJECT_DIR/data'"

if ! rsync -az --delete -e "ssh $SSH_OPTS" \
  --include='/src/***' \
  --include='/assets/***' \
  --include='/data/libs/***' \
  --include='/models/piper/***' \
  --include='/.env' \
  --include='/main.py' \
  --include='/config.yaml' \
  --include='/requirements.txt' \
  --exclude='*' \
  "$PROJECT_ROOT/" "$JETSON_HOST_VALUE:$REMOTE_PROJECT_DIR/"; then
  echo "[ERROR] rsync failed. Sync aborted."
  exit 1
fi
echo "[OK] Code synchronized"

echo "[Stage 3/4] Remote environment setup"

# ── Build the remote launch script locally, then upload + execute ──────────
# This avoids the ssh -tt + heredoc problem where the PTY echoes raw script
# source instead of executing it.  Also sidesteps nested heredoc conflicts
# (the PY heredoc inside the old REMOTE_SCRIPT heredoc).
REMOTE_LAUNCH_SCRIPT=$(mktemp "${TMPDIR:-/tmp}/caiti_remote_XXXXXX.sh")
trap 'rm -f "$REMOTE_LAUNCH_SCRIPT"' EXIT

cat > "$REMOTE_LAUNCH_SCRIPT" << 'REMOTE_SCRIPT_EOF'
#!/bin/bash
set -euo pipefail

cd "$REMOTE_PROJECT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -f "$HOME/project/.venv/bin/activate" ]]; then
  source "$HOME/project/.venv/bin/activate"
  echo "[WARN] Using fallback venv at $HOME/project/.venv"
else
  echo "[ERROR] Remote virtual environment missing at .venv/bin/activate and $HOME/project/.venv/bin/activate"
  exit 1
fi

mkdir -p data/logs
mkdir -p models/piper

VOICE_BASENAME="en_US-amy-medium"
VOICE_ONNX="models/piper/${VOICE_BASENAME}.onnx"
VOICE_JSON="models/piper/${VOICE_BASENAME}.onnx.json"

is_valid_voice_pair() {
  local onnx_path="$1"
  local json_path="$2"
  [[ -s "$onnx_path" ]] || return 1
  [[ -s "$json_path" ]] || return 1
  python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$json_path" >/dev/null 2>&1 || return 1
  return 0
}

repair_voice_pair() {
  local candidates=(
    "./${VOICE_BASENAME}.onnx"
    "$HOME/project/${VOICE_BASENAME}.onnx"
    "$HOME/project/eric-project/LLM_therapist_prototype/${VOICE_BASENAME}.onnx"
    "$HOME/project/models/piper/${VOICE_BASENAME}.onnx"
  )

  for candidate in "${candidates[@]}"; do
    local candidate_json="${candidate}.json"
    if is_valid_voice_pair "$candidate" "$candidate_json"; then
      cp "$candidate" "$VOICE_ONNX"
      cp "$candidate_json" "$VOICE_JSON"
      echo "[OK] Staged valid Piper voice from $candidate"
      return 0
    fi
  done

  return 1
}

# ── Piper voice file repair (hard force-sync) ─────────────────────────────
if [[ -f "$VOICE_JSON" && ! -s "$VOICE_JSON" ]]; then
  echo "[WARN] Piper voice config ($VOICE_JSON) is empty/corrupt. Removing for re-copy."
  rm -f "$VOICE_JSON"
fi
if [[ -f "$VOICE_ONNX" && ! -s "$VOICE_ONNX" ]]; then
  echo "[WARN] Piper voice model ($VOICE_ONNX) is empty/corrupt. Removing for re-copy."
  rm -f "$VOICE_ONNX"
fi

LOCAL_VOICE_ASSETS=(
  "assets/audio/${VOICE_BASENAME}.onnx"
  "assets/${VOICE_BASENAME}.onnx"
  "${VOICE_BASENAME}.onnx"
)
if [[ ! -s "$VOICE_JSON" ]]; then
  for local_src in "${LOCAL_VOICE_ASSETS[@]}"; do
    local_json="${local_src}.json"
    if [[ -s "$local_src" && -s "$local_json" ]]; then
      cp "$local_src" "$VOICE_ONNX"
      cp "$local_json" "$VOICE_JSON"
      echo "[OK] Force-synced Piper voice from $local_src"
      break
    fi
  done
fi

if ! is_valid_voice_pair "$VOICE_ONNX" "$VOICE_JSON"; then
  echo "[WARN] Piper voice/model config missing or invalid. Attempting repair..."
  rm -f "$VOICE_ONNX" "$VOICE_JSON"
  repair_voice_pair || true
fi

if ! is_valid_voice_pair "$VOICE_ONNX" "$VOICE_JSON"; then
  echo "[ERROR] Piper voice pair is still invalid. Expected non-empty files at:"
  echo "        $VOICE_ONNX"
  echo "        $VOICE_JSON"
  exit 1
fi

# ── Dependency drift cleanup (remove known memory-risk packages) ──────────
# These packages can slip in via transitive deps or manual experiments.
# Each one either pulls full PyTorch or conflicts with ctranslate2's CUDA.
BLOCKLIST_PKGS="openai-whisper whisper mlc-ai-nightly torch torchaudio onnxruntime-gpu"
for pkg in $BLOCKLIST_PKGS; do
  if pip show "$pkg" >/dev/null 2>&1; then
    echo "[CLEANUP] Removing memory-risk package: $pkg"
    pip uninstall -y "$pkg" 2>/dev/null || true
  fi
done

# ── Auto-install audit dependencies ───────────────────────────────────────
for _dep in psutil setproctitle; do
  if ! python3 -c "import $_dep" 2>/dev/null; then
    echo "[AUTO-INSTALL] Installing missing dependency: $_dep"
    pip install --quiet "$_dep" || echo "[WARN] Failed to install $_dep"
  fi
done

# ── CUDA / GPU environment ─────────────────────────────────────────────────
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OLLAMA_KEEP_ALIVE=0
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export ALSA_LOG_LEVEL=none
export PYTHONUNBUFFERED=1
export CONSOLE_LOG_LEVEL=${CONSOLE_LOG_LEVEL:-INFO}

# ── Model provisioning (GPU-first, non-hanging) ───────────────────────────
OLLAMA_PULL_TIMEOUT="${OLLAMA_PULL_TIMEOUT:-300}"

if command -v ollama >/dev/null 2>&1 && [[ -n "${LLM_MODEL:-}" ]]; then
  if [[ "$LLM_MODEL" == *"-cpu" ]]; then
    BASE_MODEL="${LLM_MODEL%-cpu}"
    echo "[WARN] CPU model suffix detected ($LLM_MODEL). Stripping to GPU model: $BASE_MODEL"
    export LLM_MODEL="$BASE_MODEL"
  fi

  model_exists() {
    ollama list 2>/dev/null | awk '{print $1}' | grep -q "^${1}$"
  }

  if model_exists "$LLM_MODEL"; then
    echo "[OK] Model '$LLM_MODEL' already available."
  else
    echo "[INFO] Model '$LLM_MODEL' not found. Attempting pull (timeout ${OLLAMA_PULL_TIMEOUT}s)..."
    if timeout "$OLLAMA_PULL_TIMEOUT" ollama pull "$LLM_MODEL" 2>&1; then
      echo "[OK] Pulled model: $LLM_MODEL"
    else
      echo "[WARN] Pull failed or timed out for '$LLM_MODEL'."
      BASE_TAG="${LLM_MODEL%%:*}"
      EXISTING_BASE=$(ollama list 2>/dev/null | awk '{print $1}' \
        | grep "^${BASE_TAG}:" | head -1 || true)
      if [[ -n "$EXISTING_BASE" && "$EXISTING_BASE" != "$LLM_MODEL" ]]; then
        echo "[INFO] Found local base variant: $EXISTING_BASE. Aliasing to $LLM_MODEL..."
        if ollama copy "$EXISTING_BASE" "$LLM_MODEL" 2>/dev/null; then
          echo "[OK] Aliased $EXISTING_BASE -> $LLM_MODEL"
        else
          echo "[ERROR] Alias via 'ollama copy' failed."
          echo "[ERROR] Unable to provision model. Run manually: ollama pull $LLM_MODEL"
          exit 1
        fi
      else
        echo "[ERROR] No local base variant found for '${BASE_TAG}:*'."
        echo "[ERROR] Unable to provision model. Check network and run: ollama pull $LLM_MODEL"
        exit 1
      fi
    fi
  fi

  MODEL_SIZE=$(ollama list 2>/dev/null | awk -v m="$LLM_MODEL" '$1==m {print $3}' || echo "unknown")
  echo "[INFO] Model: $LLM_MODEL, Size: ${MODEL_SIZE:-unknown}. GPU execution enforced (num_gpu>=1)."
fi

echo "[OK] VRAM cleared"
echo "[OK] CaiTI online"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL python3 -u main.py 2>&1 | tee -a data/logs/latest_runtime.log
else
  python3 -u main.py 2>&1 | tee -a data/logs/latest_runtime.log
fi
REMOTE_SCRIPT_EOF

# Upload the script to the Jetson and execute it (no heredoc through SSH PTY)
REMOTE_TMP_SCRIPT="/tmp/caiti_launch_$$.sh"
echo "[Stage 4/4] High-fidelity execution with live telemetry"
scp $SSH_OPTS "$REMOTE_LAUNCH_SCRIPT" "$JETSON_HOST_VALUE:$REMOTE_TMP_SCRIPT" >/dev/null
echo "[INFO] Launch script uploaded to $JETSON_HOST_VALUE:$REMOTE_TMP_SCRIPT"

# Execute: non-interactive SSH with the REMOTE_PROJECT_DIR env var set.
# Use -t (single) for log streaming to terminal; the script itself is a file
# so there's no heredoc/PTY conflict.
ssh -t $SSH_OPTS "$JETSON_HOST_VALUE" \
  "REMOTE_PROJECT_DIR='$REMOTE_PROJECT_DIR' bash '$REMOTE_TMP_SCRIPT'; rm -f '$REMOTE_TMP_SCRIPT'"
