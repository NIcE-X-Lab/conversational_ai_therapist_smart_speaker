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

# ── Local model download (if needed) ─────────────────────────────────────
LITERT_MODEL_DIR="models/litert"
if [[ ! -f "$LITERT_MODEL_DIR/.download_complete" ]]; then
  echo "[INFO] LiteRT model not found locally. Downloading..."
  python3 scripts/model_fetch.py
else
  echo "[OK] LiteRT model present locally."
fi

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

  # Phase 2: Release bound ports (API server, Flask)
  for port in 8000 8001 8080; do
    fuser -k "$port/tcp" 2>/dev/null || true
  done

  # Phase 4: Drop filesystem caches to reclaim pagecache RAM on Jetson
  sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

  # Phase 5: User-scoped nuclear kill — only kill processes owned by the
  # deploy user, not system daemons.  Also kill any gemma model processes.
  DEPLOY_USER="${REMOTE_USER:-arth}"
  echo "  [SANITIZE] Phase 5: pkill -9 -u $DEPLOY_USER python + gemma"
  sudo pkill -9 -u "$DEPLOY_USER" python3 2>/dev/null || true
  sudo pkill -9 -u "$DEPLOY_USER" python  2>/dev/null || true
  sudo pkill -9 -f gemma 2>/dev/null || true

  sleep 3

  # Phase 6: Detect D-state python processes owned by deploy user
  d_state=$(ps -u "$DEPLOY_USER" -o pid=,stat=,comm= 2>/dev/null | awk '$2 ~ /^D/ && /python/ {print $1, $3}' || true)
  if [ -n "$d_state" ]; then
    echo "  [CRITICAL] D-state python processes owned by $DEPLOY_USER detected (unkillable):"
    echo "$d_state" | while read -r line; do echo "    PID $line"; done
    echo "  [CRITICAL] A HARD REBOOT is required: sudo reboot"
    exit 1
  fi

  # Phase 7: Count surviving user-owned python processes (system daemons excluded)
  remaining=$(pgrep -u "$DEPLOY_USER" -c python 2>/dev/null || echo 0)
  if [ "$remaining" -gt 1 ]; then
    echo "  [WARN] $remaining python processes still owned by $DEPLOY_USER:"
    pgrep -u "$DEPLOY_USER" -a python 2>/dev/null || true
    echo "  [WARN] Proceeding — these may be transient."
  else
    echo "  [SANITIZE] Complete. Clean state for $DEPLOY_USER."
  fi
SANITIZE_EOF
echo "[OK] Aggressive sanitization complete"

ssh $SSH_OPTS "$JETSON_HOST_VALUE" "mkdir -p '$REMOTE_PROJECT_DIR' '$REMOTE_PROJECT_DIR/data'"

if ! rsync -az --delete -e "ssh $SSH_OPTS" \
  --include='/src/***' \
  --include='/assets/***' \
  --include='/data/libs/***' \
  --include='/models/litert/***' \
  --include='/models/piper/***' \
  --include='/scripts/***' \
  --include='/.env' \
  --include='/main.py' \
  --include='/config.yaml' \
  --include='/requirements.txt' \
  --include='/start_caiti.sh' \
  --include='/start_headless.sh' \
  --include='/start_jetson_sync.sh' \
  --include='/start_jetson' \
  --include='/stop_system.sh' \
  --exclude='*' \
  "$PROJECT_ROOT/" "$JETSON_HOST_VALUE:$REMOTE_PROJECT_DIR/"; then
  echo "[ERROR] rsync failed. Sync aborted."
  exit 1
fi
echo "[OK] Code and model synchronized"

# Ensure launchers are executable and discoverable from common Jetson paths.
ssh $SSH_OPTS "$JETSON_HOST_VALUE" "set -e; \
  chmod +x '$REMOTE_PROJECT_DIR/start_jetson' '$REMOTE_PROJECT_DIR/start_jetson_sync.sh' '$REMOTE_PROJECT_DIR/start_headless.sh' 2>/dev/null || true; \
  mkdir -p ~/.local/bin ~/project; \
  ln -sfn '$REMOTE_PROJECT_DIR/start_jetson' ~/.local/bin/start_jetson; \
  ln -sfn '$REMOTE_PROJECT_DIR/start_jetson_sync.sh' ~/.local/bin/start_jetson_sync; \
  ln -sfn '$REMOTE_PROJECT_DIR/start_jetson' ~/project/start_jetson; \
  ln -sfn '$REMOTE_PROJECT_DIR/start_jetson_sync.sh' ~/project/start_jetson_sync.sh; \
  grep -q '^export PATH=\$HOME/.local/bin:\$PATH$' ~/.bashrc || echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.bashrc"
echo "[OK] Jetson launcher symlinks refreshed (~/.local/bin and ~/project)"

echo "[Stage 3/4] Remote environment setup"

# ── Build the remote launch script locally, then upload + execute ──────────
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
mkdir -p models/litert

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
BLOCKLIST_PKGS="openai-whisper whisper mlc-ai-nightly torch torchaudio onnxruntime-gpu"
for pkg in $BLOCKLIST_PKGS; do
  if pip show "$pkg" >/dev/null 2>&1; then
    echo "[CLEANUP] Removing memory-risk package: $pkg"
    pip uninstall -y "$pkg" 2>/dev/null || true
  fi
done

# ── Auto-install dependencies ─────────────────────────────────────────────
for _dep in psutil setproctitle pygame; do
  if ! python3 -c "import $_dep" 2>/dev/null; then
    echo "[AUTO-INSTALL] Installing missing dependency: $_dep"
    pip install --quiet "$_dep" || echo "[WARN] Failed to install $_dep"
  fi
done

# LiteRT-LM inference engine
if ! python3 -c "import litert_lm" 2>/dev/null; then
  echo "[AUTO-INSTALL] Installing litert-lm-api..."
  pip install litert-lm-api || echo "[WARN] Failed to install litert-lm-api"
fi

# HuggingFace Hub for model downloads
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "[AUTO-INSTALL] Installing huggingface-hub..."
  pip install --quiet huggingface-hub || echo "[WARN] Failed to install huggingface-hub"
fi

# ── CUDA / GPU environment ─────────────────────────────────────────────────
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ALSA_LOG_LEVEL=none
export PYTHONUNBUFFERED=1
export CONSOLE_LOG_LEVEL=${CONSOLE_LOG_LEVEL:-INFO}

# ── LiteRT Model Check ───────────────────────────────────────────────────
LITERT_MODEL_DIR="models/litert"
if [[ -f "$LITERT_MODEL_DIR/.download_complete" ]]; then
  echo "[OK] LiteRT model present."
else
  echo "[INFO] LiteRT model not found. Running model_fetch.py..."
  if python3 scripts/model_fetch.py; then
    echo "[OK] LiteRT model downloaded."
  else
    echo "[ERROR] Model download failed. Run manually: python3 scripts/model_fetch.py"
    exit 1
  fi
fi

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
ssh -t $SSH_OPTS "$JETSON_HOST_VALUE" \
  "REMOTE_PROJECT_DIR='$REMOTE_PROJECT_DIR' bash '$REMOTE_TMP_SCRIPT'; rm -f '$REMOTE_TMP_SCRIPT'"
