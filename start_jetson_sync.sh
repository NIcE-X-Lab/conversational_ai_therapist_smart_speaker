#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if command -v realpath >/dev/null 2>&1; then
  SCRIPT_PATH="$(realpath "$SCRIPT_PATH")"
elif command -v readlink >/dev/null 2>&1; then
  SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
fi

PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f "$PROJECT_ROOT/scripts/model_fetch.py" ]]; then
  echo "[ERROR] Project root appears invalid: $PROJECT_ROOT"
  echo "        Missing required file: scripts/model_fetch.py"
  exit 1
fi

LOCK_DIR="/tmp/caiti_start_jetson.lock"

acquire_single_instance_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$$" > "$LOCK_DIR/pid"
    trap 'rm -rf "$LOCK_DIR" 2>/dev/null || true' EXIT
    return 0
  fi

  local existing_pid=""
  if [[ -f "$LOCK_DIR/pid" ]]; then
    existing_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
  fi

  if [[ -n "$existing_pid" && ! -d "/proc/$existing_pid" ]]; then
    rm -rf "$LOCK_DIR" 2>/dev/null || true
    if mkdir "$LOCK_DIR" 2>/dev/null; then
      echo "$$" > "$LOCK_DIR/pid"
      trap 'rm -rf "$LOCK_DIR" 2>/dev/null || true' EXIT
      return 0
    fi
  fi

  echo "[ERROR] Another start_jetson_sync instance is already running${existing_pid:+ (pid $existing_pid)}."
  echo "[INFO] If CaiTI is already running, monitor logs with: tail -f data/logs/latest_runtime.log"
  exit 1
}

load_env_if_present() {
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "[OK] Loaded .env"
  else
    echo "[WARN] .env not found. Falling back to existing shell environment."
  fi
}

activate_runtime_venv() {
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "[OK] Activated venv at $PROJECT_ROOT/.venv"
  elif [[ -f "$HOME/project/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/project/.venv/bin/activate"
    echo "[WARN] Using fallback venv at $HOME/project/.venv"
  else
    echo "[ERROR] Virtual environment not found at:"
    echo "        $PROJECT_ROOT/.venv/bin/activate"
    echo "        $HOME/project/.venv/bin/activate"
    exit 1
  fi
}

_sudo() {
  if [[ -n "${JETSON_PASSWORD:-}" ]]; then
    echo "${JETSON_PASSWORD}" | sudo -S "$@" 2>/dev/null
  else
    sudo "$@"
  fi
}

sync_from_laptop_best_effort() {
  local laptop_host="$1"
  local laptop_project_dir="$2"
  local ssh_opts="$3"
  local require_sync="$4"

  if [[ -z "$laptop_host" || -z "$laptop_project_dir" ]]; then
    echo "[WARN] Laptop sync variables not set. Skipping sync and using local code."
    echo "       Set LAPTOP_HOST and LAPTOP_PROJECT_DIR in .env to enable pull-sync."
    return 1
  fi

  for required_cmd in ssh rsync; do
    if ! command -v "$required_cmd" >/dev/null 2>&1; then
      echo "[WARN] Missing command '$required_cmd'. Skipping sync and using local code."
      if [[ "$require_sync" == "1" ]]; then
        echo "[ERROR] REQUIRE_LAPTOP_SYNC=1 and sync prerequisites are missing."
        exit 1
      fi
      return 1
    fi
  done

  echo "[INFO] Laptop source host: $laptop_host"
  echo "[INFO] Laptop source path: $laptop_project_dir"
  echo "[INFO] Jetson destination path: $PROJECT_ROOT"

  local source_ready=0
  for attempt in $(seq 1 12); do
    if ssh $ssh_opts "$laptop_host" "echo connected >/dev/null" 2>/dev/null; then
      source_ready=1
      break
    fi
    echo "[WARN] Laptop SSH not ready (attempt ${attempt}/12). Retrying in 5s..."
    sleep 5
  done

  if [[ "$source_ready" -ne 1 ]]; then
    echo "[WARN] Laptop unreachable via SSH. Skipping sync and using local code."
    if [[ "$require_sync" == "1" ]]; then
      echo "[ERROR] REQUIRE_LAPTOP_SYNC=1 and laptop is unreachable."
      exit 1
    fi
    return 1
  fi

  if ! ssh $ssh_opts "$laptop_host" "test -d '$laptop_project_dir'"; then
    echo "[WARN] Source directory not found on laptop: $laptop_project_dir"
    echo "[WARN] Skipping sync and using local code."
    if [[ "$require_sync" == "1" ]]; then
      echo "[ERROR] REQUIRE_LAPTOP_SYNC=1 and laptop path is invalid."
      exit 1
    fi
    return 1
  fi

  if ! rsync -az --delete -e "ssh $ssh_opts" \
    --filter='P /models/litert/***' \
    --filter='P /models/piper/***' \
    --filter='P /data/logs/***' \
    --filter='P /backend_session.log' \
    --include='/src/***' \
    --include='/assets/***' \
    --include='/data/libs/***' \
    --include='/models/litert/***' \
    --include='/models/piper/***' \
    --include='/scripts/***' \
    --include='/.env' \
    --include='/main.py' \
    --include='/LLM_therapist_Application.py' \
    --include='/config.yaml' \
    --include='/requirements.txt' \
    --include='/start_headless.sh' \
    --include='/start_caiti.sh' \
    --include='/start_jetson' \
    --include='/start_jetson_sync.sh' \
    --exclude='*' \
    "$laptop_host:${laptop_project_dir%/}/" "$PROJECT_ROOT/"; then
    echo "[WARN] rsync failed. Skipping sync and using local code."
    if [[ "$require_sync" == "1" ]]; then
      echo "[ERROR] REQUIRE_LAPTOP_SYNC=1 and rsync failed."
      exit 1
    fi
    return 1
  fi

  echo "[OK] Code and model synchronized from laptop"
  return 0
}

echo "[Stage 1/4] Jetson environment preparation"
acquire_single_instance_lock
load_env_if_present
activate_runtime_venv

LAPTOP_HOST_VALUE="${LAPTOP_HOST:-${SYNC_SOURCE_HOST:-}}"
LAPTOP_PROJECT_DIR_VALUE="${LAPTOP_PROJECT_DIR:-${SYNC_SOURCE_DIR:-}}"
SSH_OPTS="${LAPTOP_SSH_OPTS:--o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new}"
REQUIRE_LAPTOP_SYNC_VALUE="${REQUIRE_LAPTOP_SYNC:-0}"

echo "[Stage 2/4] Laptop pull-sync (best effort)"
if sync_from_laptop_best_effort "$LAPTOP_HOST_VALUE" "$LAPTOP_PROJECT_DIR_VALUE" "$SSH_OPTS" "$REQUIRE_LAPTOP_SYNC_VALUE"; then
  echo "[OK] Running latest synchronized version"
else
  echo "[OK] Running current local version on Jetson"
fi

# If .env changed during sync, reload it before runtime setup.
load_env_if_present

echo "[Stage 3/4] Aggressive process sanitization"

kill_python_processes_matching() {
  local pat="$1"
  local pids
  pids=$(ps -eo pid=,args= | awk -v pat="$pat" '$0 ~ pat && $0 ~ /python/ {print $1}' || true)

  if [[ -z "$pids" ]]; then
    return 0
  fi

  for pid in $pids; do
    # Never kill the current shell chain executing this launcher.
    if [[ "$pid" == "$$" || "$pid" == "$PPID" ]]; then
      continue
    fi

    local cmdline
    cmdline=$(ps -p "$pid" -o args= 2>/dev/null || true)
    if [[ -n "$cmdline" ]]; then
      echo "  [SANITIZE] Killing python process matching '$pat': $pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

for pat in 'LLM_therapist' 'conversational_ai_therapist' 'main\.py' 'speech_service' 'handler_rl'; do
  kill_python_processes_matching "$pat"
done

for port in 8000 8001 8080; do
  fuser -k "$port/tcp" 2>/dev/null || true
done

sync
echo 3 | _sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

if [[ "${AGGRESSIVE_PYTHON_NUKE:-0}" == "1" ]]; then
  echo "  [SANITIZE] Phase 5: AGGRESSIVE_PYTHON_NUKE=1, running global pkill -9 python3/python"
  _sudo pkill -9 python3 2>/dev/null || true
  _sudo pkill -9 python 2>/dev/null || true
else
  echo "  [SANITIZE] Skipping global python kill (set AGGRESSIVE_PYTHON_NUKE=1 to enable)."
fi

sleep 2

d_state=$(ps aux 2>/dev/null | awk '$8 ~ /^D/ {print $2, $11}' || true)
if [[ -n "$d_state" ]]; then
  echo "  [CRITICAL] D-state (Uninterruptible Sleep) processes detected:"
  echo "$d_state" | while read -r line; do echo "    PID $line"; done
  echo "  [CRITICAL] These processes cannot be killed. A hard reboot is required: sudo reboot"
fi

remaining=$(pgrep -c python 2>/dev/null || echo 0)
if [[ "$remaining" -gt 0 ]]; then
  echo "  [WARN] $remaining python processes survived sanitization."
else
  echo "  [SANITIZE] Complete. All python processes cleared."
fi

echo "[Stage 4/4] Runtime setup and live launch"
activate_runtime_venv

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

BLOCKLIST_PKGS="openai-whisper whisper mlc-ai-nightly torch torchaudio onnxruntime-gpu"
for pkg in $BLOCKLIST_PKGS; do
  if pip show "$pkg" >/dev/null 2>&1; then
    echo "[CLEANUP] Removing memory-risk package: $pkg"
    pip uninstall -y "$pkg" 2>/dev/null || true
  fi
done

for _dep in psutil setproctitle; do
  if ! python3 -c "import $_dep" 2>/dev/null; then
    echo "[AUTO-INSTALL] Installing missing dependency: $_dep"
    pip install --quiet "$_dep" || echo "[WARN] Failed to install $_dep"
  fi
done

if ! python3 -c "import litert_lm" 2>/dev/null; then
  echo "[AUTO-INSTALL] Installing litert-lm-api..."
  pip install litert-lm-api || echo "[WARN] Failed to install litert-lm-api"
fi

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "[AUTO-INSTALL] Installing huggingface-hub..."
  pip install --quiet huggingface-hub || echo "[WARN] Failed to install huggingface-hub"
fi

export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ALSA_LOG_LEVEL=none
export PYTHONUNBUFFERED=1
export CONSOLE_LOG_LEVEL="${CONSOLE_LOG_LEVEL:-INFO}"

LITERT_MODEL_DIR="models/litert"
LITERT_MARKER="$LITERT_MODEL_DIR/.download_complete"
LITERT_MODEL_CANDIDATE=""

if [[ -n "${LITERT_MODEL_PATH:-}" && -s "${LITERT_MODEL_PATH}" ]]; then
  LITERT_MODEL_CANDIDATE="${LITERT_MODEL_PATH}"
else
  LITERT_MODEL_CANDIDATE=$(find "$LITERT_MODEL_DIR" -maxdepth 1 -type f -name '*.litertlm' -size +500M 2>/dev/null | head -n 1 || true)
fi

if [[ -f "$LITERT_MARKER" || -n "$LITERT_MODEL_CANDIDATE" ]]; then
  if [[ -n "$LITERT_MODEL_CANDIDATE" && ! -f "$LITERT_MARKER" ]]; then
    echo "ok" > "$LITERT_MARKER"
    echo "[INFO] Recreated LiteRT marker from existing model file."
  fi
  echo "[OK] LiteRT model present${LITERT_MODEL_CANDIDATE:+: $LITERT_MODEL_CANDIDATE}."
else
  echo "[INFO] LiteRT model not found. Running model_fetch.py..."
  if python3 scripts/model_fetch.py; then
    echo "[OK] LiteRT model downloaded."
  else
    LITERT_MODEL_CANDIDATE=$(find "$LITERT_MODEL_DIR" -maxdepth 1 -type f -name '*.litertlm' -size +500M 2>/dev/null | head -n 1 || true)
    if [[ -n "$LITERT_MODEL_CANDIDATE" ]]; then
      echo "ok" > "$LITERT_MARKER"
      echo "[WARN] model_fetch reported failure, but found existing model. Continuing with: $LITERT_MODEL_CANDIDATE"
    else
      echo "[ERROR] Model download failed and no local LiteRT model is available."
      echo "        Run manually when network is available: python3 scripts/model_fetch.py"
      exit 1
    fi
  fi
fi

echo "[OK] CaiTI online"
echo "[INFO] Streaming runtime logs to terminal and data/logs/latest_runtime.log"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL python3 -u main.py 2>&1 | tee -a data/logs/latest_runtime.log backend_session.log
else
  python3 -u main.py 2>&1 | tee -a data/logs/latest_runtime.log backend_session.log
fi
