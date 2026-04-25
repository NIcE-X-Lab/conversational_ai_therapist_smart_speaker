#!/bin/bash
# jetson_setup.sh — one-time environment bootstrap for a fresh Jetson.
#
# Runs on: Jetson.
# Use:     once per Jetson (or after a wipe).  Idempotent — re-running
#          is safe and updates anything that drifted.
#
# Actions:
#   1. Install apt prerequisites (portaudio, venv, gpio)
#   2. Create / reuse the project venv and install requirements.txt
#   3. Stage the Piper TTS voice pair into models/piper/
#   4. Fetch the LiteRT Gemma weights if missing
#   5. Remove known memory-risk packages (whisper, torch, onnxruntime-gpu)
#
# Prerequisites:
#   - Code already present at the project root (laptop_sync.sh landed it,
#     or the repo was cloned here directly).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f requirements.txt || ! -f main.py ]]; then
    echo "[jetson_setup] ERROR: not running from a CaiTI project root"
    echo "               expected requirements.txt + main.py at: $PROJECT_ROOT"
    exit 1
fi

_sudo() {
    if [[ -n "${JETSON_PASSWORD:-}" ]]; then
        echo "${JETSON_PASSWORD}" | sudo -S "$@" 2>/dev/null
    else
        sudo "$@"
    fi
}

echo "[Stage 1/5] apt prerequisites"
_sudo apt-get update -qq || true
_sudo apt-get install -y -qq \
    portaudio19-dev \
    python3-venv \
    python3-jetson-gpio \
    espeak-ng \
    libpulse0 \
    alsa-utils \
    pulseaudio-utils \
    || echo "[jetson_setup] WARN: some apt deps did not install"

echo "[Stage 2/5] Python venv + requirements"
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Memory-risk packages routinely land as transitive deps of model packs.
# CaiTI runs LiteRT directly; pytorch / whisper / onnxruntime-gpu all
# hold VRAM that collides with Gemma.
for pkg in openai-whisper whisper mlc-ai-nightly torch torchaudio onnxruntime-gpu; do
    if pip show "$pkg" >/dev/null 2>&1; then
        echo "  [cleanup] uninstalling memory-risk package: $pkg"
        pip uninstall -y "$pkg" >/dev/null 2>&1 || true
    fi
done

for dep in psutil setproctitle huggingface_hub; do
    python - <<PY 2>/dev/null || pip install --quiet "$dep"
import importlib, sys
sys.exit(0 if importlib.util.find_spec("$dep") else 1)
PY
done

if ! python -c "import litert_lm" 2>/dev/null; then
    echo "  [install] litert-lm-api"
    pip install --quiet litert-lm-api || echo "[jetson_setup] WARN: litert-lm-api install failed"
fi

echo "[Stage 3/5] Piper voice pair"
mkdir -p models/piper
VOICE_BASENAME="en_US-amy-medium"
VOICE_ONNX="models/piper/${VOICE_BASENAME}.onnx"
VOICE_JSON="models/piper/${VOICE_BASENAME}.onnx.json"

is_valid_voice_pair() {
    [[ -s "$1" && -s "$2" ]] || return 1
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$2" >/dev/null 2>&1 || return 1
    return 0
}

if ! is_valid_voice_pair "$VOICE_ONNX" "$VOICE_JSON"; then
    for src_onnx in \
        "./${VOICE_BASENAME}.onnx" \
        "assets/audio/${VOICE_BASENAME}.onnx" \
        "assets/${VOICE_BASENAME}.onnx"; do
        src_json="${src_onnx}.json"
        if is_valid_voice_pair "$src_onnx" "$src_json"; then
            cp "$src_onnx" "$VOICE_ONNX"
            cp "$src_json" "$VOICE_JSON"
            echo "  [stage] copied Piper voice from $src_onnx"
            break
        fi
    done
fi

if ! is_valid_voice_pair "$VOICE_ONNX" "$VOICE_JSON"; then
    echo "[jetson_setup] ERROR: Piper voice pair missing or invalid:"
    echo "               $VOICE_ONNX"
    echo "               $VOICE_JSON"
    echo "               Place the files at the project root and re-run."
    exit 1
fi

echo "[Stage 4/5] LiteRT Gemma weights"
mkdir -p models/litert
LITERT_MODEL_CANDIDATE=""
if [[ -n "${LITERT_MODEL_PATH:-}" && -s "${LITERT_MODEL_PATH}" ]]; then
    LITERT_MODEL_CANDIDATE="${LITERT_MODEL_PATH}"
else
    LITERT_MODEL_CANDIDATE=$(find models/litert -maxdepth 1 -type f -name '*.litertlm' -size +500M 2>/dev/null | head -n 1 || true)
fi

if [[ -n "$LITERT_MODEL_CANDIDATE" ]]; then
    echo "  [found] $LITERT_MODEL_CANDIDATE"
    echo "ok" > models/litert/.download_complete
else
    echo "  [fetch] running scripts/model_fetch.py..."
    if python3 scripts/model_fetch.py; then
        echo "  [ok] LiteRT model downloaded"
    else
        echo "[jetson_setup] ERROR: model_fetch.py failed and no local weights found"
        echo "               network down?  run manually when available."
        exit 1
    fi
fi

echo "[Stage 5/5] done"
echo ""
echo "Next:  ./jetson_run.sh"
