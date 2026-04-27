#!/bin/bash
# start_therapist.sh — one-command CaiTI launcher with clinician-view log.
#
# Dual-mode, auto-detected by host:
#
#   ON LAPTOP (default mode when /etc/nv_tegra_release is absent):
#     1. rsync code to the Jetson            (./laptop_sync.sh)
#     2. kill stale processes on the Jetson  (ssh jetson_kill.sh)
#     3. launch main.py on the Jetson with CLINICIAN_LOG_MODE=1
#        (re-invokes this same script over ssh — the Jetson side
#         hits the Jetson branch below)
#
#   ON JETSON (/etc/nv_tegra_release present, or THERAPIST_MODE=jetson):
#     activates venv, sources .env, exports clinician-view env,
#     and tees the run to data/logs/therapist_<ts>.log.  The console
#     view matches the demo recording: only RL weights, module I/O,
#     and WARNING+ events.  Full DEBUG detail still reaches the file.
#
# Usage:
#   ./start_therapist.sh                          # clinician view on console
#   CONSOLE_LOG_LEVEL=DEBUG ./start_therapist.sh  # verbose debug on console
#   CLINICIAN_LOG_MODE=0 ./start_therapist.sh     # every logger on console (noisy)
#
# Laptop-side flags (mirror laptop_deploy.sh):
#   SKIP_SYNC=1        skip step 1 (no code changed)
#   SKIP_KILL=1        skip step 2 (Jetson known to be idle)
#   THERAPIST_MODE=jetson   force Jetson mode on a non-Jetson host
#   THERAPIST_MODE=laptop   force laptop mode on a Jetson
#
# Tune the console allow/block lists at runtime via env:
#   CLINICIAN_LOG_ALLOW="LLMClient,ResourceAudit"  # extend allowlist
#   CLINICIAN_LOG_BLOCK="Questioner"               # block extra loggers
# (Both extend the defaults in src/utils/log_util.py.)
#
# To kill: Ctrl-C in the foreground (laptop: detaches, Jetson process
# keeps running), or `pkill -f main.py` from elsewhere.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Load .env for both modes — JETSON_HOST lives there, as do LiteRT paths
# and pin assignments used downstream.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# ── Mode detection ────────────────────────────────────────────────────────
# Jetson platforms ship with /etc/nv_tegra_release as a marker file on
# every L4T install, including the Orin Nano this project targets.
# THERAPIST_MODE=laptop|jetson forces the branch for edge cases
# (ARM laptop, non-Tegra board, etc.).
MODE="${THERAPIST_MODE:-}"
if [[ -z "$MODE" ]]; then
    if [[ -f /etc/nv_tegra_release ]]; then
        MODE="jetson"
    else
        MODE="laptop"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════
# LAPTOP MODE — orchestrate sync + kill + remote clinician launch
# ══════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "laptop" ]]; then
    : "${JETSON_HOST:?JETSON_HOST is not set — add it to .env}"
    REMOTE_DIR="${JETSON_PROJECT_DIR:-~/project}"
    SSH_OPTS="${JETSON_SSH_OPTS:--o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new}"

    echo "========================================================================"
    echo " CaiTI start_therapist (laptop → $JETSON_HOST:$REMOTE_DIR)"
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
        # jetson_kill returns 1 on D-state survivors — surface but don't
        # abort; the user may want to see the warning then decide whether
        # to reboot before launching.
        ssh $SSH_OPTS "$JETSON_HOST" "bash $REMOTE_DIR/jetson_kill.sh" || \
            echo "[start_therapist] WARN: jetson_kill reported survivors (see above)"
    fi

    echo "[step 3/3] ssh start_therapist.sh (streaming, clinician view)"
    echo "           Ctrl+C here to detach; the remote process keeps running."
    echo ""
    # Re-invoke this same script on the Jetson; the Jetson branch below
    # runs.  THERAPIST_MODE=jetson is belt-and-suspenders in case the
    # remote host's /etc/nv_tegra_release check somehow fails.  Forward
    # the user's console log flags so `CONSOLE_LOG_LEVEL=DEBUG ./start_therapist.sh`
    # propagates all the way through the ssh hop.
    ssh -t $SSH_OPTS "$JETSON_HOST" \
        "THERAPIST_MODE=jetson \
         CLINICIAN_LOG_MODE='${CLINICIAN_LOG_MODE:-1}' \
         CONSOLE_LOG_LEVEL='${CONSOLE_LOG_LEVEL:-INFO}' \
         CLINICIAN_LOG_ALLOW='${CLINICIAN_LOG_ALLOW:-}' \
         CLINICIAN_LOG_BLOCK='${CLINICIAN_LOG_BLOCK:-}' \
         bash $REMOTE_DIR/start_therapist.sh"
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════
# JETSON MODE — local headless launch with clinician-view console
# ══════════════════════════════════════════════════════════════════════════

# Activate venv if present.
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "[start_therapist] WARN: .venv not found; using system Python"
fi

mkdir -p data/logs

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="data/logs/therapist_${TS}.log"
RAW_STDERR="data/logs/therapist_${TS}.stderr.log"

export LOG_FILE
export PYTHONUNBUFFERED=1
# Clinician console: demo-shape terminal view. File handler stays DEBUG.
export CONSOLE_LOG_LEVEL="${CONSOLE_LOG_LEVEL:-INFO}"
export CLINICIAN_LOG_MODE="${CLINICIAN_LOG_MODE:-1}"
# Headless deployment flag — main.py is already wired to run in this mode.
export DISABLE_INTERNAL_SPEECH="${DISABLE_INTERNAL_SPEECH:-0}"

# Suppress noisy native libraries when we're in clinician mode. These
# env vars quiet the two biggest offenders: TF-Lite's glog and the
# accompanying absl runtime. The fallback stderr redirect below catches
# anything they still print.
if [[ "${CLINICIAN_LOG_MODE}" == "1" ]]; then
    export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"   # 0=INFO,1=WARN,2=ERR,3=FATAL
    export GLOG_minloglevel="${GLOG_minloglevel:-2}"           # 0=INFO,1=WARN,2=ERR,3=FATAL
    export GRPC_VERBOSITY="${GRPC_VERBOSITY:-ERROR}"
    export ABSL_LOG_LEVEL="${ABSL_LOG_LEVEL:-ERROR}"
fi

cat <<BANNER
=======================================================
  CaiTI Smart-Speaker - Headless Start
  time:         $(date '+%Y-%m-%d %H:%M:%S %Z')
  host:         $(hostname -s)
  project:      ${PROJECT_ROOT}
  venv:         $([[ -n "${VIRTUAL_ENV:-}" ]] && echo "${VIRTUAL_ENV}" || echo "(system python)")
  log file:     ${LOG_FILE} (clinician trace)
  stderr file:  ${RAW_STDERR} (native-library noise)
  console:      ${CONSOLE_LOG_LEVEL}$([[ "$CLINICIAN_LOG_MODE" == "1" ]] && echo " (clinician view)" || echo "")
  subject_id:   ${SUBJECT_ID:-onboarded per session}
  llm model:    ${LLM_MODEL:-gemma-4-E2B-it}
=======================================================
BANNER

# Routing in clinician mode:
#   - Python clinician logger writes to STDOUT (goes to console + tee'd
#     into the main log file).
#   - Native C++ libraries (LiteRT, XNNPack, TensorFlow Lite, absl) and
#     LLM prompt dumps write to STDERR, which is redirected to the
#     *.stderr.log file so the clinician console never sees them. The
#     stderr log is kept alongside the main log so forensic debugging
#     still has every byte the runtime produced.
#
# Non-clinician mode preserves legacy behaviour: stderr merged into
# stdout so developers still see all errors on the console.
if [[ "${CLINICIAN_LOG_MODE}" == "1" ]]; then
    exec python3 -u main.py 2> >(tee -a "${RAW_STDERR}" >/dev/null) \
        | tee -a "${LOG_FILE}"
else
    exec python3 -u main.py 2>&1 | tee -a "${LOG_FILE}"
fi
