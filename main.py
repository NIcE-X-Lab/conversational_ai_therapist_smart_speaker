"""Main entry point orchestrating therapy session lifecycle over FastAPI + embedded speech loop."""

import os
import time
import threading
import queue
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.core.handler_rl import HandlerRL
from src.utils.io_record import init_record, OUTPUT_QUEUE, INPUT_QUEUE
from src.utils.log_util import get_logger
from src.utils.resource_audit import get_resource_audit
from src.utils import io_record
from src.utils.config_loader import (
    SUBJECT_ID, LLM_MODEL, LITERT_MODEL_PATH,
    DB_PATH, STT_MODEL_PATH, TTS_MODEL_PATH,
    TTS_INTERMISSION_MODEL_PATH,
    SER_ENABLED,
)

# ── Memory autopsy helper ───────────────────────────────────────────────
def _log_process_rss(label: str):
    """Log this process's RSS and VMS in MB using psutil (best-effort)."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        rss_mb = mem.rss / (1024 * 1024)
        vms_mb = mem.vms / (1024 * 1024)
        children = proc.children(recursive=True)
        child_rss = sum(c.memory_info().rss for c in children) / (1024 * 1024)
        logger.info(
            f"[RSS AUDIT] {label}: "
            f"PID={os.getpid()} RSS={rss_mb:.1f}MB VMS={vms_mb:.1f}MB "
            f"Children({len(children)})={child_rss:.1f}MB "
            f"Total={rss_mb + child_rss:.1f}MB"
        )
    except ImportError:
        logger.warning("[RSS AUDIT] psutil not installed — skipping RSS audit.")
    except Exception as e:
        logger.warning(f"[RSS AUDIT] {label}: failed ({e})")


def _ghost_hunt(rss_threshold_mb: float = 50.0):
    """Identify child processes consuming > rss_threshold_mb and log them.

    This helps detect leaked model-loading forks or any non-essential
    process eating into the Jetson's 8GB budget.
    """
    try:
        import psutil
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        ghosts_found = 0
        for child in children:
            try:
                child_rss_mb = child.memory_info().rss / (1024 * 1024)
                if child_rss_mb > rss_threshold_mb:
                    ghosts_found += 1
                    cmdline = " ".join(child.cmdline()) or child.name()
                    logger.warning(
                        f"[GHOST HUNT] Heavy child process detected: "
                        f"PID={child.pid} RSS={child_rss_mb:.1f}MB "
                        f"CMD='{cmdline[:120]}'"
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if ghosts_found == 0:
            logger.debug(f"[GHOST HUNT] No child processes > {rss_threshold_mb}MB. Clean.")
        else:
            logger.warning(
                f"[GHOST HUNT] Found {ghosts_found} child process(es) "
                f"> {rss_threshold_mb}MB. Review for memory savings."
            )
    except ImportError:
        logger.warning("[GHOST HUNT] psutil not installed — skipping.")
    except Exception as e:
        logger.warning(f"[GHOST HUNT] Failed: {e}")


logger = get_logger("MainApp")
RESOURCE_AUDIT = get_resource_audit()

DISABLE_INTERNAL_SPEECH = os.environ.get("DISABLE_INTERNAL_SPEECH", "0").strip().lower() in {"1", "true", "yes", "on"}

# H5: clinical trial fail-fast flag. When enabled, ANY critical check
# failure in the startup audit aborts the process with a diagnostic exit
# code so a clinician never sees the device "working" while a subsystem
# is silently broken.
CLINICAL_MODE = os.environ.get("CLINICAL_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}

# H6: minimum free-disk threshold. Sessions write ~2-3 MB of logs + dossier
# + DB growth; 500 MB headroom comfortably covers a long study day.
_MIN_FREE_DISK_MB = int(os.environ.get("MIN_FREE_DISK_MB", "500"))


def _check_disk_space() -> tuple[bool, str]:
    """H6: verify enough free space for at least a few sessions of logging."""
    try:
        import shutil
        path = os.path.abspath(".")
        usage = shutil.disk_usage(path)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < _MIN_FREE_DISK_MB:
            return False, (
                f"Free disk {free_mb:.0f}MB below threshold {_MIN_FREE_DISK_MB}MB "
                f"on {path}"
            )
        return True, f"Free disk: {free_mb:.0f}MB available"
    except Exception as e:
        return False, f"disk check error: {e}"


def _startup_checklist() -> bool:
    """Print a colour-coded system health checklist.

    Returns True if every critical subsystem passed. In CLINICAL_MODE the
    caller must abort on a False return.

    Critical subsystems (must pass for clinical trials):
      - LiteRT model present
      - Piper TTS model present
      - Faster-Whisper importable
      - Database reachable
      - Enough free disk space

    Non-critical: GPIO (stubbed on non-Jetson dev machines is acceptable).
    """
    import sqlite3

    lines = []
    critical_failures: list[str] = []

    # 1. GPIO (informational only — stub is acceptable off-Jetson)
    try:
        from src.drivers.gpio_manager import _GPIO_AVAILABLE, PIN_START_SESSION, PIN_END_SESSION, PIN_OPT_OUT, PIN_LED_LISTEN
        gpio_ok = _GPIO_AVAILABLE
        lines.append(
            f"[{'x' if gpio_ok else '!'}] GPIO {'Initialized' if gpio_ok else 'Stub (non-Jetson)'}"
            f" (Pins {PIN_START_SESSION}, {PIN_END_SESSION}, {PIN_OPT_OUT}, {PIN_LED_LISTEN})"
        )
    except Exception as e:
        lines.append(f"[!] GPIO FAILED: {e}")
        if CLINICAL_MODE:
            critical_failures.append(f"GPIO: {e}")

    # 2. LiteRT-LM model (CRITICAL)
    litert_ok = os.path.isfile(LITERT_MODEL_PATH)
    lines.append(
        f"[{'x' if litert_ok else '!'}] LiteRT Model "
        f"{'found' if litert_ok else 'NOT FOUND'} ({LITERT_MODEL_PATH})"
    )
    if not litert_ok:
        critical_failures.append(f"LiteRT model missing at {LITERT_MODEL_PATH}")
    else:
        try:
            size_mb = os.path.getsize(LITERT_MODEL_PATH) / (1024 * 1024)
            lines.append(f"[x] LiteRT Model Size: {size_mb:.0f}MB ({LLM_MODEL})")
            if size_mb < 100:
                lines.append("[!] LiteRT model looks truncated (<100MB); treating as missing.")
                critical_failures.append("LiteRT model truncated")
        except OSError:
            pass

    # 3. STT + TTS (CRITICAL)
    try:
        import faster_whisper  # noqa: F401
        whisper_ok = True
    except Exception:
        whisper_ok = False
    piper_ok = os.path.isfile(TTS_MODEL_PATH)
    lines.append(f"[{'x' if whisper_ok else '!'}] Faster-Whisper configured ({STT_MODEL_PATH})")
    lines.append(f"[{'x' if piper_ok else '!'}] Piper Model {'found' if piper_ok else 'NOT FOUND'} ({TTS_MODEL_PATH})")
    if not whisper_ok:
        critical_failures.append("faster-whisper not importable")
    if not piper_ok:
        critical_failures.append(f"Piper TTS model missing at {TTS_MODEL_PATH}")
    # Second voice is optional — missing file is a warning, never a
    # critical failure, since generate() silently falls back to primary.
    if TTS_INTERMISSION_MODEL_PATH:
        alt_ok = os.path.isfile(TTS_INTERMISSION_MODEL_PATH)
        lines.append(
            f"[{'x' if alt_ok else '~'}] Piper Intermission Voice "
            f"{'found' if alt_ok else 'NOT FOUND (will fall back to primary)'} "
            f"({TTS_INTERMISSION_MODEL_PATH})"
        )
    # SER status — informational only, never blocks boot.  Disabled
    # means the stub in src/models/ser.py is active and emotion tags
    # are hard-coded "neu".  See config.yaml `ser:` block to re-enable.
    # NOTE: Python 3.10 (Jetson default) disallows backslashes inside
    # f-string expression parts, so we build the SER descriptor via a
    # plain conditional rather than embedding escaped quotes in the
    # f-string. 3.12+ relaxes this rule but the production Jetson
    # image pins to 3.10.
    _ser_desc = (
        'enabled (real backend expected)'
        if SER_ENABLED
        else 'disabled (stub — emotion tag is always "neu")'
    )
    lines.append(
        f"[{'x' if SER_ENABLED else '~'}] SER {_ser_desc}"
    )

    # 4. Database (CRITICAL)
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
        lines.append(f"[x] Database Connected ({DB_PATH})")
    except Exception as e:
        lines.append(f"[!] Database FAILED: {e}")
        critical_failures.append(f"DB: {e}")

    # 5. H6 — free disk space (CRITICAL)
    disk_ok, disk_msg = _check_disk_space()
    lines.append(f"[{'x' if disk_ok else '!'}] {disk_msg}")
    if not disk_ok:
        critical_failures.append(disk_msg)

    sep = "=" * 55
    print(f"\n{sep}")
    print("  CaiTI System Boot — Connectivity Audit")
    if CLINICAL_MODE:
        print("  [CLINICAL_MODE enabled — failures will abort boot]")
    print(sep)
    for line in lines:
        print(f"  {line}")
    print(f"{sep}\n")
    RESOURCE_AUDIT.capture_point("startup_checklist_complete")
    RESOURCE_AUDIT.capture_process_inventory("startup_checklist_inventory")

    if critical_failures:
        logger.error(
            f"[BOOT-AUDIT] {len(critical_failures)} critical failure(s): {critical_failures}"
        )
        return False
    logger.debug("[BOOT-AUDIT] All critical subsystems passed.")
    return True


# ==========================================
# LOCAL MODE (FastAPI + Embedded Speech Loop)
# ==========================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
def get_status():
    current_status = "api_only_ready" if DISABLE_INTERNAL_SPEECH else "ready"
    if hasattr(app.state, 'speech_loop'):
        current_status = app.state.speech_loop.state
    elif io_record.START_SESSION_EVENT.is_set():
        current_status = "session_active"

    return {
        "status": current_status,
        # Read the live subject rather than the boot-time default so
        # /api/status reflects the name captured during onboarding for
        # the currently-running session.
        "subject_id": getattr(io_record, "SUBJECT_ID", SUBJECT_ID),
        "session_id": io_record.SESSION_ID
    }

@app.get("/api/turns")
def get_turns():
    if io_record.DB and io_record.SESSION_ID:
        history = io_record.DB.get_session_history(io_record.SESSION_ID)
        return history
    return []

@app.post("/api/action")
def post_action(action: dict):
    action_type = action.get("type")

    if not hasattr(app.state, 'speech_loop'):
        return {"status": "ignored", "reason": "internal_speech_unavailable"}
    
    if action_type == "stop":
        logger.info("Received STOP command")
        app.state.speech_loop.stop_audio()
        return {"status": "stopped"}
        
    elif action_type == "start_listening":
        logger.info("Received START_LISTENING command")
        app.state.speech_loop.manual_input_event.set()
        return {"status": "listening_triggered"}
        
    elif action_type == "set_mode":
        mode = action.get("mode")
        logger.info(f"Setting mode to {mode}")
        app.state.speech_loop.is_hands_free = (mode == "hands_free")
        return {"status": "mode_set", "mode": mode}

    return {"status": "ignored", "reason": f"unknown_action:{action_type}"}

@app.post("/api/pause")
def pause_session():
    if not hasattr(app.state, 'speech_loop'):
        return {"status": "ignored", "reason": "internal_speech_unavailable"}
    app.state.speech_loop.set_paused(True)
    return {"status": "paused"}

@app.post("/api/resume")
def resume_session():
    if not hasattr(app.state, 'speech_loop'):
        return {"status": "ignored", "reason": "internal_speech_unavailable"}
    app.state.speech_loop.set_paused(False)
    return {"status": "resumed"}

@app.post("/api/end_session")
def end_session_api():
    logger.info("[SESSION] End requested via API.")
    if hasattr(app.state, 'speech_loop'):
        app.state.speech_loop.stop_audio()
    io_record.END_SESSION_EVENT.set()
    io_record.START_SESSION_EVENT.clear()
    io_record.INPUT_QUEUE.put("SESSION_END")
    return {"status": "session_ended"}

@app.post("/api/login")
def login_user(data: dict):
    user_type = data.get("user_id", "test_user")
    logger.info(f"Logging in user: {user_type}")
    
    if user_type == "new_user":
        uid = f"user_{str(uuid.uuid4())[:8]}"
    else:
        uid = "test_user"
        
    io_record.reset_session(uid)

    io_record.END_SESSION_EVENT.clear()
    io_record.START_SESSION_EVENT.set() 
    return {"status": "logged_in", "user_id": uid, "session_id": io_record.SESSION_ID}

@app.post("/api/input")
def receive_input(data: dict):
    text = data.get("text")
    if text:
        logger.info(f"API Input received: {text}")
        io_record.INPUT_QUEUE.put(text)
    return {"status": "received"}

@app.post("/api/intent")
def classify_intent(data: dict):
    text = data.get("text", "")
    if not text: return {"intent": "none"}
        
    try:
        from src.models.llm_client import llm_complete, LLMRole
        system_prompt = (
            "You are a routing AI for a smart speaker therapist. Determine if the user's statement is a command to START or END the session.\n"
            "Rules:\n"
            "- Only answer START if the user is explicitly trying to wake you up, say hello to you, or start a new therapy session.\n"
            "- Only answer END if the user is explicitly commanding you to stop, end the session, wrap up, or say goodbye.\n"
            "- If the statement is just a normal conversational answer (even if it contains words like 'stop' or 'end'), reply NONE.\n"
        )
        user_prompt = f"User statement: \"{text}\"\n\nReply with exactly one word: START, END, or NONE\nClassification:"
        # Paper role: GENERAL (wake/sleep intent routing, extension beyond paper).
        response = llm_complete(system_prompt, user_prompt, role=LLMRole.GENERAL).strip().upper()
        
        if "START" in response: return {"intent": "start"}
        elif "END" in response: return {"intent": "end"}
        else: return {"intent": "none"}
            
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {"intent": "none"}

@app.get("/api/output")
def get_output():
    try:
        text = str(OUTPUT_QUEUE.get_nowait())
        logger.info(f"API Output served: {text}")
        return {"text": text}
    except queue.Empty:
        return {"text": None}

def run_fastapi_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning", access_log=False)

def main():
    RESOURCE_AUDIT.capture_point("main_bootstrap")
    _log_process_rss("Before startup checklist")
    boot_ok = _startup_checklist()
    if not boot_ok and CLINICAL_MODE:
        logger.error(
            "[BOOT-AUDIT] CLINICAL_MODE=1 and critical checks failed. Aborting."
        )
        # Non-zero exit code so supervising scripts (systemd, launchers)
        # can detect the failure and alert the clinician.
        raise SystemExit(2)
    # C1: init_record is idempotent; the handler also calls it on every
    # session start, but this initial call primes the DB + CSV paths
    # before the FastAPI server comes up.
    init_record()
    RESOURCE_AUDIT.capture_point("record_init_complete")
    _log_process_rss("After init_record (before audio stack)")
    
    # Start API server in background for remote monitoring/control
    api_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    api_thread.start()
    logger.debug("API Server started on port 8000")
    RESOURCE_AUDIT.capture_point("api_thread_started")

    speech_service = None
    if DISABLE_INTERNAL_SPEECH:
        logger.info("DISABLE_INTERNAL_SPEECH is enabled. Running in API-only mode.")
    else:
        try:
            # Lazy import so headless/API-only mode does not require audio stack deps.
            from src.services.speech_service import SpeechInteractionService

            speech_service = SpeechInteractionService(INPUT_QUEUE, OUTPUT_QUEUE)
            # SER is disabled by default (src/models/ser.py stub).  Only
            # Whisper + Piper dual-voice TTS contribute to the post-init
            # RSS delta; the SER seat loads nothing until a real backend
            # is dropped in and `ser.ser_enabled=true`.
            _ser_label = "SER+" if SER_ENABLED else ""
            _log_process_rss(
                f"After SpeechService init (Whisper+{_ser_label}TTS loaded)"
            )
            app.state.speech_loop = speech_service
            speech_thread = threading.Thread(target=speech_service.run, daemon=True)
            speech_thread.start()
            logger.debug("Unified SpeechInteractionService started.")
            RESOURCE_AUDIT.capture_point("speech_service_started")
            _log_process_rss("After SpeechService thread started (baseline)")
        except Exception as e:
            logger.error(f"Failed to start SpeechInteractionService; falling back to API-only mode: {e}")
            RESOURCE_AUDIT.capture_point("speech_service_start_failed", extra={"error": str(e)})

    # ── Ghost hunt: flag any child processes eating >50MB ──
    _ghost_hunt(rss_threshold_mb=50.0)
    RESOURCE_AUDIT.capture_process_inventory("post_init_ghost_hunt")

    try:
        while True:
            # Main logic loop: wait for the speech service to set START_SESSION_EVENT
            if not io_record.START_SESSION_EVENT.is_set():
                time.sleep(1)
                continue
                
            io_record.END_SESSION_EVENT.clear()
            logger.info(f"[SESSION] Clinical pipeline starting (DB session_id={io_record.SESSION_ID}).")
            
            # HandlerRL orchestrates the CBT/RL turns and uses the interstitial engine
            handler = HandlerRL()
            handler.run()

            # Bug-1 fix (tail): handler may have queued a final utterance
            # ("Great work today.", closing reflection, etc.) on OUTPUT_QUEUE
            # immediately before returning. Give the speech service a bounded
            # window to speak it before we transition to idle; once
            # START_SESSION_EVENT clears, the main speech loop drops into
            # the wake-word listener and stops consuming OUTPUT_QUEUE, so
            # any remaining item would be orphaned.
            _DRAIN_TIMEOUT_SEC = 5.0
            _drain_deadline = time.time() + _DRAIN_TIMEOUT_SEC
            while time.time() < _drain_deadline:
                if io_record.OUTPUT_QUEUE.empty():
                    # Small extra grace so an in-flight say() can finish
                    # speaking before idle transition silences things.
                    time.sleep(0.5)
                    if io_record.OUTPUT_QUEUE.empty():
                        break
                time.sleep(0.1)

            logger.info("[SESSION] Clinical pipeline finished — returning to idle, awaiting next wake.")
            io_record.START_SESSION_EVENT.clear()
            
    except KeyboardInterrupt:
        logger.info("Application interrupted.")
    finally:
        if speech_service is not None:
            speech_service.stop()
        RESOURCE_AUDIT.emit_resource_map()


if __name__ == "__main__":
    main()