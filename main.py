"""Main entry point orchestrating therapy session lifecycle over local or remoted API servers."""

import os
import time
import threading
import queue
import json
import pandas as pd
import uuid
import requests
from flask import Flask, request as flask_request, jsonify
from flask_cors import CORS

# FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.core.handler_rl import HandlerRL
from src.utils.io_record import init_record, OUTPUT_QUEUE, INPUT_QUEUE
from src.utils.log_util import get_logger
from src.utils.resource_audit import get_resource_audit
from src.utils import io_record
from src.utils.config_loader import (
    SUBJECT_ID, RECORD_CSV, OPENAI_BASE_URL, LLM_MODEL,
    DB_PATH, STT_MODEL_PATH, TTS_MODEL_PATH,
    PIN_BTN_START, PIN_BTN_END, PIN_BTN_OPT_OUT, PIN_LISTENING_LED,
    OLLAMA_KEEP_ALIVE
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

    This helps detect zombie Ollama runners, leaked model-loading forks,
    or any non-essential process eating into the Jetson's 8GB budget.
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
            logger.info(f"[GHOST HUNT] No child processes > {rss_threshold_mb}MB. Clean.")
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

APP_MODE = os.environ.get("APP_MODE", "local")
DISABLE_INTERNAL_SPEECH = os.environ.get("DISABLE_INTERNAL_SPEECH", "0").strip().lower() in {"1", "true", "yes", "on"}


def _startup_checklist():
    """Print a colour-coded system health checklist to the terminal on boot."""
    import sqlite3

    lines = []

    # 1. GPIO
    try:
        from src.drivers.gpio_manager import _GPIO_AVAILABLE, PIN_START_SESSION, PIN_END_SESSION, PIN_OPT_OUT, PIN_LED_LISTEN
        gpio_ok = _GPIO_AVAILABLE
        lines.append(
            f"[{'x' if gpio_ok else '!'}] GPIO {'Initialized' if gpio_ok else 'Stub (non-Jetson)'}"
            f" (Pins {PIN_START_SESSION}, {PIN_END_SESSION}, {PIN_OPT_OUT}, {PIN_LED_LISTEN})"
        )
    except Exception as e:
        lines.append(f"[!] GPIO FAILED: {e}")

    # 2. Ollama / LLM
    try:
        base = OPENAI_BASE_URL.replace("/v1", "")
        resp = requests.get(f"{base}/api/tags", timeout=3)
        available = [m["name"] for m in resp.json().get("models", [])]
        model_found = any(LLM_MODEL in m for m in available)
        lines.append(
            f"[{'x' if model_found else '!'}] Ollama Connected ({LLM_MODEL})"
            + ("" if model_found else f" ⚠ model not pulled yet, available: {available}")
        )
        lines.append(f"[x] Ollama Keep Alive configured ({OLLAMA_KEEP_ALIVE})")
    except Exception as e:
        lines.append(f"[!] Ollama UNREACHABLE: {e}")

    # 3. STT / TTS
    try:
        import faster_whisper  # noqa: F401
        whisper_ok = True
    except Exception:
        whisper_ok = False
    piper_ok = os.path.isfile(TTS_MODEL_PATH)
    lines.append(f"[{'x' if whisper_ok else '!'}] Faster-Whisper configured ({STT_MODEL_PATH})")
    lines.append(f"[{'x' if piper_ok else '!'}] Piper Model {'found' if piper_ok else 'NOT FOUND'} ({TTS_MODEL_PATH})")

    # 4. Database
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
        lines.append(f"[x] Database Connected ({DB_PATH})")
    except Exception as e:
        lines.append(f"[!] Database FAILED: {e}")

    sep = "=" * 55
    print(f"\n{sep}")
    print("  CaiTI System Boot — Connectivity Audit")
    print(sep)
    for line in lines:
        print(f"  {line}")
    print(f"{sep}\n")
    RESOURCE_AUDIT.capture_point("startup_checklist_complete")
    RESOURCE_AUDIT.capture_process_inventory("startup_checklist_inventory")


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
        "subject_id": SUBJECT_ID,
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
    logger.info("Ending session via API.")
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
        from src.models.llm_client import llm_complete
        system_prompt = (
            "You are a routing AI for a smart speaker therapist. Determine if the user's statement is a command to START or END the session.\n"
            "Rules:\n"
            "- Only answer START if the user is explicitly trying to wake you up, say hello to you, or start a new therapy session.\n"
            "- Only answer END if the user is explicitly commanding you to stop, end the session, wrap up, or say goodbye.\n"
            "- If the statement is just a normal conversational answer (even if it contains words like 'stop' or 'end'), reply NONE.\n"
        )
        user_prompt = f"User statement: \"{text}\"\n\nReply with exactly one word: START, END, or NONE\nClassification:"
        response = llm_complete(system_prompt, user_prompt).strip().upper()
        
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

def main_local():
    RESOURCE_AUDIT.capture_point("main_local_entry")
    _log_process_rss("Before startup checklist")
    _startup_checklist()
    init_record()
    RESOURCE_AUDIT.capture_point("record_init_complete")
    _log_process_rss("After init_record (before audio stack)")
    
    # Start API server in background for remote monitoring/control
    api_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    api_thread.start()
    logger.info("API Server started on port 8000")
    RESOURCE_AUDIT.capture_point("api_thread_started")

    speech_service = None
    if DISABLE_INTERNAL_SPEECH:
        logger.info("DISABLE_INTERNAL_SPEECH is enabled. Running in API-only mode.")
    else:
        try:
            # Lazy import so headless/API-only mode does not require audio stack deps.
            from src.services.speech_service import SpeechInteractionService

            speech_service = SpeechInteractionService(INPUT_QUEUE, OUTPUT_QUEUE)
            _log_process_rss("After SpeechService init (Whisper+SER+TTS loaded)")
            app.state.speech_loop = speech_service
            speech_thread = threading.Thread(target=speech_service.run, daemon=True)
            speech_thread.start()
            logger.info("Unified SpeechInteractionService started.")
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
            logger.info(f"Starting Clinical Pipeline (HandlerRL) for Session {io_record.SESSION_ID}")
            
            # HandlerRL orchestrates the CBT/RL turns and uses the interstitial engine
            handler = HandlerRL()
            handler.run()
            
            logger.info("Clinical Pipeline turn finished. Waiting for next trigger.")
            io_record.START_SESSION_EVENT.clear()
            
    except KeyboardInterrupt:
        logger.info("Application interrupted.")
    finally:
        if speech_service is not None:
            speech_service.stop()
        RESOURCE_AUDIT.emit_resource_map()


# ==========================================
# SERVER MODE (Flask Database Polling)
# ==========================================

flask_app = Flask(__name__)
CORS(flask_app)

_rl_thread = None
_rl_running = False
_rl_lock = threading.Lock()

def _read_record():
    return pd.read_csv(RECORD_CSV)

def _write_record(df):
    tmp_path = RECORD_CSV + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, RECORD_CSV)

def _start_rl_if_needed():
    global _rl_thread, _rl_running
    with _rl_lock:
        if _rl_thread is not None and _rl_thread.is_alive():
            return
        _rl_running = True
        def _runner():
            global _rl_running
            logger.info("RL thread started")
            HandlerRL().run()
            _rl_running = False
            logger.info("RL thread finished")
        _rl_thread = threading.Thread(target=_runner, daemon=True)
        _rl_thread.start()

def _get_question_blocking(timeout_sec=60):
    try:
        return str(OUTPUT_QUEUE.get(timeout=timeout_sec))
    except queue.Empty:
        return ""

def _log_resp(text: str):
    INPUT_QUEUE.put(text)

@flask_app.route("/gpt", methods=["POST"])
def gpt():
    payload = flask_request.get_json(force=True)
    user_input = str(payload["user_input"])
    subject_id = str(payload.get("subject_ID", ""))

    if user_input.lower().strip() == "start":
        _start_rl_if_needed()
        question = _get_question_blocking()
        return jsonify({"subject_ID": subject_id, "question": question})

    _log_resp(user_input)
    question = _get_question_blocking()
    return jsonify({"subject_ID": subject_id, "question": question})

@flask_app.route("/health", methods=["GET"])
def health():
    status = "running" if (_rl_thread is not None and _rl_thread.is_alive()) else "idle"
    return jsonify({"status": status})

def main_server():
    RESOURCE_AUDIT.capture_point("main_server_entry")
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "8080"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    flask_app.run(host=host, port=port, debug=debug)


# ==========================================
# BOOTSTRAP
# ==========================================
def main():
    RESOURCE_AUDIT.capture_point("main_bootstrap")
    if APP_MODE == "local":
        logger.info("Booting in LOCAL mode")
        main_local()
    elif APP_MODE == "server":
        logger.info("Booting in SERVER mode")
        main_server()
    else:
        logger.error(f"Unknown APP_MODE: {APP_MODE}. Exiting.")

if __name__ == "__main__":
    main()