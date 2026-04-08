"""Main entry point orchestrating therapy session lifecycle over local or remoted API servers."""

import os
import time
import threading
import queue
import json
import string
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
from src.services.speech_service import SpeechInteractionService
from src.utils.io_record import init_record, OUTPUT_QUEUE, INPUT_QUEUE, HEADER
from src.utils.log_util import get_logger
from src.utils import io_record
from src.utils.config_loader import (
    SUBJECT_ID, RECORD_CSV, OPENAI_BASE_URL, LLM_MODEL,
    DB_PATH, STT_MODEL_PATH, TTS_MODEL_PATH,
    PIN_BTN_START, PIN_BTN_END, PIN_BTN_OPT_OUT, PIN_LISTENING_LED,
    OLLAMA_KEEP_ALIVE
)

# Perception and Action modules
from src.drivers.audio import AudioRecorder
from src.models.stt import STTGenerator
from src.models.tts import TTSGenerator
from src.drivers.player import AudioPlayer

logger = get_logger("MainApp")

APP_MODE = os.environ.get("APP_MODE", "local")


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
    whisper_ok = True   # WhisperModel loaded lazily; just verify path
    piper_ok = os.path.isfile(TTS_MODEL_PATH)
    lines.append(f"[{'x' if whisper_ok else '!'}] Whisper Model configured ({STT_MODEL_PATH})")
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
    current_status = "ready"
    if hasattr(app.state, 'speech_loop'):
        current_status = app.state.speech_loop.state

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
    
    if action_type == "stop":
        logger.info("Received STOP command")
        if hasattr(app.state, 'speech_loop'):
             app.state.speech_loop.stop_audio()
        return {"status": "stopped"}
        
    elif action_type == "start_listening":
        logger.info("Received START_LISTENING command")
        if hasattr(app.state, 'speech_loop'):
             app.state.speech_loop.manual_input_event.set()
        return {"status": "listening_triggered"}
        
    elif action_type == "set_mode":
        mode = action.get("mode")
        logger.info(f"Setting mode to {mode}")
        if hasattr(app.state, 'speech_loop'):
            app.state.speech_loop.is_hands_free = (mode == "hands_free")
        return {"status": "mode_set", "mode": mode}

@app.post("/api/pause")
def pause_session():
    if hasattr(app.state, 'speech_loop'):
        app.state.speech_loop.set_paused(True)
    return {"status": "paused"}

@app.post("/api/resume")
def resume_session():
    if hasattr(app.state, 'speech_loop'):
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
    _startup_checklist()
    init_record()
    
    # Start API server in background for remote monitoring/control
    api_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    api_thread.start()
    logger.info("API Server started on port 8000")

    # Initialize and start the Unified Speech Interaction Service
    # This service handles both voice wake-word and physical button events
    speech_service = SpeechInteractionService(INPUT_QUEUE, OUTPUT_QUEUE)
    speech_thread = threading.Thread(target=speech_service.run, daemon=True)
    speech_thread.start()
    logger.info("Unified SpeechInteractionService started.")
    
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
        speech_service.stop()


# ==========================================
# SERVER MODE (Flask Database Polling)
# ==========================================

flask_app = Flask(__name__)
CORS(flask_app)

_rl_thread = None
_rl_running = False
_rl_lock = threading.Lock()

def _ensure_record_file():
    folder = os.path.dirname(RECORD_CSV)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if not os.path.exists(RECORD_CSV):
        df = pd.DataFrame([["", 0, "", 1]], columns=HEADER)
        df.to_csv(RECORD_CSV, columns=HEADER, index=False)

def _read_record():
    return pd.read_csv(RECORD_CSV)

def _write_record(df):
    tmp_path = RECORD_CSV + ".tmp"
    df.to_csv(tmp_path, columns=HEADER, index=False)
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
    t0 = time.time()
    while True:
        df = _read_record()
        if int(df.loc[0, "Question_Lock"]) == 1:
            question = str(df.loc[0, "Question"])
            df.loc[0, "Question_Lock"] = 0
            _write_record(df)
            return question
        if time.time() - t0 > timeout_sec:
            return ""
        time.sleep(0.1)

def _log_resp(text: str):
    df = _read_record()
    df.loc[0, "Resp"] = text
    df.loc[0, "Resp_Lock"] = 0
    _write_record(df)

@flask_app.route("/gpt", methods=["POST"])
def gpt():
    payload = flask_request.get_json(force=True)
    user_input = str(payload["user_input"])
    subject_id = str(payload.get("subject_ID", ""))

    if user_input.lower().strip() == "start":
        _ensure_record_file()
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
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "8080"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    flask_app.run(host=host, port=port, debug=debug)


# ==========================================
# BOOTSTRAP
# ==========================================
def main():
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