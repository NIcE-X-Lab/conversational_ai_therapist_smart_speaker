"""Utility helper managing file I/O operations and synchronous memory channels."""
import os
import json
import queue
import logging
import time
import threading
import pandas as pd
from typing import List, Tuple

from src.utils.log_util import get_logger
from src.drivers.db_manager import DBManager
from src.utils.config_loader import DB_PATH, SUBJECT_ID, RECORD_CSV

logger = get_logger("IORecord")

_DEFAULT_RECORD_CSV = RECORD_CSV
_LAST_AUTO_RECORD_CSV = None

# IPC Queues
# Agent puts questions here, Interface gets them
OUTPUT_QUEUE = queue.Queue()
# Interface puts responses here, Agent gets them
INPUT_QUEUE = queue.Queue()

END_SESSION_EVENT = threading.Event()
START_SESSION_EVENT = threading.Event()


# Database instance
DB = None
SESSION_ID = None
CURRENT_TURN_INDEX = 0

_PENDING_QUESTION_PREFIX = ""
USER_CONTEXT = ""
_LAST_USER_TRANSCRIPT = ""
_LAST_USER_EMOTION = "Neutral"
_LAST_RL_STATE = {}
_LATEST_SCREENING_SCORES = {
    "anxiety": None,
    "depression": None,
    "total": None,
}

# JSON session log path (set dynamically in init_record)
_JSON_LOG_PATH: str = ""

def get_user_context():
    return USER_CONTEXT

def set_last_user_signal(transcript: str, emotion: str = "Neutral"):
    global _LAST_USER_TRANSCRIPT, _LAST_USER_EMOTION
    _LAST_USER_TRANSCRIPT = str(transcript or "").strip()
    _LAST_USER_EMOTION = str(emotion or "Neutral").strip()

def set_rl_context(state: dict):
    global _LAST_RL_STATE
    _LAST_RL_STATE = state if isinstance(state, dict) else {}

def set_latest_screening_scores(anxiety=None, depression=None, total=None):
    global _LATEST_SCREENING_SCORES
    _LATEST_SCREENING_SCORES = {
        "anxiety": anxiety,
        "depression": depression,
        "total": total,
    }

def get_context_pack() -> dict:
    return {
        "user_transcript": _LAST_USER_TRANSCRIPT,
        "emotion_tag": _LAST_USER_EMOTION,
        "rl_state": _LAST_RL_STATE,
        "screening_scores": _LATEST_SCREENING_SCORES,
    }

def set_question_prefix(text: str):
    """
    Set a pending prefix that will be prepended to the next question output.
    """
    global _PENDING_QUESTION_PREFIX
    _PENDING_QUESTION_PREFIX = str(text) if text is not None else ""

# CSV Synchronization (Full Transcript Logging)
HEADER = ["Timestamp", "Type", "Speaker", "Text"]

def log_json_event(event_type: str, data: dict):
    """
    Append a timestamped JSON line to the session JSON log file.
    Format: {"ts": "...", "event": "...", ...data}
    """
    global _JSON_LOG_PATH
    if not _JSON_LOG_PATH:
        return
    try:
        import datetime
        folder = os.path.dirname(_JSON_LOG_PATH)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        entry = {"ts": datetime.datetime.now().isoformat(), "event": event_type}
        entry.update(data)
        with open(_JSON_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write JSON log: {e}")

def append_to_csv(log_type: str, speaker: str, text: str):
    """
    Append a single entry to record.csv to capture the full state of the conversation.
    """
    try:
        folder = os.path.dirname(RECORD_CSV)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        write_header = not os.path.exists(RECORD_CSV)
        
        with open(RECORD_CSV, 'a', encoding='utf-8') as f:
            if write_header:
                f.write(",".join(HEADER) + "\n")
            
            # Escape strings for CSV
            escaped_text = str(text).replace('"', '""')
            line = f'"{timestamp}","{log_type}","{speaker}","{escaped_text}"\n'
            f.write(line)
    except Exception as e:
        logger.error(f"Failed to sync to CSV: {e}")

def init_record(user_id_override: str = None):
    """Initialize queues, database session, and CSV."""
    global DB, SESSION_ID, CURRENT_TURN_INDEX, SUBJECT_ID
    global _LAST_USER_TRANSCRIPT, _LAST_USER_EMOTION, _LAST_RL_STATE, _LATEST_SCREENING_SCORES
    global _LAST_AUTO_RECORD_CSV

    # Reset PHQ-4 / GAD-2 screening state for this session
    try:
        from src.models.llm_client import _reset_screening
        _reset_screening()
    except Exception:
        pass  # llm_client not yet loaded on first import is okay

    if user_id_override:
        logger.info(f"Overriding SUBJECT_ID with {user_id_override}")
        SUBJECT_ID = user_id_override
    
    # Clear queues
    with OUTPUT_QUEUE.mutex:
        OUTPUT_QUEUE.queue.clear()
    with INPUT_QUEUE.mutex:
        INPUT_QUEUE.queue.clear()
    
    # Initialize DB
    try:
        DB = DBManager(DB_PATH)
        user_id = DB.get_user_id(SUBJECT_ID)
        SESSION_ID = DB.create_session(user_id)
        # Load User Context (Summaries & Preferences)
        try:
            global USER_CONTEXT
            USER_CONTEXT = DB.get_user_context_string(user_id)
            if USER_CONTEXT:
                logger.info("Loaded User Context for session.")
        except Exception as e:
            logger.error(f"Failed to load user context: {e}")

        CURRENT_TURN_INDEX = 0
        _LAST_USER_TRANSCRIPT = ""
        _LAST_USER_EMOTION = "Neutral"
        _LAST_RL_STATE = {}
        _LATEST_SCREENING_SCORES = {"anxiety": None, "depression": None, "total": None}
    except Exception as e:
        logger.error(f"Failed to initialize DB: {e}")
        
    # Resolve RECORD_CSV and _JSON_LOG_PATH.
    # Keep dynamic per-session path by default, but respect explicit overrides
    # (e.g., tests that set io_record.RECORD_CSV to a fixed file).
    global RECORD_CSV, _JSON_LOG_PATH
    import datetime
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_session_dir = os.path.join(os.path.abspath("."), "data", "logs", SUBJECT_ID)
    record_csv_is_external = RECORD_CSV not in {_DEFAULT_RECORD_CSV, _LAST_AUTO_RECORD_CSV}
    if record_csv_is_external:
        logger.info(f"Using externally configured RECORD_CSV path: {RECORD_CSV}")
    else:
        RECORD_CSV = os.path.join(base_session_dir, f"{SUBJECT_ID}_Session_{timestamp_str}.log")
        _LAST_AUTO_RECORD_CSV = RECORD_CSV
    _JSON_LOG_PATH = os.path.join(base_session_dir, f"{SUBJECT_ID}_Session_{timestamp_str}.json")
    
    # Initialize CSV
    append_to_csv("internal", "system", f"Session initialized. User: {SUBJECT_ID}, DB ID: {SESSION_ID}")

def reset_session(new_user_id: str = None):
    """Reset the session, optionally switching users."""
    init_record(new_user_id)



def log_question(text: str, meta_data: dict = None):
    """
    Log a question from the agent. 
    Prepends pending prefix, puts into Output Queue, saves to DB, and updates CSV.
    """
    global _PENDING_QUESTION_PREFIX, CURRENT_TURN_INDEX
    
    combined = text
    if _PENDING_QUESTION_PREFIX:
        combined = f"{_PENDING_QUESTION_PREFIX}\n\n{text}"
        logger.info("Combining pending prefix with next question.")
    
    # Push to queue for interface
    OUTPUT_QUEUE.put(combined)
    
    # Save to DB
    if DB and SESSION_ID:
        DB.add_turn(SESSION_ID, CURRENT_TURN_INDEX, "agent", combined, meta_data=meta_data)
        CURRENT_TURN_INDEX += 1
    
    # Sync to CSV 
    append_to_csv("turn", "agent", combined)
    log_json_event("agent_turn", {"text": combined})
    
    # Clear prefix
    _PENDING_QUESTION_PREFIX = ""
    logger.info(f"Prompted question: {combined}")

def log_reasoning(reasoning_type: str, data: dict):
    """
    Log a system reasoning event (e.g., RL states, Semantic scores) to the database.
    """
    global CURRENT_TURN_INDEX
    if DB and SESSION_ID:
        meta = {"reasoning_type": reasoning_type}
        meta.update(data)
        DB.add_turn(SESSION_ID, CURRENT_TURN_INDEX, "system", f"[{reasoning_type.upper()}]", meta_data=meta)
        CURRENT_TURN_INDEX += 1
        logger.info(f"Logged Reasoning ({reasoning_type}) to DB.")
    if reasoning_type == "rl_decision":
        set_rl_context(data)

def get_answer() -> Tuple[List, List[str]]:
    """
    Get answer from the user. 
    Blocks until input is available in Input Queue.
    Returns (dummy_DLA_result, segments).
    """
    global CURRENT_TURN_INDEX
    
    logger.info("Waiting for user answer...")
    user_input_raw = INPUT_QUEUE.get() # Blocking get
    logger.info(f"Received user input: {user_input_raw}")
    
    # Save to DB
    if DB and SESSION_ID:
        DB.add_turn(SESSION_ID, CURRENT_TURN_INDEX, "user", user_input_raw)
        CURRENT_TURN_INDEX += 1
        
    # We should read the last Question from CSV? Or just write Resp.
    # Let's try to keep it simple.
    append_to_csv("turn", "user", user_input_raw)
        
    # Process as JSON if possible
    import json
    emotion_str = "Neutral"
    try:
        parsed = json.loads(str(user_input_raw))
        user_input_text = parsed.get("transcript", "")
        emotion_str = parsed.get("detected_emotion", "Neutral")
    except Exception:
        user_input_text = str(user_input_raw)

    set_last_user_signal(user_input_text, emotion_str)

    user_input_text = user_input_text.replace(", and", ".").replace("but", ".")
    raw_segments = user_input_text.split(".")
    
    segments = []
    for i, seg in enumerate(raw_segments):
        seg = seg.strip()
        if seg:
            # Append emotion auxiliary context to the last valid segment
            if i == len(raw_segments) - 1 or len([s for s in raw_segments[i+1:] if s.strip()]) == 0:
                seg = f"{seg} [Detected Emotion: {emotion_str}]"
            segments.append(seg)
            
    DLA_result = []
    log_json_event("user_turn", {"transcript": user_input_text, "emotion": emotion_str, "segments": segments})
    return DLA_result, segments

def get_resp_log() -> str:
    """
    Get raw user response (for RV logic).
    Blocking.
    """
    global CURRENT_TURN_INDEX
    
    logger.info("Waiting for user response (raw)...")
    user_response_raw = INPUT_QUEUE.get()
    
    import json
    try:
        parsed = json.loads(str(user_response_raw))
        transcript = parsed.get("transcript", "")
        emotion = parsed.get("detected_emotion", "Neutral")
        user_response = f"{transcript} [Detected Emotion: {emotion}]"
        set_last_user_signal(transcript, emotion)
    except Exception:
        user_response = str(user_response_raw)
        set_last_user_signal(user_response, "Neutral")
    
    if DB and SESSION_ID:
        DB.add_turn(SESSION_ID, CURRENT_TURN_INDEX, "user", user_response)
        CURRENT_TURN_INDEX += 1

    append_to_csv("turn", "user", user_response)
    log_json_event("user_turn", {"response": user_response})

    logger.info(f"Received user response: {user_response}")
    return user_response


def dump_session_history_to_terminal() -> None:
    """Print full session history to terminal in chronological order."""
    if not DB or not SESSION_ID:
        logger.warning("Cannot dump session history: DB or SESSION_ID missing.")
        return
    history = DB.get_session_history(SESSION_ID)
    logger.info("========== SESSION HISTORY BEGIN ==========")
    for idx, turn in enumerate(history, start=1):
        speaker = str(turn.get("speaker", "unknown")).upper()
        text = str(turn.get("text", "")).strip()
        logger.info(f"[{idx:03d}] {speaker}: {text}")
    logger.info("=========== SESSION HISTORY END ===========")



