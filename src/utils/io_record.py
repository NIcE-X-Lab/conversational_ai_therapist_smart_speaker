import os
import queue
import logging
import time
import threading
import pandas as pd
from typing import List, Tuple

from src.utils.log_util import get_logger
from src.database.db_manager import DBManager
from src.utils.config_loader import DB_PATH, SUBJECT_ID, RECORD_CSV

logger = get_logger("IORecord")

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

def get_user_context():
    return USER_CONTEXT

def set_question_prefix(text: str):
    """
    Set a pending prefix that will be prepended to the next question output.
    """
    global _PENDING_QUESTION_PREFIX
    _PENDING_QUESTION_PREFIX = str(text) if text is not None else ""

# CSV Synchronization (Full Transcript Logging)
HEADER = ["Timestamp", "Type", "Speaker", "Text"]

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
        logger.info(f"Initialized record. Session ID: {SESSION_ID}")
    except Exception as e:
        logger.error(f"Failed to initialize DB: {e}")
        
    # Initialize CSV
    append_to_csv("internal", "system", f"Session initialized. User: {SUBJECT_ID}, DB ID: {SESSION_ID}")

def reset_session(new_user_id: str = None):
    """Reset the session, optionally switching users."""
    init_record(new_user_id)



def log_question(text: str):
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
        DB.add_turn(SESSION_ID, CURRENT_TURN_INDEX, "agent", combined)
        CURRENT_TURN_INDEX += 1
    
    # Sync to CSV 
    append_to_csv("turn", "agent", combined)
    
    # Clear prefix
    _PENDING_QUESTION_PREFIX = ""
    logger.info(f"Prompted question: {combined}")

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
        
    # Process segments
    user_input = str(user_input_raw)
    user_input = user_input.replace(", and", ".").replace("but", ".")
    user_input = user_input.split(".")
    
    segments = []
    for seg in user_input:
        seg = seg.strip()
        if seg:
            segments.append(seg)
            
    DLA_result = [] 
    
    return DLA_result, segments

def get_resp_log() -> str:
    """
    Get raw user response (for RV logic).
    Blocking.
    """
    global CURRENT_TURN_INDEX
    
    logger.info("Waiting for user response (raw)...")
    user_response = INPUT_QUEUE.get()
    
    if DB and SESSION_ID:
        DB.add_turn(SESSION_ID, CURRENT_TURN_INDEX, "user", user_response)
        CURRENT_TURN_INDEX += 1
    
    append_to_csv("turn", "user", user_response)
        
    logger.info(f"Received user response: {user_response}")
    return user_response


