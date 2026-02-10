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

# CSV Synchronization (Legacy/Frontend Requirement)
HEADER = ["Question", "Question_Lock", "Resp", "Resp_Lock"]

def _atomic_write_csv(data: dict):
    """
    Write current state to record.csv for external frontend compatibility.
    """
    try:
        df = pd.DataFrame([data])
        folder = os.path.dirname(RECORD_CSV)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        tmp_path = RECORD_CSV + ".tmp"
        df.to_csv(tmp_path, columns=HEADER, index=False)
        os.replace(tmp_path, RECORD_CSV)
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
    _atomic_write_csv({"Question": "", "Question_Lock": 0, "Resp": "", "Resp_Lock": 1})

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
    
    # Sync to CSV (Agent sets Question, Locks it)
    # We read existing check? No, just overwrite for this simple sync.
    # In a real dual-writer scenario, we'd need to be careful, but here Agent drives.
    _atomic_write_csv({"Question": combined, "Question_Lock": 1, "Resp": "", "Resp_Lock": 0})
    
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
        
    # Sync to CSV (User Resp arrived)
    # Ideally we update the existing row.
    # For now, we assume the previous LogQuestion set the state.
    # We update Resp and Lock Resp.
    # We verify if we need to preserve Question? usually yes.
    # But since we don't have global state of Question here easily unless we track it.
    # Let's just write the response.
    # Note: The 'Frontend' usually writes the response.
    # Here, 'SpeechInteractionLoop' (Frontend) wrote to INPUT_QUEUE.
    # And we (Handler/Backend) are consuming it.
    # So WE are responsible for marking it as 'Read' or 'Logged'?
    # Actually, if the 'Frontend' (Speech Script) wrote to CSV, then it would be there.
    # But here we are simulating the frontend via Queue.
    # So we write it to CSV to verify "transcripts back into data/record.csv".
    
    # We should read the last Question from CSV? Or just write Resp.
    # Let's try to keep it simple.
    _atomic_write_csv({"Question": "...", "Question_Lock": 0, "Resp": user_input_raw, "Resp_Lock": 1})
        
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
    
    _atomic_write_csv({"Question": "...", "Question_Lock": 0, "Resp": user_response, "Resp_Lock": 1})
        
    logger.info(f"Received user response: {user_response}")
    return user_response


