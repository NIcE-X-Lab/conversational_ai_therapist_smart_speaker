import os
import time
import threading
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.handler_rl import HandlerRL
from src.utils.config_loader import RECORD_CSV
from src.utils.log_util import get_logger
from src.utils.io_record import HEADER  # Reuse column names to keep consistent with backend

# Initialize logger for this server module
logger = get_logger("FlaskServer")

# Create Flask app and enable CORS for cross-origin requests
app = Flask(__name__)
CORS(app)

# Global variables for RL thread management
_rl_thread = None
_rl_running = False
_rl_lock = threading.Lock()  # Lock to ensure thread-safe RL start/stop

def _ensure_record_file():
    """
    Ensure the record.csv file exists and has the correct columns.
    If the parent folder does not exist, create it.
    If the file does not exist, create it with a default row.
    """
    folder = os.path.dirname(RECORD_CSV)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if not os.path.exists(RECORD_CSV):
        df = pd.DataFrame([["", 0, "", 1]], columns=HEADER)
        df.to_csv(RECORD_CSV, columns=HEADER, index=False)

def _read_record():
    """
    Read the record.csv file into a pandas DataFrame.
    """
    return pd.read_csv(RECORD_CSV)

def _write_record(df):
    """
    Write the given DataFrame to record.csv atomically, enforcing column order.
    This avoids I/O conflicts by writing to a temporary file and then renaming.
    """
    tmp_path = RECORD_CSV + ".tmp"
    df.to_csv(tmp_path, columns=HEADER, index=False)
    os.replace(tmp_path, RECORD_CSV)

def _start_rl_if_needed():
    """
    Start the RL handler in a background thread if not already running.
    Uses a lock to prevent race conditions.
    """
    global _rl_thread, _rl_running
    with _rl_lock:
        if _rl_thread is not None and _rl_thread.is_alive():
            return  # RL thread is already running
        _rl_running = True

        def _runner():
            global _rl_running
            logger.info("RL thread started")
            HandlerRL().run()  # Main RL workflow
            _rl_running = False
            logger.info("RL thread finished")

        _rl_thread = threading.Thread(target=_runner, daemon=True)
        _rl_thread.start()

def _get_question_blocking(timeout_sec=60):
    """
    Poll the record file for a new question (Question_Lock == 1).
    Once found, reset the lock and return the question.
    Timeout after timeout_sec seconds and return empty string if not found.
    """
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
        time.sleep(0.1)  # Sleep briefly to avoid busy-waiting

def _log_resp(text: str):
    """
    Write the user's response to the record file and unlock Resp_Lock.
    """
    df = _read_record()
    df.loc[0, "Resp"] = text
    df.loc[0, "Resp_Lock"] = 0
    _write_record(df)

@app.route("/gpt", methods=["POST"])
def gpt():
    """
    Main API endpoint for user interaction.
    - If user_input is "start", initialize the record file and RL thread, then return the first question.
    - Otherwise, log the user's response and return the next question.
    """
    payload = request.get_json(force=True)
    user_input = str(payload["user_input"])
    subject_id = str(payload.get("subject_ID", ""))

    if user_input.lower().strip() == "start":
        _ensure_record_file()
        _start_rl_if_needed()
        # Return the first question produced by RL (which now handles greeting itself)
        question = _get_question_blocking()
        return jsonify({"subject_ID": subject_id, "question": question})

    _log_resp(user_input)
    question = _get_question_blocking()
    return jsonify({"subject_ID": subject_id, "question": question})

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    Returns "running" if RL thread is alive, otherwise "idle".
    """
    status = "running" if (_rl_thread is not None and _rl_thread.is_alive()) else "idle"
    return jsonify({"status": status})

if __name__ == "__main__":
    # Entry point for running the Flask server directly.
    # Host, port, and debug mode can be set via environment variables.
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "8080"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)