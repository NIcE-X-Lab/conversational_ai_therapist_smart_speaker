# LLM_therapist_Application.py

import os
import time
import threading
import pandas as pd

from src.handler_rl import HandlerRL
from src.utils.io_record import init_record
from src.utils.config_loader import RECORD_CSV

# The columns used in the record CSV file for question/response exchange
HEADER = ["Question", "Question_Lock", "Resp", "Resp_Lock"]

def _atomic_write_record(df: pd.DataFrame):
    """
    Atomically write the DataFrame to the record CSV file.
    This prevents partial writes and ensures the file is always in a valid state.
    """
    folder = os.path.dirname(RECORD_CSV)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    tmp_path = RECORD_CSV + ".tmp"
    df.to_csv(tmp_path, columns=HEADER, index=False)
    os.replace(tmp_path, RECORD_CSV)

def console_io_loop():
    """
    Console I/O loop running in a separate thread.
    - Waits for a new question to appear in the CSV (Question_Lock == 1)
    - Prints the question to the terminal and prompts the user for input
    - Writes the user's response back to the CSV and unlocks the response
    """
    while True:
        time.sleep(0.1)
        try:
            df = pd.read_csv(RECORD_CSV, dtype={"Question": str, "Question_Lock": "int64", "Resp": str, "Resp_Lock": "int64"})
        except Exception:
            # Wait for the main process to initialize the record file
            continue

        # If a new question is available (locked), display it and get user input
        if int(df.loc[0, "Question_Lock"]) == 1:
            q = str(df.loc[0, "Question"])
            print(f"\nQUESTION: {q}", flush=True)
            print("Your answer: ", end="", flush=True)
            # Unlock the question so the main process can write the next one
            df.loc[0, "Question_Lock"] = 0
            _atomic_write_record(df)

            try:
                user_input = input()
            except KeyboardInterrupt:
                print("\n[Console IO] Interrupted by user.")
                break

            # Write the user's response and unlock the response field
            df = pd.read_csv(RECORD_CSV, dtype={"Question": str, "Question_Lock": "int64", "Resp": str, "Resp_Lock": "int64"})
            df.loc[0, "Resp"] = user_input
            df.loc[0, "Resp_Lock"] = 0
            _atomic_write_record(df)

def main():
    """
    Main entry point for the application.
    - Initializes the record CSV file
    - Starts the console I/O thread for user interaction
    - Runs the main RL-based therapist workflow
    """
    # Initialize the record file for question/response exchange
    init_record()

    # Start the console I/O thread (daemon so it exits with the main process)
    t = threading.Thread(target=console_io_loop, daemon=True)
    t.start()

    # Start the main RL workflow (this will drive the therapy session)
    HandlerRL().run()

    # Give the console thread a moment to finish up before exiting
    time.sleep(0.3)

if __name__ == "__main__":
    main()