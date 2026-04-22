import time
from typing import Dict, Any

import numpy as np
import os
import pandas as pd

from src.questioner import ask_question
from src.CBT import run_cbt
from src.utils.config_loader import (
    ITEM_N_STATES,
    GAMMA,
    ALPHA,
    QUESTION_LIB_FILENAME,
    SUBJECT_ID,
    DATA_DIR,
)
from src.utils.config_loader import RECORD_CSV
from src.utils.io_question_lib import load_question_lib, save_question_lib, generate_results
from src.utils.io_record import init_record, log_question, set_question_prefix
from src.utils.rl_qtables import (
    initialize_q_table,
    choose_action,
    get_env_feedback,
)
# Set up logger for this module
from src.utils.log_util import get_logger
from src.utils.llm_client import llm_complete
logger = get_logger("HandlerRL")

class HandlerRL:
    """
    Top-level RL workflow coordinator.
    Handles the main reinforcement learning loop for question selection and evaluation.
    All file I/O is performed via utility modules.
    """

    def __init__(self):
        # Stores the last question asked to the user
        self.last_question: str = " "
        # Stores all user responses for later result generation
        self.new_response: list = []
        # The main question library loaded from file
        self.question_lib: Dict[str, Any] = {}
        # Q-table for item selection (top-level RL)
        self.item_q_table = None
        # Action id -> label mapping for logging readability
        self.item_action_labels = {}

    def setup(self):
        """
        Initialize records, load question library, and set up Q-tables and masks.
        """
        logger.info("Initializing RL handler setup: loading records and question library.")
        init_record()
        self.question_lib = load_question_lib(QUESTION_LIB_FILENAME)
        # Define possible actions for item selection (as string indices)
        item_actions = ['{0}'.format(e) for e in np.arange(0, ITEM_N_STATES)]
        # # Initialize masks and question-level Q-tables are deprecated; single-question per item is used
        # self.all_question_mask = {}
        # self.all_question_q_table = {}
        self.item_q_table = initialize_q_table(ITEM_N_STATES, item_actions)
        self.item_actions = item_actions

        # Build action id -> label mapping for logging readability
        # Action "0" is a synthetic start/index action and not part of the question lib
        self.item_action_labels = {"0": "INIT"}
        for i in range(1, ITEM_N_STATES):
            self.item_action_labels[str(i)] = self.question_lib[str(i)]["1"]["label"]
  
        # Load persistent Q tables (if exist)
        qdir = os.path.join(DATA_DIR, "q_tables")
        qfile = os.path.join(qdir, f"item_qtable_{SUBJECT_ID}.csv")
        if os.path.exists(qfile):
            self.item_q_table = pd.read_csv(qfile, index_col=0)
            logger.info(f"Loaded item Q table for subject {SUBJECT_ID} from {qfile}.")
        else:
            logger.info(f"Item Q table for subject {SUBJECT_ID} not found at {qfile}. ")
        
        logger.info("RL handler setup complete.")

    def run(self):
        """
        Main RL loop for the entire screening process.
        Iteratively selects items and asks questions using RL, updating Q-tables and saving results.
        """
        logger.info("Starting main RL screening process.")
        self.setup()

        # Opening greeting (LLM-rewritten) delivered before the first question for all interfaces
        try:
            greeting_raw = (
                "Hello, I'm CaiTI, your intelligence therapist. Thank you for joining me today. "
                "Let's get started with a couple of questions about your recent daily life."
            )
            rewrite_system_prompt = (
                "You are a warm, concise, and professional therapist-assistant.\n\n"
                "Task: Given an opening greeting, rewrite it to sound supportive, welcoming, and clear, "
                "suitable for the very first message of a therapeutic screening conversation.\n\nRules:\n"
                "- 1–2 short sentences.\n- Friendly, non-judgmental tone.\n"
                "- No extra headers or labels; output the final greeting directly.\n"
            )
            greeting = llm_complete(rewrite_system_prompt, greeting_raw).strip()
            # Use greeting as a prefix so the first substantive question appears immediately
            set_question_prefix(greeting)
            time.sleep(0.5)
        except Exception as e:
            # If LLM call fails, fall back to raw greeting prefix without blocking the flow
            logger.warning(f"Opening greeting rewrite failed: {e}")
            set_question_prefix(greeting_raw)
            time.sleep(0.5)
        new_q_table = self.item_q_table.copy()
        S = 0  # Start state for item RL
        is_terminated = False
        # Mask for available items (first item is always available)
        item_mask = [0] + [1] * (ITEM_N_STATES - 1)
        while not is_terminated:
            # If all items have been asked, exit to CBT directly
            if sum(item_mask) == 0:
                is_terminated = True
                logger.info("All items have been asked. Proceeding to CBT.")
                break
            # Select an item to ask about using RL policy
            A = choose_action(S, self.item_q_table, item_mask, ITEM_N_STATES, self.item_actions, self.item_action_labels)
            # Mark this item as used
            item_mask[int(A)] = 0
            # Ask questions for the selected item
            openai_res, DLA_terminate, last_question_updated = ask_question(self.question_lib, int(A))
            self.last_question = last_question_updated
            # Get next state and reward for item RL
            S_, R = get_env_feedback(S, A, openai_res, DLA_terminate, item_mask)
            # Q-learning update for item Q-table
            q_predict = self.item_q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * self.item_q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            new_q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            logger.debug(
                f"Q update applied at action: Q(S={S},A={A}) {q_predict} -> {new_q_table.loc[S, A]} (target={q_target})"
            )
            S = S_
            # If the DLA process signals termination, end the loop and save results
            if DLA_terminate == 1:
                # DLA process signaled termination; proceed to save artifacts
                is_terminated = True
                save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
                save_question_lib(save_filename, self.question_lib)
                logger.info(f"Saved question library to {save_filename} after DLA termination.")
                # log_question("Goodbye. We will do the screening in another time. 886")
                logger.info("Goodbye. We will do the screening in another time. 886")        # Save results if terminated
        if is_terminated:
            # Persist question library snapshot upon termination
            save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
            save_question_lib(save_filename, self.question_lib)
            logger.info(f"Saved question library to {save_filename} after session termination.")
            
            # Save Q tables (in parallel with existing results)
            qdir = os.path.join(DATA_DIR, "q_tables")
            qfile = os.path.join(qdir, f"item_qtable_{SUBJECT_ID}.csv")
            self.item_q_table = new_q_table
            dir_preexisted = os.path.exists(qdir)
            if not dir_preexisted:
                os.makedirs(qdir, exist_ok=True)
                logger.info(f"Created q_tables directory at {qdir}.")
            file_preexisted = os.path.exists(qfile)
            self.item_q_table.to_csv(qfile)
            if file_preexisted:
                logger.info(f"Updated item Q table for subject {SUBJECT_ID} at {qfile}.")
            else:
                logger.info(f"Created new item Q table for subject {SUBJECT_ID} at {qfile}.")

        # Run CBT after the screening loop concludes
        run_cbt(self.question_lib)
        logger.info("Completed CBT flow.")
        # Persist question_lib again to capture CBT notes
        save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
        save_question_lib(save_filename, self.question_lib)
        logger.info(f"Saved question library with CBT notes to {save_filename}.")

        # Generate final results for this session
        generate_results(self.question_lib, self.new_response)
        logger.info("Generated final results for this session.")

        # Deliver concluding message (LLM-generated) only if CBT was NOT used
        # If CBT ran, its own final message is the user-visible conclusion. Avoid double messages due to lock semantics.
        try:
            cbt_used, cbt_summary = self._detect_cbt_summary()
            if not cbt_used:
                sys_prompt = (
                    "You are a warm, concise, and professional therapist-assistant.\n\n"
                    "Background: This message appears at the end of a brief screening/CBT session.\n"
                    "Goal: Generate a short closing message for the user.\n\n"
                    "Inputs you may receive:\n"
                    "- cbt_used: whether CBT was conducted in this session (true/false).\n"
                    "- session_summary: brief bullet/lines from the session (if available).\n\n"
                    "Instructions:\n"
                    "- If cbt_used is true: Congratulate the user for working on CBT today, acknowledge their effort, and say goodbye.\n"
                    "- If cbt_used is false: Indicate there is no area of concern identified today and say goodbye.\n"
                    "- 1–2 sentences only.\n"
                    "- Friendly, non-judgmental tone.\n"
                    "- No headers or labels; output the final message directly.\n"
                )
                user_payload = (
                    f"cbt_used: {str(cbt_used).lower()}\n" + (f"session_summary:\n{cbt_summary}" if cbt_summary else "")
                )
                closing = llm_complete(sys_prompt, user_payload).strip()
                time.sleep(0.5)
                log_question(closing)
                time.sleep(0.5)
                self._unlock_question_if_stuck()
            else:
                logger.info("CBT delivered its own closing; skipping RL-level closing to avoid double message.")
        except Exception as e:
            logger.warning(f"Concluding message generation failed: {e}")
            # Only attempt fallback if CBT was not used
            cbt_used, _ = self._detect_cbt_summary()
            if not cbt_used:
                fallback = "Thank you for your time today. Take care, and goodbye."
                time.sleep(0.5)
                log_question(fallback)
                time.sleep(0.5)
                self._unlock_question_if_stuck()

    def _detect_cbt_summary(self) -> tuple:
        """Return (cbt_used, summary_str) by scanning question_lib notes for CBT markers."""
        try:
            lines = []
            cbt_used = False
            for i in range(1, len(self.question_lib) + 1):
                for j in range(1, len(self.question_lib[str(i)]) + 1):
                    entry = self.question_lib[str(i)][str(j)]
                    notes = entry.get("notes", [])
                    for note in notes:
                        if isinstance(note, list) and any((isinstance(x, str) and x.startswith("CBT_")) for x in note):
                            cbt_used = True
                            for x in note:
                                if isinstance(x, str) and (
                                    x.startswith("CBT_dimension:") or
                                    x.startswith("CBT_statement:") or
                                    x.startswith("CBT_unhelpful_thoughts:") or
                                    x.startswith("CBT_challenge:") or
                                    x.startswith("CBT_reframe:") or
                                    x.startswith("CBT_stage:")
                                ):
                                    lines.append(x)
            summary = "\n".join(lines[-8:]) if lines else ""
            return cbt_used, summary
        except Exception:
            return False, ""

    def _unlock_question_if_stuck(self) -> None:
        """If Question_Lock remains set after a system message, clear it to avoid blocking."""
        try:
            df = pd.read_csv(RECORD_CSV)
            if int(df.loc[0, "Question_Lock"]) == 1:
                df.loc[0, "Question_Lock"] = 0
                tmp_path = RECORD_CSV + ".tmp"
                df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, RECORD_CSV)
                logger.info("Force-unlocked Question_Lock after system message.")
        except Exception as e:
            logger.warning(f"Failed to force-unlock Question_Lock: {e}")