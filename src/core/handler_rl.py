"""Domain logic managing Reiforcement Learning (RL) conversational flows."""
import time
from typing import Dict, Any

import numpy as np
import os
import pandas as pd

from src.core.questioner import ask_question
from src.core.CBT import run_cbt
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
import src.utils.io_record as io_rec
from src.utils.rl_qtables import (
    initialize_q_table,
    choose_action,
    get_env_feedback,
)
# Set up logger for this module
from src.utils.log_util import get_logger
from src.models.llm_client import llm_complete
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
            greeting_raw = "Hello, I'm CaiTI."
            user_ctx = io_rec.get_user_context()
            
            if user_ctx:
                rewrite_system_prompt = (
                    "You are a warm, concise, and professional therapist-assistant.\n\n"
                    "Task: Generate a welcoming opening greeting for a returning user. Transition into starting a new session.\n"
                    f"Here is the context from their previous sessions:\n{user_ctx}\n\n"
                    "Rules:\n"
                    "- Briefly and naturally acknowledge a detail from their past session summary to show you remember them.\n"
                    "- Do not list out their preferences mechanically. Just weave it into the 'Welcome back' if relevant.\n"
                    "- 2–3 short sentences maximum.\n- Friendly, non-judgmental tone.\n"
                    "- No extra headers or labels; output the final greeting directly.\n"
                )
            else:
                rewrite_system_prompt = (
                    "You are a warm, concise, and professional therapist-assistant.\n\n"
                    "Task: Generate a welcoming opening greeting for a user. Transition into starting the first session.\n"
                    "Rules:\n"
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
            set_question_prefix("Hello, I'm CaiTI. Let's get started with a couple of questions about your recent daily life.")
            time.sleep(0.5)
        new_q_table = self.item_q_table.copy()
        S = 0  # Start state for item RL
        is_terminated = False
        # Mask for available items (first item is always available)
        item_mask = [0] + [1] * (ITEM_N_STATES - 1)
        while not is_terminated:
            if io_rec.END_SESSION_EVENT.is_set():
                logger.info("Session Interrupted (End Session Event). Committing Q-Tables early.")
                is_terminated = True
                break

            # If all items have been asked, exit to CBT directly
            if sum(item_mask) == 0:
                is_terminated = True
                logger.info("All items have been asked. Proceeding to CBT.")
                break
            # Select an item to ask about using RL policy
            A = choose_action(S, self.item_q_table, item_mask, ITEM_N_STATES, self.item_actions, self.item_action_labels)
            
            # Log the RL's internal logical state to the backend database before proceeding
            q_vals = self.item_q_table.loc[S].to_dict()
            io_rec.log_reasoning("rl_decision", {"state": S, "action_chosen": A, "available_mask": item_mask, "q_values": q_vals})
            
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

        # Run CBT after the screening loop concludes if not interrupted
        if not io_rec.END_SESSION_EVENT.is_set():
            run_cbt(self.question_lib)
            logger.info("Completed CBT flow.")
            # Persist question_lib again to capture CBT notes
            save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
            save_question_lib(save_filename, self.question_lib)
            logger.info(f"Saved question library with CBT notes to {save_filename}.")
        else:
            logger.info("Session was early terminated. Skipping CBT workflow.")

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

        # Perform Session Analysis (Summaries, Prefs, Safety)
        try:
            self._generate_session_analysis()
        except Exception as e:
            logger.error(f"Post-session analysis failed: {e}")
            
        # Ensure deep memory release of the ML frames to prevent user leakage
        del self.item_q_table
        logger.info("Garbage collected ML DataFrame for HandlerRL Context wipe.")

    def _generate_session_analysis(self):
        """
        Analyze the session history for summary, preferences, and safety flags.
        Stores them in the DB.
        """
        if not io_rec.DB or not io_rec.SESSION_ID:
            logger.warning("DB or Session ID not available for analysis.")
            return

        history = io_rec.DB.get_session_history(io_rec.SESSION_ID)
        if not history:
            return

        # Format history string
        hist_text = "\\n".join([f"{h['speaker']}: {h['text']}" for h in history])
        
        prompt = (
            "Analyze the following therapy session history:\\n"
            f"{hist_text}\\n\\n"
            "Tasks:\\n"
            "1. SUMMARY: Provide a brief 2-3 sentence summary of the session's key topics and user state.\\n"
            "2. PREFERENCES: Extract any specific user preferences or facts mentioned (e.g., likes shopping, dislikes crowds). Format: KEY: VALUE\\n"
            "3. SAFETY_FLAGS: Identify any potential safety risks (e.g., self-harm, violence). If none, say NONE.\\n"
            "   Severity scale: 1 (mild) to 5 (critical).\\n\\n"
            "Response Format:\\n"
            "SUMMARY: <summary text>\\n"
            "PREFERENCES:\\n- <key>: <value>\\n"
            "SAFETY_FLAGS:\\n- <type>: <text>: <severity>\\n"
        )
        
        try:
            analysis = llm_complete("You are a clinical supervisor analyzing session notes.", prompt)
            
            # Parse and Store
            current_section = None
            for line in analysis.split('\\n'):
                line = line.strip()
                if not line: continue
                
                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()
                    if summary:
                        io_rec.DB.add_summary(io_rec.SESSION_ID, summary)
                        logger.info(f"Stored summary: {summary}")
                    current_section = "SUMMARY"
                elif line.startswith("PREFERENCES:"):
                    current_section = "PREFERENCES"
                elif line.startswith("SAFETY_FLAGS:"):
                    current_section = "SAFETY_FLAGS"
                elif line.startswith("-") and current_section == "PREFERENCES":
                    # Parse preference "Key: Value"
                    parts = line.replace("-", "").strip().split(":", 1)
                    if len(parts) == 2:
                        k, v = parts[0].strip(), parts[1].strip()
                        user_id = io_rec.DB.get_user_id(SUBJECT_ID)
                        io_rec.DB.set_preference(user_id, k, v)
                        logger.info(f"Stored preference: {k}={v}")
                elif line.startswith("-") and current_section == "SAFETY_FLAGS":
                    # Parse safety flag "Type: Text: Severity"
                    parts = line.replace("-", "").strip().split(":")
                    if len(parts) >= 3:
                        ftype = parts[0].strip()
                        try:
                            severity = int(parts[-1].strip())
                            raw = ":".join(parts[1:-1]).strip()
                            io_rec.DB.log_safety_flag(io_rec.SESSION_ID, ftype, raw, severity)
                            logger.warning(f"Logged SAFETY FLAG: {ftype} ({severity})")
                        except ValueError:
                            pass

        except Exception as e:
            logger.error(f"Session analysis failed: {e}")

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
        """Legacy method for IPC lock. No longer needed with full transcript logging."""
        pass