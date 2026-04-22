"""Domain logic managing Reiforcement Learning (RL) conversational flows."""
import io
import json
import time
from typing import Dict, Any, List

import numpy as np
import os
import pandas as pd

from src.core.questioner import ask_question
from src.core.CBT import run_cbt
from src.core.therapy_content import (
    CLINICAL_SCREENING,
    CRITICAL_DIMS,
    GAD2_THRESHOLD,
    PHQ4_THRESHOLD,
    SAFETY_RESOURCES_MESSAGE,
    score_response,
)
from src.utils.config_loader import (
    ITEM_N_STATES,
    GAMMA,
    ALPHA,
    EPSILON,
    ITEM_IMPORTANCE,
    QUESTION_LIB_FILENAME,
    SUBJECT_ID,
    DATA_DIR,
)
from src.utils.config_loader import RECORD_CSV
from src.utils.io_question_lib import load_question_lib, save_question_lib, generate_results
from src.utils.io_record import init_record, log_question, set_question_prefix, dump_session_history_to_terminal
import src.utils.io_record as io_rec
from src.utils.rl_qtables import (
    initialize_q_table,
    choose_action,
    get_env_feedback,
)
# Set up logger for this module
from src.utils.log_util import get_logger
from src.utils.resource_audit import get_resource_audit
from src.models.llm_client import llm_complete
logger = get_logger("HandlerRL")
_RESOURCE_AUDIT = get_resource_audit()

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
        # Crisis override state
        self._crisis_triggered: bool = False
        self._crisis_dim: str = ""
        # Longitudinal memory: top Score-2 dimensions from previous session,
        # loaded in setup() for warm-start + recall greeting.
        self._prior_score2_dims: List[str] = []
        self._is_returning_user: bool = False

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

        # Longitudinal memory: load persistent RL state from DB and warm-start
        # the Q-table by boosting dimensions that were scored at 2 in the most
        # recent session.  This implements the paper's "Recall and Resume"
        # protocol — returning users start with their prior problematic areas
        # pre-weighted so the RL agent revisits them first.
        self._load_longitudinal_state()

        logger.info("RL handler setup complete.")

    def _load_longitudinal_state(self):
        """Warm-start Q-table from persistent per-user DB state."""
        if not io_rec.DB:
            logger.info("No DB available; skipping longitudinal warm-start.")
            return
        try:
            user_id = io_rec.DB.get_user_id(SUBJECT_ID)
            state = io_rec.DB.load_rl_state(user_id)
        except Exception as e:
            logger.warning(f"Longitudinal state load failed: {e}")
            return

        if not state:
            logger.info(f"Subject {SUBJECT_ID}: first-time user, no longitudinal state.")
            return

        self._is_returning_user = True
        logger.info(f"Subject {SUBJECT_ID}: returning user — applying warm-start.")

        # 1. Restore persisted Q-table if present.  This takes precedence over
        # the CSV (CSV is still written for backwards compatibility).
        q_json = state.get("q_table_json")
        if q_json:
            try:
                # Newer pandas requires a file-like wrapper for read_json.
                restored = pd.read_json(io.StringIO(q_json), orient="split")
                # Ensure column/index shapes match the current item space.
                if restored.shape == self.item_q_table.shape:
                    restored.columns = restored.columns.astype(str)
                    self.item_q_table = restored
                    logger.info("Warm-started Q-table from persistent DB state.")
                else:
                    logger.warning(
                        f"Persisted Q-table shape {restored.shape} does not match "
                        f"current {self.item_q_table.shape}; ignoring."
                    )
            except Exception as e:
                logger.warning(f"Failed to restore Q-table from DB: {e}")

        # 2. Load top Score-2 dimensions from last session.  The saved format
        # is a list of {"label": ..., "name": ...} dicts; we flatten to just
        # the lower-cased label strings here.
        dims_json = state.get("top_score2_dims_json")
        if dims_json:
            try:
                dims = json.loads(dims_json)
                if isinstance(dims, list):
                    labels: List[str] = []
                    for d in dims:
                        if isinstance(d, dict):
                            lbl = str(d.get("label", "")).strip().lower()
                            if lbl:
                                labels.append(lbl)
                        elif isinstance(d, str) and d.strip():
                            labels.append(d.strip().lower())
                    self._prior_score2_dims = labels
            except Exception as e:
                logger.warning(f"Failed to parse top_score2_dims_json: {e}")

        # 3. Dimensional weighting bonus: boost Q-values for state indices
        # whose label matches a prior Score-2 dimension so choose_action
        # prefers them early in the session.
        if self._prior_score2_dims:
            boost = 3.0  # additive bonus on top of ITEM_IMPORTANCE base weight
            for i_key in self.question_lib.keys():
                try:
                    label = str(self.question_lib[i_key]["1"].get("label", "")).lower()
                except Exception:
                    continue
                if label in self._prior_score2_dims and i_key in self.item_q_table.columns:
                    self.item_q_table[i_key] = self.item_q_table[i_key] + boost
            logger.info(
                f"Boosted Q-values for prior Score-2 dimensions: {self._prior_score2_dims}"
            )

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
            user_name = str(getattr(io_rec, "SUBJECT_ID", "User") or "User")
            identity_guard = (
                "Identity Rules:\n"
                "- AI_NAME: CaiTI\n"
                f"- USER_NAME: {user_name}\n"
                "- You are CaiTI. The user is USER_NAME.\n"
                "- Never confuse the two identities.\n"
            )
            
            # Reasoner decision: determine if this is a returning user with
            # sufficient prior context for a recall-and-resume greeting.
            is_returning = bool(user_ctx and len(user_ctx) > 50)
            io_rec.log_reasoning("reasoner_decision", {
                "component": "greeting",
                "decision": "returning_user" if is_returning else "new_user",
                "user_ctx_length": len(user_ctx) if user_ctx else 0,
                "prior_score2_dims": self._prior_score2_dims,
            })

            if is_returning:
                # Recall-and-Resume protocol: if the user had Score-2 dimensions
                # last session, name the top one (human-friendly label) so the
                # greeting signals continuity of care.
                recall_hint = ""
                if self._prior_score2_dims:
                    top_label = self._prior_score2_dims[0]
                    top_human = top_label
                    for i_key in self.question_lib.keys():
                        entry = self.question_lib[i_key].get("1", {})
                        if str(entry.get("label", "")).lower() == top_label:
                            top_human = entry.get("name", top_label)
                            break
                    recall_hint = (
                        f"\nRecall hint: last session the user struggled most with "
                        f"'{top_human}'. Naturally ask how that has been since you "
                        f"last spoke.\n"
                    )

                rewrite_system_prompt = (
                    "You are a warm, concise, and professional therapist-assistant.\n\n"
                    f"{identity_guard}\n"
                    "Task: Generate a welcoming opening greeting for a returning user. Transition into starting a new session.\n"
                    f"Here is the context from their previous sessions:\n{user_ctx}\n"
                    f"{recall_hint}\n"
                    "Rules:\n"
                    "- Briefly and naturally acknowledge a detail from their past session summary to show you remember them.\n"
                    "- If a 'Recall hint' is provided, gently check in on that topic in ONE short sentence.\n"
                    "- Do not list out their preferences mechanically. Just weave it into the 'Welcome back' if relevant.\n"
                    "- 2–3 short sentences maximum.\n- Friendly, non-judgmental tone.\n"
                    "- No extra headers or labels; output the final greeting directly.\n"
                )
            else:
                rewrite_system_prompt = (
                    "You are a warm, concise, and professional therapist-assistant.\n\n"
                    f"{identity_guard}\n"
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
        # ── Interactive PHQ-4 / GAD-2 Clinical Screening ─────────────────
        # Run before the RL loop so scores can inform question selection.
        phq4_result = self._run_phq4_screening()
        if phq4_result is None:
            # Session was interrupted during screening
            logger.info("Session interrupted during PHQ-4 screening. Exiting.")
            generate_results(self.question_lib, self.new_response)
            dump_session_history_to_terminal()
            return

        # RL feedback: if PHQ-4 total >= threshold, shift to crisis/meditation mode
        if phq4_result.get("phq4_high_risk"):
            crisis_msg = (
                "Thank you for sharing that with me. Based on your responses, "
                "it seems like things have been quite difficult lately. "
                "I want you to know that support is available, and it's okay to reach out. "
                "Let's continue our conversation with extra care."
            )
            set_question_prefix(crisis_msg)
            logger.warning("[CRISIS MODE] PHQ-4 high risk detected. Therapeutic tone elevated.")

        new_q_table = self.item_q_table.copy()
        S = 0  # Start state for item RL
        is_terminated = False
        # Mask for available items (first item is always available)
        item_mask = [0] + [1] * (ITEM_N_STATES - 1)
        turn_idx = 0

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

            # Immediate re-screening for returning users: the paper requires
            # bypassing the standard epsilon-greedy exploration for the first
            # 1-2 turns and force-targeting previously problematic dimensions
            # (Score 2 last session) to see if they have improved.
            A = None
            if (
                self._is_returning_user
                and turn_idx < 2
                and self._prior_score2_dims
            ):
                for prior_dim in self._prior_score2_dims:
                    for i_key in self.question_lib.keys():
                        try:
                            label = str(self.question_lib[i_key]["1"].get("label", "")).lower()
                        except Exception:
                            continue
                        if label != prior_dim:
                            continue
                        try:
                            idx = int(i_key)
                        except ValueError:
                            continue
                        if 0 < idx < ITEM_N_STATES and item_mask[idx] == 1:
                            A = str(idx)
                            logger.info(
                                f"[RESUME] Force-targeting prior Score-2 dim '{prior_dim}' "
                                f"on turn {turn_idx} (bypassing epsilon-greedy)."
                            )
                            break
                    if A is not None:
                        break

            if A is None:
                # Select an item to ask about using fixed epsilon (paper/legacy aligned)
                A = choose_action(
                    S, self.item_q_table, item_mask, ITEM_N_STATES,
                    self.item_actions, self.item_action_labels,
                    epsilon=EPSILON,
                )

            # Log the RL's internal logical state to the backend database before proceeding
            q_vals = self.item_q_table.loc[S].to_dict()
            io_rec.log_reasoning("rl_decision", {
                "state": S,
                "action_chosen": A,
                "available_mask": item_mask,
                "q_values": q_vals,
                "epsilon": EPSILON,
                "turn_idx": turn_idx,
            })
            io_rec.set_rl_context({"state": S, "action_chosen": A, "available_mask": item_mask, "q_values": q_vals})
            
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
            turn_idx += 1

            # Crisis override: if ANY dimension is now scored at 2 AND that
            # dimension is clinically critical, deliver the safety message
            # immediately and pre-select that dimension as the CBT focus.
            # Re-checks every turn because multi-dim back-fill can record a
            # Score 2 on a critical dimension that was not the primary ask.
            if self._crisis_scan() and not self._crisis_triggered:
                self._crisis_triggered = True
                logger.warning("[CRISIS OVERRIDE] Critical-dim Score 2 detected. Delivering safety resources.")
                io_rec.log_reasoning("crisis_override", {
                    "triggered_at_turn": turn_idx,
                    "critical_dim": self._crisis_dim,
                })
                log_question(SAFETY_RESOURCES_MESSAGE)
                # Short-circuit to CBT so the session focuses on the safety topic.
                is_terminated = True
                break
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

            # Longitudinal persistence: authoritative per-user RL snapshot so
            # future sessions can recall prior Score-2 dimensions and warm-start
            # the Q-table with dimensional weighting.
            self._save_longitudinal_state(item_mask)

        # Run CBT after the screening loop concludes if not interrupted.
        # When a crisis override fired during the screening loop, surface the
        # critical dimension to the CBT selector as the default focus.
        if not io_rec.END_SESSION_EVENT.is_set():
            if self._crisis_triggered and self._crisis_dim:
                logger.info(
                    f"[CRISIS] Routing CBT to crisis dimension '{self._crisis_dim}'."
                )
                try:
                    io_rec.log_reasoning("cbt_crisis_routing", {"dim": self._crisis_dim})
                except Exception:
                    pass
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

        # Deliver concluding message — skip LLM if session was interrupted for prompt exit
        if io_rec.END_SESSION_EVENT.is_set():
            logger.info("Session was interrupted. Skipping LLM-generated closing message.")
        else:
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
                cbt_used, _ = self._detect_cbt_summary()
                if not cbt_used:
                    fallback = "Thank you for your time today. Take care, and goodbye."
                    time.sleep(0.5)
                    log_question(fallback)
                    time.sleep(0.5)
                    self._unlock_question_if_stuck()

        # Perform Session Analysis — skip LLM calls if session was interrupted
        if io_rec.END_SESSION_EVENT.is_set():
            logger.info("Session was interrupted. Skipping post-session LLM analysis for prompt exit.")
            dump_session_history_to_terminal()
        else:
            try:
                self._generate_clinical_summary()
                self._generate_session_analysis()
                dump_session_history_to_terminal()
            except Exception as e:
                logger.error(f"Post-session analysis failed: {e}")
            
        # ── Forensic report: emit immutable session resource snapshot ──────
        try:
            _RESOURCE_AUDIT.capture_point("session_end")
            report_path = _RESOURCE_AUDIT.write_report()
            if report_path:
                logger.info(f"[FORENSIC] Session resource report written to {report_path}")
        except Exception as e:
            logger.warning(f"Failed to write session forensic report: {e}")

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
                        if not ftype:
                            logger.warning(f"Empty safety flag type in line: {line}")
                            continue
                        try:
                            severity = int(parts[-1].strip())
                        except ValueError:
                            logger.warning(f"Non-integer severity in safety flag: {line}")
                            continue
                        if severity < 1 or severity > 5:
                            logger.warning(f"Safety flag severity {severity} out of range 1-5, clamping.")
                            severity = max(1, min(5, severity))
                        raw = ":".join(parts[1:-1]).strip()
                        io_rec.DB.log_safety_flag(io_rec.SESSION_ID, ftype, raw, severity)
                        logger.warning(f"Logged SAFETY FLAG: {ftype} ({severity})")

        except Exception as e:
            logger.error(f"Session analysis failed: {e}")

    def _generate_clinical_summary(self):
        """
        Generate a SOAP-formatted clinical session report at session end.

        SOAP is the standard format used in licensed-therapist session notes:
            Subjective   — the client's reported experience in their own words.
            Objective    — observable / measured data (PHQ-4, GAD-2, dimensional
                           scores 0-2 flagged as problematic).
            Assessment   — clinician's interpretation of problematic dimensions,
                           patterns, and risk status.
            Intervention — what was delivered this session (R-V validations,
                           CBT stages reached, crisis routing, next-session focus).
        """
        if not io_rec.DB or not io_rec.SESSION_ID:
            logger.warning("DB or Session ID not available for clinical summary.")
            return

        history = io_rec.DB.get_session_history(io_rec.SESSION_ID)
        if not history:
            return

        hist_text = "\n".join([f"{h['speaker']}: {h['text']}" for h in history])
        screening = io_rec.DB.get_screening_scores(io_rec.SESSION_ID) or {}
        anxiety = screening.get("anxiety")
        depression = screening.get("depression")
        total = screening.get("total")

        # Collect per-dimension objective scores from the session's question_lib.
        dim_rows = []
        for i_key in self.question_lib.keys():
            entry = self.question_lib[i_key].get("1", {})
            label = entry.get("label", "")
            name = entry.get("name", label)
            scores = [s for s in entry.get("score", []) if isinstance(s, int) and 0 <= s <= 2]
            if scores:
                dim_rows.append(f"- {name} ({label}): score={max(scores)}")
        dim_block = "\n".join(dim_rows) if dim_rows else "- (no dimensional scores recorded)"

        cbt_used, cbt_notes = self._detect_cbt_summary()
        crisis_line = (
            f"Crisis override triggered on dimension '{self._crisis_dim}'."
            if self._crisis_triggered else "No crisis override triggered."
        )

        prompt = (
            "Create a SOAP-format clinical session note for therapist handoff.\n"
            "Output EXACTLY with these four headers and no others:\n\n"
            "SUBJECTIVE:\n"
            "- <client's reported experience in their own phrasing, 2-4 bullets>\n\n"
            "OBJECTIVE:\n"
            "- PHQ-4 total: <value>\n"
            "- GAD-2 sub-total: <value>\n"
            "- PHQ-2 sub-total: <value>\n"
            "- Dimensional scores 0-2 (problematic only):\n"
            "  <one bullet per dimension with a non-zero score>\n\n"
            "ASSESSMENT:\n"
            "- <clinician interpretation of the most clinically significant "
            "findings, 2-3 bullets>\n"
            "- <risk status line, e.g. 'No self-harm indicators' or 'Crisis override fired'>\n\n"
            "INTERVENTION:\n"
            "- <Reflection-Validation moments used>\n"
            "- <CBT stages reached: Recognize / Challenge / Reframe>\n"
            "- <Recommended focus for the next session>\n\n"
            "Keep it concise, specific, and clinically neutral.\n"
            "Use ASCII characters only.\n\n"
            f"Screening snapshot => GAD-2:{anxiety}, PHQ-2:{depression}, PHQ-4:{total}\n"
            f"Problematic dimensional scores:\n{dim_block}\n"
            f"CBT used this session: {cbt_used}\n"
            f"CBT notes:\n{cbt_notes or '(none)'}\n"
            f"Safety: {crisis_line}\n\n"
            f"Session History:\n{hist_text}"
        )

        summary = llm_complete("You are a clinical documentation assistant.", prompt).strip()
        if summary:
            io_rec.DB.add_summary(io_rec.SESSION_ID, summary)
            log_question(summary)
            logger.info("SOAP-format clinical report generated and stored.")

    def _save_longitudinal_state(self, item_mask: list) -> None:
        """Persist Q-table, item_mask and top Score-2 dims for this session."""
        if not io_rec.DB:
            return
        try:
            user_id = io_rec.DB.get_user_id(SUBJECT_ID)
            # Collect labels of dimensions that hit Score 2 this session,
            # ordered by importance weight (descending) so the top-3 most
            # clinically important problematic dims are surfaced first in
            # the next session's recall greeting.
            score2_entries = []
            for i_key in self.question_lib.keys():
                try:
                    i_int = int(i_key)
                except ValueError:
                    continue
                entry = self.question_lib[i_key].get("1", {})
                label = str(entry.get("label", "")).lower()
                if not label:
                    continue
                if any((isinstance(s, int) and s == 2) for s in entry.get("score", [])):
                    imp = ITEM_IMPORTANCE[i_int] if i_int < len(ITEM_IMPORTANCE) else 0
                    score2_entries.append((imp, label, entry.get("name", label)))
            score2_entries.sort(reverse=True)
            top_dims = [{"label": lbl, "name": nm} for (_, lbl, nm) in score2_entries[:5]]

            q_json = self.item_q_table.to_json(orient="split")
            mask_json = json.dumps(item_mask)
            dims_json = json.dumps(top_dims)

            io_rec.DB.save_rl_state(
                user_id=user_id,
                q_table_json=q_json,
                item_mask_json=mask_json,
                top_score2_dims_json=dims_json,
                last_session_id=io_rec.SESSION_ID,
            )
            logger.info(
                f"Persisted longitudinal RL state for user_id={user_id}: "
                f"{len(top_dims)} Score-2 dimensions recorded."
            )
        except Exception as e:
            logger.warning(f"Longitudinal state save failed: {e}")

    def _crisis_scan(self) -> bool:
        """Return True if any CRITICAL_DIMS entry now has a score of 2.

        Side-effect: records the first matching dimension label in
        `self._crisis_dim` so the caller can prioritise it for CBT and
        persist a safety flag.
        """
        try:
            for i_key in self.question_lib.keys():
                entry = self.question_lib[i_key].get("1", {})
                label = str(entry.get("label", "")).lower()
                if label not in CRITICAL_DIMS:
                    continue
                if any((isinstance(s, int) and s == 2) for s in entry.get("score", [])):
                    self._crisis_dim = label
                    # Best-effort safety flag persistence
                    if io_rec.DB and io_rec.SESSION_ID:
                        try:
                            io_rec.DB.log_safety_flag(
                                io_rec.SESSION_ID,
                                flag_type=f"CRITICAL_DIM:{label}",
                                raw_text=f"Score 2 detected on critical dimension '{label}'",
                                severity=5,
                            )
                        except Exception as e:
                            logger.warning(f"Could not persist safety flag: {e}")
                    return True
        except Exception as e:
            logger.warning(f"crisis scan failed: {e}")
        return False

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

    def _run_phq4_screening(self) -> dict:
        """
        Interactive PHQ-4 / GAD-2 clinical screening loop.
        Asks all 4 standardised questions, scores each response (0-3),
        persists to DB immediately, and returns the score snapshot.

        Returns dict with keys: anxiety, depression, total, phq4_high_risk, gad2_positive.
        Returns None if the session is interrupted.
        """
        logger.info("[PHQ-4] Starting interactive clinical screening.")
        anxiety_scores = []
        depression_scores = []

        options_hint = "You can answer: Not at all, Several days, More than half the days, or Nearly every day."

        for i, q in enumerate(CLINICAL_SCREENING):
            if io_rec.END_SESSION_EVENT.is_set():
                logger.info("[PHQ-4] Session interrupted during screening.")
                return None

            # Present the question with the Likert options hint
            question_text = f"{q['text']}\n{options_hint}"
            log_question(question_text)

            # Wait for user response
            user_response = io_rec.get_resp_log()
            if user_response == "SESSION_END":
                logger.info("[PHQ-4] Session ended during screening.")
                return None

            clean_resp = user_response.strip()

            # Check for opt-out
            if any(kw in clean_resp.lower() for kw in ("skip", "don't want", "opt out", "no thanks", "stop")):
                logger.info(f"[PHQ-4] User opted out at question {i+1} ({q['id']}).")
                io_rec.log_reasoning("phq4_screening", {
                    "status": "opted_out",
                    "opted_out_at": q["id"],
                    "anxiety_scores": anxiety_scores,
                    "depression_scores": depression_scores,
                })
                # Mark this and all remaining questions as SKIPPED in
                # the intermission tracker so the ladder never re-asks.
                if io_rec.DB and io_rec.SESSION_ID:
                    for remaining in CLINICAL_SCREENING[i:]:
                        try:
                            io_rec.DB.upsert_intermission_screening_status(
                                session_id=io_rec.SESSION_ID,
                                question_id=remaining["id"],
                                status="SKIPPED",
                                reason="phq4_opt_out",
                            )
                        except Exception:
                            pass
                break

            # Score the response
            score = score_response(clean_resp)
            if score == -1:
                logger.info(f"[PHQ-4] Refusal detected at question {i+1} ({q['id']}).")
                # Mark remaining questions as SKIPPED for the intermission ladder
                if io_rec.DB and io_rec.SESSION_ID:
                    for remaining in CLINICAL_SCREENING[i:]:
                        try:
                            io_rec.DB.upsert_intermission_screening_status(
                                session_id=io_rec.SESSION_ID,
                                question_id=remaining["id"],
                                status="SKIPPED",
                                reason="phq4_refusal",
                            )
                        except Exception:
                            pass
                break

            logger.info(f"[PHQ-4] {q['id']}: response='{clean_resp}' -> score={score}")

            if q["scale"] == "anxiety":
                anxiety_scores.append(score)
            else:
                depression_scores.append(score)

            # Persist incrementally after each answer
            anxiety_total = sum(anxiety_scores) if anxiety_scores else None
            depression_total = sum(depression_scores) if depression_scores else None
            phq4_total = (anxiety_total or 0) + (depression_total or 0)

            io_rec.set_latest_screening_scores(anxiety_total, depression_total, phq4_total)
            if io_rec.DB and io_rec.SESSION_ID:
                try:
                    io_rec.DB.log_screening_scores(
                        io_rec.SESSION_ID,
                        anxiety_score=anxiety_total,
                        depression_score=depression_total,
                        phq4_total=phq4_total,
                    )
                except Exception as e:
                    logger.warning(f"[PHQ-4] Failed to persist score: {e}")

                # Mark question as ANSWERED in the intermission tracker DB so
                # the speech-service intermission ladder never re-asks it.
                try:
                    io_rec.DB.upsert_intermission_screening_status(
                        session_id=io_rec.SESSION_ID,
                        question_id=q["id"],
                        status="ANSWERED",
                        score=score,
                        response_text=clean_resp,
                        reason="phq4_screening",
                    )
                except Exception as e:
                    logger.warning(f"[PHQ-4] Failed to sync intermission status for {q['id']}: {e}")

            io_rec.log_reasoning("phq4_response", {
                "question_id": q["id"],
                "response": clean_resp,
                "score": score,
                "running_anxiety": anxiety_total,
                "running_depression": depression_total,
                "running_total": phq4_total,
            })

        # Final scores
        anxiety_total = sum(anxiety_scores) if anxiety_scores else 0
        depression_total = sum(depression_scores) if depression_scores else 0
        phq4_total = anxiety_total + depression_total
        gad2_positive = anxiety_total >= GAD2_THRESHOLD
        phq4_high_risk = phq4_total >= PHQ4_THRESHOLD

        result = {
            "anxiety": anxiety_total,
            "depression": depression_total,
            "total": phq4_total,
            "gad2_positive": gad2_positive,
            "phq4_high_risk": phq4_high_risk,
        }

        # Log clinical flags
        if gad2_positive:
            logger.warning(f"[CLINICAL-FLAG] GAD2_POSITIVE — anxiety={anxiety_total} >= {GAD2_THRESHOLD}")
            io_rec.append_to_csv("clinical_flag", "system", f"GAD2_POSITIVE: anxiety={anxiety_total}")
        if phq4_high_risk:
            logger.warning(f"[CLINICAL-FLAG] PHQ4_HIGH_RISK — total={phq4_total} >= {PHQ4_THRESHOLD}")
            io_rec.append_to_csv("clinical_flag", "system", f"PHQ4_HIGH_RISK: total={phq4_total}")

        io_rec.log_reasoning("phq4_screening", {
            "status": "completed",
            **result,
        })

        logger.info(f"[PHQ-4] Screening complete: {result}")
        return result