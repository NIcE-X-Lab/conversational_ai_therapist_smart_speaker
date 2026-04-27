"""Domain logic managing Reiforcement Learning (RL) conversational flows.

Q-table persistence contract (paper §5.1 vs. current deployment)
----------------------------------------------------------------
Paper/legacy: per-subject Q-table is persisted to a CSV on the user's
device at session end — `data/q_tables/item_qtable_<subject_id>.csv` —
and reloaded at the next session's start. That's the full extent of
longitudinal state in the paper.

This deployment writes TWO persistence layers in parallel, by design:

  1. CSV  (`data/q_tables/item_qtable_<subject_id>.csv`)
     - Written unconditionally at session end via `DataFrame.to_csv`.
     - Same filename scheme and format as the paper/legacy prototype.
     - Kept authoritative for tooling, inspection, and paper-parity
       artifact comparisons (e.g. `data/q_tables/item_qtable_8080.csv`
       matches the legacy prototype's dumps byte-for-byte in shape).
     - Loaded into `self.item_q_table` in `setup()` BEFORE
       `_load_longitudinal_state()` runs, so it acts as the baseline
       state before the DB warm-start overlay.

  2. SQLite `rl_state` row (via `DBManager.save_rl_state`)
     - Stores a JSON-serialised copy of the Q-table PLUS:
         • `item_mask_json`        — which dimensions were asked this
                                      session (supports resume-partial
                                      sessions without re-asking).
         • `top_score2_dims_json`  — top-5 Score-2 dimensions ordered by
                                      importance, used for the recall-
                                      and-resume greeting and for
                                      force-targeting in the first two
                                      turns of the next session.
         • `last_session_id`       — provenance.
     - The CSV format can't cleanly express mask/top-dims, so the DB
       is authoritative for "pick up where we left off" semantics.
     - At load time, `_load_longitudinal_state()` reads the DB row and,
       if the Q-table shape matches, OVERWRITES the CSV-loaded baseline
       with the DB copy. If the DB row is missing or malformed, we fall
       through to the CSV-loaded baseline (graceful degradation).

Divergence risk (documented, not mitigated)
-------------------------------------------
If a crash occurs between the CSV write and the DB write (or vice
versa), the two layers can go out of sync. Because DB is authoritative
for Q-values at load time, a newer CSV with older DB data will be
silently overwritten at next session start. This is acceptable for the
current research deployment but noted here so any future transactional
dual-write can be bolted on without spelunking to find the contract.

Practical implication: treat the CSV as a human-inspectable mirror of
the DB's Q-values and DO NOT hand-edit it — edits will be overwritten.
To force a reset, delete BOTH the CSV and the subject's `rl_state` DB
row.
"""
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
    CRISIS_OVERRIDE_ENABLED,
    CRITICAL_DIMS,
    SAFETY_RESOURCES_MESSAGE,
)
from src.utils.config_loader import (
    ITEM_N_STATES,
    GAMMA,
    ALPHA,
    EPSILON,
    ITEM_IMPORTANCE,
    QUESTION_LIB_FILENAME,
    DATA_DIR,
    WARM_START_ENABLED,
    SESSION_ANALYSIS_ENABLED,
    SOAP_REPORT_ENABLED,
    DIMENSION_OPTOUTS_ENABLED,
    SESSION_CAP_ENABLED,
)
from src.utils.config_loader import RECORD_CSV
from src.utils.io_question_lib import load_question_lib, save_question_lib, generate_results
from src.utils.io_record import init_record, log_question, set_question_prefix, dump_session_history_to_terminal
import src.utils.io_record as io_rec


def _current_subject_id() -> str:
    """Return the *per-session composed* subject id, e.g. "alice_20260425_190621".

    Used for any filename that must be unique per session (crisis fallback
    files, identity-guard labels surfaced to the LLM). SUBJECT_ID is
    installed by io_record.reset_session() after onboarding.
    """
    return str(getattr(io_rec, "SUBJECT_ID", "User") or "User")


def _current_base_subject_id() -> str:
    """Return the *stable* raw subject id, e.g. "alice".

    Used for DB lookups and the Q-table filename — these must resolve to
    the same row across a subject's multiple sessions so longitudinal
    state (warm-start Q-values, recall greeting context) keeps working.
    Falls back to the composed SUBJECT_ID if BASE isn't set (pre-session
    diagnostic path).
    """
    return str(
        getattr(io_rec, "SUBJECT_BASE_ID", None)
        or getattr(io_rec, "SUBJECT_ID", "User")
        or "User"
    )
from src.utils.rl_qtables import (
    initialize_q_table,
    choose_action,
    get_env_feedback,
)
# Set up logger for this module
from src.utils.log_util import get_logger
from src.utils.resource_audit import get_resource_audit
from src.models.llm_client import llm_complete, LLMRole, LLMError
logger = get_logger("HandlerRL")
_RESOURCE_AUDIT = get_resource_audit()

# M6: hard cap on session length. Paper §3 sessions are brief (minutes to
# ~20 for CBT); a 60-minute wall guards against runaway LLM loops or stuck
# intermissions keeping a participant on-device indefinitely.
_SESSION_MAX_SECONDS = float(os.environ.get("SESSION_MAX_SECONDS", str(60 * 60)))

class HandlerRL:
    """
    Top-level RL workflow coordinator.
    Handles the main reinforcement learning loop for question selection and evaluation.
    All file I/O is performed via utility modules.
    """

    def __init__(self):
        # Stores the last question asked to the user
        self.last_question: str = " "
        # The main question library loaded from file
        self.question_lib: Dict[str, Any] = {}
        # Q-table for item selection (top-level RL)
        self.item_q_table = None
        # Action id -> label mapping for logging readability
        self.item_action_labels = {}
        # Crisis override state — tracks every critical dim that has fired
        # so a second crisis (e.g. mid-CBT) can trigger the safety path again.
        self._crisis_triggered: bool = False
        self._crisis_dim: str = ""
        self._crisis_dims_handled: set = set()
        # Longitudinal memory: top Score-2 dimensions from previous session,
        # loaded in setup() for warm-start + recall greeting.
        self._prior_score2_dims: List[str] = []
        self._is_returning_user: bool = False
        # M6: session start timestamp for the hard cap.
        self._session_started_at: float = 0.0

    def setup(self):
        """
        Initialize records, load question library, and set up Q-tables and masks.
        """
        logger.debug("Initializing RL handler setup: loading records and question library.")
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
  
        # Step 1 of persistence contract (see module docstring): load the
        # paper/legacy-compatible CSV as the baseline Q-table. This matches
        # the legacy prototype format byte-for-byte and is what any external
        # tooling inspecting data/q_tables/ expects to see.
        # Q-table filename uses the BASE id so Alice's session-2 warm-
        # starts from the Q-values written at the end of session 1.
        subject = _current_base_subject_id()
        qdir = os.path.join(DATA_DIR, "q_tables")
        qfile = os.path.join(qdir, f"item_qtable_{subject}.csv")
        if os.path.exists(qfile):
            self.item_q_table = pd.read_csv(qfile, index_col=0)
            logger.info(f"[RL] Loaded prior Q-table for subject {subject} (longitudinal warm-start).")
        else:
            logger.info(f"[RL] No prior Q-table for subject {subject}; initialising from item_importance priors.")

        # Step 2 of persistence contract: overlay the DB's rl_state row,
        # which is authoritative for resume semantics (carries the Q-table
        # plus item_mask + top_score2_dims that CSV cannot express). If DB
        # load fails or is missing, we keep the CSV baseline from step 1.
        self._load_longitudinal_state()

        logger.debug("RL handler setup complete.")

    def _load_longitudinal_state(self):
        """Warm-start Q-table from persistent per-user DB state.

        Authoritative read path for Q-values (see module docstring). If
        present, the DB-stored Q-table overwrites the CSV baseline from
        `setup()` because it's the layer that carries item_mask and
        top_score2_dims alongside it. CSV is still the mirror that gets
        rewritten at session end for paper/legacy tooling compatibility.

        G11 — gated by rl.warm_start_enabled. Default off for legacy
        parity (legacy prototype had no DB-backed longitudinal state
        beyond the CSV Q-table, which is loaded in setup() directly and
        NOT gated). When off, the returning-user recall greeting also
        becomes inactive because `_is_returning_user` stays False.
        """
        if not WARM_START_ENABLED:
            logger.debug("Warm-start disabled (rl.warm_start_enabled=false); skipping.")
            return
        if not io_rec.DB:
            logger.debug("No DB available; skipping longitudinal warm-start.")
            return
        # DB row is keyed by the BASE id so longitudinal warm-start
        # works across a subject's multiple session timestamps.
        subject = _current_base_subject_id()
        try:
            user_id = io_rec.DB.get_user_id(subject)
            state = io_rec.DB.load_rl_state(user_id)
        except Exception as e:
            logger.warning(f"Longitudinal state load failed: {e}")
            return

        if not state:
            logger.info(f"[RL] Subject {subject}: first-time user — no prior Q-values to warm-start.")
            return

        self._is_returning_user = True
        logger.info(f"[RL] Subject {subject}: returning user — applying Q-value warm-start.")

        # 1. Restore persisted Q-table if present. Precedence order is
        # documented in the module docstring: DB is authoritative at load
        # time; CSV is authoritative for the written artifact format.
        q_json = state.get("q_table_json")
        if q_json:
            try:
                # Newer pandas requires a file-like wrapper for read_json.
                restored = pd.read_json(io.StringIO(q_json), orient="split")
                # Ensure column/index shapes match the current item space.
                if restored.shape == self.item_q_table.shape:
                    restored.columns = restored.columns.astype(str)
                    self.item_q_table = restored
                    logger.info("[RL] Q-table warm-started from DB persistent state (resume semantics).")
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

        # 3. Dimensional weighting nudge: lightly bias Q-values for state
        # indices whose label matches a prior Score-2 dimension so
        # choose_action softly prefers them, without hard-overriding the
        # paper's ε-greedy exploration. Divergence 4: reduced from the
        # earlier boost=3.0 (which dominated selection) to a small 0.3
        # additive nudge so the full RL loop still runs normally.
        if self._prior_score2_dims:
            boost = 0.3  # minor nudge on top of ITEM_IMPORTANCE base weight
            for i_key in self.question_lib.keys():
                try:
                    label = str(self.question_lib[i_key]["1"].get("label", "")).lower()
                except Exception:
                    continue
                if label in self._prior_score2_dims and i_key in self.item_q_table.columns:
                    self.item_q_table[i_key] = self.item_q_table[i_key] + boost
            logger.info(
                f"[RL] Recall-and-resume: nudged Q-values (+{boost}) for prior Score-2 dimensions: {self._prior_score2_dims}"
            )

    def run(self):
        """
        Main RL loop for the entire screening process.
        Iteratively selects items and asks questions using RL, updating Q-tables and saving results.
        """
        logger.info("[PIPELINE] Screening loop starting (Q-learning questioner over 37 dimensions).")
        self._session_started_at = time.monotonic()
        self.setup()

        # Opening greeting (LLM-rewritten) delivered before the first question for all interfaces.
        # G3 — legacy-prototype seed: gives Gemma enough material to rewrite
        # into the demo's warm variant ("Hi, I'm Caiti, and I'm here to support
        # you. Thanks for being here. Let's start with a few quick questions
        # about your recent day-to-day.").  A one-sentence seed produced too
        # terse an output under Gemma-4-E2B.
        try:
            greeting_raw = (
                "I'm CaiTI, your intelligent therapist. "
                "Thank you for joining me today. "
                "Let's get started with a couple of questions about your recent daily life."
            )
            user_ctx = io_rec.get_user_context()
            # Use the BASE name so the LLM addresses the user as "Alice"
            # rather than "alice_20260425_190621".
            user_name = _current_base_subject_id()
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

            # The speech-service onboarding layer already said
            # "Hello, <Name>. I'm CaiTI, your intelligent therapist. Thank you
            # for joining me today." before the handler even started, so the
            # rewritten greeting must NOT open with another "Hello" or
            # re-introduce CaiTI — otherwise the user hears two back-to-back
            # introductions.  This rule is enforced in every rewrite prompt
            # below.
            no_double_hello_rule = (
                "- The user has ALREADY been greeted with 'Hello, <their name>. "
                "I'm CaiTI, your intelligent therapist. Thank you for joining me "
                "today.' — do NOT repeat any of that. Start with a short warm "
                "sentence that moves the session forward (e.g. 'Let's begin with a "
                "few quick check-ins about your recent day-to-day.' or 'I'd like to "
                "start with a couple of questions about how you've been.').\n"
                "- Do NOT open with 'Hello', 'Hi', 'Hey', or any variant of "
                "'Welcome back' that names CaiTI again.\n"
            )
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
                    "Task: Generate a brief transition that moves a returning user from "
                    "the opening handshake into their new session.\n"
                    f"Here is the context from their previous sessions:\n{user_ctx}\n"
                    f"{recall_hint}\n"
                    "Rules:\n"
                    f"{no_double_hello_rule}"
                    "- Briefly and naturally acknowledge a detail from their past session summary to show you remember them.\n"
                    "- If a 'Recall hint' is provided, gently check in on that topic in ONE short sentence.\n"
                    "- Do not list out their preferences mechanically.\n"
                    "- 2–3 short sentences maximum.\n- Friendly, non-judgmental tone.\n"
                    "- No extra headers or labels; output the final text directly.\n"
                )
            else:
                rewrite_system_prompt = (
                    "You are a warm, concise, and professional therapist-assistant.\n\n"
                    f"{identity_guard}\n"
                    "Task: Generate a brief transition sentence that moves a new user "
                    "from the opening handshake into the first session.\n"
                    "Rules:\n"
                    f"{no_double_hello_rule}"
                    "- 1–2 short sentences.\n- Friendly, non-judgmental tone.\n"
                    "- No extra headers or labels; output the final text directly.\n"
                )
                
            # Paper role: GENERAL (greeting / session opener, not microbenchmarked).
            greeting = llm_complete(rewrite_system_prompt, greeting_raw, role=LLMRole.GENERAL).strip()
            # Use greeting as a prefix so the first substantive question appears immediately
            set_question_prefix(greeting)
        except Exception as e:
            # If LLM call fails, fall back to a short transition that does NOT
            # re-introduce CaiTI (the speech-service handshake has already
            # done that).  Avoids the "Hello John ... Hello I'm CaiTI" double
            # greeting observed in session 11.
            logger.warning(f"Opening greeting rewrite failed: {e}")
            set_question_prefix("Let's get started with a couple of questions about your recent daily life.")
        # Divergence 2: PHQ-4 / GAD-2 no longer runs as a sequential pre-
        # screen. The SpeechInteractionService owns the intermission ladder
        # (SCREENING / BREATHING / MUSIC) and surfaces each of the four
        # PHQ/GAD questions AT MOST ONCE per session, interleaved with
        # breathing exercises and ambient music while the LLM is still
        # generating. When the LLM's response arrives, the current
        # intermission activity is cut short at the earliest natural
        # boundary (a completed question+answer, a finished breath cycle,
        # or a music fade).  This preserves the PHQ-4/GAD-2 signal without
        # front-loading the session with clinical instruments — matching
        # the demo video's flow, where the first spoken question is
        # "How are your eating habits?" not "Over the last 2 weeks...".
        new_q_table = self.item_q_table.copy()
        S = 0  # Start state for item RL
        is_terminated = False
        # Mask for available items (first item is always available)
        item_mask = [0] + [1] * (ITEM_N_STATES - 1)

        # Paper appendix: users can opt out of sensitive dimensions
        # (e.g. "arrest"/"legal" for non-applicable populations).  We
        # apply these opt-outs at mask-init time so the RL policy never
        # selects a disabled dimension.  Reserved preference key is
        # `disabled_dim:<label>` with value "1".
        self._apply_dimension_optouts(item_mask)

        turn_idx = 0

        while not is_terminated:
            if io_rec.END_SESSION_EVENT.is_set():
                logger.info("[SESSION] End signal received mid-loop — committing Q-tables and exiting.")
                is_terminated = True
                break

            # If all items have been asked, exit to CBT directly
            if sum(item_mask) == 0:
                is_terminated = True
                logger.info("[PIPELINE] Screening loop complete — all 37 dimensions scored. Proceeding to CBT.")
                break

            # Divergence 4: the returning-user warm-start is now a minor
            # nudge on the Q-values (see _load_longitudinal_state). The
            # previous "force-target prior Score-2 dim on the first two
            # turns, bypassing epsilon-greedy" override has been removed
            # so the full paper/legacy ε-greedy RL loop runs unchanged
            # for every user, returning or not.
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
            # Q-learning update for item Q-table.
            # Paper §5.1 uses 39 states (37 dims + START + END); we model END
            # as the sentinel S_ == 'terminal' and skip bootstrapping in that
            # branch, which is equivalent to an absorbing END row of zeros.
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

            # Crisis override: when enabled, a CRITICAL_DIMS Score 2 short-
            # circuits the RL loop and delivers the safety message. Currently
            # disabled via CRISIS_OVERRIDE_ENABLED=False (see therapy_content
            # for the rationale and re-enablement checklist).
            if CRISIS_OVERRIDE_ENABLED and self._crisis_scan():
                self._crisis_triggered = True
                logger.warning(
                    f"[CRISIS OVERRIDE] Critical-dim Score 2 detected ({self._crisis_dim}). "
                    "Delivering safety resources via guaranteed path."
                )
                io_rec.log_reasoning("crisis_override", {
                    "triggered_at_turn": turn_idx,
                    "critical_dim": self._crisis_dim,
                })
                self._deliver_safety_message(self._crisis_dim)
                # Short-circuit to CBT so the session focuses on the safety topic.
                is_terminated = True
                break

            # M6: hard session-length cap — graceful termination.
            if self._session_timed_out():
                is_terminated = True
                break
            # If the DLA process signals termination, end the loop and save results
            if DLA_terminate == 1:
                # DLA process signaled termination; proceed to save artifacts
                is_terminated = True
                save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
                save_question_lib(save_filename, self.question_lib)
                logger.debug(f"Saved question library to {save_filename} after DLA termination.")
                # log_question("Goodbye. We will do the screening in another time. 886")
                logger.debug("Goodbye. We will do the screening in another time.")        # Save results if terminated
        if is_terminated:
            # Persist question library snapshot upon termination
            save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
            save_question_lib(save_filename, self.question_lib)
            logger.debug(f"Saved question library to {save_filename} after session termination.")
            
            # Persistence step 1 (see module docstring): write the CSV
            # mirror in paper/legacy format. Tooling and external
            # inspection rely on this file existing at the exact path
            # legacy used — do NOT rename or drop it without coordinating
            # with downstream consumers.
            #
            # C4: atomic tmp-file + os.replace so a process kill mid-write
            # cannot corrupt the paper-compatible CSV artefact.
            # Q-table write mirrors the load — BASE id so accumulated
            # Q-values persist across Alice's session timestamps.
            subject = _current_base_subject_id()
            qdir = os.path.join(DATA_DIR, "q_tables")
            qfile = os.path.join(qdir, f"item_qtable_{subject}.csv")
            self.item_q_table = new_q_table
            dir_preexisted = os.path.exists(qdir)
            if not dir_preexisted:
                os.makedirs(qdir, exist_ok=True)
                logger.debug(f"Created q_tables directory at {qdir}.")
            file_preexisted = os.path.exists(qfile)
            tmp_qfile = qfile + ".tmp"
            try:
                self.item_q_table.to_csv(tmp_qfile)
                os.replace(tmp_qfile, qfile)
                if file_preexisted:
                    logger.info(f"[RL] Updated Q-table for subject {subject} (persisted to CSV).")
                else:
                    logger.info(f"[RL] Created initial Q-table for subject {subject} (persisted to CSV).")
            except Exception as e:
                logger.error(f"[Q-TABLE] Atomic CSV write failed: {e}")
                try:
                    if os.path.exists(tmp_qfile):
                        os.remove(tmp_qfile)
                except Exception:
                    pass

            # Persistence step 2 (see module docstring): write the
            # authoritative DB row. This includes the same Q-values plus
            # item_mask and top_score2_dims needed for next-session
            # resume + recall greeting. Divergence between CSV and DB at
            # crash time is acknowledged; DB wins at load.
            self._save_longitudinal_state(item_mask)

        # Run CBT after the screening loop concludes if not interrupted.
        # When a crisis override fired during the screening loop, surface the
        # critical dimension to the CBT selector as the default focus.
        if not io_rec.END_SESSION_EVENT.is_set():
            if self._crisis_triggered and self._crisis_dim:
                logger.info(
                    f"[SAFETY] Crisis override active — routing CBT to dimension '{self._crisis_dim}'."
                )
                try:
                    io_rec.log_reasoning("cbt_crisis_routing", {"dim": self._crisis_dim})
                except Exception:
                    pass
            def _cbt_crisis_hook() -> bool:
                """Scan question_lib for new critical-dim Score 2 between CBT
                stages; deliver safety resources if found, return True to
                signal CBT to pause. Inert unless CRISIS_OVERRIDE_ENABLED."""
                if not CRISIS_OVERRIDE_ENABLED:
                    return False
                if self._crisis_scan():
                    self._deliver_safety_message(self._crisis_dim)
                    return True
                return False

            logger.info("[PIPELINE] CBT protocol starting (3 stages: Recognize -> Challenge -> Reframe).")
            run_cbt(self.question_lib, crisis_callback=_cbt_crisis_hook)
            logger.info("[PIPELINE] CBT protocol completed.")
            # Persist question_lib again to capture CBT notes
            save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
            save_question_lib(save_filename, self.question_lib)
        else:
            logger.info("[SESSION] Early termination — skipping CBT.")

        # Generate final results for this session
        generate_results(self.question_lib, [])
        logger.info("[PIPELINE] Per-session Report/Notes CSVs written.")

        # ── Session-end closing sequence ─────────────────────────────────
        # Two modes gated by rl.session_analysis_enabled (G12):
        #
        #   OFF (default, legacy parity):
        #     - If CBT ran, CBT's "Great work today..." is the only closing.
        #     - If CBT did NOT run, speak a short LLM-generated "no areas
        #       of concern identified today" closing via the legacy prompt.
        #     - No SOAP, no preferences extraction, no safety-flag LLM pass.
        #
        #   ON (research / clinical-trial mode):
        #     - SOAP report runs silently (if soap_report_enabled).
        #     - _generate_session_analysis produces SUMMARY/PREFERENCES/
        #       SAFETY_FLAGS; the SUMMARY line becomes the spoken closing.
        #     - Preferences and safety flags are stored silently in DB.
        #
        # On session interrupt (END_SESSION_EVENT) every LLM path is
        # skipped and we only dump history for post-mortem.
        if io_rec.END_SESSION_EVENT.is_set():
            logger.info("[SESSION] Interrupted — skipping post-session analysis for prompt exit.")
            dump_session_history_to_terminal()
        else:
            try:
                cbt_used, cbt_summary = self._detect_cbt_summary()

                if SESSION_ANALYSIS_ENABLED:
                    # SOAP first (silent; itself gated by soap_report_enabled).
                    self._generate_clinical_summary()
                    # Session analysis returns the 2-3 sentence spoken SUMMARY.
                    spoken_summary = self._generate_session_analysis() or ""
                    if cbt_used:
                        logger.debug("CBT delivered its own closing; skipping RL-level closing message.")
                        if spoken_summary:
                            log_question(spoken_summary)
                    else:
                        log_question(
                            spoken_summary
                            or "Thank you for your time today. Take care, and goodbye."
                        )
                else:
                    # Legacy closing path: if CBT wasn't used, emit a short
                    # LLM-generated "no concerns identified" goodbye using
                    # the exact legacy prompt. If CBT ran, stay silent.
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
                            "- 1-2 sentences only.\n"
                            "- Friendly, non-judgmental tone.\n"
                            "- No headers or labels; output the final message directly.\n"
                        )
                        user_payload = (
                            f"cbt_used: {str(cbt_used).lower()}\n"
                            + (f"session_summary:\n{cbt_summary}" if cbt_summary else "")
                        )
                        closing = llm_complete(
                            sys_prompt, user_payload, role=LLMRole.GENERAL
                        ).strip()
                        log_question(closing or "Thank you for your time today. Take care, and goodbye.")
                    else:
                        logger.info("CBT delivered its own closing; skipping RL-level closing.")
                dump_session_history_to_terminal()
            except Exception as e:
                logger.error(f"Post-session closing failed: {e}")
                try:
                    cbt_used, _ = self._detect_cbt_summary()
                    if not cbt_used:
                        log_question("Thank you for your time today. Take care, and goodbye.")
                except Exception:
                    pass
                dump_session_history_to_terminal()
            
        # ── Forensic report: emit immutable session resource snapshot ──────
        try:
            _RESOURCE_AUDIT.capture_point("session_end")
            report_path = _RESOURCE_AUDIT.write_report()
            if report_path:
                logger.debug(f"[FORENSIC] Session resource report written to {report_path}")
        except Exception as e:
            logger.warning(f"Failed to write session forensic report: {e}")

        # Ensure deep memory release of the ML frames to prevent user leakage
        del self.item_q_table
        logger.debug("Garbage collected ML DataFrame for HandlerRL Context wipe.")

        # Mark the DB session row closed so crash-recovery doesn't mis-flag
        # this as a dangling session on next boot.
        try:
            io_rec.mark_session_finalised()
        except Exception as e:
            logger.warning(f"mark_session_finalised failed: {e}")

    def _generate_session_analysis(self) -> str:
        """
        Analyze the session history for summary, preferences, and safety flags.
        Stores them in the DB and returns the 2-3 sentence SUMMARY string so
        the caller can use it as the spoken session-closing message
        (Divergence 7).  Returns "" on any failure; callers must be tolerant.
        """
        if not io_rec.DB or not io_rec.SESSION_ID:
            logger.warning("DB or Session ID not available for analysis.")
            return ""

        history = io_rec.DB.get_session_history(io_rec.SESSION_ID)
        if not history:
            return ""

        spoken_summary = ""

        # Format history string
        hist_text = "\n".join([f"{h['speaker']}: {h['text']}" for h in history])

        # G10 — the SUMMARY line is spoken to the user as the closing
        # farewell (see run()'s closing block). It must read as a warm,
        # supportive goodbye, not a clinical case note. PREFERENCES and
        # SAFETY_FLAGS remain structured clinical output — those are
        # never spoken, only persisted to the DB for future sessions
        # and the clinician's handoff.
        prompt = (
            "You are reviewing a therapy session you just had with the user.\n\n"
            "Session history:\n"
            f"{hist_text}\n\n"
            "Produce three sections in order. Follow the exact labels below.\n\n"
            "1. SUMMARY (SPOKEN to the user as the closing goodbye):\n"
            "   - 1 to 2 short sentences, second person (\"you\"), warm and\n"
            "     supportive tone, no clinical jargon, no bullet points.\n"
            "   - Briefly acknowledge what the user worked on today and\n"
            "     close with a gentle well-wishing goodbye.\n"
            "   - Do NOT list diagnoses, scores, or dimension labels.\n"
            "   - Example: \"Thank you for sharing about your medication\n"
            "     routine today - that takes courage. Take care of yourself,\n"
            "     and I'll be here whenever you want to talk again.\"\n\n"
            "2. PREFERENCES (stored silently for future sessions):\n"
            "   - Extract any specific facts or preferences the user mentioned\n"
            "     (e.g., likes shopping, dislikes crowds, works nights).\n"
            "   - One bullet per preference, KEY: VALUE format.\n\n"
            "3. SAFETY_FLAGS (stored silently for the clinician):\n"
            "   - Any potential safety risks (self-harm, violence, substance\n"
            "     crisis). Severity 1 (mild) to 5 (critical). Say NONE if absent.\n"
            "   - Format: TYPE: TEXT: SEVERITY.\n\n"
            "Response Format (use EXACTLY these labels, no extra headers):\n"
            "SUMMARY: <one or two warm spoken sentences>\n"
            "PREFERENCES:\n- <key>: <value>\n"
            "SAFETY_FLAGS:\n- <type>: <text>: <severity>\n"
        )

        try:
            # Paper role: GENERAL (post-session analysis, extension beyond paper).
            # System prompt tuned so the LLM understands the mixed audience:
            # SUMMARY is spoken to the user; PREFERENCES/SAFETY_FLAGS are
            # clinician artefacts.
            analysis = llm_complete(
                "You are CaiTI, a warm AI therapist-assistant. You are "
                "wrapping up a session and producing both a spoken farewell "
                "for the user AND silent structured notes for the clinician. "
                "Keep the spoken SUMMARY gentle and human; keep PREFERENCES "
                "and SAFETY_FLAGS clinical and precise.",
                prompt,
                role=LLMRole.GENERAL,
            )

            # Parse and Store
            current_section = None
            for line in analysis.split('\n'):
                line = line.strip()
                if not line: continue
                
                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()
                    if summary:
                        io_rec.DB.add_summary(io_rec.SESSION_ID, summary)
                        spoken_summary = summary
                        logger.debug(f"Stored summary: {summary}")
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
                        user_id = io_rec.DB.get_user_id(_current_base_subject_id())
                        io_rec.DB.set_preference(user_id, k, v)
                        logger.debug(f"Stored preference: {k}={v}")
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
            return ""

        return spoken_summary

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

        G13 — gated by rl.soap_report_enabled. Legacy had no SOAP
        concept. Default off for legacy parity. Re-enable only when a
        clinician is on-call to review the artefact.
        """
        if not SOAP_REPORT_ENABLED:
            return
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

        # Paper p.21: longitudinal trend across recent sessions.  The current
        # session's score is included at the head so the TREND block reads
        # newest-first when the DB has not yet committed this session's row
        # (it commits on log_screening_scores but ordering depends on
        # start_time, so we prepend the in-memory snapshot defensively).
        trend_block = ""
        try:
            user_id = io_rec.DB.get_user_id(_current_base_subject_id())
            recent = io_rec.DB.get_recent_screening_scores(user_id, limit=5)
            if recent and len(recent) > 1:
                # Drop duplicate of the just-logged session, keep the rest in
                # chronological order (oldest -> newest) for readability.
                prior = list(reversed(recent))
                phq4_series = [r["total"] for r in prior if r["total"] is not None]
                gad2_series = [r["anxiety"] for r in prior if r["anxiety"] is not None]
                phq2_series = [r["depression"] for r in prior if r["depression"] is not None]
                trend_block = (
                    f"Trend (oldest -> newest, last {len(prior)} sessions):\n"
                    f"  PHQ-4 totals: {phq4_series}\n"
                    f"  GAD-2 totals: {gad2_series}\n"
                    f"  PHQ-2 totals: {phq2_series}\n"
                )
        except Exception as e:
            logger.warning(f"Could not compute PHQ-4 trend block: {e}")

        prompt = (
            "Create a SOAP-format clinical session note for therapist handoff.\n"
            "Output EXACTLY with these four headers and no others:\n\n"
            "SUBJECTIVE:\n"
            "- <client's reported experience in their own phrasing, 2-4 bullets>\n\n"
            "OBJECTIVE:\n"
            "- PHQ-4 total: <value>\n"
            "- GAD-2 sub-total: <value>\n"
            "- PHQ-2 sub-total: <value>\n"
            "- Trend across recent sessions (if data available): "
            "<one-line description, e.g. 'PHQ-4 down from 9 to 6 over last 3 sessions'>\n"
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
            f"{trend_block}"
            f"Problematic dimensional scores:\n{dim_block}\n"
            f"CBT used this session: {cbt_used}\n"
            f"CBT notes:\n{cbt_notes or '(none)'}\n"
            f"Safety: {crisis_line}\n\n"
            f"Session History:\n{hist_text}"
        )

        # Paper role: GENERAL (SOAP clinical summary, extension beyond paper).
        summary = llm_complete(
            "You are a clinical documentation assistant.",
            prompt,
            role=LLMRole.GENERAL,
        ).strip()
        if summary:
            # Divergence 7: SOAP report is for the clinician handoff CSV
            # and DB (auditability), NOT for the user's spoken channel.
            # Do not call log_question here — the participant hears the
            # session-analysis SUMMARY from _generate_session_analysis as
            # the closing message instead.
            io_rec.DB.add_summary(io_rec.SESSION_ID, summary)
            logger.info("[CLOSING] SOAP clinical report generated and stored in DB.")

    def _apply_dimension_optouts(self, item_mask: list) -> None:
        """Mask out dimensions the user has opted out of via preferences.

        Paper (appendix): users can decline entire dimensions (e.g. the
        "law-abiding / arrest" family) if they are not applicable.  The
        preference key format is `disabled_dim:<label>` with value "1".
        This is a purely additive mask update — existing zeros (e.g. the
        INIT slot at index 0) are left alone; enabled entries are cleared
        to 0 only when a matching opt-out exists.

        G14 — gated by rl.dimension_optouts_enabled. Legacy prototype had
        no opt-out mechanism and always asked every dim, so default off.
        """
        if not DIMENSION_OPTOUTS_ENABLED:
            return
        if not io_rec.DB:
            return
        try:
            user_id = io_rec.DB.get_user_id(_current_base_subject_id())
            prefs = io_rec.DB.get_all_preferences(user_id) or {}
        except Exception as e:
            logger.warning(f"Could not read user preferences for opt-out: {e}")
            return

        disabled_labels = {
            k.split(":", 1)[1].strip().lower()
            for k, v in prefs.items()
            if k.startswith("disabled_dim:") and str(v).strip() == "1"
        }
        if not disabled_labels:
            return

        masked = []
        for i_key in self.question_lib.keys():
            try:
                idx = int(i_key)
            except ValueError:
                continue
            if not (0 < idx < ITEM_N_STATES):
                continue
            label = str(self.question_lib[i_key]["1"].get("label", "")).lower()
            if label in disabled_labels and item_mask[idx] == 1:
                item_mask[idx] = 0
                masked.append(label)

        if masked:
            logger.info(f"[RL] User-opted-out dimensions masked out of Q-table: {masked}")
            try:
                io_rec.log_reasoning("dimension_optout", {"masked": masked})
            except Exception:
                pass

    def _save_longitudinal_state(self, item_mask: list) -> None:
        """Persist Q-table, item_mask and top Score-2 dims for this session.

        Authoritative write path for resume semantics (see module
        docstring). Called in parallel with the CSV mirror write so
        both layers stay in sync under normal termination. Failure is
        logged and swallowed — the CSV mirror remains as a fallback if
        the DB write fails mid-session.

        G11 — gated by rl.warm_start_enabled (paired with the load side).
        When warm-start is off, there is no consumer of the DB rl_state
        row, so we skip the write entirely. The CSV mirror is still
        written by run() regardless; that's the paper-parity artefact.
        """
        if not WARM_START_ENABLED:
            return
        if not io_rec.DB:
            return
        try:
            user_id = io_rec.DB.get_user_id(_current_base_subject_id())
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
                f"[RL] Persisted longitudinal state: {len(top_dims)} Score-2 dimensions recorded for next session."
            )
        except Exception as e:
            logger.warning(f"Longitudinal state save failed: {e}")

    def _crisis_scan(self) -> bool:
        """Return True if any CRITICAL_DIMS entry now has a score of 2 that
        hasn't already been handled this session.

        C7: `self._crisis_dims_handled` lets the scan fire more than once per
        session so a fresh critical dim detected mid-CBT still triggers the
        safety path.

        Side-effect: records the first NEW matching dimension label in
        `self._crisis_dim` so the caller can prioritise it and persist flags.
        """
        try:
            for i_key in self.question_lib.keys():
                entry = self.question_lib[i_key].get("1", {})
                label = str(entry.get("label", "")).lower()
                if label not in CRITICAL_DIMS:
                    continue
                if label in self._crisis_dims_handled:
                    continue
                if any((isinstance(s, int) and s == 2) for s in entry.get("score", [])):
                    self._crisis_dim = label
                    # Legacy flag table (preserved for back-compat auditors).
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
                        # M3: authoritative clinical-flag log.
                        try:
                            io_rec.DB.log_clinical_flag(
                                io_rec.SESSION_ID,
                                flag_type="CRITICAL_DIM_SCORE_2",
                                details={"critical_dim": label},
                            )
                        except Exception as e:
                            logger.warning(f"Could not persist clinical flag: {e}")
                    return True
        except Exception as e:
            logger.warning(f"crisis scan failed: {e}")
        return False

    def _deliver_safety_message(self, critical_dim: str) -> bool:
        """C2: guaranteed delivery of SAFETY_RESOURCES_MESSAGE.

        Best-effort tries in this order:
          1. Normal TTS path via OUTPUT_QUEUE (`log_question`).
          2. Plain-text fallback written to `data/safety/` so an on-call
             clinician monitoring the Jetson can see the message landed
             even if speech is down.
        Every attempt writes a row to `safety_deliveries` for audit.
        Returns True if at least one delivery method succeeded.
        """
        success_any = False
        errors = []

        # Attempt 1: speak via TTS queue.
        try:
            log_question(SAFETY_RESOURCES_MESSAGE)
            success_any = True
            if io_rec.DB and io_rec.SESSION_ID:
                try:
                    io_rec.DB.log_safety_delivery(
                        session_id=io_rec.SESSION_ID,
                        critical_dim=critical_dim,
                        method="tts_queue",
                        success=True,
                        message_text=SAFETY_RESOURCES_MESSAGE,
                    )
                except Exception as e:
                    logger.warning(f"[SAFETY] Could not audit TTS delivery: {e}")
        except Exception as e:
            err = f"tts_queue failed: {e}"
            errors.append(err)
            logger.error(f"[SAFETY] {err}")
            if io_rec.DB and io_rec.SESSION_ID:
                try:
                    io_rec.DB.log_safety_delivery(
                        session_id=io_rec.SESSION_ID,
                        critical_dim=critical_dim,
                        method="tts_queue",
                        success=False,
                        message_text=SAFETY_RESOURCES_MESSAGE,
                        error_text=str(e),
                    )
                except Exception:
                    pass

        # Attempt 2: always write the file fallback so a clinician sees it.
        try:
            import datetime
            safety_dir = os.path.join(os.path.abspath("."), "data", "safety")
            os.makedirs(safety_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            subject = _current_subject_id()
            fallback_path = os.path.join(
                safety_dir,
                f"crisis_{subject}_{io_rec.SESSION_ID}_{critical_dim}_{ts}.txt",
            )
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(f"[CRISIS CHECKPOINT]\n")
                f.write(f"subject_id: {subject}\n")
                f.write(f"session_id: {io_rec.SESSION_ID}\n")
                f.write(f"critical_dim: {critical_dim}\n")
                f.write(f"timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"tts_errors: {errors or 'none'}\n\n")
                f.write(SAFETY_RESOURCES_MESSAGE)
                f.write("\n")
            logger.warning(f"[SAFETY] File fallback written: {fallback_path}")
            success_any = True
            if io_rec.DB and io_rec.SESSION_ID:
                try:
                    io_rec.DB.log_safety_delivery(
                        session_id=io_rec.SESSION_ID,
                        critical_dim=critical_dim,
                        method="file_fallback",
                        success=True,
                        message_text=fallback_path,
                    )
                except Exception:
                    pass
        except Exception as e:
            errors.append(f"file_fallback failed: {e}")
            logger.error(f"[SAFETY] File fallback failed: {e}")
            if io_rec.DB and io_rec.SESSION_ID:
                try:
                    io_rec.DB.log_safety_delivery(
                        session_id=io_rec.SESSION_ID,
                        critical_dim=critical_dim,
                        method="file_fallback",
                        success=False,
                        message_text=SAFETY_RESOURCES_MESSAGE,
                        error_text=str(e),
                    )
                except Exception:
                    pass

        if not success_any:
            logger.error(
                f"[SAFETY] ALL DELIVERY METHODS FAILED for critical_dim={critical_dim}. "
                f"Errors: {errors}"
            )

        self._crisis_dims_handled.add(critical_dim)
        return success_any

    def _session_timed_out(self) -> bool:
        """M6: True once we've exceeded the hard session-length cap.

        G15 — gated by rl.session_cap_enabled. Legacy had no cap; default
        off for legacy parity. Clinical trials should re-enable to prevent
        a stuck LLM from holding a participant indefinitely.
        """
        if not SESSION_CAP_ENABLED:
            return False
        if self._session_started_at <= 0:
            return False
        elapsed = time.monotonic() - self._session_started_at
        if elapsed >= _SESSION_MAX_SECONDS:
            logger.warning(
                f"[SESSION CAP] Session exceeded {_SESSION_MAX_SECONDS:.0f}s "
                f"(elapsed {elapsed:.0f}s). Forcing graceful termination."
            )
            if io_rec.DB and io_rec.SESSION_ID:
                try:
                    io_rec.DB.log_clinical_flag(
                        io_rec.SESSION_ID,
                        flag_type="SESSION_CAP_REACHED",
                        details={"elapsed_sec": elapsed, "cap_sec": _SESSION_MAX_SECONDS},
                    )
                except Exception:
                    pass
            return True
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
