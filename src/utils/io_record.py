"""Utility helper managing file I/O operations and synchronous memory channels.

Clinical-trial hardening applied in this module
-----------------------------------------------
- C1: `init_record` is idempotent — guarded by an init sentinel so handler-
  side + main-side calls in the same session cannot create duplicate DB
  session rows or reset CURRENT_TURN_INDEX mid-flight.
- H1: Session mutation (SUBJECT_ID, DB, SESSION_ID) is protected by a
  single `_INIT_LOCK` so the speech thread and handler thread cannot see
  a torn state snapshot.
- H3: OUTPUT_QUEUE is bounded (`maxsize=50`) with drop-oldest-on-overflow.
  INPUT_QUEUE is intentionally UNBOUNDED — dropping a user reply is a
  clinical data-loss event and is unacceptable.
- H4: CURRENT_TURN_INDEX is atomic via `_TURN_INDEX_LOCK`; the
  `_next_turn_index()` helper is the ONLY way to advance it.
- M1: `REDACT_PII=1` env flag replaces transcript text with length tags
  in all user-facing logger output. Structured DB/JSON logs are kept
  untouched so clinicians can still audit.
- M2: On `init_record()` we ask the DB to close any sessions left open
  from a prior crash (end_reason='crash_recovery') before creating the
  new session.
"""
import atexit
import os
import json
import queue
import time
import threading
from typing import List, Tuple

from src.utils.log_util import get_logger
from src.drivers.db_manager import DBManager
from src.utils.config_loader import (
    DB_PATH,
    SUBJECT_ID,
    RECORD_CSV,
    REPORT_FILE as _DEFAULT_REPORT_FILE,
    NOTES_FILE as _DEFAULT_NOTES_FILE,
    format_result_paths,
)


def _segment_utterance(text: str) -> List[str]:
    """Legacy/paper-aligned utterance segmentation (zero dep)."""
    if not text:
        return []
    normalised = text.replace(", and", ".").replace(" but ", ". ")
    pieces = []
    for chunk in normalised.replace("!", ".").replace("?", ".").split("."):
        stripped = chunk.strip()
        if stripped:
            pieces.append(stripped)
    return pieces


logger = get_logger("IORecord")

# ── Redaction (M1) ────────────────────────────────────────────────────────
REDACT_PII = os.environ.get("REDACT_PII", "0").strip().lower() in {"1", "true", "yes", "on"}


def _redact(text) -> str:
    """Return a log-safe rendering of `text` when REDACT_PII is enabled."""
    s = "" if text is None else str(text)
    if not REDACT_PII:
        return s
    # Keep length so log readers can still see "user said ~20 chars" without PII.
    return f"[REDACTED_PII len={len(s)}]"


_DEFAULT_RECORD_CSV = RECORD_CSV
_LAST_AUTO_RECORD_CSV = None

# ── Subject identity (two layers) ─────────────────────────────────────────
# SUBJECT_BASE_ID : the raw onboarded name ("alice"). Used for DB
#                   `users.subject_id` so longitudinal state (Q-table
#                   warm-start, user context from prior sessions) still
#                   links across a subject's visits. Also used as the
#                   Q-table filename stem (item_qtable_alice.csv).
#
# SUBJECT_ID      : the *per-session* composed id "alice_YYYYMMDD_HHMMSS".
#                   Used for every per-session artefact filename — dossier
#                   JSON, Report/Notes CSVs, log directory, crisis file —
#                   so three sessions from Alice produce three distinct
#                   file entries and never collide with similarly-named
#                   subjects like "Alex".
#
# Before the first session both hold the boot-time default from config.
SUBJECT_BASE_ID: str = SUBJECT_ID
REPORT_FILE: str = _DEFAULT_REPORT_FILE
NOTES_FILE: str = _DEFAULT_NOTES_FILE
# Timestamp for the currently active session, shared with dossier + CSV
# filenames so every per-session artefact lines up.
SESSION_TIMESTAMP: str = ""

# ── IPC Queues ───────────────────────────────────────────────────────────
# INPUT_QUEUE: user -> handler. UNBOUNDED by design. Dropping a user
# response is a clinical-data-loss event and not tolerable; overflow
# indicates a dead handler and should surface as a clear error, not
# silent data loss.
INPUT_QUEUE = queue.Queue()

# OUTPUT_QUEUE: handler -> speech. Bounded with drop-oldest-on-overflow
# because speech is an output-only channel. A lost agent utterance is a
# UX regression, not a safety issue.
_OUTPUT_QUEUE_MAXSIZE = 50
OUTPUT_QUEUE: queue.Queue = queue.Queue(maxsize=_OUTPUT_QUEUE_MAXSIZE)


def _safe_output_put(item):
    """Enqueue to OUTPUT_QUEUE with drop-oldest-on-overflow.

    Never blocks. Logs a warning the first time we overflow per session so
    a clogged TTS pipeline cannot silently pile up indefinitely.
    """
    try:
        OUTPUT_QUEUE.put_nowait(item)
    except queue.Full:
        try:
            dropped = OUTPUT_QUEUE.get_nowait()
            logger.warning(
                f"[QUEUE_OVERFLOW] OUTPUT_QUEUE full ({_OUTPUT_QUEUE_MAXSIZE}). "
                f"Dropping oldest agent utterance to make room. "
                f"Dropped preview: {_redact(str(dropped)[:60])}..."
            )
        except queue.Empty:
            pass
        try:
            OUTPUT_QUEUE.put_nowait(item)
        except queue.Full:
            logger.error("[QUEUE_OVERFLOW] OUTPUT_QUEUE still full after drop; utterance lost.")


END_SESSION_EVENT = threading.Event()
START_SESSION_EVENT = threading.Event()


# ── Global session state (protected by _INIT_LOCK) ───────────────────────
DB = None
SESSION_ID = None
CURRENT_TURN_INDEX = 0

_PENDING_QUESTION_PREFIX = ""
_LAST_AGENT_LOGGED = ""
USER_CONTEXT = ""
_LAST_USER_TRANSCRIPT = ""
_LAST_USER_EMOTION = "Neutral"
_LAST_RL_STATE = {}
_LATEST_SCREENING_SCORES = {
    "anxiety": None,
    "depression": None,
    "total": None,
}
_JSON_LOG_PATH: str = ""

# ── Concurrency locks ────────────────────────────────────────────────────
# H1: single lock serialising every mutation of session state.
_INIT_LOCK = threading.Lock()
# H4: atomic turn_index advance.
_TURN_INDEX_LOCK = threading.Lock()

# C1: idempotent-init sentinel. Cleared by reset_session() on explicit reset.
_INIT_DONE = False


def _next_turn_index() -> int:
    """H4: atomically claim and advance CURRENT_TURN_INDEX."""
    global CURRENT_TURN_INDEX
    with _TURN_INDEX_LOCK:
        idx = CURRENT_TURN_INDEX
        CURRENT_TURN_INDEX += 1
        return idx


# ── Accessors (unchanged signatures) ─────────────────────────────────────

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


def set_question_prefix(text: str):
    """Set a pending prefix that will be prepended to the next question output."""
    global _PENDING_QUESTION_PREFIX
    _PENDING_QUESTION_PREFIX = str(text) if text is not None else ""


# CSV header
HEADER = ["Timestamp", "Type", "Speaker", "Text"]


def log_json_event(event_type: str, data: dict):
    """Append a timestamped JSON line to the session JSON log."""
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
    """Append a single CSV row capturing the full conversation state."""
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
            escaped_text = str(text).replace('"', '""')
            line = f'"{timestamp}","{log_type}","{speaker}","{escaped_text}"\n'
            f.write(line)
    except Exception as e:
        logger.error(f"Failed to sync to CSV: {e}")


# ── Session lifecycle ────────────────────────────────────────────────────

def init_record(user_id_override: str = None, force: bool = False):
    """Initialize queues, DB session, CSV.

    C1 + H1: idempotent + thread-safe. If already initialised in this
    session, `init_record()` is a no-op unless called with force=True
    (which is what `reset_session()` does).
    """
    global DB, SESSION_ID, CURRENT_TURN_INDEX, SUBJECT_ID, SUBJECT_BASE_ID
    global _LAST_USER_TRANSCRIPT, _LAST_USER_EMOTION, _LAST_RL_STATE, _LATEST_SCREENING_SCORES
    global _LAST_AUTO_RECORD_CSV, RECORD_CSV, _JSON_LOG_PATH, _INIT_DONE
    global REPORT_FILE, NOTES_FILE, SESSION_TIMESTAMP

    with _INIT_LOCK:
        if _INIT_DONE and not force:
            logger.debug(
                f"init_record already ran this session (SESSION_ID={SESSION_ID}); "
                "skipping duplicate init."
            )
            return

        # ── Compose the two-layer subject identity ───────────────────
        # 1. BASE: the raw onboarded name — stable across sessions so
        #    DB-driven longitudinal state (Q-table, user context) still
        #    resolves Alice's 3 visits to one `users` row.
        # 2. SUBJECT_ID: base_name + session timestamp — used for every
        #    per-session filename so three sessions for Alice produce
        #    Report_alice_<ts1>.csv, Report_alice_<ts2>.csv, and
        #    similarly-named subjects ("Alex") can't ever collide.
        import datetime
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        SESSION_TIMESTAMP = timestamp_str

        if user_id_override:
            logger.debug(f"Overriding SUBJECT_BASE_ID with {user_id_override}")
            SUBJECT_BASE_ID = str(user_id_override)
        SUBJECT_ID = f"{SUBJECT_BASE_ID}_{timestamp_str}"
        logger.info(f"[SESSION] Subject onboarded: {SUBJECT_BASE_ID} (session id: {SUBJECT_ID})")

        # Clear queues (on force / fresh session)
        with OUTPUT_QUEUE.mutex:
            OUTPUT_QUEUE.queue.clear()
        with INPUT_QUEUE.mutex:
            INPUT_QUEUE.queue.clear()

        try:
            DB = DBManager(DB_PATH)
            # DB row is keyed by the stable BASE id, not the timestamped
            # one, so a returning subject's longitudinal Q-table warm-
            # start, user context, preferences all still resolve.
            user_id = DB.get_user_id(SUBJECT_BASE_ID)

            # M2: crash recovery — close any sessions left open for this user.
            try:
                recovered = DB.close_open_sessions_for_user(user_id, reason="crash_recovery")
                if recovered:
                    logger.warning(
                        f"[RECOVERY] Closed {len(recovered)} dangling session(s) "
                        f"for user_id={user_id}: {recovered}"
                    )
                    try:
                        DB.log_clinical_flag(
                            session_id=recovered[0],
                            flag_type="CRASH_RECOVERY",
                            details={"closed_sessions": recovered, "user_id": user_id},
                        )
                    except Exception as e:
                        logger.warning(f"[RECOVERY] Could not log recovery flag: {e}")
            except Exception as e:
                logger.warning(f"[RECOVERY] Skipping crash-recovery scan: {e}")

            SESSION_ID = DB.create_session(user_id)

            try:
                global USER_CONTEXT
                USER_CONTEXT = DB.get_user_context_string(user_id)
                if USER_CONTEXT:
                    logger.debug("Loaded User Context for session.")
            except Exception as e:
                logger.error(f"Failed to load user context: {e}")

            CURRENT_TURN_INDEX = 0
            _LAST_USER_TRANSCRIPT = ""
            _LAST_USER_EMOTION = "Neutral"
            _LAST_RL_STATE = {}
            _LATEST_SCREENING_SCORES = {"anxiety": None, "depression": None, "total": None}
        except Exception as e:
            logger.error(f"Failed to initialize DB: {e}")

        # Per-session log directory, keyed by the composed SUBJECT_ID so
        # each session gets its own folder ("data/logs/alice_20260425_190621/")
        # — you can archive or hand off one session's artefacts cleanly.
        base_session_dir = os.path.join(os.path.abspath("."), "data", "logs", SUBJECT_ID)
        record_csv_is_external = RECORD_CSV not in {_DEFAULT_RECORD_CSV, _LAST_AUTO_RECORD_CSV}
        if record_csv_is_external:
            logger.debug(f"Using externally configured RECORD_CSV path: {RECORD_CSV}")
        else:
            RECORD_CSV = os.path.join(base_session_dir, f"{SUBJECT_ID}.log")
            _LAST_AUTO_RECORD_CSV = RECORD_CSV
        _JSON_LOG_PATH = os.path.join(base_session_dir, f"{SUBJECT_ID}.json")

        # Per-session Report/Notes CSV destinations. The composed
        # SUBJECT_ID already carries the session timestamp, so this
        # produces Report_alice_20260425_190621.csv etc. automatically.
        REPORT_FILE, NOTES_FILE = format_result_paths(SUBJECT_ID)
        logger.debug(f"[RESULTS] Per-session CSVs: {REPORT_FILE}, {NOTES_FILE}")

        _init_dossier()

        append_to_csv("internal", "system", f"Session initialized. User: {SUBJECT_ID}, DB ID: {SESSION_ID}")
        _INIT_DONE = True


def reset_session(new_user_id: str = None):
    """Explicit reset — tears down the init sentinel and re-initialises."""
    global _INIT_DONE
    with _INIT_LOCK:
        _INIT_DONE = False
    init_record(new_user_id, force=True)


def mark_session_finalised(reason: str = "normal"):
    """Close the current SESSION_ID row in DB cleanly on session end.

    Called by the handler on normal exit, by the speech service on
    user-initiated end, and by the `atexit` hook below on interpreter
    shutdown. Idempotent; safe to call twice.

    Phase B (strategy 1a): on a `reason="normal"` close (clean graceful
    end with the runtime still alive), we trigger the therapist-report
    CSV generator. On `reason="atexit"` the report call is a backstop —
    the internal run-once guard in `therapist_report._EXPORTED_SESSIONS`
    prevents duplicate files.
    """
    global _INIT_DONE, _DOSSIER
    current_session_id = SESSION_ID

    # Phase B: attempt therapist report before closing the session row,
    # so the export runs while all state (DB, subject_id) is still
    # available. Best-effort — report failure must not block session
    # closure or the atexit shutdown path.
    if DB and current_session_id:
        try:
            from src.utils.therapist_report import generate_therapist_report
            generate_therapist_report(current_session_id, db=DB)
        except Exception as e:
            logger.warning(f"therapist_report generation failed (non-fatal): {e}")

    if DB and current_session_id:
        try:
            DB.close_session(current_session_id, reason=reason)
        except Exception as e:
            logger.warning(f"Could not close session {current_session_id}: {e}")
    # Flush and close the open dossier so `interactions[]` is persisted
    # even when the process exits abruptly (SIGTERM from systemd, Ctrl-C).
    try:
        if _DOSSIER is not None and not _DOSSIER._closed:
            _DOSSIER.save_and_close()
    except Exception as e:
        logger.warning(f"Could not save dossier on shutdown: {e}")
    with _INIT_LOCK:
        _INIT_DONE = False


def _atexit_close_session():
    """atexit hook: stamp end_time on any still-open session at shutdown.

    Phase A: without this hook, SIGTERM/SIGINT leaves `sessions.end_time`
    NULL — producing the "dangling session" clutter that Phase A is meant
    to eliminate. The subsequent boot's crash-recovery path will mark
    those rows with reason='crash_recovery', which is misleading for a
    clean Ctrl-C exit.
    """
    try:
        mark_session_finalised(reason="atexit")
    except Exception as e:
        # atexit handlers must never raise — logging is best-effort.
        try:
            logger.warning(f"atexit session closure failed: {e}")
        except Exception:
            pass


atexit.register(_atexit_close_session)


# ── Queue-level logging ──────────────────────────────────────────────────

def log_question(text: str, meta_data: dict = None):
    """Log an agent question → OUTPUT_QUEUE + DB + CSV + dossier.

    Respects `_PENDING_QUESTION_PREFIX` so RV Validator / CBT recap can
    prepend context to the next question.
    """
    global _PENDING_QUESTION_PREFIX

    combined = text
    if _PENDING_QUESTION_PREFIX:
        combined = f"{_PENDING_QUESTION_PREFIX}\n\n{text}"
        logger.debug("Combining pending prefix with next question.")

    _safe_output_put(combined)

    if DB and SESSION_ID:
        idx = _next_turn_index()
        DB.add_turn(SESSION_ID, idx, "agent", combined, meta_data=meta_data)

    append_to_csv("turn", "agent", combined)
    log_json_event("agent_turn", {"text": combined})

    if _DOSSIER and not _DOSSIER._closed:
        _DOSSIER.record_interaction(
            raw_transcription=_LAST_USER_TRANSCRIPT,
            llm_response=combined,
            rl_decision_logic=_LAST_RL_STATE,
            ser_metrics={
                "emotion_tag": _LAST_USER_EMOTION,
                "screening_scores": _LATEST_SCREENING_SCORES,
            },
        )

    _PENDING_QUESTION_PREFIX = ""
    # Agent output is the model's output — not PII. Logged in full.
    logger.info(f"[AGENT] {combined}")
    # Track the last agent-tagged line so speech_service.say() can dedup
    # the [TTS] echo when it speaks the same handler-driven text.
    global _LAST_AGENT_LOGGED
    _LAST_AGENT_LOGGED = combined


def log_reasoning(reasoning_type: str, data: dict):
    """Log a system reasoning event (RL states, Semantic scores) to DB."""
    if DB and SESSION_ID:
        idx = _next_turn_index()
        meta = {"reasoning_type": reasoning_type}
        meta.update(data)
        DB.add_turn(SESSION_ID, idx, "system", f"[{reasoning_type.upper()}]", meta_data=meta)
        # DEBUG: internal reasoning breadcrumb, useful for forensics but
        # not part of the demo's spoken-turn log taxonomy. Divergence 7.
        logger.debug(f"Logged Reasoning ({reasoning_type}) to DB.")
    if reasoning_type == "rl_decision":
        set_rl_context(data)


def get_answer() -> Tuple[List, List[str]]:
    """Block on INPUT_QUEUE; return (DLA_result=[], segments)."""
    logger.debug("Waiting for user answer...")
    user_input_raw = None
    while user_input_raw is None:
        if END_SESSION_EVENT.is_set():
            logger.info("END_SESSION_EVENT detected while waiting for user answer.")
            return [], ["SESSION_END"]
        try:
            user_input_raw = INPUT_QUEUE.get(timeout=0.5)
        except queue.Empty:
            continue
    logger.info(f"[USER] {_redact(user_input_raw)}")

    if DB and SESSION_ID:
        idx = _next_turn_index()
        DB.add_turn(SESSION_ID, idx, "user", user_input_raw)

    append_to_csv("turn", "user", user_input_raw)

    emotion_str = "Neutral"
    try:
        parsed = json.loads(str(user_input_raw))
        user_input_text = parsed.get("transcript", "")
        emotion_str = parsed.get("detected_emotion", "Neutral")
    except Exception:
        user_input_text = str(user_input_raw)

    set_last_user_signal(user_input_text, emotion_str)
    segments = _segment_utterance(user_input_text)

    DLA_result = []
    log_json_event("user_turn", {"transcript": user_input_text, "emotion": emotion_str, "segments": segments})
    return DLA_result, segments


def get_resp_log() -> str:
    """Block on INPUT_QUEUE; return the raw user response string (RV path)."""
    logger.debug("Waiting for user response (raw)...")
    user_response_raw = None
    while user_response_raw is None:
        if END_SESSION_EVENT.is_set():
            logger.info("END_SESSION_EVENT detected while waiting for user response.")
            return "SESSION_END"
        try:
            user_response_raw = INPUT_QUEUE.get(timeout=0.5)
        except queue.Empty:
            continue

    try:
        parsed = json.loads(str(user_response_raw))
        transcript = parsed.get("transcript", "")
        emotion = parsed.get("detected_emotion", "Neutral")
        user_response = transcript
        set_last_user_signal(transcript, emotion)
    except Exception:
        user_response = str(user_response_raw)
        set_last_user_signal(user_response, "Neutral")

    if DB and SESSION_ID:
        idx = _next_turn_index()
        DB.add_turn(SESSION_ID, idx, "user", user_response)

    append_to_csv("turn", "user", user_response)
    log_json_event("user_turn", {"response": user_response})

    logger.info(f"[USER] {_redact(user_response)}")
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
        # Redact user turns in terminal dump (agent + system turns are AI
        # output and safe to print in full).
        if speaker == "USER":
            text = _redact(text)
        logger.info(f"[{idx:03d}] {speaker}: {text}")
    logger.info("=========== SESSION HISTORY END ===========")


# ══════════════════════════════════════════════════════════════════════════════
#  Session Dossier — per-interaction structured JSON log
# ══════════════════════════════════════════════════════════════════════════════

class SessionDossier:
    """Accumulates structured per-interaction records into a single JSON
    dumped to `data/sessions/session_<SUBJECT_ID>.json` on close.

    `subject_id` is expected to be the composed per-session id from
    io_record.SUBJECT_ID (e.g. "alice_20260425_190621"), which already
    carries the session timestamp — so the output filename is unique per
    session without appending a second timestamp.
    """

    def __init__(self, subject_id: str, session_id):
        import datetime
        self._subject = subject_id
        self._session_id = session_id
        self._dir = os.path.join(os.path.abspath("."), "data", "sessions")
        self._path = os.path.join(self._dir, f"session_{subject_id}.json")
        self._interactions: list = []
        self._meta: dict = {
            "subject_id": subject_id,
            "session_id": session_id,
            "started_at": datetime.datetime.now().isoformat(),
            "ended_at": None,
        }
        self._lock = threading.Lock()
        self._closed = False
        logger.debug(f"[DOSSIER] Initialized: {self._path}")

    def record_interaction(
        self,
        raw_transcription: str = "",
        llm_response: str = "",
        rl_decision_logic: dict | None = None,
        ser_metrics: dict | None = None,
        resource_telemetry: dict | None = None,
    ):
        if self._closed:
            return
        import datetime
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "raw_transcription": raw_transcription,
            "llm_response": llm_response,
            "rl_decision_logic": rl_decision_logic or {},
            "ser_metrics": ser_metrics or {},
            "resource_telemetry": resource_telemetry or {},
        }
        with self._lock:
            self._interactions.append(entry)

    def save_and_close(self):
        if self._closed:
            return
        import datetime
        self._meta["ended_at"] = datetime.datetime.now().isoformat()
        self._meta["total_interactions"] = len(self._interactions)
        payload = {
            "meta": self._meta,
            "interactions": self._interactions,
        }
        try:
            os.makedirs(self._dir, exist_ok=True)
            # Atomic dossier write — tmp + replace so a crash mid-dump
            # cannot corrupt the session file.
            tmp_path = self._path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, self._path)
            logger.debug(f"[DOSSIER] Saved {len(self._interactions)} interactions to {self._path}")
        except Exception as e:
            logger.error(f"[DOSSIER] Failed to save: {e}")
        finally:
            self._closed = True

    @property
    def path(self) -> str:
        return self._path


_DOSSIER: SessionDossier | None = None


def get_dossier() -> SessionDossier | None:
    return _DOSSIER


def _init_dossier():
    global _DOSSIER
    if _DOSSIER is not None and not _DOSSIER._closed:
        _DOSSIER.save_and_close()
    _DOSSIER = SessionDossier(SUBJECT_ID, SESSION_ID)
