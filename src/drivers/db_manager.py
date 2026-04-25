"""Low-level driver managing exact transactions with SQLite persistence.

Clinical-trial hardening:
- Every DB method wraps its connection in a `_connect()` context manager so
  exceptions NEVER leak file handles (connection leaks under load cause
  SQLite busy-lock cascades that silently corrupt clinical records).
- Writes use a single transaction boundary per method; `_connect()` commits
  on success and rolls back on error.
- Adds `clinical_flags` (M3), `safety_deliveries` (C2), and session-close
  helper `close_open_sessions` (M2 crash recovery) tables.
"""

import os
import sqlite3
import json
from contextlib import contextmanager

from src.utils.log_util import get_logger

logger = get_logger("DBManager")

# 5-second lock timeout prevents indefinite hangs when another writer is
# in-flight. On exceeding this, we raise instead of blocking the clinical
# pipeline — a clear error is safer than a silent freeze mid-session.
_SQLITE_TIMEOUT = 5.0


class DBManager:
    def __init__(self, db_path):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        """Context-managed sqlite connection with guaranteed close + rollback.

        Any exception inside the `with` block triggers a rollback and still
        closes the handle before re-raising. This is the ONLY path by which
        the rest of the module should acquire a connection.
        """
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _migrate_schema(self, c):
        """Apply additive schema migrations for existing trial databases.

        Each migration is idempotent (check-then-add) so running against a
        DB created by an older build is safe.
        """
        # Migration: sessions.end_reason (added for C2 / M2).
        try:
            c.execute("SELECT end_reason FROM sessions LIMIT 1")
        except Exception:
            try:
                c.execute("ALTER TABLE sessions ADD COLUMN end_reason TEXT")
                logger.info("[MIGRATE] Added sessions.end_reason column.")
            except Exception as e:
                logger.warning(f"[MIGRATE] sessions.end_reason add failed: {e}")

    def _init_db(self):
        """Create all tables if they do not already exist (idempotent)."""
        with self._connect() as conn:
            c = conn.cursor()

            # Users table
            c.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          subject_id TEXT UNIQUE,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

            # Sessions table
            c.execute('''CREATE TABLE IF NOT EXISTS sessions
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id INTEGER,
                          start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          end_time TIMESTAMP,
                          end_reason TEXT,
                          FOREIGN KEY(user_id) REFERENCES users(id))''')

            # Turns table (stores the dialogue)
            c.execute('''CREATE TABLE IF NOT EXISTS turns
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          session_id INTEGER,
                          turn_index INTEGER,
                          speaker TEXT,
                          text TEXT,
                          timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          meta_data TEXT,
                          FOREIGN KEY(session_id) REFERENCES sessions(id))''')

            # Summaries table
            c.execute('''CREATE TABLE IF NOT EXISTS summaries
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          session_id INTEGER,
                          summary_text TEXT,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          FOREIGN KEY(session_id) REFERENCES sessions(id))''')

            # User preferences
            c.execute('''CREATE TABLE IF NOT EXISTS user_preferences
                         (user_id INTEGER,
                          key TEXT,
                          value TEXT,
                          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          PRIMARY KEY (user_id, key),
                          FOREIGN KEY(user_id) REFERENCES users(id))''')

            # Feedback
            c.execute('''CREATE TABLE IF NOT EXISTS feedback
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          session_id INTEGER,
                          turn_index INTEGER,
                          rating TEXT,
                          comments TEXT,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          FOREIGN KEY(session_id) REFERENCES sessions(id))''')

            # Safety flags (preserved for legacy writes)
            c.execute('''CREATE TABLE IF NOT EXISTS safety_flags
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          session_id INTEGER,
                          turn_index INTEGER,
                          flag_type TEXT,
                          raw_text TEXT,
                          severity INTEGER,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          FOREIGN KEY(session_id) REFERENCES sessions(id))''')

            # Clinical screening sub-scores
            c.execute('''CREATE TABLE IF NOT EXISTS clinical_screening
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id INTEGER,
                      anxiety_score INTEGER,
                      depression_score INTEGER,
                      phq4_total INTEGER,
                      gad2_positive INTEGER DEFAULT 0,
                      phq4_high_risk INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY(session_id) REFERENCES sessions(id))''')

            # Intermission per-question tracker
            c.execute('''CREATE TABLE IF NOT EXISTS intermission_screening
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id INTEGER,
                      question_id TEXT,
                      status TEXT,
                      score INTEGER,
                      response_text TEXT,
                      reason TEXT,
                      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      UNIQUE(session_id, question_id),
                      FOREIGN KEY(session_id) REFERENCES sessions(id))''')

            # Persistent RL state (longitudinal Q-table)
            c.execute('''CREATE TABLE IF NOT EXISTS persistent_rl_state
                     (user_id INTEGER PRIMARY KEY,
                      q_table_json TEXT,
                      item_mask_json TEXT,
                      top_score2_dims_json TEXT,
                      last_session_id INTEGER,
                      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY(user_id) REFERENCES users(id))''')

            # M3 — consolidated clinical flag log (single authoritative audit
            # trail for auditors). Mirrors legacy CSV writes but in one place.
            c.execute('''CREATE TABLE IF NOT EXISTS clinical_flags
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id INTEGER,
                      flag_type TEXT,
                      details_json TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY(session_id) REFERENCES sessions(id))''')

            # C2 — safety-message delivery audit: every crisis resource
            # broadcast records whether the message was successfully spoken,
            # the delivery method (TTS vs file-fallback vs cached-wav), and
            # the critical dimension that triggered it.
            c.execute('''CREATE TABLE IF NOT EXISTS safety_deliveries
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id INTEGER,
                      critical_dim TEXT,
                      method TEXT,
                      success INTEGER DEFAULT 0,
                      message_text TEXT,
                      error_text TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY(session_id) REFERENCES sessions(id))''')

            # Phase B — clinical_scores: one row per (session, dimension)
            # capturing the FINAL score the Response Analyzer assigned to
            # that dimension. This is the canonical row the Therapist
            # Export report joins against. `evidence_turn_id` points to
            # the user turn that sourced the final score.
            c.execute('''CREATE TABLE IF NOT EXISTS clinical_scores
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id INTEGER NOT NULL,
                      dim_index INTEGER NOT NULL,
                      dim_label TEXT NOT NULL,
                      dim_name TEXT,
                      score INTEGER NOT NULL,
                      evidence_text TEXT,
                      evidence_turn_id INTEGER,
                      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      UNIQUE(session_id, dim_label),
                      FOREIGN KEY(session_id) REFERENCES sessions(id),
                      FOREIGN KEY(evidence_turn_id) REFERENCES turns(id))''')

            # Phase B — clinical_score_attempts: every scored attempt per
            # dimension, including UNRESOLVED (-2) and OPT_OUT (-1)
            # sentinels. A therapist reviewing a session needs the full
            # history ("user scored 1 on first ask, then 2 on the RV
            # follow-up") not only the final. Append-only; `clinical_scores`
            # is the consolidated last-word row per dimension.
            c.execute('''CREATE TABLE IF NOT EXISTS clinical_score_attempts
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id INTEGER NOT NULL,
                      dim_index INTEGER NOT NULL,
                      dim_label TEXT NOT NULL,
                      dim_name TEXT,
                      score INTEGER NOT NULL,
                      evidence_text TEXT,
                      evidence_turn_id INTEGER,
                      source TEXT,
                      attempt_index INTEGER,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY(session_id) REFERENCES sessions(id),
                      FOREIGN KEY(evidence_turn_id) REFERENCES turns(id))''')

            # Phase B — intervention_logs: MI reflections + CBT stage
            # completions. One row per intervention event. `technique`
            # captures the specific MI move (e.g. "simple_reflection",
            # "guide_redirect") or CBT stage label (e.g. "recognize",
            # "challenge", "reframe"). `outcome` is one of {started,
            # success, failed, escalated}.
            c.execute('''CREATE TABLE IF NOT EXISTS intervention_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id INTEGER NOT NULL,
                      kind TEXT NOT NULL,
                      stage TEXT,
                      technique TEXT,
                      outcome TEXT,
                      dim_label TEXT,
                      detail_json TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY(session_id) REFERENCES sessions(id))''')

            # Apply any additive migrations for pre-existing DBs.
            self._migrate_schema(c)

        logger.info(f"Database initialized at {self.db_path}")

    # ── Core session / user ──────────────────────────────────────────────

    def get_user_id(self, subject_id):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("SELECT id FROM users WHERE subject_id=?", (subject_id,))
            result = c.fetchone()
            if result:
                return result[0]
            c.execute("INSERT INTO users (subject_id) VALUES (?)", (subject_id,))
            return c.lastrowid

    def create_session(self, user_id):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("INSERT INTO sessions (user_id) VALUES (?)", (user_id,))
            return c.lastrowid

    def close_session(self, session_id, reason: str = "normal"):
        """Mark a session closed with a reason (used on normal exit and M2 recovery)."""
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "UPDATE sessions SET end_time=CURRENT_TIMESTAMP, end_reason=? WHERE id=?",
                (reason, session_id),
            )

    def close_open_sessions_for_user(self, user_id: int, reason: str = "crash_recovery"):
        """M2: close any session rows left open (end_time IS NULL) from prior crashes.

        Returns the list of closed session IDs so callers can record in logs
        that recovery happened.
        """
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT id FROM sessions WHERE user_id=? AND end_time IS NULL",
                (user_id,),
            )
            ids = [r[0] for r in c.fetchall()]
            if ids:
                c.executemany(
                    "UPDATE sessions SET end_time=CURRENT_TIMESTAMP, end_reason=? WHERE id=?",
                    [(reason, sid) for sid in ids],
                )
        return ids

    # ── Turns ────────────────────────────────────────────────────────────

    def add_turn(self, session_id, turn_index, speaker, text, meta_data=None):
        meta_json = json.dumps(meta_data) if meta_data else None
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO turns (session_id, turn_index, speaker, text, meta_data) VALUES (?, ?, ?, ?, ?)",
                (session_id, turn_index, speaker, text, meta_json),
            )

    def get_session_history(self, session_id):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT speaker, text, meta_data FROM turns WHERE session_id=? ORDER BY turn_index",
                (session_id,),
            )
            rows = c.fetchall()
        return [
            {
                "speaker": r[0],
                "text": r[1],
                "meta_data": json.loads(r[2]) if r[2] else None,
            }
            for r in rows
        ]

    # ── Summaries / preferences / feedback ───────────────────────────────

    def add_summary(self, session_id, summary_text):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO summaries (session_id, summary_text) VALUES (?, ?)",
                (session_id, summary_text),
            )

    def set_preference(self, user_id, key, value):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO user_preferences (user_id, key, value)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
                (user_id, key, value),
            )

    def get_preference(self, user_id, key):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT value FROM user_preferences WHERE user_id=? AND key=?",
                (user_id, key),
            )
            result = c.fetchone()
        return result[0] if result else None

    def get_all_preferences(self, user_id):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("SELECT key, value FROM user_preferences WHERE user_id=?", (user_id,))
            rows = c.fetchall()
        return {k: v for k, v in rows}

    def log_feedback(self, session_id, rating, comments, turn_index=None):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (session_id, turn_index, rating, comments) VALUES (?, ?, ?, ?)",
                (session_id, turn_index, rating, comments),
            )

    # ── Safety / clinical flags ──────────────────────────────────────────

    def log_safety_flag(self, session_id, flag_type, raw_text, severity, turn_index=None):
        """Legacy safety_flags row (kept for backwards compatibility)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO safety_flags (session_id, turn_index, flag_type, raw_text, severity) VALUES (?, ?, ?, ?, ?)",
                (session_id, turn_index, flag_type, raw_text, severity),
            )

    def log_clinical_flag(self, session_id, flag_type: str, details: dict | None = None):
        """M3: authoritative clinical flag log for auditors.

        A single table recording every clinically significant event in one
        place. `details` is serialised as JSON so new flag types don't
        require schema changes.
        """
        details_json = json.dumps(details) if details else None
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO clinical_flags (session_id, flag_type, details_json) VALUES (?, ?, ?)",
                (session_id, flag_type, details_json),
            )

    def log_safety_delivery(
        self,
        session_id: int,
        critical_dim: str,
        method: str,
        success: bool,
        message_text: str,
        error_text: str | None = None,
    ):
        """C2: audit trail proving whether a crisis resource message was delivered.

        If a subsequent session review shows a CRITICAL_DIMS trigger with no
        corresponding successful safety_deliveries row, that's a clinical
        incident requiring investigation.
        """
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO safety_deliveries
                   (session_id, critical_dim, method, success, message_text, error_text)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    critical_dim,
                    method,
                    int(bool(success)),
                    message_text,
                    error_text,
                ),
            )

    def get_safety_deliveries(self, session_id: int):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """SELECT critical_dim, method, success, message_text, error_text, created_at
                   FROM safety_deliveries WHERE session_id=? ORDER BY id ASC""",
                (session_id,),
            )
            rows = c.fetchall()
        return [
            {
                "critical_dim": r[0],
                "method": r[1],
                "success": bool(r[2]),
                "message_text": r[3],
                "error_text": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    # ── User context for greeting ────────────────────────────────────────

    def get_user_context_string(self, user_id, limit=3):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("SELECT key, value FROM user_preferences WHERE user_id=?", (user_id,))
            prefs = c.fetchall()
            c.execute(
                """SELECT summaries.summary_text, summaries.created_at FROM summaries
                   JOIN sessions ON summaries.session_id = sessions.id
                   WHERE sessions.user_id=?
                   ORDER BY sessions.start_time DESC LIMIT ?""",
                (user_id, limit),
            )
            sums = c.fetchall()

        prefs_str = "\n".join([f"- {k}: {v}" for k, v in prefs])
        sums_str = "\n".join([f"- {s[1]}: {s[0]}" for s in sums])
        context = ""
        if prefs_str:
            context += f"\n[User Preferences]\n{prefs_str}\n"
        if sums_str:
            context += f"\n[Recent Session Summaries]\n{sums_str}\n"
        return context

    # ── Clinical screening ───────────────────────────────────────────────

    def log_screening_scores(
        self,
        session_id: int,
        anxiety_score: int | None,
        depression_score: int | None,
        phq4_total: int | None,
    ):
        from src.core.therapy_content import GAD2_THRESHOLD, PHQ4_THRESHOLD
        gad2_pos = int((anxiety_score or 0) >= GAD2_THRESHOLD)
        phq4_risk = int((phq4_total or 0) >= PHQ4_THRESHOLD)

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO clinical_screening
                   (session_id, anxiety_score, depression_score, phq4_total, gad2_positive, phq4_high_risk)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, anxiety_score, depression_score, phq4_total, gad2_pos, phq4_risk),
            )
        logger.info(
            f"Screening scores stored — anxiety={anxiety_score}, "
            f"depression={depression_score}, PHQ-4={phq4_total}, "
            f"GAD2_pos={bool(gad2_pos)}, PHQ4_risk={bool(phq4_risk)}"
        )

    def get_recent_screening_scores(self, user_id: int, limit: int = 5):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """SELECT cs.session_id, cs.anxiety_score, cs.depression_score,
                          cs.phq4_total, cs.gad2_positive, cs.phq4_high_risk,
                          cs.created_at
                   FROM clinical_screening cs
                   JOIN sessions s ON cs.session_id = s.id
                   WHERE s.user_id = ?
                   ORDER BY s.start_time DESC, cs.created_at DESC, cs.id DESC
                   LIMIT ?""",
                (user_id, limit),
            )
            rows = c.fetchall()
        return [
            {
                "session_id": r[0],
                "anxiety": r[1],
                "depression": r[2],
                "total": r[3],
                "gad2_positive": bool(r[4]),
                "phq4_high_risk": bool(r[5]),
                "created_at": r[6],
            }
            for r in rows
        ]

    def get_screening_scores(self, session_id: int):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """SELECT anxiety_score, depression_score, phq4_total, gad2_positive, phq4_high_risk
                   FROM clinical_screening WHERE session_id=? ORDER BY created_at DESC LIMIT 1""",
                (session_id,),
            )
            row = c.fetchone()
        if row:
            return {
                "anxiety": row[0],
                "depression": row[1],
                "total": row[2],
                "gad2_positive": bool(row[3]),
                "phq4_high_risk": bool(row[4]),
            }
        return None

    def upsert_intermission_screening_status(
        self,
        session_id: int,
        question_id: str,
        status: str,
        score: int | None = None,
        response_text: str | None = None,
        reason: str | None = None,
    ):
        norm_status = str(status or "").upper().strip()
        # Phase A: UNRESOLVED represents a non-empty-but-unparseable STT result
        # (distinct from the user explicitly opting out, which stays SKIPPED).
        if norm_status not in {"ANSWERED", "SKIPPED", "UNRESOLVED"}:
            raise ValueError(f"Invalid intermission status: {status}")

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO intermission_screening
                   (session_id, question_id, status, score, response_text, reason, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(session_id, question_id) DO UPDATE SET
                       status=excluded.status,
                       score=excluded.score,
                       response_text=excluded.response_text,
                       reason=excluded.reason,
                       updated_at=CURRENT_TIMESTAMP""",
                (session_id, question_id, norm_status, score, response_text, reason),
            )

    def get_intermission_screening_statuses(self, session_id: int):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """SELECT question_id, status, score, response_text, reason
                   FROM intermission_screening WHERE session_id=?""",
                (session_id,),
            )
            rows = c.fetchall()
        result = {}
        for qid, status, score, response_text, reason in rows:
            result[qid] = {
                "status": status,
                "score": score,
                "response_text": response_text,
                "reason": reason,
            }
        return result

    # ── Persistent longitudinal RL state ─────────────────────────────────

    def save_rl_state(
        self,
        user_id: int,
        q_table_json: str,
        item_mask_json: str,
        top_score2_dims_json: str,
        last_session_id: int | None = None,
    ):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO persistent_rl_state
                   (user_id, q_table_json, item_mask_json, top_score2_dims_json,
                    last_session_id, updated_at)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(user_id) DO UPDATE SET
                       q_table_json=excluded.q_table_json,
                       item_mask_json=excluded.item_mask_json,
                       top_score2_dims_json=excluded.top_score2_dims_json,
                       last_session_id=excluded.last_session_id,
                       updated_at=CURRENT_TIMESTAMP""",
                (user_id, q_table_json, item_mask_json, top_score2_dims_json, last_session_id),
            )

    def load_rl_state(self, user_id: int):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """SELECT q_table_json, item_mask_json, top_score2_dims_json,
                          last_session_id, updated_at
                   FROM persistent_rl_state WHERE user_id=?""",
                (user_id,),
            )
            row = c.fetchone()
        if not row:
            return None
        return {
            "q_table_json": row[0],
            "item_mask_json": row[1],
            "top_score2_dims_json": row[2],
            "last_session_id": row[3],
            "updated_at": row[4],
        }

    # ── Phase B: Clinical scores + intervention logs ─────────────────────

    def record_clinical_score(
        self,
        session_id: int,
        dim_index: int,
        dim_label: str,
        score: int,
        dim_name: str | None = None,
        evidence_text: str | None = None,
        evidence_turn_id: int | None = None,
        source: str = "response_analyzer",
        attempt_index: int | None = None,
    ):
        """Phase B: persist a scoring event.

        Writes BOTH:
          - `clinical_score_attempts` (append-only, full history of every
            scored attempt including UNRESOLVED / OPT_OUT sentinels).
          - `clinical_scores` (one row per (session, dim); the last write
            wins, representing the FINAL score the Response Analyzer
            settled on). UNRESOLVED/OPT_OUT sentinels are recorded in
            attempts but DO NOT overwrite a previously valid (0/1/2)
            final score — a valid final score is more clinically
            meaningful than a trailing ambiguous retry.
        """
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """INSERT INTO clinical_score_attempts
                   (session_id, dim_index, dim_label, dim_name, score,
                    evidence_text, evidence_turn_id, source, attempt_index)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, dim_index, dim_label, dim_name, int(score),
                 evidence_text, evidence_turn_id, source, attempt_index),
            )

            # Only propagate valid 0/1/2 scores into the canonical
            # clinical_scores row. Sentinels (-1 OPT_OUT, -2 UNRESOLVED)
            # are kept out of the final-score table so a therapist export
            # reflects scored dimensions only.
            if int(score) in (0, 1, 2):
                c.execute(
                    """INSERT INTO clinical_scores
                       (session_id, dim_index, dim_label, dim_name, score,
                        evidence_text, evidence_turn_id, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                       ON CONFLICT(session_id, dim_label) DO UPDATE SET
                           dim_index=excluded.dim_index,
                           dim_name=excluded.dim_name,
                           score=excluded.score,
                           evidence_text=excluded.evidence_text,
                           evidence_turn_id=excluded.evidence_turn_id,
                           updated_at=CURRENT_TIMESTAMP""",
                    (session_id, dim_index, dim_label, dim_name, int(score),
                     evidence_text, evidence_turn_id),
                )

    def get_clinical_scores(self, session_id: int):
        """Return the consolidated final-score rows for a session."""
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """SELECT dim_index, dim_label, dim_name, score,
                          evidence_text, evidence_turn_id, updated_at
                   FROM clinical_scores WHERE session_id=?
                   ORDER BY dim_index ASC""",
                (session_id,),
            )
            rows = c.fetchall()
        return [
            {
                "dim_index": r[0],
                "dim_label": r[1],
                "dim_name": r[2],
                "score": r[3],
                "evidence_text": r[4],
                "evidence_turn_id": r[5],
                "updated_at": r[6],
            }
            for r in rows
        ]

    def get_clinical_score_attempts(self, session_id: int):
        """Return every scored attempt (append-only history) for a session."""
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """SELECT dim_index, dim_label, dim_name, score,
                          evidence_text, evidence_turn_id, source,
                          attempt_index, created_at
                   FROM clinical_score_attempts WHERE session_id=?
                   ORDER BY dim_index ASC, created_at ASC, id ASC""",
                (session_id,),
            )
            rows = c.fetchall()
        return [
            {
                "dim_index": r[0],
                "dim_label": r[1],
                "dim_name": r[2],
                "score": r[3],
                "evidence_text": r[4],
                "evidence_turn_id": r[5],
                "source": r[6],
                "attempt_index": r[7],
                "created_at": r[8],
            }
            for r in rows
        ]

    def record_intervention_log(
        self,
        session_id: int,
        kind: str,
        stage: str | None = None,
        technique: str | None = None,
        outcome: str | None = None,
        dim_label: str | None = None,
        detail: dict | None = None,
    ):
        """Phase B: record an MI reflection or CBT stage event.

        `kind`      is one of {"MI", "CBT"}.
        `stage`     CBT stage ("recognize" / "challenge" / "reframe"),
                    or an MI sub-phase label.
        `technique` specific move ("simple_reflection", "guide_redirect",
                    "oars_validation"...) or stage-specific tag.
        `outcome`   {"started", "success", "failed", "escalated", "paused"}.
        `dim_label` focal dimension.
        `detail`    free-form dict serialised as JSON.
        """
        detail_json = json.dumps(detail) if detail else None
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO intervention_logs
                   (session_id, kind, stage, technique, outcome, dim_label, detail_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, kind, stage, technique, outcome, dim_label, detail_json),
            )

    def get_intervention_logs(self, session_id: int):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """SELECT kind, stage, technique, outcome, dim_label,
                          detail_json, created_at
                   FROM intervention_logs WHERE session_id=?
                   ORDER BY id ASC""",
                (session_id,),
            )
            rows = c.fetchall()
        return [
            {
                "kind": r[0],
                "stage": r[1],
                "technique": r[2],
                "outcome": r[3],
                "dim_label": r[4],
                "detail": json.loads(r[5]) if r[5] else None,
                "created_at": r[6],
            }
            for r in rows
        ]
