"""Low-level driver managing exact transactions with SQLite persistence."""
import os
import sqlite3
import datetime
import json
from src.utils.log_util import get_logger

logger = get_logger("DBManager")

# All connections use a 5-second timeout to prevent indefinite hangs when
# another process holds the database lock.
_SQLITE_TIMEOUT = 5.0


class DBManager:
    def __init__(self, db_path):
        self.db_path = db_path
        # Ensure parent directory exists before opening the database.
        parent = os.path.dirname(db_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
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
                      FOREIGN KEY(user_id) REFERENCES users(id))''')

        # Turns table (stores the dialogue)
        c.execute('''CREATE TABLE IF NOT EXISTS turns
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id INTEGER,
                      turn_index INTEGER,
                      speaker TEXT, -- 'user' or 'agent'
                      text TEXT,
                      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      meta_data TEXT, -- JSON string for extra info (scores, etc.)
                      FOREIGN KEY(session_id) REFERENCES sessions(id))''')

        # Summaries table
        c.execute('''CREATE TABLE IF NOT EXISTS summaries
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id INTEGER,
                      summary_text TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY(session_id) REFERENCES sessions(id))''')

        # User Preferences table
        c.execute('''CREATE TABLE IF NOT EXISTS user_preferences
                     (user_id INTEGER,
                      key TEXT,
                      value TEXT,
                      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      PRIMARY KEY (user_id, key),
                      FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        # Feedback table
        c.execute('''CREATE TABLE IF NOT EXISTS feedback
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id INTEGER,
                      turn_index INTEGER,
                      rating TEXT,
                      comments TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY(session_id) REFERENCES sessions(id))''')

        # Safety Flags table
        c.execute('''CREATE TABLE IF NOT EXISTS safety_flags
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id INTEGER,
                      turn_index INTEGER,
                      flag_type TEXT,
                      raw_text TEXT,
                      severity INTEGER,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY(session_id) REFERENCES sessions(id))''')

        # Clinical Screening table (PHQ-4 / GAD-2 sub-scores)
        c.execute('''CREATE TABLE IF NOT EXISTS clinical_screening
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id INTEGER,
                  anxiety_score INTEGER,     -- GAD-2 sub-total (0-6)
                  depression_score INTEGER,  -- PHQ-2 sub-total (0-6)
                  phq4_total INTEGER,        -- composite (0-12)
                  gad2_positive INTEGER DEFAULT 0,
                  phq4_high_risk INTEGER DEFAULT 0,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(session_id) REFERENCES sessions(id))''')

        # Intermission Screening status table (per PHQ question in waiting ladder)
        c.execute('''CREATE TABLE IF NOT EXISTS intermission_screening
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id INTEGER,
                  question_id TEXT,
                  status TEXT,               -- ANSWERED or SKIPPED
                  score INTEGER,
                  response_text TEXT,
                  reason TEXT,
                  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  UNIQUE(session_id, question_id),
                  FOREIGN KEY(session_id) REFERENCES sessions(id))''')

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def get_user_id(self, subject_id):
        """Get user ID by subject_id, creating if not exists."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE subject_id=?", (subject_id,))
        result = c.fetchone()
        if result:
            user_id = result[0]
        else:
            c.execute("INSERT INTO users (subject_id) VALUES (?)", (subject_id,))
            conn.commit()
            user_id = c.lastrowid
        conn.close()
        return user_id

    def create_session(self, user_id):
        """Create a new session for the user."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute("INSERT INTO sessions (user_id) VALUES (?)", (user_id,))
        conn.commit()
        session_id = c.lastrowid
        conn.close()
        return session_id

    def add_turn(self, session_id, turn_index, speaker, text, meta_data=None):
        """Record a turn in the conversation."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        meta_json = json.dumps(meta_data) if meta_data else None
        c.execute("INSERT INTO turns (session_id, turn_index, speaker, text, meta_data) VALUES (?, ?, ?, ?, ?)",
                  (session_id, turn_index, speaker, text, meta_json))
        conn.commit()
        conn.close()

    def get_session_history(self, session_id):
        """Retrieve all turns for a session."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute("SELECT speaker, text, meta_data FROM turns WHERE session_id=? ORDER BY turn_index", (session_id,))
        rows = c.fetchall()
        conn.close()
        history = []
        for r in rows:
            history.append({
                "speaker": r[0],
                "text": r[1],
                "meta_data": json.loads(r[2]) if r[2] else None
            })
        return history

    # --- New Methods for Extensions ---

    def add_summary(self, session_id, summary_text):
        """Add a summary for a session."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute("INSERT INTO summaries (session_id, summary_text) VALUES (?, ?)", (session_id, summary_text))
        conn.commit()
        conn.close()

    def set_preference(self, user_id, key, value):
        """Set a user preference (upsert)."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute("""INSERT INTO user_preferences (user_id, key, value) 
                     VALUES (?, ?, ?) 
                     ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
                  (user_id, key, value))
        conn.commit()
        conn.close()

    def get_preference(self, user_id, key):
        """Get a user preference."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute("SELECT value FROM user_preferences WHERE user_id=? AND key=?", (user_id, key))
        result = c.fetchone()
        conn.close()
        return result[0] if result else None

    def log_feedback(self, session_id, rating, comments, turn_index=None):
        """Log user feedback."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute("INSERT INTO feedback (session_id, turn_index, rating, comments) VALUES (?, ?, ?, ?)",
                  (session_id, turn_index, rating, comments))
        conn.commit()
        conn.close()

    def log_safety_flag(self, session_id, flag_type, raw_text, severity, turn_index=None):
        """Log a safety flag."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute("INSERT INTO safety_flags (session_id, turn_index, flag_type, raw_text, severity) VALUES (?, ?, ?, ?, ?)",
                  (session_id, turn_index, flag_type, raw_text, severity))
        conn.commit()
        conn.close()

    def get_user_context_string(self, user_id, limit=3):
        """
        Retrieve a formatted string of user context: preferences and recent summaries.
        """
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()

        # Get Preferences
        c.execute("SELECT key, value FROM user_preferences WHERE user_id=?", (user_id,))
        prefs = c.fetchall()
        prefs_str = "\n".join([f"- {k}: {v}" for k, v in prefs])
        
        # Get Recent Summaries (need join)
        # Summaries table has session_id, need to join with sessions table to filter by user_id
        c.execute("""SELECT summaries.summary_text, summaries.created_at FROM summaries 
                     JOIN sessions ON summaries.session_id = sessions.id
                     WHERE sessions.user_id=? 
                     ORDER BY sessions.start_time DESC LIMIT ?""", (user_id, limit))
        sums = c.fetchall()
        sums_str = "\n".join([f"- {s[1]}: {s[0]}" for s in sums])
        
        conn.close()
        
        context = ""
        if prefs_str:
            context += f"\n[User Preferences]\n{prefs_str}\n"
        if sums_str:
            context += f"\n[Recent Session Summaries]\n{sums_str}\n"

        return context

    def log_screening_scores(
        self,
        session_id: int,
        anxiety_score: int | None,
        depression_score: int | None,
        phq4_total: int | None,
    ):
        """
        Persist PHQ-4 / GAD-2 sub-scores for a session.
        Automatically sets gad2_positive and phq4_high_risk flags.
        """
        from src.core.therapy_content import GAD2_THRESHOLD, PHQ4_THRESHOLD
        gad2_pos  = int((anxiety_score or 0) >= GAD2_THRESHOLD)
        phq4_risk = int((phq4_total   or 0) >= PHQ4_THRESHOLD)

        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute(
            """INSERT INTO clinical_screening
               (session_id, anxiety_score, depression_score, phq4_total, gad2_positive, phq4_high_risk)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, anxiety_score, depression_score, phq4_total, gad2_pos, phq4_risk),
        )
        conn.commit()
        conn.close()
        logger.info(
            f"Screening scores stored — anxiety={anxiety_score}, "
            f"depression={depression_score}, PHQ-4={phq4_total}, "
            f"GAD2_pos={bool(gad2_pos)}, PHQ4_risk={bool(phq4_risk)}"
        )

    def get_screening_scores(self, session_id: int):
        """Fetch the latest screening scores for a session."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute(
            """SELECT anxiety_score, depression_score, phq4_total, gad2_positive, phq4_high_risk 
               FROM clinical_screening WHERE session_id=? ORDER BY created_at DESC LIMIT 1""",
            (session_id,)
        )
        row = c.fetchone()
        conn.close()
        if row:
            return {
                "anxiety": row[0],
                "depression": row[1],
                "total": row[2],
                "gad2_positive": bool(row[3]),
                "phq4_high_risk": bool(row[4])
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
        """Create or update per-question intermission screening status."""
        norm_status = str(status or "").upper().strip()
        if norm_status not in {"ANSWERED", "SKIPPED"}:
            raise ValueError(f"Invalid intermission status: {status}")

        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute(
            """INSERT INTO intermission_screening
               (session_id, question_id, status, score, response_text, reason, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(session_id, question_id) DO UPDATE SET
                   status=excluded.status,
                   score=excluded.score,
                   response_text=excluded.response_text,
                   reason=excluded.reason,
                   updated_at=CURRENT_TIMESTAMP""",
            (
                session_id,
                question_id,
                norm_status,
                score,
                response_text,
                reason,
            ),
        )
        conn.commit()
        conn.close()

    def get_intermission_screening_statuses(self, session_id: int):
        """Return the latest intermission status for each screening question."""
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute(
            """SELECT question_id, status, score, response_text, reason
               FROM intermission_screening WHERE session_id=?""",
            (session_id,),
        )
        rows = c.fetchall()
        conn.close()
        result = {}
        for qid, status, score, response_text, reason in rows:
            result[qid] = {
                "status": status,
                "score": score,
                "response_text": response_text,
                "reason": reason,
            }
        return result
