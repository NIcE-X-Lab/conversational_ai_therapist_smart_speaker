import sqlite3
import datetime
import json
from src.utils.log_util import get_logger

logger = get_logger("DBManager")

class DBManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
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

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def get_user_id(self, subject_id):
        """Get user ID by subject_id, creating if not exists."""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO sessions (user_id) VALUES (?)", (user_id,))
        conn.commit()
        session_id = c.lastrowid
        conn.close()
        return session_id

    def add_turn(self, session_id, turn_index, speaker, text, meta_data=None):
        """Record a turn in the conversation."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        meta_json = json.dumps(meta_data) if meta_data else None
        c.execute("INSERT INTO turns (session_id, turn_index, speaker, text, meta_data) VALUES (?, ?, ?, ?, ?)",
                  (session_id, turn_index, speaker, text, meta_json))
        conn.commit()
        conn.close()

    def get_session_history(self, session_id):
        """Retrieve all turns for a session."""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO summaries (session_id, summary_text) VALUES (?, ?)", (session_id, summary_text))
        conn.commit()
        conn.close()

    def set_preference(self, user_id, key, value):
        """Set a user preference (upsert)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""INSERT INTO user_preferences (user_id, key, value) 
                     VALUES (?, ?, ?) 
                     ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
                  (user_id, key, value))
        conn.commit()
        conn.close()

    def get_preference(self, user_id, key):
        """Get a user preference."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM user_preferences WHERE user_id=? AND key=?", (user_id, key))
        result = c.fetchone()
        conn.close()
        return result[0] if result else None

    def log_feedback(self, session_id, rating, comments, turn_index=None):
        """Log user feedback."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO feedback (session_id, turn_index, rating, comments) VALUES (?, ?, ?, ?)",
                  (session_id, turn_index, rating, comments))
        conn.commit()
        conn.close()

    def log_safety_flag(self, session_id, flag_type, raw_text, severity, turn_index=None):
        """Log a safety flag."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO safety_flags (session_id, turn_index, flag_type, raw_text, severity) VALUES (?, ?, ?, ?, ?)",
                  (session_id, turn_index, flag_type, raw_text, severity))
        conn.commit()
        conn.close()
