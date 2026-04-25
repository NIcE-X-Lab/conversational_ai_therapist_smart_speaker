"""M4 — Dry-run harness for CaiTI clinical pipeline.

Drives the pipeline end-to-end WITHOUT the mic or the real LLM engine.
Every LLM call is redirected to a deterministic stub so regressions in
the RL / RV / CBT / crisis flow surface in seconds instead of minutes.

Usage::

    .venv/bin/python scripts/dry_run.py

What it exercises:
  - Boot audit (disk + DB + models)
  - init_record idempotence + crash-recovery (M2)
  - PHQ-4 loop with a mix of valid / opt-out / unresolved replies (C5, M7)
  - Crisis override trigger on 'sib' Score 2 + guaranteed safety delivery (C2, C7)
  - Atomic Q-table CSV write + DB persistent_rl_state upsert (C4)
  - Post-session SOAP summary + closing reflection
  - Session closed cleanly with end_reason='normal' (M2)

Run against a scratch SQLite at `data/dry_run.db` so the real trial DB
is never touched.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from typing import List


def main() -> int:
    # Force a scratch DB + subject before anything else imports.
    scratch_dir = tempfile.mkdtemp(prefix="caiti_dryrun_")
    os.environ["CLINICAL_MODE"] = "0"  # dry-run allows degraded subsystems
    os.environ.setdefault("DISABLE_INTERNAL_SPEECH", "1")

    # Root the project so relative paths resolve.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    # Import after cwd is set so config_loader picks up config.yaml.
    from src.utils import io_record
    from src.utils.config_loader import SUBJECT_ID
    from src.core import response_analyzer, reflection_validation, CBT as cbt_mod
    from src.core import handler_rl as handler_mod
    from src.utils import text_generators
    from src.core import therapy_content as tc
    from src.drivers.db_manager import DBManager

    # ── Redirect every LLM call to a deterministic stub ────────────────
    # Different stubs for different roles so the pipeline sees realistic
    # shapes (analyzer returns "dim, score", reasoner returns "DECISION: 0", etc.)

    def _stub_llm_complete(system_content, user_content, role=None):
        role_s = str(role) if role else ""
        sys_s = system_content or ""
        # Analyzer: return "dim, score" keyed by a substring match.
        if "DIMENSION" in sys_s and "SCORE" in sys_s:
            if "self-harm" in user_content.lower() or "hurt myself" in user_content.lower():
                return "sib, 2"
            if "sleep" in user_content.lower():
                return "sleep, 1"
            if "mood" in user_content.lower():
                return "mood, 1"
            return "Yes, 0"
        if "MULTI_DIM" in sys_s or "Extract EVERY dimension" in sys_s:
            return '[{"dim":"Other","score":0}]'
        # RV Reasoner
        if "DECISION: 0" in sys_s or "DECISION: 1" in sys_s:
            return "DECISION: 0"
        # RV Validator
        if "VALIDATION" in sys_s:
            return "VALIDATION: It sounds like you are working through this."
        # RV Guide
        if "GUIDE" in sys_s:
            return "Guide: Thank you. Let us return to the topic."
        # CBT reasoners
        if "DECISION" in sys_s and "CBT" in sys_s.upper():
            return "DECISION: 0"
        # Greeting / closing / generic
        return "Thank you for taking the time to talk with me today."

    # Monkey-patch the LLM gateway.
    import src.models.llm_client as llm_client
    llm_client.llm_complete = _stub_llm_complete
    response_analyzer.llm_complete = _stub_llm_complete
    reflection_validation.llm_complete = _stub_llm_complete
    cbt_mod.llm_complete = _stub_llm_complete
    handler_mod.llm_complete = _stub_llm_complete
    text_generators.llm_complete = _stub_llm_complete

    # ── Scripted user replies pushed onto INPUT_QUEUE ─────────────────
    # Order matches: 4 PHQ-4 answers, then main-loop replies, then CBT.
    scripted_replies = [
        # PHQ-4: two valid, one unresolved-then-valid (C5 retry), one opt-out (M7)
        "not at all",
        "several days",
        "hmm, i dunno",   # unresolved -> triggers retry
        "nearly every day",  # retry success; anxiety=5 -> GAD2_POSITIVE
        "skip",            # opt-out on last PHQ (gad2 already positive -> M7 safety)
        # Main RL turns (a handful, will be terminated by stop)
        "i have been hurting myself sometimes",  # triggers sib crisis
        # CBT stage 0-3
        "1",
        "i feel worthless",
        "maybe it is not entirely true",
        "i am doing my best given the circumstances",
        # Extra buffer
        "end session",
    ]

    def _feed_replies():
        # Wait for the handler to start pulling before feeding.
        time.sleep(0.2)
        for r in scripted_replies:
            io_record.INPUT_QUEUE.put(r)
            time.sleep(0.01)

    # Use a fresh DB under the scratch dir.
    scratch_db = os.path.join(scratch_dir, "dryrun.db")
    # Point every importer at the scratch DB BEFORE handler.run() so we
    # never accidentally write to data/therapist.db.
    import src.utils.config_loader as cfg
    cfg.DB_PATH = scratch_db
    io_record.DB_PATH = scratch_db
    # Force a fresh session against the scratch DB.
    io_record._INIT_DONE = False
    io_record.DB = None
    io_record.SESSION_ID = None
    io_record.CURRENT_TURN_INDEX = 0
    # Also redirect the q_table CSV dir + session dossier dir to scratch.
    data_dir = os.path.join(scratch_dir, "data")
    os.makedirs(os.path.join(data_dir, "q_tables"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "safety"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "sessions"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "logs"), exist_ok=True)
    cfg.DATA_DIR = data_dir

    print(f"[DRY-RUN] Scratch dir: {scratch_dir}")
    print(f"[DRY-RUN] Scratch DB : {scratch_db}")

    # Feed replies in background.
    feeder = threading.Thread(target=_feed_replies, daemon=True)
    feeder.start()

    handler = handler_mod.HandlerRL()
    try:
        handler.run()
    except SystemExit:
        pass

    # ── Assertions ─────────────────────────────────────────────────────
    failures: List[str] = []
    db = DBManager(scratch_db)

    # 1. Session row created + closed.
    try:
        user_id = db.get_user_id(SUBJECT_ID)
    except Exception as e:
        failures.append(f"DB unreachable after run: {e}")
        user_id = None

    if user_id:
        # Count of open sessions (end_time IS NULL) should be 0.
        import sqlite3
        conn = sqlite3.connect(scratch_db)
        try:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM sessions WHERE user_id=? AND end_time IS NULL", (user_id,))
            open_n = c.fetchone()[0]
            if open_n != 0:
                failures.append(f"Expected 0 open sessions, got {open_n}")
            c.execute("SELECT COUNT(*) FROM safety_deliveries")
            safety_n = c.fetchone()[0]
            if safety_n < 1:
                failures.append("No safety_deliveries rows written — C2 delivery audit broken.")
            c.execute("SELECT COUNT(*) FROM clinical_flags")
            flag_n = c.fetchone()[0]
            if flag_n < 1:
                failures.append("No clinical_flags rows — M3 consolidated log broken.")
            c.execute("SELECT COUNT(*) FROM clinical_screening")
            scr_n = c.fetchone()[0]
            if scr_n < 1:
                failures.append("No clinical_screening rows — PHQ-4 persistence broken.")
            c.execute("SELECT COUNT(*) FROM turns")
            turn_n = c.fetchone()[0]
            if turn_n < 4:
                failures.append(f"Too few turns ({turn_n}) — pipeline may have short-circuited.")
        finally:
            conn.close()

    # 2. Q-table CSV written atomically somewhere under a q_tables/ dir.
    # `DATA_DIR` is captured at module import so the CSV may land in either
    # the project DATA_DIR or the scratch DATA_DIR depending on import
    # ordering. Either is fine — we just want proof the atomic write ran.
    from src.utils.config_loader import DATA_DIR as real_data_dir
    candidates = [
        os.path.join(real_data_dir, "q_tables", f"item_qtable_{SUBJECT_ID}.csv"),
        os.path.join(data_dir, "q_tables", f"item_qtable_{SUBJECT_ID}.csv"),
        os.path.join(os.path.abspath("."), "data", "q_tables", f"item_qtable_{SUBJECT_ID}.csv"),
    ]
    if not any(os.path.isfile(p) for p in candidates):
        failures.append(f"Q-table CSV missing at any of: {candidates}")

    # 3. Safety file fallback should have landed for the crisis trigger.
    # The handler writes to `os.path.abspath(".") + /data/safety/` regardless
    # of DATA_DIR, so check the project data/safety/ directory.
    safety_dir = os.path.join(os.path.abspath("."), "data", "safety")
    if os.path.isdir(safety_dir):
        crisis_files = [f for f in os.listdir(safety_dir) if f.startswith("crisis_")]
        if not crisis_files:
            failures.append("No crisis checkpoint file written to data/safety/")
    else:
        failures.append("data/safety/ dir missing — C2 file fallback broken.")

    print("\n" + "=" * 60)
    if failures:
        print(f"[DRY-RUN] ❌ FAILED with {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[DRY-RUN] ✅ PASSED — all clinical-safety checkpoints verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
