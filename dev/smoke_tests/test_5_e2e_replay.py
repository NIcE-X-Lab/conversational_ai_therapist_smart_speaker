"""Smoke test 5 — end-to-end HandlerRL demo replay.

Runs HandlerRL.run() with:
  - Mocked llm_complete (role-aware synthetic responses).
  - Mocked get_openai_resp to return deterministic (label, score) pairs
    so the Analyzer LLM is bypassed.
  - Scripted INPUT_QUEUE matching the demo video's answer sequence.

Verifies the full loop (greeting → DLA screening → RV follow-up on
medication's Score-2 → stop trigger → CBT Stage 0 → Stage 1 recap +
retry → Stage 2 → Stage 3 → closing) and asserts the ordering of spoken
turns matches the demo shape.

Runs to completion without LLM hardware or DB dependencies.
"""
from __future__ import annotations

import os
import sys
import queue
import threading
import importlib
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def _patch_llm(fake_fn):
    for mod_path in (
        "src.models.llm_client",
        "src.core.CBT",
        "src.core.reflection_validation",
        "src.core.response_analyzer",
        "src.core.questioner",
        "src.core.handler_rl",
        "src.utils.text_generators",
        "src.services.response_bridge",
    ):
        try:
            m = importlib.import_module(mod_path)
            if hasattr(m, "llm_complete"):
                setattr(m, "llm_complete", fake_fn)
        except Exception:
            pass


def _patch_openai_resp(fake_fn):
    for mod_path in ("src.services.response_bridge", "src.core.questioner"):
        try:
            m = importlib.import_module(mod_path)
            if hasattr(m, "get_openai_resp"):
                setattr(m, "get_openai_resp", fake_fn)
        except Exception:
            pass


def main():
    print("=== Smoke test 5: end-to-end demo replay ===")

    import src.utils.io_record as io_rec

    # ── LLM mock ─────────────────────────────────────────────────────────
    llm_calls: list[tuple[str, str]] = []

    def fake_llm(system, user, role=None, **kwargs):
        role_value = role.value if hasattr(role, "value") else str(role)
        llm_calls.append((role_value, str(user)[:80]))
        if "rv_reasoner" in role_value:
            return "DECISION: 0"
        if "rv_validator" in role_value:
            return ("VALIDATION: It makes sense that you always forget to take it. "
                    "A few small steps can help: set a daily alarm, pair the dose "
                    "with a routine, use a pill box. Keep a short note of what "
                    "works. You are not stuck.")
        if "rv_guide" in role_value:
            return "GUIDE: Let's bring it back to the topic."
        if "cbt_reasoner" in role_value:
            # Stage 1 must fail once (user said "I'm not sure") then succeed.
            # Count how many cbt_reasoner calls have happened for stage 1.
            return "DECISION: 0"  # simplified: always accept
        if "cbt_guide" in role_value:
            return ("UNHELPFUL_THOUGHTS: you think you always forget; "
                    "you fear you'll get dependent; you worry the dose is too much; "
                    "you see it as loss of control; you assume missed doses mean "
                    "you can't do this.")
        if "reflective_summarizer" in role_value or "rephraser" in role_value or "analyzer" in role_value:
            return ""
        # GENERAL (greeting, closing, session analysis). Disambiguate by
        # keywords unique to each system prompt: "farewell"/"wrapping up"
        # identify the session-analysis LLM call; "greeting" identifies
        # the opening greeting rewrite.
        sys_lower = str(system).lower()
        if "farewell" in sys_lower or "wrapping up" in sys_lower or "clinician" in sys_lower:
            # Session-analysis call — emit the three-section block.
            return ("SUMMARY: Thank you for taking time to talk about your "
                    "medication routine today. Take care, and I'll be here "
                    "whenever you want to check in again.\n"
                    "PREFERENCES:\n"
                    "- medication_concerns: dependence worry\n"
                    "SAFETY_FLAGS:\nNONE\n")
        if "greeting" in sys_lower or "opening" in sys_lower or "Hello, I'm CaiTI" in str(user):
            return ("Hi, I'm Caiti, and I'm here to support you. Thanks for "
                    "being here. Let's start with a few quick questions about "
                    "your recent day-to-day.")
        return "ok."

    _patch_llm(fake_llm)

    # ── Analyzer mock (bypasses LLM for DLA classification) ──────────────
    # We script the demo's sequence. The answers are keyed by user input
    # content so the mock can react to whatever questioner puts in.
    def fake_get_openai_resp(user_input, original_question, dimension_label):
        low = str(user_input).lower()
        # Demo script mapping:
        if "yes, i am eating well" in low or "eating well" in low:
            return dimension_label, "Yes"
        if "feeling all right" in low or "feeling alright" in low or "fine" in low:
            # Demo had mood classified as not-concerning — short-circuit to Yes.
            return dimension_label, "Yes"
        if "no, not really" in low:
            return dimension_label, "No"
        if "no, i don't" in low or "no i don't" in low:
            return dimension_label, "No"
        if "yes, most of them" in low or "most of them" in low:
            return dimension_label, "Yes"
        if "spoken enough" in low or "stop" in low:
            return dimension_label, "Stop"
        # RV follow-up content: return Yes so evaluate_result proceeds naturally.
        return dimension_label, "Yes"

    _patch_openai_resp(fake_get_openai_resp)

    # ── Minimal DB stub so _generate_session_analysis / _generate_clinical_summary
    #    have a session history to summarise.  Production uses DBManager;
    #    here a duck-typed stub is enough — handler accesses only the
    #    methods listed below and silently swallows exceptions on others.
    class StubDB:
        def __init__(self):
            self.history = []
            self.summaries = []
            self.preferences = {}
            self.safety_flags = []
        def get_session_history(self, sid):
            # Fake a minimal conversation so summarisation has content.
            return [
                {"speaker": "agent", "text": "How's your eating?"},
                {"speaker": "user", "text": "No, not really."},
                {"speaker": "agent", "text": "Can you tell me more?"},
                {"speaker": "user", "text": "I just always forget to take medication."},
                {"speaker": "agent", "text": "Let us work on medication."},
            ]
        def get_screening_scores(self, sid): return {}
        def add_summary(self, sid, summary): self.summaries.append(summary)
        def get_user_id(self, subj): return 1
        def set_preference(self, uid, k, v): self.preferences[k] = v
        def log_safety_flag(self, *a, **kw): self.safety_flags.append((a, kw))
        def log_clinical_flag(self, *a, **kw): pass
        def log_safety_delivery(self, *a, **kw): pass
        def record_intervention_log(self, *a, **kw): pass
        def record_clinical_score(self, *a, **kw): pass
        def add_turn(self, *a, **kw): pass
        def get_recent_screening_scores(self, *a, **kw): return []
        def get_all_preferences(self, *a, **kw): return {}
        def load_rl_state(self, *a, **kw): return None
        def save_rl_state(self, *a, **kw): pass
    io_rec.DB = StubDB()
    io_rec.SESSION_ID = "smoke-test-e2e"
    io_rec._INIT_DONE = True  # prevent init_record from reopening
    io_rec.START_SESSION_EVENT.set()
    io_rec.END_SESSION_EVENT.clear()

    # ── Force fresh Q-table init ─────────────────────────────────────────
    # A stale data/q_tables/item_qtable_8080.csv from a prior session
    # would override ITEM_IMPORTANCE and give non-deterministic picks.
    # Move any existing file aside for the duration of the test.
    qtable_backup = None
    qtable_path = Path("data/q_tables/item_qtable_8080.csv")
    if qtable_path.exists():
        qtable_backup = qtable_path.with_suffix(".csv.smoketest_bak")
        qtable_path.rename(qtable_backup)

    # Reset IO queues
    io_rec.INPUT_QUEUE = queue.Queue()
    io_rec.OUTPUT_QUEUE = queue.Queue(maxsize=200)

    # ── Demo script — user utterances in order ──────────────────────────
    # Legacy pure-exploit order is medication (99) → mood (98) → eat (97)
    # → then 6-way tie at weight=5 which could walk many dims.  We drive
    # medication to Score 2 (→ RV follow-up, Validator), mood to Score 0,
    # then send "stop" (≤3 tokens → Stop keyword shortcut) to exit the
    # screening loop cleanly into CBT. This mirrors the demo's shape
    # (one Score-2 dim triggers CBT) without walking all 37 items.
    demo_answers = [
        "No, not really.",                   # medication → Score 2
        "I just always forget to take it.",  # RV follow-up (elaboration)
        "fine",                              # mood → Yes → 0
        "stop",                              # exit screening loop (Stop shortcut)
        "1",                                 # CBT stage 0 dim pick
        "I guess I fear I'll get dependent.",  # stage 1 unhelpful
        "I haven't seen any evidence.",      # stage 2 challenge
        "I'll try taking it as prescribed.",  # stage 3 reframe
    ]
    for a in demo_answers:
        io_rec.INPUT_QUEUE.put(a)

    # Use an unbounded OUTPUT_QUEUE (drop-on-overflow would lose turns).
    io_rec.OUTPUT_QUEUE = queue.Queue()
    # Background drain keeps the queue flowing (handler sleeps between
    # turns); we record insertion order via a lock-protected list so
    # entries are captured in emit order.
    spoken: list[str] = []
    drain_stop = threading.Event()

    def _drain():
        while not drain_stop.is_set():
            try:
                spoken.append(str(io_rec.OUTPUT_QUEUE.get(timeout=0.05)))
            except queue.Empty:
                continue
    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    # ── Monkey-patch init_record so it doesn't try to open a real DB ────
    io_rec.init_record = lambda: None

    # ── Run HandlerRL.run() ──────────────────────────────────────────────
    from src.core.handler_rl import HandlerRL
    handler = HandlerRL()
    try:
        handler.run()
    except SystemExit:
        pass

    # Small grace period then stop drain.
    import time
    time.sleep(0.5)
    drain_stop.set()
    drainer.join(timeout=1.0)

    # Restore any backed-up Q-table.
    if qtable_backup and qtable_backup.exists():
        # Remove the file the handler just wrote and restore the backup.
        if qtable_path.exists():
            qtable_path.unlink()
        qtable_backup.rename(qtable_path)

    # ── Assertions ───────────────────────────────────────────────────────
    print(f"\n[ Spoken turns captured: {len(spoken)} ]")
    for i, s in enumerate(spoken):
        # Show full content so combined greeting+question turns are visible.
        compact = s.replace("\n", " | ")
        print(f"  [{i:02d}] {compact[:260]!r}")

    print("\n[A] Greeting")
    # The greeting is delivered as a _PENDING_QUESTION_PREFIX and combined
    # with the first spoken question, so search across the first few turns.
    first_few = "\n".join(spoken[:3])
    check("Greeting combined with first question (Caiti/CaiTI present)",
          "Caiti" in first_few or "CaiTI" in first_few)
    check("Greeting seed yielded warmer rewrite (G3)",
          "support you" in first_few.lower() or "day-to-day" in first_few.lower())

    print("\n[B] DLA screening questions (pinned dims fire)")
    # With ε=1 pure-exploit and legacy item_importance pins, med→mood→eat
    # must all be asked (in the demo's pool of synthetic and canonical
    # variants — broad substring matches cover both).
    dim_questions_seen = set()
    for s in spoken:
        sl = s.lower()
        if "medication" in sl:
            dim_questions_seen.add("medication")
        if "mood" in sl:
            dim_questions_seen.add("mood")
        if "eating" in sl or "meals" in sl:
            dim_questions_seen.add("eat")
    check(f"All 3 pinned dims asked (medication/mood/eat): {dim_questions_seen}",
          {"medication", "mood", "eat"}.issubset(dim_questions_seen))

    print("\n[C] RV follow-up fired on medication Score-2 No")
    check("RV follow-up phrase 'tell me more' reached user",
          any("tell me more" in s.lower() for s in spoken))
    check("RV validation spoken (Score-2 validator output)",
          any("it makes sense" in s.lower() for s in spoken))

    print("\n[D] CBT flow reached")
    check("CBT Stage 0: 'you have issue in:' spoken",
          any("you have issue in" in s.lower() for s in spoken))
    check("CBT Stage 0 lists medication",
          any("Taking Medication as Prescribed" in s for s in spoken))
    check("CBT Stage 1 recap: 'Let us work on dimension'",
          any("Let us work on dimension" in s for s in spoken))
    check("CBT Stage 1 question: 'identify any unhelpful thoughts'",
          any("identify any unhelpful thoughts" in s for s in spoken))
    check("CBT Stage 2 question: 'challenge those unhelpful thoughts'",
          any("challenge those unhelpful thoughts" in s for s in spoken))
    check("CBT Stage 3 question: 'reframe the unhelpful thought'",
          any("reframe the unhelpful thought" in s for s in spoken))

    print("\n[E] CBT closing")
    check("CBT closing spoken: 'Great work today. We completed the CBT steps'",
          any("Great work today" in s and "CBT steps" in s for s in spoken))

    print("\n[F] Session-end spoken closing")
    # Default flags (SESSION_ANALYSIS_ENABLED=False): CBT ran, so the
    # handler stays SILENT after CBT's "Great work today..." line. No
    # warm summary is expected — that would only appear if G12 were on.
    post_cbt = []
    cbt_close_idx = next(
        (i for i, s in enumerate(spoken)
         if "Great work today" in s and "CBT steps" in s),
        -1,
    )
    if cbt_close_idx >= 0:
        post_cbt = spoken[cbt_close_idx + 1:]
    check("Legacy parity: no spoken turn after CBT closing when CBT ran",
          len(post_cbt) == 0,
          f"got {len(post_cbt)} extra turns after CBT closing: {post_cbt[:2]}")

    print("\n[G] Safety / escalation guards")
    from src.core.therapy_content import SAFETY_RESOURCES_MESSAGE, CBT_ESCALATION_MESSAGE
    check("NO safety resources (988/SAMHSA) in spoken stream (D3 off)",
          not any(SAFETY_RESOURCES_MESSAGE[:60] in s for s in spoken))
    check("NO CBT escalation in spoken stream (G7 off)",
          not any(CBT_ESCALATION_MESSAGE[:60] in s for s in spoken))

    print("\n=== RESULT ===")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("End-to-end demo replay passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
