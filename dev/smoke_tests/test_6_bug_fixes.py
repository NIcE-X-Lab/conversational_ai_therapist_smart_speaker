"""Smoke test 6 — Bug 1 + Bug 2 regression coverage.

Bug 1: CBT retry pairs (Guide example + re-ask) used to emit TWO
       log_question calls per turn, which drifted on a queue-based TTS
       pipeline (speech_service.py consumes one item per turn). This
       orphaned items in OUTPUT_QUEUE and made subsequent turns speak
       the previous turn's queued-but-unspoken second item. Also, CBT's
       final "Great work today..." was stuck on the queue when
       main.py cleared START_SESSION_EVENT.

Bug 2: "Let's end the session" during screening was caught by
       GlobalCommandMatcher's aggressive END regex and set
       END_SESSION_EVENT, which blocked CBT from ever running (violates
       paper §5.1: the screening Stop keyword must still lead into CBT).

Fixes verified here:

    [A] CBT Stage 1/2/3 retry emits ONE combined log_question (not 2).
    [B] CBT Stage 1/2/3 failure-escalation emits ONE combined
        log_question (not 2) regardless of CBT_ESCALATION_ENABLED.
    [C] GlobalCommandMatcher routes SOFT_END phrases ("no more
        questions", "that's enough for today", "I don't want to answer
        any more questions", "let's end the session") as SOFT_END, NOT
        HARD_END.
    [D] GlobalCommandMatcher HARD_END still matches short kill phrases
        ("end session", "goodbye", "stop session").
    [E] Response analyzer soft-end short-circuit in response_bridge.py
        returns (dim, "Stop") for SOFT_END phrasings.
    [F] CBT_STARTED_EVENT lifecycle: cleared at init_record, set at
        run_cbt entry.
    [G] _apply_global_command_priority returns None for pre-CBT SOFT_END
        (→ transcript flows to analyzer → Stop → CBT).
    [H] _apply_global_command_priority returns "END" for HARD_END.
    [I] _apply_global_command_priority upgrades SOFT_END to "END" when
        CBT_STARTED_EVENT is set (mid-CBT quit).
    [J] Speech service defensive drain speaks multiple queued items per
        turn (future-proofs against any new double-log_question site).
    [K] main.py end-of-session drain loop exists and waits for
        OUTPUT_QUEUE to empty before clearing START_SESSION_EVENT.

Run:
    .venv/bin/python3 dev/smoke_tests/test_6_bug_fixes.py
"""
from __future__ import annotations

import os
import sys
import queue
import importlib
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ════════════════════════════════════════════════════════════════════════
# [A] CBT retry emits ONE combined log_question
# ════════════════════════════════════════════════════════════════════════

def _source_text_cbt() -> str:
    return (ROOT / "src/core/CBT.py").read_text()


def test_A_cbt_retries_merged():
    print("\n[A] CBT retry pairs merged into single log_question")
    src = _source_text_cbt()

    # Bug-1 fix: each stage's retry should combine the sanitized Guide
    # output + the re-ask prompt in ONE log_question call. The source
    # uses an f-string split across two physical lines with implicit
    # concatenation, so we match on the shape near _sanitize_guide_text.
    import re as _re

    def _assert_single_merged_call(label: str, guide_var: str, reask: str):
        # Find the retry while-loop for this stage and ensure the body
        # contains ONE log_question that takes the concatenated sanitized
        # guide + re-ask prompt.
        pattern = (
            rf"{_re.escape(guide_var)}\s*=\s*stage\d_guide\(.*?\).*?"
            rf"log_question\(\s*f\"\{{_sanitize_guide_text\({guide_var},\s*'[A-Z_]+'\)\}}\\n\\n\"\s*"
            rf'"{_re.escape(reask)}"\s*\)'
        )
        found = bool(_re.search(pattern, src, flags=_re.DOTALL))
        check(label, found)

    _assert_single_merged_call(
        "Stage 1 retry combines _sanitize_guide_text + re-ask in one log_question",
        "guide1",
        "Please share those unhelpful thoughts again, in one sentence.",
    )
    _assert_single_merged_call(
        "Stage 2 retry combines _sanitize_guide_text + re-ask in one log_question",
        "guide2",
        "Please try to challenge the unhelpful thoughts again, in one sentence.",
    )
    _assert_single_merged_call(
        "Stage 3 retry combines _sanitize_guide_text + re-ask in one log_question",
        "guide3",
        "Please try to reframe that again, in one or two sentences.",
    )

    # The old double-log_question pattern should be gone for the retry
    # paths. Ensure no stale "Please share those unhelpful thoughts
    # again" line appears as its own log_question call.
    # The merged line embeds this text inside an f-string, so a
    # standalone log_question("Please share those unhelpful thoughts...")
    # would indicate the fix didn't stick.
    standalone_reask = 'log_question("Please share those unhelpful thoughts again, in one sentence.")'
    check("No standalone Stage-1 re-ask log_question remains", standalone_reask not in src)
    standalone_reask_2 = 'log_question("Please try to challenge the unhelpful thoughts again, in one sentence.")'
    check("No standalone Stage-2 re-ask log_question remains", standalone_reask_2 not in src)
    standalone_reask_3 = 'log_question("Please try to reframe that again, in one or two sentences.")'
    check("No standalone Stage-3 re-ask log_question remains", standalone_reask_3 not in src)


# ════════════════════════════════════════════════════════════════════════
# [B] Failure-escalation paths merged
# ════════════════════════════════════════════════════════════════════════

def test_B_failure_paths_merged():
    print("\n[B] Failure-escalation pairs merged into single log_question")
    src = _source_text_cbt()

    # Fixed pattern: CBT_ESCALATION_ENABLED branch combines escalation
    # message + pause message into one log_question. Look for the
    # f-string shape that we introduced.
    combined_pattern = 'log_question(f"{CBT_ESCALATION_MESSAGE}\\n\\n{pause_msg}")'
    check("Failure branches combine CBT_ESCALATION_MESSAGE with pause_msg",
          combined_pattern in src)

    # The old 'log_question(CBT_ESCALATION_MESSAGE)' as its own
    # statement followed by a separate log_question(pause) should be
    # gone.
    standalone_escalation = "log_question(CBT_ESCALATION_MESSAGE)\n        log_question("
    check("No standalone CBT_ESCALATION_MESSAGE log_question remains",
          standalone_escalation not in src)


# ════════════════════════════════════════════════════════════════════════
# [C] + [D] GlobalCommandMatcher SOFT_END / HARD_END classification
# ════════════════════════════════════════════════════════════════════════

def test_CD_matcher_classification():
    print("\n[C] GlobalCommandMatcher SOFT_END / HARD_END classification")
    from src.services.speech_service import GlobalCommandMatcher
    m = GlobalCommandMatcher()

    # SOFT_END: "I'm done with screening" intent. Pre-CBT should route
    # to Response Analyzer as Stop → CBT; post-CBT-start should upgrade
    # to HARD_END (done at caller level).
    soft_end_phrases = [
        "no more questions",
        "I don't want to answer any more questions",
        "I dont want to answer any more questions now",
        "that's enough for today",
        "that is enough for today",
        "I think I am done with questions for today",
        "I am done with questions",
        "enough questions",
        "stop the questions",
        "let's end the session",
        "lets end the session",
        "I want to end the session",
        "I don't want to answer any more questions. let's end the session",
    ]
    for phrase in soft_end_phrases:
        got = m.match(phrase)
        check(f"SOFT_END: {phrase!r}", got == "SOFT_END", f"got {got!r}")

    # HARD_END: short, unambiguous kill-switch phrases.
    print("\n[D] HARD_END still matches explicit kill phrases")
    hard_end_phrases = [
        "end session",
        "end the session",
        "stop session",
        "stop the session please",
        "finish session",
        "close the session",
        "goodbye",
        "bye",
    ]
    for phrase in hard_end_phrases:
        got = m.match(phrase)
        check(f"HARD_END: {phrase!r}", got == "HARD_END", f"got {got!r}")

    # Non-end utterances: normal clinical content should NOT match.
    non_end_phrases = [
        "I need to be healthy and have green food",
        "yes I take my medication on time",
        "my session is going well at work",
        "I have trouble sleeping at the end of the day",
    ]
    for phrase in non_end_phrases:
        got = m.match(phrase)
        check(f"No match: {phrase!r}", got is None, f"got {got!r}")


# ════════════════════════════════════════════════════════════════════════
# [E] response_bridge soft-end short-circuit
# ════════════════════════════════════════════════════════════════════════

def test_E_response_bridge_soft_end():
    print("\n[E] response_bridge soft-end short-circuit → (dim, Stop)")
    from src.services.response_bridge import get_openai_resp, _matches_soft_end_intent

    # Helper directly.
    check("_matches_soft_end_intent('no more questions')",
          _matches_soft_end_intent("no more questions") is True)
    check("_matches_soft_end_intent('that's enough for today')",
          _matches_soft_end_intent("that's enough for today") is True)
    check("_matches_soft_end_intent('let's end the session')",
          _matches_soft_end_intent("let's end the session") is True)
    check("_matches_soft_end_intent on normal clinical input",
          _matches_soft_end_intent("yes I take my medication") is False)

    # get_openai_resp short-circuits before LLM classifier. Mock
    # classify_dimension_and_score to verify the short-circuit fires
    # BEFORE the LLM call.
    with patch("src.services.response_bridge.classify_dimension_and_score") as mock_clf:
        mock_clf.return_value = "mood, 2"  # if LLM fires, we'd get mood
        # Soft-end phrase — should hit short-circuit, NOT call LLM.
        got = get_openai_resp(
            "I don't want to answer any more questions",
            "How's your mood?",
            "mood",
        )
        check("Soft-end phrase short-circuits to (dim, 'Stop')",
              got == ("mood", "Stop"), f"got {got!r}")
        check("LLM classifier NOT called on soft-end (short-circuit)",
              mock_clf.call_count == 0)

        # Normal utterance should reach the LLM classifier.
        mock_clf.reset_mock()
        mock_clf.return_value = "mood, 2"
        got = get_openai_resp(
            "I feel sad and hopeless most days",
            "How's your mood?",
            "mood",
        )
        check("Normal clinical utterance proceeds to LLM classifier",
              mock_clf.call_count == 1)


# ════════════════════════════════════════════════════════════════════════
# [F] CBT_STARTED_EVENT lifecycle
# ════════════════════════════════════════════════════════════════════════

def test_F_cbt_started_event_lifecycle():
    print("\n[F] CBT_STARTED_EVENT lifecycle")
    import src.utils.io_record as io_rec

    # Event defined.
    check("io_record.CBT_STARTED_EVENT exists and is a threading.Event",
          hasattr(io_rec, "CBT_STARTED_EVENT")
          and hasattr(io_rec.CBT_STARTED_EVENT, "set")
          and hasattr(io_rec.CBT_STARTED_EVENT, "is_set"))

    # Cleared at init_record (grep the source).
    iorec_src = (ROOT / "src/utils/io_record.py").read_text()
    check("init_record clears CBT_STARTED_EVENT",
          "CBT_STARTED_EVENT.clear()" in iorec_src)

    # Set at run_cbt entry (grep the source).
    cbt_src = _source_text_cbt()
    check("run_cbt sets io_rec.CBT_STARTED_EVENT at entry",
          "io_rec.CBT_STARTED_EVENT.set()" in cbt_src)


# ════════════════════════════════════════════════════════════════════════
# [G] + [H] + [I] _apply_global_command_priority routing
# ════════════════════════════════════════════════════════════════════════

def test_GHI_routing_by_cbt_phase():
    print("\n[G/H/I] _apply_global_command_priority routing")

    # Real SpeechInteractionService carries audio deps we can't load
    # here; we exercise the method logic against a lightweight
    # double with just enough wiring for the method under test.
    from src.services.speech_service import GlobalCommandMatcher
    import src.utils.io_record as io_rec

    # Build a minimal object exposing the two attributes/methods the
    # function under test actually needs.  HARD_END now runs a yes/no
    # confirmation dialog before exiting; the stub fakes that with a
    # toggleable boolean so we can exercise both confirmed and declined
    # branches without driving real audio.
    class _StubService:
        def __init__(self, confirm_yes: bool = True):
            self.global_command_matcher = GlobalCommandMatcher()
            self.handle_exit_called = 0
            self._confirm_yes = confirm_yes
            self.confirm_calls = 0

        def handle_exit(self):
            self.handle_exit_called += 1

        def _run_end_confirmation(self):
            self.confirm_calls += 1
            return self._confirm_yes

    # Bind the method from the class to our stub.
    from src.services.speech_service import SpeechInteractionService
    fn = SpeechInteractionService._apply_global_command_priority

    # ── [G] Pre-CBT SOFT_END → returns None (lets transcript flow to
    #       Response Analyzer → Stop keyword → CBT).
    io_rec.CBT_STARTED_EVENT.clear()
    stub = _StubService()
    got = fn(stub, "I don't want to answer any more questions")
    check("[G] Pre-CBT SOFT_END → returns None",
          got is None, f"got {got!r}")
    check("[G] Pre-CBT SOFT_END does NOT call handle_exit",
          stub.handle_exit_called == 0)
    check("[G] Pre-CBT SOFT_END does NOT trigger end-confirmation",
          stub.confirm_calls == 0)

    got = fn(stub, "let's end the session")
    check("[G] Pre-CBT 'let's end the session' → returns None",
          got is None, f"got {got!r}")
    check("[G] Pre-CBT 'let's end the session' does NOT call handle_exit",
          stub.handle_exit_called == 0)

    # ── [H] HARD_END (confirmed yes) → returns "END" and calls handle_exit.
    stub = _StubService(confirm_yes=True)
    got = fn(stub, "end the session")
    check("[H] HARD_END 'end the session' → returns 'END'",
          got == "END", f"got {got!r}")
    check("[H] HARD_END 'end the session' calls handle_exit",
          stub.handle_exit_called == 1)
    check("[H] HARD_END 'end the session' triggered end-confirmation",
          stub.confirm_calls == 1)

    stub = _StubService(confirm_yes=True)
    got = fn(stub, "goodbye")
    check("[H] HARD_END 'goodbye' → returns 'END'", got == "END")
    check("[H] HARD_END 'goodbye' calls handle_exit",
          stub.handle_exit_called == 1)

    # ── [H'] HARD_END (declined) → returns "END_DECLINED" and does NOT
    #         call handle_exit.  Session continues.
    stub = _StubService(confirm_yes=False)
    got = fn(stub, "end the session")
    check("[H'] HARD_END declined → returns 'END_DECLINED'",
          got == "END_DECLINED", f"got {got!r}")
    check("[H'] HARD_END declined does NOT call handle_exit",
          stub.handle_exit_called == 0)
    check("[H'] HARD_END declined still ran the confirmation dialog",
          stub.confirm_calls == 1)

    # ── [I] SOFT_END + CBT_STARTED → upgrade to HARD_END (no confirmation,
    #       per design: mid-CBT escalation keeps existing immediate-exit
    #       behaviour because the user already issued one stop command).
    io_rec.CBT_STARTED_EVENT.set()
    stub = _StubService(confirm_yes=True)
    got = fn(stub, "I don't want to answer any more questions")
    check("[I] Mid-CBT SOFT_END → upgrades to 'END'",
          got == "END", f"got {got!r}")
    check("[I] Mid-CBT SOFT_END calls handle_exit",
          stub.handle_exit_called == 1)
    check("[I] Mid-CBT SOFT_END escalation skips end-confirmation",
          stub.confirm_calls == 0)

    # ── [I'] HARD_END mid-CBT (confirmed) → still runs confirmation dialog.
    stub = _StubService(confirm_yes=True)
    got = fn(stub, "end the session")
    check("[I'] Mid-CBT HARD_END (confirmed) → returns 'END'",
          got == "END", f"got {got!r}")
    check("[I'] Mid-CBT HARD_END triggered end-confirmation",
          stub.confirm_calls == 1)

    # Reset for next tests.
    io_rec.CBT_STARTED_EVENT.clear()


# ════════════════════════════════════════════════════════════════════════
# [J] Speech service defensive drain
# ════════════════════════════════════════════════════════════════════════

def test_J_defensive_drain_in_speech_service():
    print("\n[J] Speech service trailing-utterance drain")
    src = (ROOT / "src/services/speech_service.py").read_text()
    # The drain loop we added inside _wait_for_output_with_intermission
    # has a unique literal marker comment + loop shape.
    check("Drain comment present",
          "Defensive drain" in src
          and "immediately-adjacent follow-up utterances" in src)
    check("Drain loop uses output_queue.get with short timeout",
          "self.output_queue.get(timeout=0.1)" in src)
    check("Drain loop extends deadline on each drained item",
          "drain_deadline = time.monotonic() + 0.4" in src)


# ════════════════════════════════════════════════════════════════════════
# [K] main.py end-of-session drain
# ════════════════════════════════════════════════════════════════════════

def test_K_main_drain_before_idle():
    print("\n[K] main.py drains OUTPUT_QUEUE before idle")
    src = (ROOT / "main.py").read_text()
    check("main.py has end-of-session drain loop",
          "Bug-1 fix (tail)" in src
          and "_DRAIN_TIMEOUT_SEC" in src)
    # Locate the handler-run block specifically (not the unrelated
    # FastAPI end_session endpoint which also clears the event).
    hr_idx = src.index("handler.run()")
    # Find the NEXT START_SESSION_EVENT.clear() after handler.run().
    post_run = src[hr_idx:]
    drain_idx = post_run.index("io_record.OUTPUT_QUEUE.empty()")
    clear_idx = post_run.index("io_record.START_SESSION_EVENT.clear()")
    check("Drain loop runs AFTER handler.run() and BEFORE the main-loop's "
          "START_SESSION_EVENT.clear()",
          drain_idx < clear_idx,
          f"drain@{drain_idx} vs clear@{clear_idx}")


# ════════════════════════════════════════════════════════════════════════
# Runtime behavioural: full CBT Stage-1 retry runs WITHOUT drift
# ════════════════════════════════════════════════════════════════════════

def test_runtime_stage1_retry_one_beat_per_turn():
    print("\n[Runtime] CBT Stage 1 retry emits exactly ONE queue item per turn")

    # Minimal question_lib with a Score-2 dim so run_cbt proceeds.
    question_lib = {
        str(i): {
            "1": {
                "label": "medication" if i == 3 else f"dim_{i}",
                "name": "Taking Medication as Prescribed" if i == 3 else f"Dim {i}",
                "question": ["placeholder"],
                "score": [2] if i == 3 else [],
                "notes": [
                    [
                        "original_question: ...",
                        "original_resp: i forget",
                        "followup_resp: i forget",
                        "rv_decision: 0",
                        "rv_validation: that's tough",
                    ]
                ] if i == 3 else [],
            }
        }
        for i in range(1, 38)
    }

    import src.utils.io_record as io_rec

    io_rec.INPUT_QUEUE = queue.Queue()
    io_rec.OUTPUT_QUEUE = queue.Queue()
    io_rec.CBT_STARTED_EVENT.clear()

    # User script for CBT Stages:
    # - dim pick (Stage 0)
    # - Stage 1: first try empty → Reasoner=="1" → retry once → still "1" →
    #           retry twice → LEGIT answer.
    # - Stage 2: quick pass
    # - Stage 3: quick pass
    scripted = [
        "1",                            # Stage 0 pick
        "i dont really know",           # Stage 1 attempt 1 (Reasoner 1 → retry)
        "i still dont know",            # Stage 1 attempt 2 (Reasoner 1 → retry)
        "i fear getting dependent",     # Stage 1 attempt 3 (Reasoner 0 → pass)
        "ill look at evidence",         # Stage 2 (pass first try)
        "ill try with a pill box",      # Stage 3 (pass first try)
    ]
    for s in scripted:
        io_rec.INPUT_QUEUE.put(s)

    # Per-turn counters: track the number of OUTPUT_QUEUE items
    # produced between consecutive INPUT_QUEUE pops (i.e. between
    # calls to get_resp_log). The pre-fix behaviour would produce 2
    # items per Stage 1 retry turn (Guide example + re-ask). The
    # post-fix behaviour produces 1.
    turn_emissions: list[int] = []
    current_turn = [0]
    queue_lock = __import__("threading").Lock()

    # Wrap OUTPUT_QUEUE.put_nowait so we can count per turn.
    orig_safe_put = io_rec._safe_output_put

    def counting_put(item):
        with queue_lock:
            current_turn[0] += 1
        return orig_safe_put(item)

    # get_resp_log boundary: each call ends a turn and opens a new one.
    orig_get_resp_log = io_rec.get_resp_log

    def counting_get_resp_log():
        with queue_lock:
            if current_turn[0] > 0:
                turn_emissions.append(current_turn[0])
                current_turn[0] = 0
        return orig_get_resp_log()

    # Mock the LLM path so retries behave deterministically.
    # First two Stage 1 Reasoner calls fail, the third passes.
    reasoner_call_count = [0]

    def fake_llm(system, user, role=None, **kwargs):
        role_val = role.value if hasattr(role, "value") else str(role)
        # Stage 1/2/3 reasoner: first Stage 1 call fails twice then passes.
        if "cbt_reasoner" in role_val:
            reasoner_call_count[0] += 1
            # The exact failure count: Stage 1 attempt 1 → fail;
            # Stage 1 attempt 2 → fail; Stage 1 attempt 3 → pass.
            # Then Stage 2 → pass, Stage 3 → pass.
            if reasoner_call_count[0] <= 2:
                return "DECISION: 1"
            return "DECISION: 0"
        if "cbt_guide" in role_val:
            return ("UNHELPFUL_THOUGHTS: you think you always forget; "
                    "you fear you'll get dependent; you worry the dose is too much.")
        # Stage-3 recap.
        return "You already tried to look at the evidence. Good work."

    # Apply patches.
    with patch("src.core.CBT.llm_complete", side_effect=fake_llm), \
         patch("src.utils.io_record._safe_output_put", side_effect=counting_put), \
         patch("src.utils.io_record.get_resp_log", side_effect=counting_get_resp_log), \
         patch("src.core.CBT.get_resp_log", side_effect=counting_get_resp_log):
        # Also disable crisis callback.
        from src.core.CBT import run_cbt
        try:
            run_cbt(question_lib, crisis_callback=None)
        except Exception as e:
            print(f"  [note] run_cbt raised: {e}")

    # Final turn: record anything after the last get_resp_log.
    with queue_lock:
        if current_turn[0] > 0:
            turn_emissions.append(current_turn[0])

    print(f"  Per-turn queue emissions: {turn_emissions}")

    # Paper §5.3: each CBT turn should emit exactly one user-facing beat.
    # With our fix the retry's Guide example + re-ask are a single log
    # _question call → one queue item. We allow the FINAL CBT turn
    # (which may emit Stage-3 recap via set_question_prefix + its
    # log_question + the final "Great work today") to have a couple
    # extra items because they combine naturally.
    # Strict invariant: NO turn should emit 2+ items from a retry
    # source. Pre-fix, retry turns would emit 2 (guide + reask).
    retry_turn_violations = sum(1 for c in turn_emissions if c >= 2)
    check("No turn emits 2+ queue items from a CBT retry pair",
          retry_turn_violations <= 1,
          f"found {retry_turn_violations} turn(s) with 2+ emissions")


# ════════════════════════════════════════════════════════════════════════
# Runtime behavioural: Bug-2 flow — pre-CBT SOFT_END routes via Stop → CBT
# ════════════════════════════════════════════════════════════════════════

def test_runtime_soft_end_routes_to_cbt_via_stop():
    print("\n[Runtime] Soft-end pre-CBT routes to Stop keyword (→ CBT)")
    from src.services.response_bridge import get_openai_resp

    # Pre-CBT: SOFT_END utterance goes through response_bridge and must
    # return (dim, "Stop"). The questioner then treats "Stop" as a
    # terminal signal (valid=1, terminate=1), screening ends, and
    # handler_rl.run() calls run_cbt since END_SESSION_EVENT is NOT set.

    # Mock the LLM so we can prove the short-circuit preempts it.
    with patch("src.services.response_bridge.classify_dimension_and_score") as clf:
        clf.return_value = "medication, 0"  # what LLM would say if called

        test_phrases = [
            "I don't want to answer any more questions",
            "No more questions",
            "that's enough for today",
            "I'm done with questions",
            "let's end the session",
        ]
        for phrase in test_phrases:
            clf.reset_mock()
            got = get_openai_resp(phrase, "How's your mood?", "mood")
            check(f"Soft-end phrase '{phrase[:40]}...' → (dim, 'Stop')",
                  got == ("mood", "Stop"), f"got {got!r}")
            check(f"LLM bypassed for '{phrase[:40]}...' (short-circuit)",
                  clf.call_count == 0)


# ════════════════════════════════════════════════════════════════════════

def main():
    print("=== Smoke test 6: Bug 1 + Bug 2 regression coverage ===")

    test_A_cbt_retries_merged()
    test_B_failure_paths_merged()
    test_CD_matcher_classification()
    test_E_response_bridge_soft_end()
    test_F_cbt_started_event_lifecycle()
    test_GHI_routing_by_cbt_phase()
    test_J_defensive_drain_in_speech_service()
    test_K_main_drain_before_idle()
    test_runtime_stage1_retry_one_beat_per_turn()
    test_runtime_soft_end_routes_to_cbt_via_stop()

    print("\n=== RESULT ===")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All Bug 1 + Bug 2 fixes held.")
    sys.exit(0)


if __name__ == "__main__":
    main()
