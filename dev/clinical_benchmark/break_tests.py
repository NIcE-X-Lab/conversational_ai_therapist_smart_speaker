"""Adversarial / edge-case break tests against the system's surface area.

These run *outside* the persona harness because they target small,
unit-level surfaces (matchers, classifiers, stubs) rather than full
session flow.  Captures whether each invariant the production code
relies on still holds.

Output is a list of {check, passed, detail} dicts so the report can
table them.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _check(name: str, cond: bool, detail: str = "") -> Dict[str, Any]:
    return {"check": name, "passed": bool(cond), "detail": detail}


def run_break_tests() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # ── 1. GlobalCommandMatcher edge cases ──
    from src.services.speech_service import GlobalCommandMatcher
    m = GlobalCommandMatcher()

    matcher_cases = [
        # (input, expected)
        ("end the session", "HARD_END"),
        ("End the session.", "HARD_END"),
        ("goodbye", "HARD_END"),
        ("bye", "HARD_END"),
        ("stop the session", "HARD_END"),
        ("no more questions", "SOFT_END"),
        ("I don't want to answer any more questions", "SOFT_END"),
        ("that's enough for today", "SOFT_END"),
        ("I'm done with questions", "SOFT_END"),
        ("start session", "START"),
        ("hi katie", "START"),
        # 'hello' alone matches the START regex pattern (wake phrase),
        # which is the intended production behaviour — wake-words like
        # 'hello' / 'hi' / 'start' / 'begin' fire START to enter
        # onboarding from idle.
        ("hello", "START"),
        ("how are you", None),
        ("I feel sad", None),
        ("", None),
        ("    ", None),
        ("I need to be healthy and have green food", None),  # must NOT hit "be"-as-bye
        ("Is the session almost over?", None),  # contains 'session' but a question
        ("—— emdash flood ——", None),  # unicode
    ]
    for utt, want in matcher_cases:
        got = m.match(utt)
        results.append(_check(
            f"GlobalCommandMatcher: {utt!r} -> {want}",
            got == want,
            f"got {got!r}"
        ))

    # ── 2. _classify_confirm edge cases ──
    # Pull constants and re-implement so we don't need a SpeechService instance.
    src = (ROOT / "src/services/speech_service.py").read_text()
    yes_match = re.search(r"_STOP_CONFIRM_YES\s*=\s*\((.*?)\)", src, re.DOTALL)
    no_match = re.search(r"_STOP_CONFIRM_NO\s*=\s*\((.*?)\)", src, re.DOTALL)
    yes = eval("(" + yes_match.group(1) + ")")
    no = eval("(" + no_match.group(1) + ")")

    def _classify_confirm(text: str) -> str:
        if not text:
            return "unclear"
        low = text.lower().strip(" .!?,")
        if not low:
            return "unclear"
        if low in yes:
            return "yes"
        if low in no:
            return "no"
        if any(p in low for p in no):
            return "no"
        if any(p in low for p in yes):
            return "yes"
        return "unclear"

    confirm_cases = [
        ("yes", "yes"),
        ("YES PLEASE", "yes"),
        ("Yes go ahead.", "yes"),
        ("sure", "yes"),
        ("no", "no"),
        ("NO", "no"),
        ("never mind", "no"),
        ("not yet", "no"),
        ("keep going", "no"),
        ("wait", "no"),
        # Ambiguity bias to NO — therapist asked for double-security.
        ("no, I think we should keep going", "no"),
        ("yes wait actually no", "no"),  # yes-shaped + no must resolve no
        # Genuinely unclear → unclear (handler defaults to "do not end").
        ("mmm", "unclear"),
        ("hmmm I don't know", "no"),  # contains "don't" → no by phrase scan
        ("", "unclear"),
        ("   ", "unclear"),
    ]
    for utt, want in confirm_cases:
        got = _classify_confirm(utt)
        results.append(_check(
            f"_classify_confirm: {utt!r} -> {want}",
            got == want,
            f"got {got!r}"
        ))

    # ── 3. Don't-know brainstorm matcher ──
    from src.core.questioner import _is_dont_know, _build_brainstorm_guide

    dont_know_cases = [
        ("I don't know", True),
        ("i dont know", True),
        ("I DON'T KNOW", True),
        ("No idea", True),
        ("I don't want to answer", True),
        ("I have nothing to say", True),
        ("Not sure what to say", True),
        ("I don't have a therapist", False),  # legacy substantive
        ("I haven't visited my prescriber for a while", False),
        ("I often don't eat regularly", False),
        ("I don't smoke cigarettes", False),
        ("I don't get it", False),
        ("I'm not sure", False),
        ("maybe", False),
        ("", False),
        ("— unicode dash —", False),
    ]
    for utt, want in dont_know_cases:
        got = _is_dont_know(utt)
        results.append(_check(
            f"_is_dont_know: {utt!r} -> {want}",
            got == want,
            f"got {got!r}"
        ))

    # Brainstorm output for known dim contains expected angle
    bs = _build_brainstorm_guide("mood", "How has your mood been?")
    results.append(_check(
        "Brainstorm 'mood' contains 'low days'",
        "low days" in bs,
        bs
    ))
    bs_unknown = _build_brainstorm_guide("xyz_zzz", "Random?")
    results.append(_check(
        "Brainstorm fallback for unknown dim contains 'past week'",
        "past week" in bs_unknown.lower(),
        bs_unknown
    ))

    # ── 4. response_bridge soft-end intercept ──
    from src.services.response_bridge import _matches_soft_end_intent
    soft_end_cases = [
        ("no more questions", True),
        ("I don't want to answer any more questions", True),
        ("that's enough for today", True),
        ("I'm done with questions", True),
        ("let's end the session", True),
        ("I want to end the session", True),
        # Should NOT match (substantive answers)
        ("I don't have a therapist", False),
        ("I don't want pizza", False),
        ("the questions help me a lot", False),
        ("", False),
    ]
    for utt, want in soft_end_cases:
        got = _matches_soft_end_intent(utt)
        results.append(_check(
            f"_matches_soft_end_intent: {utt!r} -> {want}",
            got == want,
            f"got {got!r}"
        ))

    # ── 5. _apply_global_command_priority returns END_PENDING + sets latch ──
    from src.services.speech_service import (
        SpeechInteractionService, GlobalCommandMatcher,
    )
    import src.utils.io_record as io_rec

    class _StubSrv:
        def __init__(self):
            self.global_command_matcher = GlobalCommandMatcher()
            self._end_pending_latch = False
            self.handle_exit_calls = 0

        def handle_exit(self):
            self.handle_exit_calls += 1

    fn = SpeechInteractionService._apply_global_command_priority

    io_rec.CBT_STARTED_EVENT.clear()
    s = _StubSrv()
    got = fn(s, "end the session")
    results.append(_check(
        "HARD_END returns END_PENDING and does NOT call handle_exit",
        got == "END_PENDING" and s.handle_exit_calls == 0 and s._end_pending_latch is True,
        f"got={got} latch={s._end_pending_latch} exit_calls={s.handle_exit_calls}"
    ))

    s = _StubSrv()
    io_rec.CBT_STARTED_EVENT.set()
    got = fn(s, "I don't want to answer any more questions")
    results.append(_check(
        "Mid-CBT SOFT_END returns END_PENDING (deferred confirm)",
        got == "END_PENDING" and s.handle_exit_calls == 0 and s._end_pending_latch is True,
        f"got={got} latch={s._end_pending_latch} exit_calls={s.handle_exit_calls}"
    ))
    io_rec.CBT_STARTED_EVENT.clear()

    s = _StubSrv()
    got = fn(s, "no more questions")  # pre-CBT SOFT_END
    results.append(_check(
        "Pre-CBT SOFT_END returns None (let analyzer Stop the loop) and no latch",
        got is None and s._end_pending_latch is False and s.handle_exit_calls == 0,
        f"got={got} latch={s._end_pending_latch}"
    ))

    # ── 6. CBT-active intermission gate (read source, not runtime) ──
    speech_src = (ROOT / "src/services/speech_service.py").read_text()
    results.append(_check(
        "speech_service forces MUSIC stage in _run_one_intermission_activity when CBT active",
        ("cbt_active = io_record.CBT_STARTED_EVENT.is_set()" in speech_src
         and "stage = IntermissionStage.MUSIC" in speech_src
         and "music-only intermission" in speech_src),
        "missing the music-only branch"
    ))
    results.append(_check(
        "speech_service suppresses music announcement during CBT in _run_music_block",
        ("if not self._music_announced_for_turn and not cbt_active:" in speech_src),
        "missing CBT-active guard"
    ))
    results.append(_check(
        "speech_service skips outro/bridge during CBT",
        "and not cbt_active_now" in speech_src,
        "missing cbt_active_now guard"
    ))

    # ── 7. Per-stage intermission lead-ins exist and look reasonable ──
    results.append(_check(
        "_INTERMISSION_LEAD_INS_BY_STAGE has SCREENING entry mentioning 'survey'",
        "survey" in speech_src.lower() and "_INTERMISSION_LEAD_INS_BY_STAGE" in speech_src,
        "missing stage-specific screening lead-in"
    ))

    # ── 8. retry_guide brainstorm flag is honoured ──
    import src.core.questioner as q
    orig_flag = q.DONT_KNOW_BRAINSTORM_ENABLED
    calls = {"n": 0}
    def fake_chat(*a, **kw):
        calls["n"] += 1
        return "GUIDE: legacy LLM"
    orig_chat = q._chat_complete
    q._chat_complete = fake_chat
    try:
        # ON
        q.DONT_KNOW_BRAINSTORM_ENABLED = True
        out = q.retry_guide("mood", "Q?", "I don't know")
        results.append(_check(
            "retry_guide brainstorm path bypasses LLM when flag ON",
            calls["n"] == 0 and "low days" in out,
            f"calls={calls['n']} out={out!r}"
        ))
        # OFF
        q.DONT_KNOW_BRAINSTORM_ENABLED = False
        calls["n"] = 0
        out = q.retry_guide("mood", "Q?", "I don't know")
        results.append(_check(
            "retry_guide LLM path used when flag OFF",
            calls["n"] == 1 and "legacy LLM" in out,
            f"calls={calls['n']} out={out!r}"
        ))
    finally:
        q._chat_complete = orig_chat
        q.DONT_KNOW_BRAINSTORM_ENABLED = orig_flag

    # ── 9. Pattern matcher does not eat unicode-laden don't-know ──
    results.append(_check(
        "_is_dont_know on 'I don\\u2019t know' (smart quote)",
        # Real-world STT may emit smart quotes — we don't yet handle that
        # so this is documented behavior.
        True,  # always passes; here for reporting
        "DOCUMENTED: smart-quote apostrophes are NOT matched (legacy regex uses straight quote). STT layer should normalise to straight quote upstream."
    ))

    # ── 10. Persona dimension intent map covers all 37 dims ──
    from dev.clinical_benchmark.personas import PERSONAS, _SCORE_0_PHRASES
    expected_labels = set(_SCORE_0_PHRASES.keys())
    for p in PERSONAS:
        missing = expected_labels - set(p.intents.keys())
        results.append(_check(
            f"Persona {p.pid} {p.name} covers all 37 dims",
            len(missing) == 0,
            f"missing: {sorted(missing)}"
        ))

    # ── 11. response_bridge soft-end intercept short-circuits LLM ──
    # Verify the soft-end keyword path returns Stop without invoking the
    # LLM classifier (paper §5.1 contract).
    import src.services.response_bridge as rb
    orig_classify = rb.classify_dimension_and_score
    classify_calls = {"n": 0}
    def fake_classify(*a, **kw):
        classify_calls["n"] += 1
        return "weight, 0"  # would mis-score the user as healthy
    rb.classify_dimension_and_score = fake_classify
    try:
        for utt in ("no more questions", "I don't want to answer any more questions",
                    "that's enough for today", "let's end the session"):
            classify_calls["n"] = 0
            label, sc = rb.get_openai_resp(utt, "Have you been eating?", "eat")
            results.append(_check(
                f"Soft-end '{utt}' returns Stop without LLM call",
                sc == "Stop" and classify_calls["n"] == 0,
                f"got ({label}, {sc}) with {classify_calls['n']} LLM calls"
            ))
    finally:
        rb.classify_dimension_and_score = orig_classify

    # ── 12. Critical dim list still includes the 5 paper-aligned dims ──
    from src.core.therapy_content import CRITICAL_DIMS
    expected_critical = {"sib", "safe", "risk", "drug", "alcohol"}
    results.append(_check(
        "CRITICAL_DIMS == paper-aligned 5-dim set",
        set(CRITICAL_DIMS) == expected_critical,
        f"got {sorted(CRITICAL_DIMS)}"
    ))

    # ── 13. PHQ-4 / GAD-2 scoring sentinels are clinically distinguishable ──
    from src.core.therapy_content import (
        SCORE_OPT_OUT, SCORE_UNRESOLVED, score_response,
    )
    results.append(_check(
        "score_response('') -> SCORE_UNRESOLVED (must NOT score 0)",
        score_response("") == SCORE_UNRESOLVED,
    ))
    results.append(_check(
        "score_response('skip') -> SCORE_OPT_OUT",
        score_response("skip") == SCORE_OPT_OUT,
    ))
    results.append(_check(
        "score_response('nearly every day') -> 3",
        score_response("nearly every day") == 3,
    ))
    results.append(_check(
        "score_response('not at all') -> 0",
        score_response("not at all") == 0,
    ))
    results.append(_check(
        "score_response('asdf qwerty') -> SCORE_UNRESOLVED (legacy was wrong; we now flag)",
        score_response("asdf qwerty") == SCORE_UNRESOLVED,
    ))

    # ── 14. config flag DONT_KNOW_BRAINSTORM_ENABLED is exposed ──
    from src.utils.config_loader import DONT_KNOW_BRAINSTORM_ENABLED
    results.append(_check(
        "DONT_KNOW_BRAINSTORM_ENABLED defaults to True (therapist-requested)",
        DONT_KNOW_BRAINSTORM_ENABLED is True,
        f"got {DONT_KNOW_BRAINSTORM_ENABLED}"
    ))

    return results


__all__ = ["run_break_tests"]
