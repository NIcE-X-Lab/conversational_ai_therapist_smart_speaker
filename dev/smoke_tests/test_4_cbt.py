"""Smoke test 4 — CBT flow.

Builds a synthetic question_lib with one Score-2 dim (medication),
scripts user inputs for Stage 0/1/2/3, mocks llm_complete, runs run_cbt,
and verifies:

  1. Stage 0 wording is LEGACY verbatim ("you have issue in:", "Which
     dimension would you like to work on today?", "Tell me the dimension
     number. For example: 1").
  2. Dim picker parses "1" correctly.
  3. Stage 1 recap speaks "Let us work on dimension '<name>'. From our
     record, you mentioned that: <statement>".
  4. Each stage runs ONE Reasoner + ONE Validator call (no retries when
     Reasoner says 0).
  5. CBT_ESCALATION_MESSAGE NEVER reaches the spoken queue (G7 off).
  6. Final note appends CBT_stage: success.
  7. Closing message: "Great work today. We completed the CBT steps..."

Run:
    .venv/bin/python dev/smoke_tests/test_4_cbt.py
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


def main():
    print("=== Smoke test 4: CBT flow ===")
    import src.utils.io_record as io_rec
    from src.core import CBT as cbt_mod
    from src.core.therapy_content import CBT_ESCALATION_MESSAGE

    # ── Build question_lib with ONLY medication at score=2, rest clean ──
    # run_cbt iterates range(1, len(lib)+1), so we keep the full 37 dims.
    import json
    full_lib = json.loads(Path("data/libs/question_lib_v4.json").read_text())
    qlib = copy.deepcopy(full_lib)
    for k in qlib:
        qlib[k]["1"]["score"] = []
        qlib[k]["1"]["notes"] = []
    # Set medication score=2 so it's the lone Score-2 candidate.
    qlib["3"]["1"]["score"] = [2]
    # Notes: synthesize an RV follow-up entry so the stage-1 recap has a
    # statement to extract from followup_resp_1.
    qlib["3"]["1"]["notes"].append([
        "original_question: Have you been taking medication according to doctor's recommendation?",
        "original_resp: No, not really.",
        "followup_resp: I just always forget to take it",
        "rv_decision: 0",
        "rv_guide: ",
        "followup_resp_1: I just always forget to take it, and I don't wanna take too much of it",
        "rv_validation: It makes sense that you always forget.",
        "therapist_resp: ",
    ])

    # ── LLM mock, role-aware ─────────────────────────────────────────────
    llm_calls: list[tuple[str, str]] = []

    def fake_llm(system, user, role=None, **kwargs):
        role_value = role.value if hasattr(role, "value") else str(role)
        llm_calls.append((role_value, str(user)[:80]))
        if "cbt_reasoner" in role_value:
            return "DECISION: 0"  # always accept user's stage output
        if "cbt_guide" in role_value:
            return ("UNHELPFUL_THOUGHTS: you think you always forget; "
                    "you fear that if you take it regularly, you'll get dependent; "
                    "you worry that the prescribed dose is too much; "
                    "you see medication as loss of control; "
                    "you assume a few missed doses mean you can't do this at all.")
        if "analyzer" in role_value or "reflective_summarizer" in role_value or "rephraser" in role_value:
            return ""
        if "rv_reasoner" in role_value:
            return "DECISION: 0"
        if "rv_validator" in role_value:
            return "VALIDATION: ok."
        # GENERAL / CBT prompter
        return "QUESTION: ok."

    _patch_llm(fake_llm)

    # ── Script user inputs: "1" (pick med) → unhelpful → challenge → reframe
    io_rec.INPUT_QUEUE = queue.Queue()
    for reply in (
        "1",                                                           # stage 0
        "I guess I fear I'll get dependent on it.",                    # stage 1
        "I haven't seen any real evidence I'd actually get dependent.",  # stage 2
        "I'll try taking it as prescribed and see what happens.",      # stage 3
    ):
        io_rec.INPUT_QUEUE.put(reply)

    # ── Drain OUTPUT_QUEUE in background ─────────────────────────────────
    spoken: list[str] = []
    drain_stop = threading.Event()

    def _drain():
        while not drain_stop.is_set():
            try:
                spoken.append(str(io_rec.OUTPUT_QUEUE.get(timeout=0.1)))
            except queue.Empty:
                continue
    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    # ── Run CBT ──────────────────────────────────────────────────────────
    try:
        cbt_mod.run_cbt(qlib)
    finally:
        drain_stop.set()
        drainer.join(timeout=1.0)

    # ── Assertions ───────────────────────────────────────────────────────
    print("\n[A] Stage 0 wording (G6 legacy)")
    stage0 = next((s for s in spoken if "Thank you for answering the questions" in s), "")
    check("Stage 0 spoken", bool(stage0))
    check("Stage 0 says 'you have issue in:'",
          "you have issue in:" in stage0)
    check("Stage 0 says 'Which dimension would you like to work on today?'",
          "Which dimension would you like to work on today?" in stage0)
    check("Stage 0 says 'Tell me the dimension number. For example: 1'",
          "Tell me the dimension number. For example: 1" in stage0)
    check("Stage 0 does NOT say 'concerns in'",
          "concerns in" not in stage0)
    check("Stage 0 does NOT say 'Which area'",
          "Which area" not in stage0)

    print("\n[B] Stage 1 recap + question")
    s1 = next((s for s in spoken if "Let us work on dimension" in s), "")
    check("Stage 1 recap spoken", bool(s1))
    check("Stage 1 recap mentions 'Taking Medication as Prescribed'",
          "Taking Medication as Prescribed" in s1)
    check("Stage 1 recap includes statement from followup_resp_1",
          "always forget to take it" in s1 or "get off of it" in s1)
    check("Stage 1 question asked",
          any("identify any unhelpful thoughts" in s for s in spoken))

    print("\n[C] Stage 2 & 3 questions")
    check("Stage 2 question asked",
          any("challenge those unhelpful thoughts" in s for s in spoken))
    check("Stage 3 question asked",
          any("reframe the unhelpful thought" in s for s in spoken))

    print("\n[D] Closing")
    check("Closing 'Great work today. We completed the CBT steps...'",
          any("Great work today" in s and "CBT steps" in s for s in spoken))

    print("\n[E] G7 — CBT_ESCALATION_MESSAGE NEVER spoken")
    esc_spoken = any(CBT_ESCALATION_MESSAGE[:60] in s for s in spoken)
    check("No CBT escalation message in spoken output", not esc_spoken)

    print("\n[F] LLM call accounting")
    reasoner_calls = [c for c in llm_calls if "cbt_reasoner" in c[0]]
    guide_calls = [c for c in llm_calls if "cbt_guide" in c[0]]
    # Per demo: 1 Reasoner per stage (S1, S2, S3) = 3. Stage Guides only
    # fire on retry (not here — Reasoner says 0). recap_stage3_challenge
    # uses CBT_GUIDE role and fires once right before Stage 3's prompt.
    check("CBT Reasoner called exactly 3 times (S1/S2/S3, no retries)",
          len(reasoner_calls) == 3, f"got {len(reasoner_calls)}")
    check("CBT_GUIDE role fired once (stage-3 recap, no retries)",
          len(guide_calls) == 1, f"got {len(guide_calls)}")

    print("\n[G] Notes append")
    notes = qlib["3"]["1"]["notes"]
    final_note = notes[-1] if notes else []
    joined = "\n".join(final_note) if isinstance(final_note, list) else ""
    check("Final note tagged CBT_stage: success",
          "CBT_stage: success" in joined)
    check("Final note has CBT_dimension: medication",
          "CBT_dimension: medication" in joined)
    check("Final note has CBT_unhelpful_thoughts",
          "CBT_unhelpful_thoughts:" in joined)
    check("Final note has CBT_challenge",
          "CBT_challenge:" in joined)
    check("Final note has CBT_reframe",
          "CBT_reframe:" in joined)

    print("\n=== RESULT ===")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("All CBT flow invariants held.")
    sys.exit(0)


if __name__ == "__main__":
    main()
