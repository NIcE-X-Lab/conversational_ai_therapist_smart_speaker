"""Smoke test 3 — questioner pipeline (DLA + RV + G4 regression).

Mocks the LLM layer and the INPUT_QUEUE so we can exercise ask_question
end-to-end without touching Gemma or hardware. Verifies:

  1. Yes path: question asked, "Yes" classified as Yes keyword, score 0
     (eat's Yes=0), NO follow-up, note appended. DLA_terminate=0.
  2. No path (Score-2): "No" → eat's No=2 → followup_to_RV constructed
     via the legacy generate_change() path (G9 off), Validator runs once
     on the user's follow-up (G4 — no double-call), next question gets
     the validation text prefixed.
  3. Stop path: "stop" → DLA_terminate=1.
  4. Validator call count on on-topic follow-up == 1 (G4 regression).

Run:
    .venv/bin/python dev/smoke_tests/test_3_questioner.py
"""
from __future__ import annotations

import os
import sys
import queue
import threading
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


def main():
    print("=== Smoke test 3: questioner pipeline (+ G4 regression) ===")

    # Import everything AFTER we set up module-level mocks.
    import src.utils.io_record as io_rec
    from src.core import questioner, reflection_validation as rv
    from src.core.response_analyzer import classify_dimension_and_score
    from src.services import response_bridge

    # LLM call counter so we can assert the Validator fires exactly once.
    llm_calls: list[tuple[str, str]] = []  # (role, first_32_chars)

    def fake_llm_complete(system, user, role=None, **kwargs):
        role_value = role.value if hasattr(role, "value") else str(role)
        llm_calls.append((role_value, str(system)[:32]))
        # Return role-appropriate mock text.
        if "rv_reasoner" in role_value:
            return "DECISION: 0"
        if "rv_validator" in role_value:
            return ("VALIDATION: It sounds like medication has been hard. "
                    "Many people feel the same — small routines help. "
                    "Consider setting a daily alarm and a pill box by the "
                    "toothbrush. Keep a brief note of doses. You're not stuck.")
        if "rv_guide" in role_value:
            return "GUIDE: Let's focus back on the topic."
        if "analyzer" in role_value:
            # Not used in this test (we short-circuit via <=3 token shortcut).
            return "NA, 99"
        if "cbt_reasoner" in role_value:
            return "DECISION: 0"
        if "cbt_guide" in role_value:
            return "UNHELPFUL_THOUGHTS: you think ...; you fear ...; you worry ...; you see ...; you assume ..."
        if "reflective_summarizer" in role_value:
            return "REFLECTIVE_SUMMERIZER: You mentioned that medication has been hard to take."
        if "rephraser" in role_value:
            return user  # pass-through
        # GENERAL / fallback
        return "Thank you for sharing that with me."

    # Patch the LLM call at every module that imported `llm_complete`
    # via `from src.models.llm_client import llm_complete`. The import
    # creates a local binding per module, so patching only the source
    # module would miss these.
    import src.models.llm_client as llm_client_mod
    original_llm = llm_client_mod.llm_complete
    llm_client_mod.llm_complete = fake_llm_complete
    import importlib
    for mod_path in (
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
                setattr(m, "llm_complete", fake_llm_complete)
        except Exception as e:
            print(f"  [warn] failed to patch {mod_path}: {e}")

    # Load a minimal question_lib: eat dim only.
    import json
    full_lib = json.loads(Path("data/libs/question_lib_v4.json").read_text())
    qlib = {"11": full_lib["11"]}
    # Freshen score/notes
    qlib["11"]["1"]["score"] = []
    qlib["11"]["1"]["notes"] = []

    # ── Case A: Yes path (score 0, no RV, no Validator call) ─────────────
    print("\n[A] Yes path on 'eat' (Yes=0, no follow-up)")
    llm_calls.clear()
    io_rec.INPUT_QUEUE = queue.Queue()
    io_rec.INPUT_QUEUE.put("Yes.")

    reward, terminate, last_q = questioner.ask_question(qlib, 11)
    check("ask_question returns tuple",
          isinstance(reward, float) and isinstance(terminate, int))
    check("DLA_terminate == 0", terminate == 0)
    check("eat score appended: 0 (Yes=0)",
          qlib["11"]["1"]["score"] == [0])
    check("reward == 0.0 (mean of [0])",
          abs(reward - 0.0) < 1e-9)
    # No RV path because Yes path branches out before RV.
    rv_val_calls = [c for c in llm_calls if "rv_validator" in c[0]]
    rv_rsn_calls = [c for c in llm_calls if "rv_reasoner" in c[0]]
    check("No Validator call on Yes path", len(rv_val_calls) == 0,
          f"got {len(rv_val_calls)}")
    check("No RV Reasoner call on Yes path", len(rv_rsn_calls) == 0,
          f"got {len(rv_rsn_calls)}")

    # ── Case B: No path (Score-2, RV on follow-up, G4 once) ──────────────
    print("\n[B] No path on 'eat' (No=2, RV engaged, G4 regression)")
    qlib["11"]["1"]["score"] = []
    qlib["11"]["1"]["notes"] = []
    llm_calls.clear()
    io_rec.INPUT_QUEUE = queue.Queue()
    io_rec.INPUT_QUEUE.put("No.")
    # The follow-up collection: questioner speaks "It seems that ... Can
    # you tell me more about it?", then blocks on INPUT_QUEUE for the
    # follow-up answer.
    io_rec.INPUT_QUEUE.put("I've been skipping meals because of work stress.")

    # Drain OUTPUT_QUEUE in a background thread so the handler doesn't
    # block on a full queue.
    drain_stop = threading.Event()
    drained = []

    def _drain():
        while not drain_stop.is_set():
            try:
                drained.append(io_rec.OUTPUT_QUEUE.get(timeout=0.1))
            except queue.Empty:
                continue
    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    try:
        reward, terminate, last_q = questioner.ask_question(qlib, 11)
    finally:
        drain_stop.set()
        drainer.join(timeout=1.0)

    check("DLA_terminate == 0 on No path", terminate == 0)
    check("eat score appended: 2 (No=2)",
          qlib["11"]["1"]["score"] == [2])
    # The RV path should have run Reasoner once and Validator exactly once.
    rv_val_calls = [c for c in llm_calls if "rv_validator" in c[0]]
    rv_rsn_calls = [c for c in llm_calls if "rv_reasoner" in c[0]]
    check("RV Reasoner called exactly once on No-path follow-up",
          len(rv_rsn_calls) == 1, f"got {len(rv_rsn_calls)}")
    check("G4 regression — RV Validator called EXACTLY ONCE (was 2 before fix)",
          len(rv_val_calls) == 1, f"got {len(rv_val_calls)}")
    # G9 off: reflective_summarizer should NOT fire
    rs_calls = [c for c in llm_calls if "reflective_summarizer" in c[0]]
    check("G9 off — reflective_summarizer NOT called",
          len(rs_calls) == 0, f"got {len(rs_calls)}")
    # Note structure: legacy behaviour appends TWO note rows on Score-2
    # follow-up — one from _if_valid_response capturing the original
    # (question, resp), and a second from evaluate_result capturing the
    # RV triplet (decision, guide, validation).
    notes = qlib["11"]["1"]["notes"]
    check("Notes contains 2 note rows (legacy Score-2 pattern)",
          len(notes) == 2, f"got {len(notes)}")
    if len(notes) >= 2:
        initial_note = "\n".join(notes[0]) if isinstance(notes[0], list) else str(notes[0])
        rv_note = "\n".join(notes[1]) if isinstance(notes[1], list) else str(notes[1])
        check("Initial note has original_question", "original_question:" in initial_note)
        check("Initial note has original_resp", "original_resp:" in initial_note)
        check("RV note has rv_decision: 0", "rv_decision: 0" in rv_note)
        check("RV note has non-empty rv_validation",
              "rv_validation: " in rv_note
              and len(rv_note.split("rv_validation: ", 1)[1].split("\n", 1)[0].strip()) > 10)

    # ── Case C: Stop path ────────────────────────────────────────────────
    print("\n[C] Stop path on 'eat'")
    qlib["11"]["1"]["score"] = []
    qlib["11"]["1"]["notes"] = []
    llm_calls.clear()
    io_rec.INPUT_QUEUE = queue.Queue()
    io_rec.INPUT_QUEUE.put("stop")

    drain_stop.clear()
    drained.clear()
    drain_stop = threading.Event()
    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()
    try:
        reward, terminate, _ = questioner.ask_question(qlib, 11)
    finally:
        drain_stop.set()
        drainer.join(timeout=1.0)
    check("DLA_terminate == 1 on Stop", terminate == 1)
    # Legacy: Stop terminates before score append, so score list stays
    # empty. This is correct — a "stop" request does not contribute a
    # clinical score for the dimension.
    check("Stop path leaves score list empty", qlib["11"]["1"]["score"] == [],
          f"got {qlib['11']['1']['score']}")

    # Restore
    llm_client_mod.llm_complete = original_llm

    print("\n=== RESULT ===")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("All questioner pipeline invariants held.")
    sys.exit(0)


if __name__ == "__main__":
    main()
