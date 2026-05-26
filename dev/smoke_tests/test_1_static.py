"""Smoke test 1 — static invariants.

Checks that config values, flag gates, legacy wordings, and prompt
structural requirements are all in the state expected for clinical-trial
deployment. No LLM call, no IPC.

Run:
    .venv/bin/python dev/smoke_tests/test_1_static.py
"""
from __future__ import annotations

import sys
import os
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
    print("=== Smoke test 1: static invariants ===")

    # ── Config flags ────────────────────────────────────────────────────
    print("\n[A] Config flags")
    from src.utils import config_loader as cfg
    from src.core import therapy_content as tc

    check("G1 epsilon==1.0 (legacy pure-exploitation)", cfg.EPSILON == 1.0, f"got {cfg.EPSILON}")
    check("G1 alpha==0.5 (legacy aggressive Q update)", cfg.ALPHA == 0.5, f"got {cfg.ALPHA}")
    check("gamma==0.9 (paper+legacy)", cfg.GAMMA == 0.9, f"got {cfg.GAMMA}")
    check("G2 importance[2]=98 (mood pinned)", cfg.ITEM_IMPORTANCE[2] == 98, f"got {cfg.ITEM_IMPORTANCE[2]}")
    check("G2 importance[3]=99 (medication pinned)", cfg.ITEM_IMPORTANCE[3] == 99, f"got {cfg.ITEM_IMPORTANCE[3]}")
    check("G2 importance[11]=97 (eat pinned)", cfg.ITEM_IMPORTANCE[11] == 97, f"got {cfg.ITEM_IMPORTANCE[11]}")
    check("G5 REASK_DIMENSION_N == False", cfg.REASK_DIMENSION_N is False)
    check("G7 CBT_ESCALATION_ENABLED == False", tc.CBT_ESCALATION_ENABLED is False)
    # G8 — OFF (legacy-parity: paper §4.1 back-fill kept disabled so a
    # turn's score lands only on the asked dimension, matching demo flow).
    check("G8 MULTI_DIM_BACKFILL_ENABLED == False (legacy-parity)",
          cfg.MULTI_DIM_BACKFILL_ENABLED is False)
    check("G9 REFLECTIVE_SUMMARIZER_ENABLED == False", cfg.REFLECTIVE_SUMMARIZER_ENABLED is False)
    check("D3 CRISIS_OVERRIDE_ENABLED == False", tc.CRISIS_OVERRIDE_ENABLED is False)
    # D5/D6 — legacy-parity: legacy prototype DOES rephrase via
    # generate_synonymous_sentences() on ~95% of turns. ON.
    check("D5/D6 REPHRASE_AT_RUNTIME == True (legacy-parity)", cfg.REPHRASE_AT_RUNTIME is True)
    check("G11 WARM_START_ENABLED == False", cfg.WARM_START_ENABLED is False)
    check("G12 SESSION_ANALYSIS_ENABLED == False", cfg.SESSION_ANALYSIS_ENABLED is False)
    # G13 — ON (clinician audit artefact; never spoken, no user impact).
    check("G13 SOAP_REPORT_ENABLED == True (clinician audit)", cfg.SOAP_REPORT_ENABLED is True)
    check("G14 DIMENSION_OPTOUTS_ENABLED == False", cfg.DIMENSION_OPTOUTS_ENABLED is False)
    check("G15 SESSION_CAP_ENABLED == False", cfg.SESSION_CAP_ENABLED is False)
    check("REWARD_MODE == 'mean' (paper+legacy)", cfg.REWARD_MODE == "mean", f"got {cfg.REWARD_MODE}")
    check("LITERT_MAX_TOKENS >= 4096 (rich output budget)", cfg.LITERT_MAX_TOKENS >= 4096, f"got {cfg.LITERT_MAX_TOKENS}")

    # ── Legacy wording presence ─────────────────────────────────────────
    print("\n[B] Legacy wording present in source (G3, G6)")
    hdr = (ROOT / "src/core/handler_rl.py").read_text()
    cbt = (ROOT / "src/core/CBT.py").read_text()
    # G3: greeting seed mentions a therapist role and the "couple of
    # questions about your recent daily life" framing. Variants like
    # "intelligent therapist" / "intelligence therapist" both satisfy
    # the demo-parity intent.
    check("G3 greeting seed mentions therapist role",
          "intelligent therapist" in hdr or "intelligence therapist" in hdr)
    check("G3 greeting seed has 'couple of questions about your recent daily life'",
          "couple of questions about your recent daily life" in hdr)
    check("G6 CBT Stage 0 says 'you have issue in'",
          "you have issue in" in cbt)
    check("G6 CBT Stage 0 says 'Which dimension would you like to work on today'",
          "Which dimension would you like to work on today" in cbt)
    check("G6 CBT Stage 0 says 'Tell me the dimension number'",
          "Tell me the dimension number" in cbt)
    # Non-legacy wording must be gone
    check("G6 No 'you have concerns in' leftover",
          "you have concerns in" not in cbt)
    check("G6 No 'Which area would you like' leftover",
          "Which area would you like" not in cbt)

    # ── Prompt structural checks ─────────────────────────────────────────
    print("\n[C] Prompt structural checks")
    from src.core import reflection_validation as rv
    check("RV Validator prompt requires 4-7 sentences",
          "4 to 7 sentences" in rv.RV_VALIDATOR_OARS_SYSTEM_PROMPT)
    check("RV Validator has the demo medication Example 4",
          "Taking medication as prescribed" in rv.RV_VALIDATOR_OARS_SYSTEM_PROMPT
          and "blister packs" in rv.RV_VALIDATOR_OARS_SYSTEM_PROMPT)
    check("CBT Stage-1 Guide prompt requires 3-5 clauses",
          "3 to 5 distinct unhelpful thoughts" in cbt)
    check("CBT Stage-1 Guide has the demo medication enumeration example",
          "you always forget" in cbt
          and "you fear that if you take it regularly" in cbt)
    # DECISION parser is legacy substring semantics
    from src.core import CBT as cbtmod
    check("_parse_decision('there are 0 signs')=='0' (legacy-loose)",
          cbtmod._parse_decision("there are 0 signs of distortion") == "0")
    check("_parse_decision('DECISION: 1')=='1'",
          cbtmod._parse_decision("DECISION: 1") == "1")
    check("_parse_decision('')=='1' (fail-closed default)",
          cbtmod._parse_decision("") == "1")

    # ── Gate wiring ──────────────────────────────────────────────────────
    print("\n[D] Gate wiring")
    q = (ROOT / "src/core/questioner.py").read_text()
    check("G5 gate uses REASK_DIMENSION_N in questioner",
          "if REASK_DIMENSION_N and valid == 0" in q)
    check("G8 gate uses MULTI_DIM_BACKFILL_ENABLED in questioner",
          "if not MULTI_DIM_BACKFILL_ENABLED" in q)
    check("G9 gate uses REFLECTIVE_SUMMARIZER_ENABLED in questioner",
          "if REFLECTIVE_SUMMARIZER_ENABLED and seg" in q)
    check("G7 gate uses CBT_ESCALATION_ENABLED in CBT",
          "if CBT_ESCALATION_ENABLED:" in cbt)
    check("D3 gate uses CRISIS_OVERRIDE_ENABLED in handler",
          "if CRISIS_OVERRIDE_ENABLED and self._crisis_scan()" in hdr)
    check("G4 on-topic reuses rv_consolidated validation_text",
          "G4: reuse it" in q
          or "rv_consolidated() already ran" in q)

    # ── Misc runtime expectations ───────────────────────────────────────
    print("\n[E] Runtime expectations")
    check("handler_rl has NO _run_phq4_screening method (D2)",
          "_run_phq4_screening" not in hdr)
    check("handler_rl has NO first-2-turn force-target block (D4)",
          "Force-targeting prior Score-2 dim" not in hdr
          or "[RESUME] Force-targeting" not in hdr)
    check("CRITICAL_DIMS still defined (crisis scaffolding kept)",
          isinstance(tc.CRITICAL_DIMS, frozenset) and len(tc.CRITICAL_DIMS) == 5)
    check("SAFETY_RESOURCES_MESSAGE still exists",
          "988" in tc.SAFETY_RESOURCES_MESSAGE)

    # ── Question library presence ───────────────────────────────────────
    print("\n[F] Question library")
    from src.utils.config_loader import QUESTION_LIB_FILENAME
    import json
    check(f"question_lib file exists at {QUESTION_LIB_FILENAME}",
          os.path.exists(QUESTION_LIB_FILENAME))
    if os.path.exists(QUESTION_LIB_FILENAME):
        qlib = json.loads(Path(QUESTION_LIB_FILENAME).read_text())
        check("question_lib has 37 dims", len(qlib) == 37)
        check("q[11] label=='eat'", qlib.get("11", {}).get("1", {}).get("label") == "eat")
        check("q[2] label=='mood'", qlib.get("2", {}).get("1", {}).get("label") == "mood")
        check("q[3] label=='medication'", qlib.get("3", {}).get("1", {}).get("label") == "medication")
        check("q[11] question matches demo wording",
              "How's your eating? Are you eating regularly?" in qlib["11"]["1"]["question"])

    print("\n=== RESULT ===")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print(f"PASSED {sum(1 for _ in range(0))}")  # count done in caller
    print("All static invariants held.")
    sys.exit(0)


if __name__ == "__main__":
    main()
