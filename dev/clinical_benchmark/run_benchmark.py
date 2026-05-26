"""Main entry point for the clinical-trial benchmark.

Runs all 10 personas through full HandlerRL sessions (screening + CBT)
with the LLM stubbed but every code path exercised, then runs a battery
of adversarial/break tests against `_apply_global_command_priority`,
`_classify_confirm`, the brainstorm matcher, and the response_bridge
soft-end intercept.

Outputs:
  • dev/clinical_benchmark/results/<timestamp>/P{NN}_<name>.json
  • dev/clinical_benchmark/results/<timestamp>/break_tests.json
  • dev/clinical_benchmark/results/<timestamp>/REPORT.md
  • dev/clinical_benchmark/results/<timestamp>/coverage.json

Usage:
    python dev/clinical_benchmark/run_benchmark.py
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dev.clinical_benchmark.personas import PERSONAS  # noqa: E402
from dev.clinical_benchmark.harness import run_persona, RunTelemetry  # noqa: E402
from dev.clinical_benchmark.break_tests import run_break_tests  # noqa: E402
from dev.clinical_benchmark.report import build_report  # noqa: E402


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "dev" / "clinical_benchmark" / "results" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Clinical-trial benchmark — run dir: {run_dir} ===\n")

    # ── 1. Run all 10 personas ──
    persona_runs = []
    for persona in PERSONAS:
        print(f"[PERSONA] {persona.pid} {persona.name} — {persona.profile}")
        try:
            telemetry = run_persona(persona, run_dir=run_dir)
            persona_runs.append(telemetry)
            print(
                f"   done: {len(telemetry.agent_turns)} agent turns, "
                f"{len(telemetry.user_turns)} user turns, "
                f"{sum(telemetry.llm_calls_by_role.values())} LLM calls, "
                f"{len(telemetry.exceptions)} exceptions, "
                f"CBT success={telemetry.cbt_success}, "
                f"crisis={telemetry.crisis_triggered}\n"
            )
        except Exception as e:
            print(f"   FATAL: {type(e).__name__}: {e}")
            traceback.print_exc()

    # ── 2. Run break tests ──
    print("\n=== Break / adversarial tests ===\n")
    break_results = run_break_tests()
    (run_dir / "break_tests.json").write_text(json.dumps(break_results, indent=2))

    # ── 3. Build report ──
    print("\n=== Building report ===\n")
    report_md = build_report(persona_runs, break_results, run_dir=run_dir)
    (run_dir / "REPORT.md").write_text(report_md)

    print(f"Report written: {run_dir / 'REPORT.md'}\n")
    print("=" * 70)
    print(report_md)
    print("=" * 70)


if __name__ == "__main__":
    main()
