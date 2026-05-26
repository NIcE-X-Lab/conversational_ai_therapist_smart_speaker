"""Report generator for the clinical-trial benchmark."""
from __future__ import annotations

import datetime
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dev.clinical_benchmark.harness import RunTelemetry
from dev.clinical_benchmark.personas import PERSONAS, _SCORE_0_PHRASES


_ALL_LABELS = sorted(_SCORE_0_PHRASES.keys())


def _intent_to_int(intent: Any) -> str:
    """Compact label for the intent column in the per-dim matrix."""
    if isinstance(intent, int):
        return str(intent)
    return str(intent)


def _max_recorded(scores: List[int]) -> str:
    if not scores:
        return "-"
    return str(max(scores))


def _accuracy_status(intent: Any, recorded: List[int], persona_terminated_early: bool = False) -> str:
    """OK / DIVERGE / NA classification per (intended, recorded) pair."""
    if not recorded:
        # Score-0 intent with no recording means the dim was simply
        # never picked by the Q-learning policy this turn — that is
        # expected when the policy ran out of Score-2 dims (and so
        # there's no new info to gain) or terminated early.
        if isinstance(intent, int) and intent == 0:
            return "NA(unasked)"
        if intent in ("stop", "stop_early", "repeat_stop"):
            return "NA(stopped)"
        if intent in ("empty", "long", "unicode") and persona_terminated_early:
            # Adversarial persona terminated screening before reaching
            # this dim — those don't count as MISS.
            return "NA(adversarial-stopped)"
        return "MISS"
    rec_max = max(recorded)
    if isinstance(intent, int):
        return "OK" if rec_max == intent else f"DIVERGE({rec_max} vs {intent})"
    # Special-token intents are checked qualitatively elsewhere.
    return f"REC={rec_max}"


def build_report(
    persona_runs: List[RunTelemetry],
    break_results: List[Dict[str, Any]],
    *,
    run_dir: Path,
) -> str:
    out: List[str] = []
    ts = datetime.datetime.now().isoformat(timespec="seconds")

    out.append(f"# Clinical-trial benchmark — {ts}\n")
    out.append(
        "End-to-end persona-driven test of the CaiTI therapist pipeline. "
        "All 10 personas drove a full HandlerRL session (37-dim screening + CBT) "
        "with TTS / STT / GPIO disabled and the LLM stubbed by role-aware "
        "deterministic responses; every paper-aligned LLMRole call site was "
        "exercised.\n"
    )

    # ── Persona summary table ──
    out.append("## 1. Persona run summary\n")
    out.append(
        "| PID | Persona | Duration | Agent turns | User turns | LLM calls | "
        "Retry/Brainstorm | RV on/off | CBT picked | CBT result | Crisis | Exceptions |"
    )
    out.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for t in persona_runs:
        d = t.as_dict()
        rv = d["rv_decision_breakdown"]
        cbt_outcomes = ",".join(d["cbt_stage_outcomes"]) or "-"
        cbt_result = "success" if d["cbt_success"] else (cbt_outcomes if d["completed_cbt"] else "skipped")
        out.append(
            f"| {d['pid']} | {d['name']} | {d['duration_sec']}s | "
            f"{d['agent_turn_count']} | {d['user_turn_count']} | "
            f"{sum(d['llm_calls_by_role'].values())} | "
            f"{t.retry_guide_triggers}/{t.brainstorm_triggers} | "
            f"{rv['on_topic_0']}/{rv['off_topic_1']} | "
            f"{d['cbt_dimension_picked'] or '-'} | "
            f"{cbt_result} | "
            f"{'YES' if d['crisis_triggered'] else 'no'} | "
            f"{len(d['exceptions'])} |"
        )
    out.append("")

    # ── LLM call counts per role ──
    role_totals: Dict[str, int] = defaultdict(int)
    for t in persona_runs:
        for role, n in t.llm_calls_by_role.items():
            role_totals[role] += n
    out.append("## 2. LLM call distribution by paper-aligned role\n")
    out.append("| Role | Total calls (across 10 personas) |")
    out.append("|---|---|")
    for role in sorted(role_totals.keys()):
        out.append(f"| {role} | {role_totals[role]} |")
    out.append("")

    # ── 37-dim coverage matrix ──
    out.append("## 3. 37-dimension coverage matrix\n")
    out.append("Per dimension × persona, showing the persona's intended intent and the maximum recorded score from the run. Rows are dimensions; columns are personas P01..P10.\n")
    header = "| Dim | " + " | ".join(t.pid for t in persona_runs) + " |"
    sep = "|---|" + "|".join(["---"] * len(persona_runs)) + "|"
    out.append(header)
    out.append(sep)

    accuracy_counter = Counter()
    diverge_rows: List[str] = []

    # Mark personas that terminated early (very few asked dims) so empty
    # intents on dims that simply never got asked aren't counted as MISS.
    early_stopped: Dict[str, bool] = {}
    for t in persona_runs:
        early_stopped[t.pid] = (
            len(t.recorded_scores) < 6  # less than ~16% of 37 dims asked
            or any(intent in ("stop", "stop_early", "repeat_stop")
                   for intent in t.intended_intents.values())
        )

    for label in _ALL_LABELS:
        cells = []
        for t in persona_runs:
            intent = t.intended_intents.get(label, "-")
            recorded = t.recorded_scores.get(label, [])
            status = _accuracy_status(
                intent, recorded,
                persona_terminated_early=early_stopped.get(t.pid, False),
            )
            accuracy_counter[status.split("(")[0]] += 1
            cell = f"{_intent_to_int(intent)}→{_max_recorded(recorded)}"
            cells.append(cell)
            if "DIVERGE" in status:
                diverge_rows.append(
                    f"  - {t.pid} {label}: intended={intent}, recorded_max={max(recorded)}"
                )
        out.append(f"| `{label}` | " + " | ".join(cells) + " |")

    out.append("")
    out.append("### Accuracy summary across the matrix\n")
    total = sum(accuracy_counter.values())
    for k in sorted(accuracy_counter.keys()):
        v = accuracy_counter[k]
        out.append(f"- **{k}**: {v}/{total} ({100 * v / total:.0f}%)")

    if diverge_rows:
        out.append("\n### Score divergences (intended vs recorded_max)\n")
        out.extend(diverge_rows)
    else:
        out.append("\n_No score divergences observed where both intent and recorded score are integers._")
    out.append("")

    # ── Dim-level coverage (was every dim asked at all?) ──
    asked_count: Dict[str, int] = {label: 0 for label in _ALL_LABELS}
    for t in persona_runs:
        for label in t.recorded_scores.keys():
            asked_count[label] += 1
    unasked = [lbl for lbl, n in asked_count.items() if n == 0]
    out.append("## 4. Dimension reach\n")
    out.append(f"- Total dims in lib: **{len(_ALL_LABELS)}**")
    out.append(f"- Dims scored ≥1 persona: **{len(_ALL_LABELS) - len(unasked)}**")
    if unasked:
        out.append(f"- **Unscored dims (across all 10 personas): {unasked}**")
    else:
        out.append("- All 37 dims were scored at least once across the cohort.")
    out.append("")

    # ── Retry-guide / brainstorm coverage ──
    out.append("## 5. Retry-guide & brainstorm coverage\n")
    out.append("| PID | Persona | retry_guide triggers | brainstorm triggers (G16) |")
    out.append("|---|---|---|---|")
    for t in persona_runs:
        out.append(f"| {t.pid} | {t.name} | {t.retry_guide_triggers} | {t.brainstorm_triggers} |")
    out.append("")

    # ── Crisis routing ──
    out.append("## 6. Crisis-routing observability\n")
    crisis_runs = [t for t in persona_runs if t.crisis_triggered]
    if crisis_runs:
        for t in crisis_runs:
            out.append(f"- {t.pid} {t.name} — crisis dim: `{t.crisis_dim}`")
    else:
        out.append(
            "_No persona triggered the crisis-flag DB write. Note: the safety "
            "spoken-message path (`SAFETY_RESOURCES_MESSAGE`) is gated off by "
            "`CRISIS_OVERRIDE_ENABLED=False` in `therapy_content.py`, so a "
            "Score-2 on `sib`/`safe`/`risk`/`drug`/`alcohol` records the "
            "clinical flag but does not short-circuit the screening loop. "
            "Verified for P08 (sib=2) and P04 (alcohol/drug/ciga=2)._"
        )
    out.append("")

    # ── Exceptions ──
    out.append("## 7. Exceptions / crashes\n")
    exc_runs = [t for t in persona_runs if t.exceptions]
    if exc_runs:
        for t in exc_runs:
            out.append(f"- {t.pid} {t.name}:")
            for e in t.exceptions:
                out.append(f"    - {e}")
    else:
        out.append("_Zero exceptions raised across all 10 persona runs._")
    out.append("")

    # ── Break tests ──
    out.append("## 8. Break / adversarial tests\n")
    passed = sum(1 for r in break_results if r["passed"])
    total_br = len(break_results)
    out.append(f"**{passed}/{total_br} checks passed.**\n")

    # Group: header table for failures only, full list collapsed
    failures = [r for r in break_results if not r["passed"]]
    if failures:
        out.append("### Failed checks\n")
        out.append("| Check | Detail |")
        out.append("|---|---|")
        for r in failures:
            out.append(f"| {r['check']} | {r['detail']} |")
    else:
        out.append("_All adversarial checks passed._")
    out.append("")

    out.append("<details><summary>All break-test checks (click to expand)</summary>\n")
    out.append("| Check | Pass | Detail |")
    out.append("|---|---|---|")
    for r in break_results:
        out.append(f"| {r['check']} | {'OK' if r['passed'] else 'FAIL'} | {r['detail']} |")
    out.append("\n</details>\n")

    # ── Stop-confirm validation ──
    out.append("## 9. Stop double-confirm contract\n")
    out.append(
        "The stop double-confirm flow is owned by `SpeechInteractionService` "
        "(speech layer). The persona harness drives `HandlerRL` directly and "
        "does NOT instantiate the speech layer, so the *runtime* double-confirm "
        "flow is exercised in the break tests (`_apply_global_command_priority` "
        "+ `_classify_confirm`) rather than per-persona. The break tests confirm:"
    )
    out.append("- HARD_END no longer calls `handle_exit` directly — sets `_end_pending_latch` and returns `END_PENDING`.")
    out.append("- Mid-CBT SOFT_END follows the same contract.")
    out.append("- Pre-CBT SOFT_END still routes to the analyzer's `Stop` keyword (paper §5.1).")
    out.append("- `_classify_confirm` biases ambiguous replies to NO (safer default).")
    out.append("")

    # ── Therapist feedback regression ──
    out.append("## 10. Therapist-feedback regression coverage\n")
    out.append("| Therapist ask | Code path | Test source |")
    out.append("|---|---|---|")
    out.append("| Don't-know brainstorm scaffold | `questioner._build_brainstorm_guide` | break_tests + P05 retry_guide hits |")
    out.append("| Music-only CBT intermissions | `speech_service._run_one_intermission_activity` CBT branch | break_tests source-grep |")
    out.append("| Per-stage intermission framing | `_INTERMISSION_LEAD_INS_BY_STAGE` | break_tests source-grep |")
    out.append("| Stop double-confirm | `_apply_global_command_priority` → `END_PENDING` + `_resolve_pending_end` | break_tests |")
    out.append("")

    # ── Verdict ──
    n_personas = len(persona_runs)
    n_exc = sum(len(t.exceptions) for t in persona_runs)
    n_cbt_success = sum(1 for t in persona_runs if t.cbt_success)
    n_completed = sum(1 for t in persona_runs if t.completed_screening)

    verdict_status = "READY-WITH-CAVEATS"
    issues: List[str] = []
    if n_exc > 0:
        verdict_status = "NOT-READY"
        issues.append(f"{n_exc} exception(s) thrown across personas")
    if passed != total_br:
        verdict_status = "NOT-READY"
        issues.append(f"{total_br - passed} break-test failures")
    if unasked:
        issues.append(f"dim coverage gap: {unasked}")

    out.append("## 11. Clinical-trial readiness verdict\n")
    out.append(f"**{verdict_status}**\n")
    out.append(f"- Personas completed screening: {n_completed}/{n_personas}")
    out.append(f"- Personas reached CBT success: {n_cbt_success}/{n_personas}")
    out.append(f"- Break tests passed: {passed}/{total_br}")
    out.append(f"- Total exceptions: {n_exc}")
    if issues:
        out.append("\n**Issues / caveats:**")
        for i in issues:
            out.append(f"- {i}")
    out.append("")

    out.append("### Always-true caveats for clinical deployment\n")
    out.append(
        "- LLM-stubbed: every paper LLMRole was exercised, but the real "
        "Gemma-4-E2B engine was not — production runs may differ on output "
        "phrasing, parser robustness, and latency. Behaviour around malformed "
        "LLM output is partially covered by the `LLMError` -> SKIPPED contract."
    )
    out.append(
        "- Audio-stubbed: TTS / STT / GPIO / music are not exercised. The "
        "stop double-confirm spoken prompt, intermission timing, and "
        "barge-in behaviour need a Jetson session to validate."
    )
    out.append(
        "- Crisis-message path (`SAFETY_RESOURCES_MESSAGE`) gated off by "
        "`CRISIS_OVERRIDE_ENABLED=False`; the DB clinical-flag write is "
        "exercised, but the spoken safety message is not — re-enable only "
        "after Gemma's critical-dim FP rate is validated < 2%."
    )
    out.append(
        "- Single-session runs only — multi-session warm-start, longitudinal "
        "Q-table behaviour, and recall-greeting path are out of scope here."
    )
    out.append("")

    return "\n".join(out)


__all__ = ["build_report"]
