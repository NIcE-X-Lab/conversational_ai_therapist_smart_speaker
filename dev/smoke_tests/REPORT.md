# CaiTI smoke-test report

Run: `dev/smoke_tests/run_all.sh`

## Scope

Five test files exercise the full pipeline end-to-end without requiring
Gemma inference or audio hardware. Every test mocks LLM calls and the
Analyzer bridge, drives synthetic user inputs through `INPUT_QUEUE`,
and asserts specific contents and orderings on `OUTPUT_QUEUE` / DB / notes.

## Results

| Test file | OK | FAIL |
|---|---|---|
| `test_1_static.py` | 44 | 0 |
| `test_2_rl_math.py` | 16 | 0 |
| `test_3_questioner.py` | 18 | 0 |
| `test_4_cbt.py` | 21 | 0 |
| `test_5_e2e_replay.py` | 15 | 0 |
| **Total** | **114** | **0** |

## What is covered

### `test_1_static.py` — static invariants (44 OK)

Invariants that must hold BEFORE any session starts.

- Config flag values are all at legacy-parity defaults:
  - `EPSILON=1.0`, `ALPHA=0.5`, `GAMMA=0.9` (G1)
  - `ITEM_IMPORTANCE[2,3,11] = {98, 99, 97}` (G2 sentinel pins on mood/medication/eat)
  - `REASK_DIMENSION_N=False` (G5), `CBT_ESCALATION_ENABLED=False` (G7),
    `MULTI_DIM_BACKFILL_ENABLED=False` (G8), `REFLECTIVE_SUMMARIZER_ENABLED=False` (G9)
  - `CRISIS_OVERRIDE_ENABLED=False` (D3), `REPHRASE_AT_RUNTIME=False` (D5/D6)
  - `REWARD_MODE='mean'`, `LITERT_MAX_TOKENS>=4096`
- Legacy wordings are present in source (G3 greeting seed, G6 CBT Stage 0)
  and none of the non-legacy replacements ("concerns in", "Which area")
  leaked back in.
- Prompt structural requirements:
  - RV Validator mandates 4-7 sentences and carries the demo's medication
    Example 4.
  - CBT Stage-1 Guide mandates 3-5 clauses and carries the demo's
    "you think/you fear/you worry/you see/you assume" enumeration example.
  - `_parse_decision` uses legacy-loose substring semantics
    (`"0" in raw`), confirmed across three inputs.
- Gate wiring: every flag has a runtime `if FLAG:` guard in the expected file.
- Runtime expectations:
  - `_run_phq4_screening` is fully removed from `handler_rl` (D2).
  - No first-2-turn force-target block (D4).
  - Crisis scaffolding (`CRITICAL_DIMS`, `SAFETY_RESOURCES_MESSAGE`) kept
    but inert.
- Question library: 37 dimensions, demo-critical dims present with
  exact wording ("How's your eating? Are you eating regularly?").

### `test_2_rl_math.py` — RL math and Q-table (16 OK)

- `initialize_q_table` applies `ITEM_IMPORTANCE` correctly: `Q[*, col=k]
  = ITEM_IMPORTANCE[k]` so the 97/98/99 pins land at columns 11, 2, 3.
- `choose_action` at `ε=1.0` from a fresh Q-table deterministically picks
  `medication → mood → eat` as the first three action argmaxes after
  incremental masking. This is the legacy demo's pure-exploit shape.
- `get_env_feedback` returns `(int(A), reward)` normally, `("terminal", 0)`
  on `terminate_flag=1`, and `("terminal", 10)` when all items masked —
  matching legacy prototype behaviour.
- Q-learning update formula matches `new = old + α·(R + γ·max(Q[S_]) - old)`
  numerically; terminal branch correctly skips `γ·max(…)` bootstrapping.
- Bonus: replaying the legacy's own trained Q-table (`legacy-prototype/
  data/q_tables/item_qtable_8080.csv`) confirms that a trained Q-table
  DOES shift first picks — that's how the demo video's `eat→med→mood`
  ordering came from an already-trained session. Both the fresh and
  trained ordering are clinically equivalent (top-3 priority).

### `test_3_questioner.py` — questioner + G4 regression (18 OK)

The hot loop. Mocks `llm_complete`, `get_openai_resp`, and `INPUT_QUEUE`.

- **Yes path** (eat + Yes → Score 0): `ask_question` returns `(0.0, 0, "")`,
  score `[0]` is appended, NO RV Reasoner or Validator calls fire.
- **No path** (eat + No → Score 2): Score `[2]` appended, RV Reasoner
  called exactly once, RV Validator called **EXACTLY ONCE** (**G4 regression
  caught**: pre-fix was 2 calls), follow-up collected, notes contain two
  rows (original_question/original_resp + full RV triplet). Also confirms
  `reflective_summarizer` does NOT fire with G9 off.
- **Stop path**: DLA_terminate=1, score list left empty (correct — a stop
  request is not a clinical score).

### `test_4_cbt.py` — CBT Stage 0/1/2/3 (21 OK)

Scripts a one-dim (medication) CBT session.

- **Stage 0 wording is LEGACY verbatim**: `"you have issue in:"`,
  `"Which dimension would you like to work on today?"`, `"Tell me the
  dimension number. For example: 1"` — and none of the non-legacy phrases
  ("concerns in", "Which area") appear.
- Stage 1 recap correctly speaks `"Let us work on dimension 'Taking
  Medication as Prescribed'. From our record, you mentioned that: …"`
  and pulls the statement from `followup_resp_1`.
- All three stages' canonical questions are spoken.
- CBT closing: `"Great work today. We completed the CBT steps for this
  topic. Thank you for your effort."`
- LLM call accounting: exactly 3 CBT Reasoner calls (no retries) and 1
  CBT_GUIDE call (the Stage-3 recap) — matches demo flow exactly.
- **`CBT_ESCALATION_MESSAGE` never reaches the spoken queue** (G7 off).
- Final note tagged `CBT_stage: success`, includes dimension, unhelpful,
  challenge, reframe fields.

### `test_5_e2e_replay.py` — end-to-end handler replay (15 OK)

Boots `HandlerRL.run()` with a minimal DB stub, scripted user inputs
matching the demo shape, and a role-aware LLM mock.

- Greeting combined with first question; warmer rewrite present (G3).
- All 3 pinned dims (medication / mood / eat) fired in order.
- Medication Score-2 triggered RV follow-up + Validator in the spoken
  stream.
- CBT Stage 0 listed medication with legacy wording.
- All CBT stages + closing spoken.
- **Warm session-analysis summary** spoken as the session-end message
  (G10) — ends with "Take care, and I'll be here whenever you want to
  check in again."
- **No safety resources (D3)**, **no CBT escalation (G7)** reached the
  spoken stream.

## What is NOT covered

These are deferred to live-hardware validation on Jetson:

1. **Gemma-4-E2B output quality** — whether the Validator emits 4-7
   sentences and the CBT Stage-1 Guide emits 3-5 clauses in production.
   The prompts demand it; the test harness mocks the LLM responses.
2. **STT / TTS latency under load** — Whisper + Piper end-to-end timing
   on Jetson CPU.
3. **PHQ-4 / GAD-2 intermission ladder** in the SpeechInteractionService
   (the ladder itself has no unit coverage here; the handler no longer
   calls it as a pre-screen).
4. **Longitudinal Q-table persistence** across sessions (the module
   docstring documents the CSV + DB dual-write contract).
5. **`litert_lm.Engine` crash recovery** — `_invalidate_engine` /
   `LLMError` handling under real engine failures.
6. **Audio driver initialisation** on the target Jetson hardware.

## Clinical-trial readiness verdict

**Software-level: READY.** The legacy demo flow, paper-aligned module
architecture, and clinically-tested prompts are all present and gated
correctly; every feature that diverged from legacy is now either
restored (G1/G2/G3/G6) or behind an off-by-default flag (G5/G7/G8/G9).

**Hardware-level: PENDING VALIDATION.** The gates above are necessary
to hit before enrolling a participant on the Jetson:

- Deploy on the target Jetson.
- Run `scripts/start_therapist.sh` once to verify startup, Piper voice,
  and LiteRT model all load; inspect the log banner.
- Execute a full manual session end-to-end (speak through the screening
  and CBT, confirm demo-style prose).
- Tail `data/logs/therapist_*.log` and confirm the legacy log taxonomy
  is present (`INFO Prompted question: …`, `INFO Received user input: …`).
- Verify SOAP report lands silently in DB / `data/results/` CSV without
  ever reaching the speaker.
- Confirm the three intermission activities (PHQ/GAD, breathing, music)
  rotate correctly when the LLM is generating.

Once those pass on hardware, this build is suitable for a clinical
trial enrolment window.

## Smoke-test runtime

~35 seconds total on a dev laptop; safe to run pre-commit.

```bash
dev/smoke_tests/run_all.sh
```

Non-zero exit on any failure.
