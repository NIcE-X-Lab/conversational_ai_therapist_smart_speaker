# Demo-parity refactor plan — v1.1 branch

## What the demo does (ground truth, from `mexa_llmtherapist_demo.mp4` + `mexa_llm_therapist_demo_screenshots.pdf`)

1. Greeting: "Hi, I'm Caiti, and I'm here to support you. Thanks for being here. Let's start with a few quick questions about your recent day-to-day."
2. First DLA question: "How are your eating habits? Are you eating regularly?"
3. Each Score-2 DLA dim triggers one RV follow-up ("Can you elaborate?") then one long MI Validator paragraph (3-5+ sentences, paper-style).
4. User says "I think we've spoken enough today" → DLA terminate.
5. CBT Stage 0 presents Score-2 candidates with numeric picker; user enters "1".
6. Stage 1 recap + prompt; user says "not sure"; Stage 1 Guide emits the long enumeration ("You think you always forget... you fear that if you take it regularly... you worry that even taking the prescribed amount...").
7. Stages 2/3: CHALLENGE + REFRAME with recaps between them.
8. Closing: "Great work today. We completed the CBT steps for this topic. Thank you for your effort."

No PHQ-4 at the start. No crisis override. No SOAP in the spoken channel. No returning-user warm-start visible (subject 8080 session from a fresh Q-table).

---

## Divergences to resolve (user's numbering)

### D1 — Gemma tuning so Gemma-4-E2B emits the demo's rich MI/CBT prose

**Files**: `src/utils/config_loader.py`, `src/models/llm_client.py`, `.env` guidance

- Remove the hard `LITERT_MAX_TOKENS = 512` cap for roles that need paragraph output. Introduce per-role max-token budgets:
  - `RV_VALIDATOR`: 512 tokens (3-5 sentence empathic validation + strategies).
  - `RV_GUIDE`, `CBT_GUIDE`: 512 tokens (long enumerations).
  - `CBT_REASONER`, `RV_REASONER`, `ANALYZER`: keep at ~96 tokens (they only emit `DECISION: 0/1` or short classifications).
  - `REPHRASER`, `REFLECTIVE_SUMMARIZER`: 160 tokens (1-2 sentences).
  - `GENERAL`: 400 tokens (greeting / closing / session analysis).
- If the installed `litert_lm.Engine` exposes a `max_output_tokens` / `config` knob, wire it through. If not, document the cap is upstream-bounded by engine defaults and remove the env-level cap so the engine uses its native max.
- Leave temperature at the engine default so Gemma's Validator stays coherent; the prose length is primarily controlled by max-tokens, not temperature.
- Soften `_parse_decision` back to the legacy substring semantics so a Gemma reply that emits `0` or `1` *without* the literal `DECISION:` header is accepted — the demo's legacy behaviour, and the reason it didn't retry.

### D2 — PHQ/GAD as intermission-only, not pre-screening

**Files**: `src/core/handler_rl.py`

The speech service already delivers PHQ-4 via `IntermissionLadderManager` as one of the latency-fillers (SCREENING / BREATHING / MUSIC) — that pool runs automatically when the LLM is still generating. So the `_run_phq4_screening()` block at the top of `HandlerRL.run()` (lines ~342-359) is a *duplicate* sequential pre-screen. Remove it:

- Delete the `phq4_result = self._run_phq4_screening()` call and the `if phq4_result.get("phq4_high_risk")` block that follows.
- Delete the `_run_phq4_screening()` method entirely (lines ~1134-1401).
- Delete the PHQ-4 "high risk" crisis-msg injection.
- Intermission ladder stays as-is: each of the 4 PHQ/GAD questions fires at most once per session, interleaved with breathing exercises and music, cut short as soon as the LLM's response is ready.

### D3 — Crisis override toggle (off by default)

**Files**: `src/core/therapy_content.py`, `src/core/handler_rl.py`, `src/core/CBT.py`

- Add `CRISIS_OVERRIDE_ENABLED = False` to `therapy_content.py` with a block comment documenting intent as a future safety-critical addition.
- Gate every `_crisis_scan()` call site and every `crisis_callback` invocation behind `if CRISIS_OVERRIDE_ENABLED:`.
- Keep `CRITICAL_DIMS` / `SAFETY_RESOURCES_MESSAGE` / `_deliver_safety_message` / `_crisis_scan` definitions intact but inert; add module-level docstring note explaining the future-addition posture.

### D4 — Warm-start = minor nudge, no first-turn bypass

**Files**: `src/core/handler_rl.py`

- Change `boost = 3.0` → `boost = 0.3` in `_load_longitudinal_state()`. That still shifts Q-value ordering for prior Score-2 dims without hard-overriding exploration.
- Delete the "first 2 turns force-target prior Score-2 dim" block (lines ~392-418). Q-learning now runs its full ε-greedy loop unchanged.
- Keep the recall greeting: it's a user-facing improvement, not a flow change.

### D5 — Flow must roughly mirror demo

Natural consequence of D1-D4 + D6. Verified by re-reading the greeting → DLA → RV → MI-Validator → CBT-0/1/2/3 → closing path once the edits are in.

### D6 — Rephraser off, Reasoner loose, Validator long

**Files**: `config.yaml`, `src/core/CBT.py`

- Set `rl.rephrase_at_runtime: false` in `config.yaml` so the therapist-authored question wording is spoken verbatim (matches the demo's exact "How are your eating habits? Are you eating regularly?" wording).
- Replace `_parse_decision` regex body with the legacy-parity form: `return "0" if (raw and "0" in raw) else "1"`. Documented why (Gemma's output sometimes drops the literal header; substring semantics was what the demo used and paper implemented).

### D7 — Log taxonomy, silent SOAP, spoken session analysis, new startup script

**Files**: `src/utils/log_util.py`, `src/utils/io_record.py`, `src/core/handler_rl.py`, new `scripts/start_therapist.sh`

- Lift the log format already in use (`_LOG_FORMAT = '%(asctime)s arth-desktop %(name)s[%(process)d] %(levelname)s %(message)s'`) — it already matches the legacy demo's pattern: `2026-02-06 10:37:41,... Yuangs-MacBook-Pro-10 LLM_therapist[14759] INFO RL thread started`. Keep this.
- Default console level stays `INFO`; move the high-noise breadcrumbs to `DEBUG`:
  - Heartbeat log inside `llm_complete` from INFO → DEBUG.
  - `[LLM_CLIENT] Requesting in-process LiteRT inference...` INFO → DEBUG.
  - `io_record.log_reasoning`'s "Logged Reasoning ({type}) to DB" INFO → DEBUG.
  - Every `log_reasoning` call site at INFO remains on DB, just doesn't flood console.
  - `Appended note to question_lib[...]['notes']` DEBUG was already DEBUG; leave alone.
  - `Received user input` / `Received user response` stays INFO (these are the demo-visible breadcrumbs).
  - `_RESOURCE_AUDIT.capture_point / capture_process_inventory / emit_resource_map` logs → DEBUG.
- Silence SOAP from the spoken stream: `_generate_clinical_summary` currently calls `log_question(summary)` at the end. Remove that call — SOAP still goes into DB + the session CSV/log file, but is never spoken.
- Move `_generate_session_analysis` output to the spoken closing: the 2-3 sentence SUMMARY section becomes the user-facing closing sentence(s). If CBT was used its existing "Great work today..." line stays, and the analysis summary plays *after* the SOAP silent write.
- New `scripts/start_therapist.sh` that:
  - Activates `.venv`.
  - Prints the legacy-style header (`CaiTI Smart-Speaker - Headless Start`).
  - Exports `DISABLE_INTERNAL_SPEECH=0`, `CONSOLE_LOG_LEVEL=INFO`, `LOG_FILE=data/logs/therapist_<ts>.log`.
  - Runs `python main.py`.

---

## Out of scope

- Flutter frontend (user explicitly disowned it).
- Swapping backend to OpenAI (user kept Gemma as the sole LLM).
- Any DB schema changes; intermission ladder, clinical_scores, intervention_logs etc. stay.

---

## Risks I'll flag at implementation

- Gemma-4-E2B may simply not be capable of the demo's GPT-5-class MI output even with max-tokens lifted. If that happens, post-implementation, I'll note where the ceiling sits so a larger model swap stays a one-line change (`ROLE_MODEL_MAP[RV_VALIDATOR] = "some-bigger-model"`).
- Reverting `_parse_decision` to substring semantics intentionally trades strictness for demo parity. Any free-text "0" in the Reasoner's justification will pass as `DECISION=0`. Documented in the code comment.
- The spoken session analysis can drift or drone; kept to 1-2 sentences max via prompt instruction.
