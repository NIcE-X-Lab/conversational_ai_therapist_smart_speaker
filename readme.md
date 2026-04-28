# CaiTI — Conversational AI Therapist Interface

A fully local, paper-aligned reproduction of the **CaiTI** system (Nie et al., 2025, ACM TCH, [DOI 10.1145/3712299](https://doi.org/10.1145/3712299)) running entirely on-device on an NVIDIA Jetson Orin Nano. CaiTI delivers **daily-functioning screening** and **psychotherapeutic interventions** (Motivational Interviewing + Cognitive Behavioral Therapy) through natural spoken conversation — no cloud, no external LLM API, no PHI leaving the device.

The pipeline implements the paper's architecture end-to-end:
Q-learning questioner over 37 daily-functioning dimensions → Response Analyzer → Reflection-Validation (Reasoner + Guide + Validator) → 3-stage CBT (Recognize → Challenge → Reframe). Clinical state lives in the question library and SQLite; LLM calls are **self-contained per task** (no rolling context injection), matching the legacy prototype's design.

---

## Table of Contents

1. [Paper Alignment at a Glance](#paper-alignment-at-a-glance)
2. [Clinical-Trial Hardening](#clinical-trial-hardening)
3. [Latency & Performance](#latency--performance)
4. [System Architecture](#system-architecture)
5. [Repository Layout](#repository-layout)
6. [Module Reference](#module-reference)
7. [Session Lifecycle](#session-lifecycle)
8. [LLM Design (Self-Contained Per Task)](#llm-design-self-contained-per-task)
9. [Persistence Model](#persistence-model)
10. [Configuration](#configuration)
11. [Feature Gates & Future Development](#feature-gates--future-development)
12. [FastAPI Endpoints](#fastapi-endpoints)
13. [Running the System](#running-the-system)
14. [Jetson Deployment](#jetson-deployment)
15. [Testing](#testing)
16. [Clinical-Trial Operations Playbook](#clinical-trial-operations-playbook)
17. [Key Technologies](#key-technologies)

---

## Paper Alignment at a Glance

| Paper component | Paper section | Implementation |
|---|---|---|
| 37 daily-functioning dimensions | §3.1, Table 1 | `data/libs/question_lib_v4.json` (indices 1–37) |
| Q-learning questioner | §5.1, p.11 | `src/utils/rl_qtables.py`, `src/core/handler_rl.py` — runs on legacy-prototype values (ε=1, α=0.5, γ=0.9) which is what the published demo recording uses; paper-strict values (ε=0.9, α=0.1) are kept as a commented alternate in `config.yaml`. See [Feature Gates](#feature-gates--future-development). |
| 7–11 therapist-authored question variants per dim + runtime Rephraser | §5.1 | `question_lib_v4.json` (`question[]` + `question_synthetic[]`) + `src/utils/text_generators.py::generate_synonymous_sentences` |
| Response Analyzer (Dim, Score ∈ {0, 1, 2}) | §5.2 | `src/core/response_analyzer.py::classify_dimension_and_score` + `src/services/response_bridge.py` |
| Reflective Summarizer (1st→3rd person) | §5.2 | `src/core/response_analyzer.py::reflective_summarizer`, `src/utils/text_generators.py::generate_change` |
| Reflection-Validation: Reasoner → Guide / Validator | §5.3, Fig. 9 | `src/core/reflection_validation.py::{rv_reasoner, rv_guide, rv_validator_mi}` |
| 3-stage CBT (Recognize → Challenge → Reframe) with Reasoner+Guide retries | §5.4 | `src/core/CBT.py` |
| PHQ-4 / GAD-2 standardised clinical screening | §4.2 | `src/core/therapy_content.py` + `handler_rl._run_phq4_screening` |
| Per-subject Q-table persistence | §5.1 | `data/q_tables/item_qtable_<id>.csv` (mirrors legacy) + SQLite `persistent_rl_state` (resume-rich) |
| Recall-and-Resume greeting for returning users | §5.1 end | `handler_rl._load_longitudinal_state` + greeting block |
| Task-specific LLM decomposition (Analyzer, Rephraser, RV Reasoner/Guide/Validator, CBT Reasoner/Guide) | §5, Fig. 10 | `src/models/llm_client.py::LLMRole` + `ROLE_MODEL_MAP` (all roles currently map to local Gemma; enum is in place for per-role model swap) |

**Divergences (documented, intentional):**

- **LLM backend.** Paper uses GPT-4 / fine-tuned GPT-3.5 per role via OpenAI API. We run a single local Gemma 4 E2B via **LiteRT-LM** for on-device privacy. Role plumbing stays paper-faithful (`LLMRole` enum at every call site) so multi-model deployments can map different roles to different model backends later.
- **Crisis override.** Adds a safety-net short-circuit: if any `CRITICAL_DIMS` entry (`sib`, `safe`, `risk`, `drug`, `alcohol`) scores 2, the RL loop is short-circuited to a `SAFETY_RESOURCES_MESSAGE` (988 Lifeline + SAMHSA) before CBT. The paper describes similar safety handling narratively; we implement it as an explicit check in `handler_rl._crisis_scan`.
- **Multi-dim back-fill.** `questioner._apply_segment_level_backfill` + `_apply_multi_dim_updates` credit secondary dimensions mentioned in a single substantive utterance (paper §4.1 "minimal questioning"). The per-segment back-fill is free (reuses existing classification); the LLM back-fill is gated to ≥20 tokens + ≥2 segments.
- **Reward aggregation.** `config.yaml::rl.reward_mode` defaults to `"mean"` (paper-strict + legacy-faithful np.mean). `"hybrid" = (max + mean) / 2` is available as a research alternate for dims where a single severe segment should bias revisit priority (see `config.yaml` for the full rationale). Switch via `reward_mode: hybrid`.
- **Smart-speaker UX layer.** Paper targets multiple form factors; this repo adds a Jetson-specific hardware layer — GPIO buttons, LED, ambient background music bed with automatic ducking + smooth fades, and a **parallel STT + GIL-safe proactive intermission** pipeline (SCREENING / BREATHING / MUSIC, SCREENING-first while PHQ-4 questions remain, then last-activity-deprioritised cycling). The moment the user stops speaking, a daemon STT worker starts transcribing the saved WAV while the main thread drops straight into the intermission — the user hears "While I'm processing that…" / breathing / music within ~100 ms of the mic closing. When Whisper finishes (~2 s later) the worker pushes the transcript onto `input_queue`, so the handler's LLM call runs in parallel with the rest of the intermission rather than serialised behind it. Music sits at a foreground listen level (0.85) whenever Piper TTS is not speaking and the mic is not listening, and auto-ducks to a 2 % whisper in both of those cases so the voice path stays clean. An **output-ready watcher** peeks the handler's `output_queue` during the intermission and cuts BREATHING/MUSIC TTS short (via `stop_playback_event`) the instant the next LLM response is ready — with a 6 s minimum-engagement floor so meditation never gets chopped after a single breath. SCREENING is never interrupted (clinical data). Short user utterances (< 3 s) fall back to a serial path that preserves the fragment-merge retry. Random-segment playback keeps a long ambient track fresh; breathing scripts are session-scoped so all 5 meditations play before any repeat. Median turn latency on Jetson GPU backend: **~16 s** (vs. ~82 s in the initial CPU-backend build). These are UX additions; they do not change clinical behaviour.

---

## Clinical-Trial Hardening

This build adds 22 layered safeguards on top of the paper's pipeline, each one addressing a specific failure mode that would otherwise compromise a clinical trial. Every fix was audited against the paper's direction; five actively strengthen paper intent (C5, C7, M1, M6, M7).

### Critical safeguards (must-have for any participant)

| ID | Safeguard | Behaviour |
|---|---|---|
| **C1** | Idempotent `init_record` | A `_INIT_DONE` sentinel guarded by `_INIT_LOCK` ensures main-side + handler-side calls do not create duplicate DB session rows or reset `CURRENT_TURN_INDEX` mid-flight. Explicit reset via `reset_session()` or `init_record(force=True)`. |
| **C2** | Guaranteed safety-message delivery | `HandlerRL._deliver_safety_message()` always attempts (1) TTS via `OUTPUT_QUEUE`, (2) a plain-text file at `data/safety/crisis_<subject>_<session>_<dim>_<timestamp>.txt`. Every attempt writes a row to `safety_deliveries` with `method`, `success`, and any `error_text`. Auditor query: `SELECT * FROM safety_deliveries` — every CRITICAL_DIM event should have ≥1 `success=1` row. |
| **C3** | DB connections never leak | `DBManager._connect()` is a single context manager with guaranteed rollback + close. Every method uses it. Exception in any write no longer leaks a handle or triggers the SQLite busy-lock cascade. Schema migrations land in `_migrate_schema()` so an existing trial DB gains new columns (e.g. `sessions.end_reason`) without losing data. |
| **C4** | Atomic Q-table CSV write | `item_qtable_<subject>.csv` is written to `<path>.tmp` and `os.replace`-d. A process kill mid-write can never produce a corrupt paper-compatible artefact. |
| **C5** | Empty/ambiguous STT is never scored | `score_response` returns `SCORE_UNRESOLVED` (-2) for empty or unparseable input, `SCORE_OPT_OUT` (-1) for explicit refusal. PHQ-4 loop re-prompts up to 2× with clearer anchor phrasing, then marks SKIPPED (`reason='stt_unresolved'`). Fixes the false-negative where legacy scored unparseable replies as 0 ("Not at all"). |
| **C6** | `LLMError` for hard engine failures | Engine crashes raise a typed exception instead of returning a placeholder string that silently poisoned downstream classification. Clinically-important callers can now treat failure as SKIPPED rather than silently accepting a "none" result. |
| **C7** | Crisis re-scan between CBT stages | `run_cbt(..., crisis_callback=_cbt_crisis_hook)`. The handler supplies a callback that re-scans after every CBT stage collects a user reply — if a NEW critical-dim Score 2 is detected mid-CBT (e.g. user mentions self-harm in their CHALLENGE step), CBT pauses and the safety path fires again via `_crisis_dims_handled` re-entrant tracking. |

### High-priority safeguards

| ID | Safeguard | Behaviour |
|---|---|---|
| **H1** | Thread-safe session state | Single `_INIT_LOCK` protects `SUBJECT_ID`, `DB`, `SESSION_ID` mutations. No more torn snapshots between speech thread and handler thread. |
| **H2** | LLM engine auto-recovery | `_invalidate_engine()` drops the singleton on any generate-path failure. `_ENGINE_MAX_FAILURES=3` per session; past that, `_init_engine` raises `LLMError` instead of spinning. `engine_is_healthy()` public check lets callers decide whether to short-circuit. |
| **H3** | Bounded `OUTPUT_QUEUE`, unbounded `INPUT_QUEUE` | `OUTPUT_QUEUE` maxsize=50 with drop-oldest via `_safe_output_put()` (logs `[QUEUE_OVERFLOW]` on drop). `INPUT_QUEUE` is intentionally unbounded — **dropping a user reply is a clinical-data-loss event and unacceptable**; overflow there signals a dead handler, which surfaces as a log warning rather than silently losing PHI. |
| **H4** | Atomic `turn_index` | `_next_turn_index()` increments under `_TURN_INDEX_LOCK`. No collisions → no duplicate `turn_index` in the DB → session reconstruction is always deterministic. |
| **H5** | `CLINICAL_MODE=1` fail-fast boot | If any of LiteRT model, Piper model, Faster-Whisper, DB, or disk-space check fails, `main()` exits with code 2. Supervising scripts (systemd, launchers) detect and alert the clinician. GPIO failures are non-fatal (stub is fine off-Jetson). |
| **H6** | Pre-flight disk-space check | `_check_disk_space()` against `MIN_FREE_DISK_MB` (default 500 MB, recommended 1000 MB for trials). Aborts boot in `CLINICAL_MODE`. |
| **H7** | Crisis regression tests | `tests/test_clinical_safety.py` — 9 tests covering crisis-scan triggering, re-entrance, safety message delivery across TTS + file fallback + `safety_deliveries` audit, C5 sentinels, M2 crash recovery, M3 clinical_flags round-trip, H3 bounded queue, H4 atomic turn index. |

### Medium-priority safeguards

| ID | Safeguard | Behaviour |
|---|---|---|
| **M1** | PII redaction in logs | `REDACT_PII=1` replaces user transcript content in logger output (`io_record._redact`) with `[REDACTED_PII len=N]`. DB and structured JSON logs stay untouched so clinicians retain the audit trail. |
| **M2** | Crash recovery on boot | `DBManager.close_open_sessions_for_user()` is called during `init_record()`. Any session row left with `end_time IS NULL` (crashed session) is stamped closed with `end_reason='crash_recovery'` and surfaces a `clinical_flags.CRASH_RECOVERY` row. A fresh session begins with a new DB id. `atexit` hook handles clean SIGTERM/Ctrl-C closure. |
| **M3** | Consolidated `clinical_flags` table | Single authoritative audit trail. Every clinical event (`GAD2_POSITIVE`, `PHQ4_HIGH_RISK`, `PHQ4_OPT_OUT_ELEVATED`, `CRITICAL_DIM_SCORE_2`, `CRASH_RECOVERY`, `SESSION_CAP_REACHED`) writes a row via `DBManager.log_clinical_flag(session_id, flag_type, details)` with JSON `details_json` for extensibility. |
| **M4** | Dry-run harness | `scripts/dry_run.py` exercises the full pipeline end-to-end against a scratch SQLite with deterministic LLM stubs. Verifies session opens + closes cleanly, `safety_deliveries` row written, `clinical_flags` row written, PHQ-4 persistence, turn count, Q-table CSV written, crisis checkpoint file present. Exit code 0 on pass, 1 on any regression. Run before every trial day. |
| **M5** | Cached TTS fallback WAV | `assets/audio/tts_fallback.wav` (3 s, 880 Hz alert beeps) is copied to the output path when BOTH Piper and espeak-ng fail. The device is never silent during a trial. Generic "check the device" signal for clinicians. |
| **M6** | Max session length cap | `SESSION_MAX_SECONDS` (default 3600). `_session_timed_out()` checks at the top of every RL-loop iteration; timeout triggers graceful termination + `clinical_flags.SESSION_CAP_REACHED`. Matches paper's brief-session model. |
| **M7** | Safety scan on PHQ-4 opt-out | If the user opts out of PHQ-4 partway and their partial sub-scores already crossed `GAD2_THRESHOLD` or `PHQ4_THRESHOLD`, the safety path fires: `clinical_flags.PHQ4_OPT_OUT_ELEVATED` + `_deliver_safety_message("phq4_partial_<last_qid>")`. Paper's safety logic does not hinge on completing all 4 questions. |

### Per-session subject identity (two-layer model)

Every session is a distinct file entry, even for repeat visits from the same subject or near-collision names like "Alice" / "Alex". This is achieved with a two-layer identity installed at onboarding (`io_record.init_record`):

| Layer | Example | Used for |
|---|---|---|
| **`SUBJECT_BASE_ID`** (stable across a subject's visits) | `alice` | `users.subject_id` in SQLite, Q-table filename (`item_qtable_alice.csv`) — so longitudinal warm-start still finds prior Q-values, user context, preferences on the next visit. |
| **`SUBJECT_ID`** (composed per session as `<base>_<YYYYMMDD_HHMMSS>`) | `alice_20260425_190621` | Every per-session artefact filename — Report / Notes CSV, session dossier JSON, per-session log directory, crisis file-fallback. Also what `/api/status` reports as the live subject. |

**Artefacts produced per session** (for onboarded name `alice` starting at `20260425_190621`):

```
data/results/Report_alice_20260425_190621.csv       # CSV Report (per session)
data/results/Notes_alice_20260425_190621.csv        # CSV Notes  (per session)
data/sessions/session_alice_20260425_190621.json    # Session dossier JSON
data/logs/alice_20260425_190621/                    # Per-session log dir
        alice_20260425_190621.log                 # CSV transcript
        alice_20260425_190621.json                # NDJSON event stream
data/clinical/clinical_report_alice_<session_id>.csv  # Therapist-facing export
data/safety/crisis_alice_20260425_190621_<dim>_<ts>.txt  # (if crisis fires)
```

**Subject-stable artefacts** (intentionally NOT per-session — longitudinal continuity):

```
data/q_tables/item_qtable_alice.csv                 # Q-table accumulates across Alice's sessions
users row (alice)                                    # One DB users row → many sessions rows
```

Three sequential sessions from Alice therefore produce three distinct Report / Notes / dossier files, but a single `item_qtable_alice.csv` that warm-starts each new visit.

### Participant experience end-to-end

1. **Boot (`CLINICAL_MODE=1`):** Jetson aborts cleanly if LiteRT / Piper / Whisper / SQLite / disk-space fails; no silent half-working device.
2. **Session start:** crashed sessions swept into `end_reason='crash_recovery'`. Fresh session gets unique ID + atomic turn index.
3. **PHQ-4:** 3 attempts per question; empty STT never scores as 0. Opt-out with elevated scores auto-delivers safety resources.
4. **RL loop:** atomic Q-table CSV persistence. Session exceeds 60 min → graceful termination with flag.
5. **Crisis detection** (any time — main loop or mid-CBT): TTS 988/SAMHSA + file fallback + `safety_deliveries` audit + `clinical_flags.CRITICAL_DIM_SCORE_2`.
6. **LLM failure:** `LLMError` raised; bounded to 3 retries per session; callers can mark the turn SKIPPED.
7. **TTS hard failure:** Piper → espeak → cached alert WAV. Never silent.
8. **Session end:** DB row closed with `end_reason='normal'`. Dossier JSON atomically flushed. `atexit` handles SIGTERM/Ctrl-C cleanly.

---

## Latency & Performance

The initial build (first "Smith" session, 2026-04-27) exhibited 60–120 s median per-turn latency on Jetson Orin Nano. After four rounds of targeted changes the median turn landed at **~16 s (5× faster)** with no change to clinical behaviour.

### Current per-turn budget (Jetson, GPU backend, all tuning applied)

| Stage | Time | Notes |
|---|---|---|
| Mic recording (record-until-silence) | user-driven | not on our critical path |
| Save WAV to disk | ~0.1 s | pygame mixer write |
| **STT decode (parallel)** | 1.5–2.5 s | runs in a daemon worker; main thread is already in the intermission |
| **ANALYZER LLM call** | 2.0 s | single-segment warm call, ~1,263-token system prompt |
| Multi-dim ANALYZER (if ≥ 20 tok + ≥ 2 seg) | 2.4 s | credits secondary dimensions, runs in parallel with intermission |
| Intermission TTS playback | 5–15 s | BREATHING/MUSIC are interruptible; SCREENING runs to completion |
| Music fade + bridge phrase | 5–8 s | `time.sleep(0.9)` + Piper spawn + 3-word bridge |
| Final question TTS gen + playback | 3–5 s | Piper ~1.8 s spawn + ~25 ms/word |
| **End-to-end turn median** | **~16 s** | measured across 7 turns of the Max session |

### Per-role LLM latency (real production prompts, Jetson GPU warm)

| Role | Prompt tokens | Time |
|---|---|---|
| Short GENERAL | ~50 | 0.9 s |
| REPHRASER | ~350 | 1.6 s |
| ANALYZER | ~1,263 | 2.0 s |
| MULTI-DIM ANALYZER | ~395 | 2.4 s |
| RV_REASONER | ~588 | 2.1 s |
| RV_GUIDE | ~1,140 | 3.9 s |
| RV_VALIDATOR (4–7 sentence output) | ~1,485 | 9.4 s |
| Cold start (first call of the session) | — | ~4.6 s |

Re-measure anytime with `./.venv/bin/python scripts/_audit_latency.py`.

### How we got the 5× reduction

1. **GPU backend for LiteRT-LM** (`LITERT_BACKEND=gpu` in `.env`). Gemma-4-E2B runs 4–8× faster per call on the Orin Nano's Ampere GPU than on its 6-core ARM CPU (XNNPack path). Whisper stays on CPU — they share the same 8 GB unified-memory pool and moving both to GPU risks OOM. This single change collapsed ANALYZER from ~18 s → 2 s and RV_VALIDATOR from ~29 s → 9 s.
2. **Parallel STT + intermission** (`_run_parallel_turn` in `speech_service.py`). Record → save WAV → spawn STT worker → immediately run intermission on the main thread. The worker pushes the transcript onto `input_queue` as soon as decode completes, so the handler's LLM call runs in parallel with the rest of the intermission instead of serialising behind it. Closed the ~4 s gap between mic-close and the user hearing "While I'm processing that…".
3. **Early-exit watcher on BREATHING / MUSIC** (`_start_output_ready_watcher`). Meditation and music blocks previously ran their full 30 s hold even after the LLM response was ready. The watcher peeks `output_queue` every 100 ms and trips `stop_playback_event` + `llm_done` as soon as the handler enqueues its response (respecting a 6 s minimum-engagement floor so meditation never gets chopped after a single breath). SCREENING remains uninterruptible — PHQ/GAD answers are clinical data.
4. **Rephraser-probability tuning** (`rl.rephrase_probability: 0.20`). Legacy fires the Rephraser on ~95 % of turns; at 0.20 most turns speak the therapist-authored variant verbatim. Saves ~1.3 s/turn averaged across a session. Flip back to 0.95 for legacy wording frequency.
5. **LITERT_MAX_TOKENS tightened from 4096 → 3072** (`.env`). Engine-level ceiling on prompt + output; 3072 covers RV_VALIDATOR worst-case (~2.2 k tokens total) while cutting KV-cache footprint on the unified-memory pool.

### Validated in the Max session (7 turns, 2026-04-27 22:51–22:56)

| Turn | Duration (mic-close → next question audible) | Intermission | Notes |
|---|---|---|---|
| 1 (medication) | ~23 s | SCREENING | first turn, Rephraser for opening |
| 2 (mood) | ~28 s | SCREENING (worry control) | 15 s PHQ TTS playback dominates |
| 3 (eat) | ~18 s | SCREENING | shorter user input |
| 4 (work) | ~14 s | BREATHING (cut at 7.4 s) | early-exit watcher fired |
| 5 (retry answer) | ~13 s | MUSIC (cut at 6.0 s) | early-exit watcher fired |
| 6 (showup → care) | ~16 s | SCREENING | Rephraser in-flight + Analyzer |
| 7 (care) | ~10 s | BREATHING (full run) | short user input, breathing not cut |

**Median ~16 s.** A PDF walkthrough (per-module + per-turn timeline, before/after comparison, critical-path breakdown) is produced by `scripts/_build_latency_report_pdf.py`.

### Remaining levers (not yet applied)

| Lever | Saves per turn | Effort |
|---|---|---|
| Shrink ANALYZER prompt (drop 14 examples + confusing-dims paragraph) | 1–2 s | 1 h |
| Shrink RV_VALIDATOR (keep only Example 4, require 2–3 sentences instead of 4–7) | 3–6 s on RV turns | 1 h |
| Shorten bridge phrases to 3–4 words | 2–3 s | 10 min |
| Drop the 0.9 s handoff `time.sleep` to 0.3 s | 0.6 s | 1 min |
| Token-streaming LLM → Piper | ~2 s | 1 day |
| Swap Whisper `base.en` for `tiny.en` | ~0.5 s | 5 min (quality trade) |

Applying the top three would land the median turn at ~8–10 s.

---

## System Architecture

```
                    +-----------------------------+
                    |  GPIO buttons + LED         |
                    |  (gpio_manager.py)          |
                    |  Pin 11 Start / 13 End /    |
                    |  15 Opt-out / 18 LED        |
                    +-------+---------------------+
                            |
     Mic --> AudioRecorder --> STTGenerator --+
            (PyAudio + VAD)   (Faster-Whisper) |
                                               |
                  +----------------------------v----------------------------+
                  |            SpeechInteractionService                      |
                  |            (speech_service.py)                           |
                  |                                                          |
                  |  - Wake-word detection, onboarding, turn taking          |
                  |  - Intermission ladder during LLM wait                   |
                  |  - VRAM handoff: suspend/resume Whisper around LLM call  |
                  +----+----------------------+------------------------------+
                       |                      |
                INPUT_QUEUE               OUTPUT_QUEUE --> TTSGenerator (Piper)
                       |                      ^             |
                       v                      |             v
                  +----+----------------------+---+     AudioPlayer
                  |           main.py              |     (pygame.mixer)
                  |  FastAPI :8000                  |         |
                  |  - /api/* remote control        |   BackgroundMusicThread
                  |  - Runs HandlerRL each session  |   (ambient bed w/ ducking)
                  +----+--------------------------+-+
                       |
                       v
                  +----+----------------------+
                  |       HandlerRL            |
                  |  - Q-learning orchestrator |
                  |  - PHQ-4 / GAD-2 screening |
                  |  - Crisis override         |
                  |  - Longitudinal Q persist  |
                  +----+---------+-------------+
                       |         |
           +-----------+         +---------------+
           v                                     v
   +-------+--------+                   +--------+-------+
   |   Questioner   |                   |       CBT       |
   |  ask_question  |                   |  3-stage loop   |
   +-------+--------+                   +--------+-------+
           |                                     |
           v                                     |
   +-------+---------+                           |
   | ResponseAnalyzer |-----+                    |
   | classify_dim+score|    |                    |
   +-------+---------+      |                    |
           |                v                    |
           |   +------------+------------+       |
           |   | ReflectionValidation    |       |
           |   | rv_reasoner / rv_guide /|       |
           |   | rv_validator_mi         |       |
           |   +------------+------------+       |
           |                |                    |
           +----------------+--------------------+
                            |
                            v
                    +-------+------+
                    |  LLMClient    |---> LiteRT-LM (Gemma 4 E2B, in-process)
                    |  llm_complete |     CPU (XNNPack) / GPU (ML Drift)
                    +-------+-------+
                            |
                    +-------+-------+
                    |  DBManager    |---> SQLite (data/therapist.db)
                    +---------------+
```

Every LLM call passes through `llm_complete(system, user, role=LLMRole.XYZ)` with a **self-contained** payload — the system prompt carries instructions + few-shot examples, the user prompt carries exactly the fields the task needs. No rolling conversation history is auto-injected. This mirrors the legacy prototype's design: continuity across turns lives in `question_lib[i][j]["score"]` / `["notes"]` and in explicit function arguments, not in prompt history.

---

## Repository Layout

```
.
├── main.py                    # FastAPI entrypoint + speech loop bootstrap
├── config.yaml                # RL / audio / STT / TTS parameters
├── requirements.txt           # Pinned pip deps
├── .env                       # Runtime config (LiteRT path, GPIO, etc.)
├── readme.md                  # This document
│
├── src/
│   ├── core/                  # Clinical domain logic
│   │   ├── handler_rl.py          # Q-learning orchestrator + screening + crisis override
│   │   ├── questioner.py          # Question selection, delivery, segment back-fill
│   │   ├── response_analyzer.py   # Dim/Score classifier + Reflective Summarizer + Rephraser
│   │   ├── reflection_validation.py # RV Reasoner + Guide + Validator (paper p.13 Fig.9)
│   │   ├── CBT.py                 # 3-stage CBT with per-stage Reasoner + Guide
│   │   ├── therapy_content.py     # PHQ-4 / GAD-2 content + MEDITATIONS + CRITICAL_DIMS + SAFETY msg
│   │   ├── intermission_manager.py # Ladder state machine (screening -> breathing -> music)
│   │   └── context_manager.py     # Session-end closing-reflection helper
│   │
│   ├── models/
│   │   ├── llm_client.py          # LiteRT-LM in-process Gemma engine + LLMRole enum
│   │   ├── stt.py                 # Faster-Whisper wrapper (with suspend/resume for VRAM handoff)
│   │   └── tts.py                 # Piper subprocess wrapper with espeak-ng fallback
│   │
│   ├── services/
│   │   ├── speech_service.py      # Real-time mic/GPIO/turn-taking loop + intermission driver
│   │   └── response_bridge.py     # Dim/Score parser (handles LLM output variants) + Yes/No shortcuts
│   │
│   ├── drivers/
│   │   ├── audio.py               # PyAudio + VAD recorder + BackgroundMusicThread (ducking)
│   │   ├── player.py              # pygame.mixer WAV playback for TTS
│   │   ├── gpio_manager.py        # Jetson.GPIO singleton (graceful no-op stub off-Jetson)
│   │   └── db_manager.py          # SQLite schema + CRUD + longitudinal RL state
│   │
│   └── utils/
│       ├── config_loader.py       # Aggregates config.yaml + .env
│       ├── io_record.py           # IPC queues, session events, CSV/JSON logging, SessionDossier
│       ├── io_question_lib.py     # Question library JSON I/O + results CSV emit
│       ├── rl_qtables.py          # Q-table init / choose_action / get_env_feedback
│       ├── text_generators.py     # Grammatical transforms (1st->3rd person, Q->declarative, synonymous)
│       ├── inference_guard.py     # Heavy-inference lock + VRAM cache cleanup + GPU telemetry
│       ├── resource_audit.py      # Optional runtime forensics (module init, prompt budget, peaks)
│       ├── therapist_report.py    # Phase B: clinical_report_*.csv generator (crisis rows first)
│       └── log_util.py            # Colored logging setup
│
├── models/                    # Model weights (git-ignored)
│   ├── litert/                # Gemma 4 E2B (.litertlm)
│   └── piper/                 # Piper TTS voice (.onnx + .onnx.json)
│
├── data/                      # Runtime data (all subdirectories auto-created on first boot)
│   ├── therapist.db           # SQLite (authoritative relational store; turns + clinical_scores + intervention_logs + ...)
│   ├── libs/                  # question_lib_v4.json (37 dims + synthetic pool) — only file kept here
│   ├── q_tables/              # Per-subject Q-tables (CSV, keyed by BASE id — stable across a subject's sessions); rebuilt from config.yaml item_importance on participant reset
│   ├── results/               # Per-session Report_{SUBJECT}_{TS}.csv / Notes_{SUBJECT}_{TS}.csv (handler_rl end-of-session emit)
│   ├── sessions/              # SessionDossier JSON — one file per session, session_{SUBJECT}_{TS}.json
│   ├── clinical/              # Phase B: clinical_report_{SUBJECT}_{SESSION_ID}.csv — therapist-facing export
│   └── logs/                  # data/logs/{SUBJECT}_{TS}/ per-session log dir — {SUBJECT}_{TS}.log (CSV) + .json (NDJSON)
│
├── assets/audio/              # waiting_music.wav (fallback) + ambient_therapy.mp3 (long-form ambient bed, git-ignored; deploy-time asset)
├── scripts/                   # Python tools + clinical-ops helpers (no launchers)
│   ├── new_participant_init.sh    # Archive per-subject Q-table; optional new subject_id (DB preserved)
│   ├── model_fetch.py             # Fetch Gemma 4 E2B LiteRT model (~2.6 GB)
│   └── expand_question_lib.py     # Seed / LLM-extend question_lib synthetic variants
├── dev/tests/                 # Pipeline + DB + parser unit tests + clinical export tests
├── tests/                     # Question library invariant tests
├── legacy-prototype/          # Original Nie et al. CSV-bridge reference (unchanged)
│
├── laptop_deploy.sh           # ⭐ laptop: one-command dev loop — sync + kill + run + tail
├── laptop_sync.sh             # laptop: rsync code to Jetson (no kill, no launch)
├── laptop_pull.sh             # laptop: single-mirror data pull Jetson → pulled_data/latest/ (non-destructive on Jetson)
├── jetson_setup.sh            # Jetson: one-time env bootstrap (apt, venv, Piper, LiteRT)
├── jetson_kill.sh             # Jetson: hard kill of all CaiTI processes + port release
└── jetson_run.sh              # Jetson: launch main.py (assumes env ready, ports clear)
```

---

## Module Reference

### `src/core/handler_rl.py` — RL orchestrator

The top-level clinical pipeline. `HandlerRL.run()` is the session's main thread.

**Flow:**
1. `setup()` — load `question_lib_v4.json`, build fresh Q-table weighted by `ITEM_IMPORTANCE`, load per-subject CSV Q-table (paper baseline), then overlay the DB's `persistent_rl_state` row (authoritative, carries `item_mask` + `top_score2_dims`).
2. Greeting — one LLM call building a **recall-and-resume** system prompt for returning users (includes the top-Score-2 dim from the last session) or a plain welcome for first-timers. Greeting text goes through `set_question_prefix` so it's prepended to the first real question.
3. `_run_phq4_screening()` — asks all 4 PHQ-4 / GAD-2 questions (GAD-2 + PHQ-2), scores each 0–3 via `therapy_content.score_response`, persists incrementally via `DB.log_screening_scores`, and records each `intermission_screening` row so the speech-loop intermission ladder never re-asks them. Triggers `GAD2_POSITIVE` / `PHQ4_HIGH_RISK` clinical flags.
4. **Main RL loop** — epsilon-greedy `choose_action` over the item_mask-filtered Q-table. Every turn calls `questioner.ask_question` and applies the standard Q-update `Q(s, a) += α · (r + γ · max Q(s', ·) − Q(s, a))`. Returning users get the first 1–2 turns force-targeted to their prior Score-2 dims (bypasses ε-greedy).
5. **Crisis override** — after each turn, `_crisis_scan()` checks whether any `CRITICAL_DIMS` entry just hit score 2. If yes: log a `safety_flags` row, log_question the `SAFETY_RESOURCES_MESSAGE` (988 / SAMHSA), short-circuit into CBT with the critical dim pre-selected.
6. `run_cbt(question_lib)` — 3-stage CBT if any dim scored 2 (see `CBT.py`).
7. **Post-session** — `_save_longitudinal_state()` writes DB `persistent_rl_state` (Q-table JSON + item_mask + top-5 Score-2 dims ordered by importance), writes paper-format CSV at `data/q_tables/item_qtable_<subject>.csv`, emits `_generate_session_analysis` (SUMMARY / PREFERENCES / SAFETY_FLAGS extraction for DB), `_generate_clinical_summary` (SOAP note), a closing LLM reply, and dumps `SessionDossier` JSON.

**Persistence contract** (documented at module head): CSV is paper/legacy compatible and human-inspectable; DB `persistent_rl_state` is authoritative for Q-values at load time. Don't hand-edit the CSV — it will be overwritten by the DB on next session start.

### `src/core/questioner.py` — Question selection & evaluation

`ask_question(question_lib, S)` delivers one dimension's question and classifies the user's reply.

- **Variant pool** — combines `question[]` (therapist-authored, legacy) with `question_synthetic[]` (post-hoc structural paraphrases added by `scripts/expand_question_lib.py`). Weighted draw: 60 % legacy / 40 % synthetic, so a single therapist-authored question is still picked 60 % of the time even when synthetic paraphrases exist.
- **Runtime Rephraser** — gated by `config.yaml::rl.rephrase_at_runtime` + `rephrase_probability` (default 95 %). One extra LLM call runs `generate_synonymous_sentences` on the selected variant to reproduce the paper's "varied rephrasings" behaviour. Set `rephrase_at_runtime: false` for a latency-sensitive deploy that relies on the pre-generated pool alone.
- **Classify** — `classify_segments` runs the Response Analyzer per segment via `response_bridge.get_openai_resp` (short Yes/No/Stop/Maybe/Question shortcuts skip the LLM entirely when the reply is ≤3 words).
- **`_if_valid_response`** — handles the three result shapes: matching dim + score, general Yes/No/Stop keyword (score comes from `entry["Yes"]` / `["No"]`), or ambiguous Maybe/Question. Score 2 replies compose a follow-up ("You mentioned that ... Can you tell me more about it?") via `reflective_summarizer` with a `generate_change` fallback.
- **RV follow-up** — after any score-2 follow-up is collected, `evaluate_result` calls the split RV pipeline (Reasoner → Guide or Validator, see below) and queues the Validator's prose as the `set_question_prefix` for the next question.
- **Retry guide** — if the first classification is invalid and the user didn't explicitly stop, `retry_guide()` generates a one-line clarification and re-collects.
- **Multi-dim back-fill** — after a valid primary answer, `_apply_segment_level_backfill` opportunistically credits non-primary dims already classified in the per-segment output (no extra LLM call). `_apply_multi_dim_updates` runs the `MULTI_DIM_SYSTEM_PROMPT` classifier on the joined utterance if it's ≥20 tokens and ≥2 segments — catches dimensions the per-segment classifier missed.
- **Reward** — per-turn reward is configurable via `rl.reward_mode`: `"mean"` (paper-strict) or `"hybrid"` = (max + mean) / 2 (default; prevents a single severe segment being diluted to near-zero).

### `src/core/response_analyzer.py` — Response Analyzer (paper §5.2)

- `classify_dimension_and_score(user_input, original_question)` — the paper's Analyzer. Single `INIT_ASKER_SYSTEM_PROMPT_V2` system prompt listing every dim + definitions + few-shot examples; user payload is `Question: X\nAnswer: Y`.
- `classify_multi_dimensions(user_input, original_question)` — strict-JSON multi-dim classifier used by questioner's multi-dim back-fill.
- `reflective_summarizer(original_question, user_response)` — 1st→3rd person restate ("You mentioned that ...") per paper §5.2.
- `rephrase_question(original_question)` — structural rephrase preserving clinical intent + timeframe. Used by external tooling; questioner uses `generate_synonymous_sentences` at runtime instead.

### `src/core/reflection_validation.py` — R-V pipeline (paper §5.3, Fig. 9)

Three distinct LLM tasks, each a self-contained call:

- `rv_reasoner` — decides DECISION 0 (follow-up is topically related) vs 1 (unrelated). Single-line output.
- `rv_validator_mi` — on DECISION 0, produces an empathic **VALIDATION** block (3–5 sentences of acknowledgement + grounded support, paper-length). Used to seed the prefix on the next question.
- `rv_guide` — on DECISION 1, produces a **Guide** block (2–4 sentences: brief acknowledgement + redirect with a focused follow-up question anchored to the client's phrasing).

The consolidated wrapper `rv_consolidated` dispatches to Guide-or-Validator based on the Reasoner's decision; the Validator is always run on the final follow-up text even after a Guide redirect (matches legacy). `_extract_labelled` captures everything after the `VALIDATION:` / `GUIDE:` label to end-of-output, so multi-sentence paragraphs are preserved.

### `src/core/CBT.py` — 3-stage CBT protocol (paper §5.4)

Triggered when one or more dims scored 2. `run_cbt(question_lib)`:

1. **Stage 0 — Select.** List every Score-2 dim; user picks by number. `_extract_choice_number()` accepts digits ("2"), cardinal words ("two"), and ordinal words ("second", "the third one") — voice-dictated answers through Whisper rarely emit numerals for small integers.
2. **Stage 1 — Recognize.** Pull the user's recorded statement (prefers `followup_resp_1` > `followup_resp` > `original_resp` from the RV notes), ask them to identify unhelpful thoughts. `stage1_reasoner` checks validity (returns DECISION 0/1). Up to 2 retries via `stage1_guide`.
3. **Stage 2 — Challenge.** `stage2_reasoner` / `stage2_guide` loop.
4. **Stage 3 — Reframe.** `recap_stage3_challenge` recaps the user's CHALLENGE as a prefix, then `stage3_reasoner` / `stage3_guide` loop.

If any stage exhausts its 3-attempt budget, the `CBT_ESCALATION_MESSAGE` (SAMHSA / 988) is spoken before pausing. Every CBT step writes a structured note row into `question_lib[i][j]["notes"]` with `CBT_stage: success` or `CBT_stage: N_failed`.

`_parse_decision` is strict — requires an explicit `DECISION: 0` / `DECISION: 1` line and fail-closes to "retry" on ambiguity, avoiding the legacy false-pass on any stray `0` in prose.

**Guide output sanitizer.** Each `stageN_guide` LLM call is wrapped in `_sanitize_guide_text(raw, target_label)` before being spoken. Addresses three Gemma-4-E2B small-LLM failure modes: (1) the model echoes the prompt's STATEMENT / UNHELPFUL_THOUGHTS header back before the actual guidance (confusing — sounds like CaiTI is reading the user's own words back); (2) it prefixes the output with the target label itself (`CHALLENGE: ...`, fine in the DB note but awkward spoken); (3) it mimics the few-shot examples' first-person register ("I can challenge this thought by asking myself..." — sounds like CaiTI is narrating the user's internal monologue). The sanitizer extracts only the content after the target label, strips re-echoed header lines, swaps first-person pronouns to second-person ("I can" → "you can", "myself" → "yourself"), recapitalises sentence-initial pronouns, and prefixes with "Here's an example challenge you could try:" so the user hears it clearly as offered guidance, not a question aimed at them. Legacy / GPT-4 didn't need this — the sanitizer is a Gemma-specific output-hygiene shim, not a divergence from the clinical contract.

### `src/core/therapy_content.py` — Clinical content constants

- `CLINICAL_SCREENING` — 4 PHQ-4 questions (2 GAD-2 anxiety + 2 PHQ-2 depression), each with id / text / options / scale.
- `GAD2_THRESHOLD = 3`, `PHQ4_THRESHOLD = 6` — clinical flag triggers.
- `MEDITATIONS` — 5 guided breathing scripts used by the intermission ladder.
- `CRITICAL_DIMS = {sib, safe, risk, drug, alcohol}` — dimensions that trigger the safety override on Score 2.
- `SAFETY_RESOURCES_MESSAGE` / `CBT_ESCALATION_MESSAGE` — 988 Lifeline + SAMHSA 1-800-662-4357 + 911 escalation text.
- `score_response(text)` — maps free-form Likert speech ("several days", "nearly every day") to 0–3, returns −1 on opt-out.

### `src/core/intermission_manager.py` — Intermission activity selector

`IntermissionLadderManager` + `IntermissionTracker` drive the intermission pipeline that `speech_service._wait_for_output_with_intermission` runs while the LLM thinks. Replaces the original strict ladder with **randomised cycling**:

- `next_activity(exclude=...)` picks one of `{SCREENING, BREATHING_EXERCISE, MUSIC}` at random on each iteration, deprioritising the last stage played so users don't get the same activity twice in a row when alternatives exist. SCREENING is only eligible while at least one PHQ/GAD question remains unanswered. MUSIC is the guaranteed fallback — if every stage is excluded (e.g. after the user declines all others this turn), MUSIC is returned so the user never hears silence.
- `mark_activity(stage)` records the last-played activity for the next pick's deprioritisation.
- **SCREENING** — asks the next unanswered PHQ-4 / GAD-2 question. Tracker guarantees no repeats **for the entire session** and supports checkpoint restore from DB so mid-session restarts don't reset state. Answered / skipped questions never regress. If the user says "no" / "skip" / "pass", the question is marked SKIPPED and the turn falls through to a fresh random pick from `{BREATHING, MUSIC}`.
- **BREATHING_EXERCISE** — one meditation from the pool with **session-scoped removal**: each played script is added to `_used_breathing_idx` and excluded from subsequent picks, so a user working through a long session hears all 5 scripts before any repeat. When the pool is exhausted the used set clears (wrap-around) and the next pick still avoids the most recent script. Breathing is passive — the block speaks the guidance and holds silently for the LLM's remaining latency; there is **no listen step** (removed so the user isn't left wondering why the device is waiting for them to respond to a meditation, and to eliminate one class of STT false positives on the global command gate).
- **MUSIC** — ambient fallback via `BackgroundMusicThread` with random-segment playback (see `src/drivers/audio.py` below). The MUSIC block fades the bed up to `_MUSIC_BED_INTERMISSION` (85 %) via `BackgroundMusicThread.fade_to` — at listen-level loudness matching Piper TTS — and triggers a fresh `jump_to_random_segment()` on each invocation so two back-to-back MUSIC beats sound different.

**Handoff back to the LLM reply.** When the LLM output arrives, the intermission loop fades the music down to `_MUSIC_BED_HANDOFF` (5 %) for the bridge phrase, speaks a randomised bridge ("Thank you for reflecting on that with me…"), speaks the LLM response (during which `set_ai_speaking` auto-ducks the bed to 2 %), then fades back up to `_MUSIC_BED_AMBIENT` (85 %) ready for the next turn. Music is never hard-cut — every transition goes through `fade_to`. The bridge phrase is suppressed on the very first turn (`is_session_start=True`) because the user hasn't shared anything yet — the opening dimension question lands clean after the post-greeting intermission.

**Background-music loudness contract.** Two audibly distinct states:

| State | Volume | When it applies |
|---|---|---|
| **LOUD** (foreground listen level) | 0.85 | Idle, post-turn rest, between meditation guidance phrases, MUSIC intermission, any "hold" window while the LLM thinks and the mic is NOT listening |
| **DUCKED** (whisper) | 0.02 | Any time Piper TTS is playing (`set_ai_speaking` set by `AudioPlayer.play`) OR the mic is listening (`set_user_speaking` set from the moment `record_until_silence` opens the stream — not just after VAD detects voice) |

`_target_volume()` short-circuits to `speaking_volume` (0.02) whenever either flag is set, so raising the base levels cannot interfere with TTS intelligibility or mic capture. The four per-state constants keep distinct names so a deployment can tune one independently, but by default three of them share the same loud value:

| Constant | Value | When it applies |
|---|---|---|
| `_MUSIC_BED_AMBIENT` | 0.85 | Idle / post-turn resting level (LOUD) |
| `_MUSIC_BED_BREATHING` | 0.85 | Between meditation guidance phrases (LOUD) |
| `_MUSIC_BED_INTERMISSION` | 0.85 | MUSIC intermission block (LOUD, TTS-match) |
| `_MUSIC_BED_HANDOFF` | 0.05 | ~1 s dip right before the LLM response TTS |

### `src/core/context_manager.py` — Closing-reflection helper

Thin singleton. `record_turn` / `reset` are no-op stubs kept for API compatibility. The only live method is `generate_closing_reflection()`, which pulls the last 12 turns from the session DB and passes them explicitly into one self-contained LLM call. Invoked by `speech_service.handle_end_session` at end-of-session before the goodbye music.

### `src/models/llm_client.py` — LLM gateway

Single public function: `llm_complete(system_content, user_content, role=LLMRole.XYZ) -> str`.

- **Engine** — lazy singleton `litert_lm.Engine` loading `./models/litert/gemma-4-E2B-it.litertlm`. CPU (XNNPack) default, GPU (ML Drift) when `LITERT_BACKEND=gpu` and supported.
- **Role plumbing** — `LLMRole` enum mirrors the paper's task decomposition (Analyzer, Rephraser, ReflectiveSummarizer, R-V Reasoner / Guide / Validator, CBT Reasoner / Guide, General). `ROLE_MODEL_MAP` currently maps every role to the one on-device Gemma; update the map to split roles across backends later without touching call sites.
- **No auto-injection.** Every call is self-contained. `llm_complete` does not read session history, user context, or RL state — the caller passes exactly what the task needs.
- **Heartbeat** — a background thread logs every 10 s during inference so the terminal never looks dead on slow Jetson runs.
- **Heavy-inference lock** — wraps the generate call in `inference_guard.heavy_stage` so STT and LLM never contend for VRAM simultaneously.

### `src/models/stt.py` — Speech-to-Text

`STTGenerator` wraps `faster-whisper` (`base.en`, `int8`, `beam_size=2`). Returns `{"transcript": "...", "detected_emotion": "neu"}` — the `detected_emotion` slot is a legacy compatibility shim; SER has been removed from this deployment (paper doesn't require SER for the clinical pipeline).

**VRAM handoff** — `suspend_all()` fully deletes the Whisper model + `gc.collect()` + empties CUDA cache so the 2–3 GB Gemma model has room. `resume_all()` re-loads for the next listen cycle. Orchestrated by `speech_service` around every LLM call.

### `src/models/tts.py` — Text-to-Speech

`TTSGenerator` runs `piper` as a subprocess with `en_US-amy-medium.onnx`, piping text on stdin. Configurable `length_scale` (speech rate) and `sentence_silence` via env. Falls back to `espeak-ng` if Piper or its config is broken, so a session never fails for lack of TTS.

### `src/services/speech_service.py` — Real-time orchestrator

The main audio loop. States: `idle`, `onboarding`, `main_listen`, `main_process`, `intermission_screening`, `intermission_exercise`, `music_fallback`, `speaking`.

- **Idle** — polls 1 s audio windows for the wake phrase (`hello|hi|hey|start|wake` + `katie`) or GPIO Start button.
- **Onboarding** — asks for the user's name, then delivers a **personalised handshake greeting**: `"Hello, {name}. I'm CaiTI, your intelligent therapist. Thank you for joining me today."` (the name is dropped gracefully when the Name Guard fell back to `User`). The earlier "Hello, I'm CaiTI." → music swell → "Who am I speaking with today?" opener still frames the name prompt. A multi-layer Name Guard rejects common filler ("of course", "good morning", "I'm fine", etc.) and strips phrases like "my name is". Bypass keywords (e.g. "start", "hello") fall back to `User`. At the end of onboarding, `_first_output_pending` is armed so the main loop routes the very first LLM utterance through the intermission pipeline.
- **Active session (parallel turn)** — every clinical turn goes through `_run_parallel_turn()`, which returns one of `"delivered"` / `"silence"` / `"session_end"` / `"start_echo"`. The body is (1) `_record_utterance_to_wav()` — mic-only capture, saves a per-turn unique WAV so a concurrent SCREENING block can't overwrite it; (2) if the captured audio is ≥ `_PARALLEL_MIC_WINDOW_MIN_SEC` (3 s), spawn a daemon `_start_transcription_worker()` on the saved WAV; (3) call `_run_one_intermission_activity()` on this thread immediately, so the user hears "While I'm processing that…" / breathing / music within ~100 ms of the mic closing. The worker pushes the transcript onto `input_queue` the moment decode completes, so the handler's LLM call runs in parallel with the intermission instead of serialising behind it. Short utterances (< 3 s) take a serial fallback path that preserves the fragment-merge retry in `_listen_with_retry`. `_first_output_pending` routes the very first LLM utterance through the intermission pipeline the same way (PHQ-4 / breathing / music fills the pre-first-question gap; bridge phrase suppressed on session start).
- **VRAM handoff** — moved INSIDE the STT worker: `stt.suspend_all()` runs after decode completes so the main thread can never suspend Whisper mid-decode. STT is resumed at the top of `_run_parallel_turn` and inside SCREENING's listen step, re-suspended by the worker at utterance end. Also resumed at idle-loop entry so the wake-detect transcribe never fails with "Model not loaded".
- **Intermission early-exit watcher** — BREATHING and MUSIC blocks run behind `_start_output_ready_watcher(llm_done)`, a daemon thread that peeks `output_queue.empty()` every 100 ms. As soon as the handler's response lands, the watcher sets `stop_playback_event` (cuts any playing meditation TTS gracefully, respecting `_INTERMISSION_MIN_ENGAGEMENT_SEC=6.0`) and trips `llm_done` so the block's hold timer returns immediately. SCREENING is deliberately NOT watched — cutting the user mid-PHQ-answer would destroy clinical data.
- **Intermission pipeline — GIL-safe proactive activity** — LiteRT-LM's Gemma inference holds the Python GIL for the full 2–10 s of each GPU-backed LLM call, which means any `Event.wait(3)` on this thread is effectively paused until the handler releases the GIL. The activity's TTS + listen run in GIL-free pygame / PyAudio C extensions, so they play over the top of the handler's LLM work in parallel. SCREENING is picked first while any PHQ-4 / GAD-2 question remains; once all four are resolved, the ladder cycles BREATHING ↔ MUSIC with last-activity deprioritisation. Breathing is passive (no listen step, just speak + hold). User-decline chains still fire on SCREENING (paper-aligned fallback). When the LLM response arrives the music fades down for the bridge phrase (suppressed on session start), the therapist's reply is spoken, and the music fades back up to the ambient base.
- **End session** — via voice (`end session` / `goodbye` via `GlobalCommandMatcher`), GPIO End button, or the FastAPI `/api/end_session`. Triggers: `stop_audio` → `generate_closing_reflection` → speak reflection → save `SessionDossier` → play goodbye music → back to idle. Also clears `_first_output_pending` so a mid-first-turn end doesn't leave the flag armed. The global command matcher uses **exact-token match** for `goodbye`/`bye` (short-token fuzzy matching is unreliable — "be" ↔ "bye" scored 0.80 under the old threshold and ended a session mid-CBT, fixed). The STT worker's command gate is authoritative: on `__CMD_END__` the worker does NOT queue the transcript to `input_queue` (so the handler never sees a sentinel as a clinical answer) and the main thread sets `stop_playback_event` to cut any in-flight intermission.

### `src/services/response_bridge.py` — Classifier output parser

Pure-Python parser for the Analyzer's output. `get_openai_resp(user_input, original_question, dimension_label)` returns one of:

- `(dimension_label, "Yes" | "No" | "Stop" | "Maybe" | "Question")` — short-response shortcut (skips the LLM entirely for ≤3-word replies containing one of these tokens).
- `(dim, score)` — parsed from plain text (`weight, 2`), prefixed text (`DLA_3_talk, 1`), or JSON-like output (`{"dim": "mood", "score": 2}`).
- `("NA", 99)` — parse failure / `Other, N` fallback.

### `src/drivers/audio.py` — Audio perception + background music

- `AudioRecorder` — PyAudio + WebRTC-VAD recorder. `record_until_silence` waits up to 5 s for speech to start, then records until `silence_duration` (2 s default) of VAD silence, plus a 0.4 s trailing pad. Filters out noise bursts shorter than `min_speech_sec` (0.3 s). Computes RMS so the caller can discard buffers below a noise gate.
- `BackgroundMusicThread` — ambient `pygame.mixer.music` loop. Always-on; `start()` / `stop()` are idempotent. Plays `assets/audio/ambient_therapy.mp3` with `waiting_music.wav` fallback.
  - **Auto-duck** — whenever `set_ai_speaking(True)` (Piper TTS playing) or `set_user_speaking(True)` (mic stream open — fires from the moment `record_until_silence` opens the stream, not just after VAD detects voice) is set, `_target_volume()` short-circuits to `speaking_volume` (2 % default) regardless of any base/fade target. Restores to the current base (85 %) as soon as both flags clear. This is the single source of "music ducks while the therapist speaks OR the mic is listening."
  - **Smooth fades** — `fade_to(target_volume, duration)` ramps linearly between volumes over `duration` seconds (the worker thread evaluates the ramp every ~200 ms). Used by the intermission pipeline to hold `_MUSIC_BED_INTERMISSION` / `_MUSIC_BED_BREATHING` / `_MUSIC_BED_AMBIENT` (all 85 % by default, foreground listen level matching Piper TTS) during any "hold" window, then dip to `_MUSIC_BED_HANDOFF` (5 %) ~1 s before the LLM reply TTS so the bridge phrase lands clean, then restore to `_MUSIC_BED_AMBIENT` (85 %) after. See the [Intermission Manager section](#srccoreintermission_managerpy--intermission-activity-selector) for the full loudness contract + constants table.
  - **Random-segment playback** — on startup and on every natural loop rollover, the track begins at a **random offset** picked uniformly from `[0, duration × 0.90]`. Duration is probed once per track via a zero-dependency MPEG frame-header parser (`_probe_audio_duration`) so no `mutagen` / `ffprobe` install is needed. For the shipped 5 h ambient track this means users rarely hear the same opening seconds twice.
  - **Explicit segment jumps** — `jump_to_random_segment()` sets a thread-safe event that the worker observes on its next poll and seeks to a new random offset. Called from `_run_music_block` so every MUSIC intermission plays a different part of the track.
  - **Seek hardening** — SDL2's MP3 decoder silently fails on very deep seeks on some hardware (confirmed empirically on the shipped Jetson + USB DAC chain at offsets past ~95 % of the 5 h track: `play(start=offset)` returned `get_busy()==False` and produced no audio). Two defences: (a) the 90 % safe-fraction cap on the random-offset upper bound avoids the problem region entirely for organic jumps; (b) every `play(start=offset)` is verified via `get_busy()` + `get_pos() >= 0` within ~200 ms and on failure retries at half the offset, then zero.
  - **Anti-click transitions** — every `play()` uses `fade_ms=400` (the click-masking window empirically required on the USB audio chain) and every stop / jump goes through `fadeout()`, waiting for the ramp to complete before overwriting the stream. Eliminates the mid-waveform click transient on hard seeks.

### `src/drivers/player.py` — TTS playback

`AudioPlayer.play(wav, stop_event)` plays through the shared `pygame.mixer` (no subprocess). Emits `DUCK` → `set_ai_speaking(True)` before playback and `RESTORE` on completion so the background music drops while the agent speaks.

### `src/drivers/gpio_manager.py` — Jetson hardware I/O

Singleton. Pin map (BOARD numbering):

| Pin | Role | Edge / Polarity |
|---|---|---|
| 11 | Start Session | FALLING / active-low (300 ms debounce) |
| 13 | End Session | FALLING / active-low (300 ms debounce) |
| 15 | Opt-Out | FALLING / active-low (300 ms debounce) |
| 16 | Button 4 (spare) | FALLING / active-low (300 ms debounce) |
| 18 | Listening LED | OUT |

Every polarity is overridable per pin via the `PIN_BTN_*_ACTIVE_LOW` / `PIN_LISTENING_LED_ACTIVE_LOW` env flags. Events are pushed onto a thread-safe queue plus a fallback level poll (`_poll_buttons_fallback`) catches any missed edge interrupts. Off-Jetson, the module silently downgrades to a `_GPIOStub` so the rest of the stack runs identically on a laptop.

### `src/drivers/db_manager.py` — SQLite persistence

Single-file DB at `data/therapist.db`. All writes go through a context-managed connection (`_connect()`) that commits on success and rolls back on exception, so connection handles never leak under clinical load.

| Table | Purpose |
|---|---|
| `users` | Subject IDs |
| `sessions` | Session start/end + `end_reason` (`normal` / `atexit` / `crash_recovery`) + user link |
| `turns` | Full dialogue with speaker, turn_index, meta_data (JSON) |
| `summaries` | Per-session LLM summaries (SOAP notes + session analysis) |
| `user_preferences` | Extracted key-value facts, upserted |
| `feedback` | User feedback rows |
| `safety_flags` | Legacy flag rows (kept for back-compat auditors) |
| `clinical_flags` | M3 authoritative clinical-flag log (consolidated audit trail) |
| `safety_deliveries` | C2 audit: every crisis-resources broadcast (method / success / error) |
| `clinical_screening` | Per-session PHQ-4 / GAD-2 totals + `gad2_positive` / `phq4_high_risk` flags |
| `intermission_screening` | Per-question `ANSWERED` / `SKIPPED` / `UNRESOLVED` (Phase A) status — the no-repeat source of truth for the speech-loop ladder |
| `clinical_scores` | **Phase B** — final score per (session, dim). One row per scored dimension; valid 0/1/2 only (sentinels don't land here). `evidence_turn_id` FKs the user turn that sourced the score. |
| `clinical_score_attempts` | **Phase B** — append-only history of **every** scored attempt including UNRESOLVED (-2) / OPT_OUT (-1) sentinels, with `source` and `attempt_index`. |
| `intervention_logs` | **Phase B** — MI reflections + CBT stage events. `kind` ∈ {`MI`, `CBT`}; `outcome` ∈ {`started`, `success`, `failed`, `delivered`, `escalated`}. |
| `persistent_rl_state` | One row per user: Q-table JSON + item_mask JSON + top-5 Score-2 dims JSON + last_session_id |

**Helper methods (Phase B):**

- `record_clinical_score(session_id, dim_index, dim_label, score, ...)` — sentinel-aware writer. Appends to `clinical_score_attempts` unconditionally; upserts into `clinical_scores` only when `score ∈ {0, 1, 2}` (never letting an OPT_OUT / UNRESOLVED retry overwrite a valid final).
- `record_intervention_log(session_id, kind, stage, technique, outcome, dim_label, detail)` — MI/CBT event logger.
- `get_clinical_scores(session_id)` / `get_clinical_score_attempts(session_id)` / `get_intervention_logs(session_id)` — read-side helpers used by the therapist-report generator.
- `close_session(session_id, reason)` — used by the **atexit hook** (Phase A) to stamp `sessions.end_time` + `end_reason`, preventing the "dangling session" clutter that used to appear on SIGTERM / Ctrl-C.

`get_recent_screening_scores(user_id, limit=5)` powers the SOAP-format longitudinal TREND block; `get_user_context_string` powers the recall-and-resume greeting.

### `src/utils/config_loader.py` — Config aggregator

Reads `config.yaml` (RL + audio + STT + TTS + DB) and `.env` (LiteRT model + GPIO pins + TTS pacing + hardware polarity). Fails fast if `config.yaml` is missing `app` / `paths` / `rl`.

### `src/utils/io_record.py` — IPC + session state

- Queues: `INPUT_QUEUE` (user → handler), `OUTPUT_QUEUE` (handler → speech). Both `queue.Queue`.
- Events: `START_SESSION_EVENT`, `END_SESSION_EVENT`.
- **Two-layer subject identity.** `init_record()` composes `SUBJECT_ID = "<base>_<YYYYMMDD_HHMMSS>"` from `SUBJECT_BASE_ID` at session start. `SUBJECT_ID` keys every per-session artefact (Report / Notes / dossier / log dir / crisis file); `SUBJECT_BASE_ID` stays stable across visits and keys the DB `users.subject_id` row + Q-table filename so longitudinal state still resolves for returning subjects.
- Per-session `REPORT_FILE` / `NOTES_FILE` module attrs are re-computed inside `init_record()` via `config_loader.format_result_paths(SUBJECT_ID)`, so `generate_results()` writes to a session-unique path every time.
- `log_question(text)` / `get_answer()` / `get_resp_log()` — the speech loop writes user input to INPUT_QUEUE; the handler writes questions to OUTPUT_QUEUE via `log_question`.
- `set_question_prefix(text)` — prepends a one-shot prefix (e.g. RV Validator output) to the next agent utterance.
- `_segment_utterance(text)` — legacy-faithful segmenter that splits on `.!?` + connectives `", and"` / `" but "`. Zero deps.
- Multi-destination logging: SQLite `turns` rows, per-session `.log` CSV, per-session `.json` NDJSON, `SessionDossier` (structured per-interaction JSON dumped to `data/sessions/` at end of session).

### `src/utils/io_question_lib.py` — Question library I/O

`load_question_lib(path)` / `save_question_lib(path, lib)` + `generate_results` which atomically writes `Report_<SUBJECT_ID>.csv` and `Notes_<SUBJECT_ID>.csv` to `data/results/`. The destinations default to `io_record.REPORT_FILE` / `NOTES_FILE`, which are rewritten at session start so the composed `<name>_<timestamp>` id produces a distinct file for every session.

### `src/utils/rl_qtables.py` — Q-learning primitives

- `initialize_q_table(n_states, actions)` — `DataFrame(n_states × |actions|)` seeded with `ITEM_IMPORTANCE` weights along the diagonal.
- `choose_action(state, q_table, mask, ...)` — ε-greedy over masked actions (mask zeros out already-asked dims).
- `get_env_feedback(S, A, reward, terminate_flag, item_mask)` — returns `'terminal'` when the mask is empty or a termination flag fires; otherwise the next state is the selected action.

### `src/utils/text_generators.py` — Grammatical transforms

Legacy-compatible LLM transforms. All self-contained (no context injection):

- `generate_synonymous_sentences` — runtime Rephraser for screening questions (role = `REPHRASER`).
- `generate_change` — 1st → 3rd person (role = `REFLECTIVE_SUMMARIZER`).
- `generate_change_positive` / `_negative` — question → declarative (used for the Score-2 follow-up template on Yes / No branches).

### `src/utils/inference_guard.py` — Resource guard

- `heavy_stage(name)` — single global threading.Lock. Serializes STT and LLM so they never hold VRAM at the same time.
- `clear_inference_cache(reason)` — `gc.collect()` + optional `torch.cuda.empty_cache()` when PyTorch is present (used by STT; the LLM client doesn't call this any more on the hot path).
- `get_system_memory_snapshot()` — `free -h` + sysfs GPU freq/load / `nvidia-smi` / `tegrastats` cascade for debugging Jetson OOMs.

### `src/utils/resource_audit.py` — Optional forensics

Captures module-init peaks, prompt-budget utilisation, process inventory, and zone peaks. Writes a JSON report at session end. Call sites in `main.py` + `handler_rl.py` + `stt.py`. Safe to disable with `RESOURCE_AUDIT_ENABLED=0` if the audit overhead is not wanted.

### `src/utils/therapist_report.py` — Clinical-grade CSV export (Phase B)

`generate_therapist_report(session_id, db=None)` writes `data/clinical/clinical_report_{SUBJECT_ID}_{SESSION_ID}.csv` — a flat, therapist-facing export that joins `turns`, `clinical_scores`, `clinical_score_attempts`, `intervention_logs`, and `safety_deliveries` for one session.

**Output row types, in order:**

1. **`Crisis`** — every dimension with final Score = 2 is emitted at the top of the file so heightened-attention items land in front of the reviewer.
2. **`ScoredDim`** — canonical per-dimension final score, with `attempts=N` and the full `trail=[...]` of attempt scores in the Notes column.
3. **`Intervention`** — MI reflections + CBT stages in chronological order. `InterventionDetail` carries `kind / stage / technique / outcome`; `Notes` carries the most therapist-relevant field from the detail payload (validation_text, guide_text, reframe, etc.).
4. **`SafetyDelivery`** — audit rows proving that the `SAFETY_RESOURCES_MESSAGE` was actually spoken for every `CRITICAL_DIM_SCORE_2` flag (or recording the fallback / error when it wasn't).
5. **`Turn`** — raw dialogue last, so a clinician can drop back into the transcript after reviewing metrics.

**Trigger policy (Phase A + B):**

- Primary: `mark_session_finalised("normal")` triggers the report **before** closing the session row, with the full runtime still alive (clean LLM / DB handles).
- Backstop: `atexit._atexit_close_session` calls `mark_session_finalised("atexit")` on interpreter shutdown. A run-once guard (`_EXPORTED_SESSIONS`) absorbs the duplicate so no double-write happens.

**Regeneration:** the run-once guard is in-memory only; spawn a fresh Python process (or `_EXPORTED_SESSIONS.clear()`) to re-export an existing session's CSV.

---

## Session Lifecycle

```
IDLE
  |
  |  [wake word / GPIO Start / API /api/login]
  v
ONBOARDING            ---> name captured, reset_session(name) installs:
                              SUBJECT_BASE_ID = "alice"                  (stable across visits)
                              SUBJECT_ID      = "alice_20260425_190621"  (per-session unique)
                           then START_SESSION_EVENT.set()
                           |-- personalised handshake greeting:
                           |     "Hello, <name>. I'm CaiTI, your intelligent
                           |      therapist. Thank you for joining me today."
                           |-- _first_output_pending = True
                           |     (arms the main loop to route the first LLM
                           |      utterance through the intermission pipeline)
  |
  v
POST-GREETING GAP     ---> _wait_for_output_with_intermission(is_session_start=True)
                           |-- PHQ-4 / breathing / music fills the silence
                           |   while HANDLER_RL initialises + generates
                           |   the opening dimension question
                           |-- watchdog speaks the first LLM utterance
                           |   on arrival (bridge phrase suppressed)
  |
  v
HANDLER_RL.run()
  |
  |-- setup:          load question_lib + Q-table (CSV baseline, DB overlay)
  |-- greeting:       LLM w/ recall-and-resume for returning users
  |-- PHQ-4 screen:   4 questions, persisted + intermission tracker updated
  |-- crisis check:   phq4_high_risk or gad2_positive -> elevate tone
  |
  |-- ε-greedy LOOP
  |   |
  |   |-- choose_action over item_mask
  |   |
  |   |-- questioner.ask_question(S)
  |   |    |-- pick variant (60/40 legacy/synthetic) + optional Rephraser
  |   |    |-- log_question -> OUTPUT_QUEUE -> speech_service.say()
  |   |    |
  |   |    |-- get_answer() <-- INPUT_QUEUE <-- STT transcript
  |   |    |
  |   |    |-- classify_segments (Response Analyzer per segment)
  |   |    |-- evaluate_result:
  |   |    |     - Yes/No polarity OR matched dim+score OR retry_guide
  |   |    |     - if score 2: follow-up + RV Reasoner -> Guide/Validator
  |   |    |       Validator prose queued as set_question_prefix
  |   |    |
  |   |    |-- segment-level back-fill (free)
  |   |    |-- LLM multi-dim back-fill (>=20 tokens, >=2 segments)
  |   |
  |   |-- Q-update: Q(S,A) += alpha * (r + gamma * max Q(S',.) - Q(S,A))
  |   |
  |   |-- crisis_scan -> if CRITICAL_DIMS hit 2: safety msg + short-circuit to CBT
  |   |
  |   LOOP until item_mask empty OR DLA_terminate OR session end event
  |
  |-- save question_lib snapshot
  |-- save Q-table (CSV + DB persistent_rl_state)
  |
  |-- run_cbt(question_lib) if any score 2
  |   |-- Stage 0: user picks dim
  |   |-- Stage 1 Recognize (Reasoner + up to 2 Guide retries, else CBT_ESCALATION)
  |   |-- Stage 2 Challenge
  |   |-- Stage 3 Reframe
  |
  |-- generate_results -> data/results/{Report,Notes}_<SUBJECT_ID>.csv
  |                       (per session — composed id carries the timestamp)
  |-- closing message LLM
  |-- _generate_session_analysis -> DB summaries + preferences + safety flags
  |-- _generate_clinical_summary -> SOAP note in DB + spoken
  |-- dump_session_history_to_terminal
  |
  v
END_SESSION_EVENT
  |
  |-- speech_service.handle_end_session():
  |    - stop_audio / stop music
  |    - generate_closing_reflection (LLM, last 12 turns)
  |    - speak reflection + "Goodbye"
  |    - save SessionDossier JSON
  |    - goodbye music -> idle ambient bed
  v
IDLE
```

---

## LLM Design (Self-Contained Per Task)

Every module runs the LLM for its own subtask, independently. No rolling history, no auto-injected context pack.

| Call site | LLMRole | What the payload carries |
|---|---|---|
| `response_analyzer.classify_dimension_and_score` | `ANALYZER` | `Question: X\nAnswer: Y` |
| `response_analyzer.classify_multi_dimensions` | `ANALYZER` | Same + strict-JSON output contract |
| `response_analyzer.reflective_summarizer` | `REFLECTIVE_SUMMARIZER` | `{Original Question, User Response}` |
| `response_analyzer.rephrase_question` | `REPHRASER` | `{Original Question}` |
| `text_generators.generate_synonymous_sentences` | `REPHRASER` | Chosen question variant |
| `text_generators.generate_change{_positive,_negative}` | `REFLECTIVE_SUMMARIZER` / `REPHRASER` | User's segment / asked question |
| `reflection_validation.rv_reasoner` | `RV_REASONER` | `{Topic, Original Q/R, Follow-up}` |
| `reflection_validation.rv_guide` | `RV_GUIDE` | Same |
| `reflection_validation.rv_validator_mi` | `RV_VALIDATOR` | Same |
| `CBT.stage{1,2,3}_reasoner` | `CBT_REASONER` | `STATEMENT; UNHELPFUL_THOUGHTS; [CHALLENGE; [REFRAME]]` |
| `CBT.stage{1,2,3}_guide`, `recap_stage3_challenge`, `stage0_prompter` | `CBT_GUIDE` | Same fields the stage needs |
| `handler_rl` greeting / closing / session_analysis / clinical_summary | `GENERAL` | Constructed per call (history + screening block passed explicitly) |
| `main.classify_intent` (wake/sleep router) | `GENERAL` | Transcript + one-word output contract |
| `context_manager.generate_closing_reflection` | `GENERAL` | Last 12 turns from DB, passed in payload |

Continuity across turns lives in `question_lib[i][j]["score"]` / `["notes"]` (read back by Python to build the next prompt) and in DB rows (`user_preferences`, `summaries`, `persistent_rl_state`) — never in prompt-side history.

---

## Persistence Model

Session data fans out into **five parallel stores**, plus a session-close artefact and two RL-state stores. They are not backups of each other — each is optimised for a different reader.

### Five session streams (fired turn-by-turn)

| # | Store | Location | Written by | Read by |
|---|---|---|---|---|
| 1 | SQLite — dialogue | `data/therapist.db` → `turns` | `DB.add_turn()` | query, therapist report |
| 2 | SQLite — clinical | `data/therapist.db` → `clinical_scores`, `clinical_score_attempts`, `intervention_logs`, `clinical_flags`, `safety_deliveries`, `clinical_screening`, `intermission_screening` | `DB.record_clinical_score()`, `DB.record_intervention_log()`, `log_*_flag()`, `log_safety_delivery()` | therapist report, audits, longitudinal analysis |
| 3 | JSON dossier | `data/sessions/session_{SUBJECT_ID}.json` (where `SUBJECT_ID` is the composed `<base>_{TS}`) | `SessionDossier.record_interaction()` (via `log_question`) | archival / human inspection |
| 4 | JSON event stream | `data/logs/{SUBJECT_ID}/{SUBJECT_ID}.json` (NDJSON) | `log_json_event()` | timeline replay, debugging |
| 5 | CSV transcript | `data/logs/{SUBJECT_ID}/{SUBJECT_ID}.log` | `append_to_csv()` | spreadsheet review |

All five are coordinated through `src/utils/io_record.py`.

### Session-close artefact (Phase B)

6. **Therapist report CSV** — `data/clinical/clinical_report_{SUBJECT}_{SESSION}.csv`, generated automatically by `mark_session_finalised("normal")`. Crisis rows at top, scored dims, interventions, safety-delivery audit, raw turns. See [`src/utils/therapist_report.py`](#srcutilstherapist_reportpy--clinical-grade-csv-export-phase-b).

### Two RL-state stores (side-by-side by design)

**Per-subject Q-table CSV** — `data/q_tables/item_qtable_<base_subject_id>.csv`

- Written unconditionally at the end of every session.
- **Keyed by the BASE subject id** (no timestamp), so Alice's session-2 warm-starts from the Q-values saved at the end of session 1. This is a deliberate divergence from the per-session filename scheme used for Report / Notes / dossier artefacts.
- Paper / legacy byte-compatible format (same filename scheme, same shape).
- Loaded as the baseline Q-table at session start.

**SQLite `persistent_rl_state` row**

- JSON-serialised Q-table + `item_mask_json` + `top_score2_dims_json` + `last_session_id`.
- Authoritative for resume semantics: carries info (mask + top-dims) that CSV cannot cleanly express.
- Read at session start by `_load_longitudinal_state`; if the shape matches, it OVERWRITES the CSV-loaded baseline. If missing / malformed, the CSV baseline is kept.

**Divergence risk.** Between a CSV write and a DB write, a crash can desynchronise the two. Since DB wins at load time, a newer CSV with older DB data will be silently overwritten. Acceptable for this deployment; documented at the head of `src/core/handler_rl.py`.

To reset Q-state for the next participant without deleting DB history, use `./scripts/new_participant_init.sh [SUBJECT_ID]` — it archives the per-subject Q-table CSV with a timestamped suffix; the next boot rebuilds from `config.yaml → rl.item_importance` (the therapist-authored empirical priors). `clinical_scores`, `intervention_logs`, `turns`, dossiers, and reports are preserved for longitudinal analysis.

### Session lifecycle & atomic closure (Phase A)

- **atexit hook** — `src/utils/io_record.py` registers `_atexit_close_session` so `sessions.end_time` and `end_reason` are always stamped on interpreter shutdown (SIGTERM from systemd, Ctrl-C, clean exit). Pre-Phase-A sessions were left with `end_time IS NULL` if the process didn't go through the handler's graceful close; now clean exits are explicitly tagged `end_reason='atexit'`.
- **Crash recovery** — on the next boot, `DB.close_open_sessions_for_user` catches any sessions that still made it through with `end_time IS NULL` and tags them `end_reason='crash_recovery'`, so the operator can tell a clean shutdown from a kill signal.
- **STT silence → UNRESOLVED** — `speech_service` no longer collapses an ambiguous utterance into the opt-out bucket. Non-empty-but-unparseable replies persist with `status='UNRESOLVED'`, `reason='stt_unresolved'`, so clinicians can distinguish "user mumbled" from "user opted out" from "user scored 0".
- **Continuous crisis scan** — `CBT.run_cbt` runs `_crisis_intervened()` at each stage entry (Recognize / Challenge / Reframe) **and** after each user response, so a critical dim that hits Score = 2 mid-CBT triggers the safety path immediately rather than at the next user turn.

Clinical state (scores, RV notes, CBT stage results) is written to **both** the relational SQL tables **and** the per-session `question_lib_v4.json` snapshot that `handler_rl` persists at session close.

Additive schema migrations live in `DBManager._migrate_schema`; a trial DB created by an older build is upgraded in place at next boot (idempotent `ALTER TABLE` guarded by column-probe).

---

## Data Storage & How to Access It

### Quickest — read the therapist CSV

```bash
cat data/clinical/clinical_report_*.csv
```

Executive summary per session: Crisis rows first, then all scored dimensions, interventions, safety-delivery audit, raw dialogue.

### SQLite — ad-hoc queries

```bash
python3 -c "
import sqlite3
c = sqlite3.connect('data/therapist.db')
for r in c.execute('SELECT end_reason, COUNT(*) FROM sessions GROUP BY end_reason'):
    print(r)
"
```

Useful one-liners (all hit `data/therapist.db`):

```sql
-- How sessions are closing (clean vs crash-recovered vs dangling):
SELECT end_reason, COUNT(*) FROM sessions GROUP BY end_reason;

-- Latest N sessions for a subject (DB keys on the BASE id; pass the
-- onboarded name here, NOT the composed per-session SUBJECT_ID):
SELECT s.id, s.start_time, s.end_time, s.end_reason
FROM sessions s JOIN users u ON s.user_id = u.id
WHERE u.subject_id = 'alice'
ORDER BY s.id DESC LIMIT 10;

-- All Score=2 (crisis) dimensions across history:
SELECT cs.session_id, cs.dim_label, cs.evidence_text, cs.updated_at
FROM clinical_scores cs WHERE cs.score = 2
ORDER BY cs.updated_at DESC;

-- Every CRITICAL_DIM flag joined to its safety delivery (should never be NULL):
SELECT cf.session_id, cf.flag_type, sd.method, sd.success
FROM clinical_flags cf
LEFT JOIN safety_deliveries sd ON cf.session_id = sd.session_id
WHERE cf.flag_type = 'CRITICAL_DIM_SCORE_2';

-- Full scoring trail for a session (attempts + final):
SELECT a.dim_label, a.attempt_index, a.score, a.source, a.evidence_text
FROM clinical_score_attempts a
WHERE a.session_id = :sid
ORDER BY a.dim_index, a.created_at;

-- CBT stage progress for a session:
SELECT stage, technique, outcome, dim_label, created_at
FROM intervention_logs WHERE session_id = :sid AND kind = 'CBT' ORDER BY id;

-- Any UNRESOLVED PHQ-4 questions from STT layer (Phase A):
SELECT session_id, question_id, reason
FROM intermission_screening
WHERE status = 'UNRESOLVED' AND reason = 'stt_unresolved';
```

### Programmatic helpers

```python
from src.drivers.db_manager import DBManager
db = DBManager("data/therapist.db")

db.get_session_history(session_id)                   # ordered turns
db.get_clinical_scores(session_id)                   # one row per dim, final score
db.get_clinical_score_attempts(session_id)           # full scoring trail (incl. sentinels)
db.get_intervention_logs(session_id)                 # MI + CBT events
db.get_safety_deliveries(session_id)                 # crisis-message delivery audit
db.get_intermission_screening_statuses(session_id)   # PHQ-4/GAD-2 per-question
db.get_recent_screening_scores(user_id, limit=5)     # longitudinal history
```

### Regenerate the CSV report for an existing session

```python
from src.drivers.db_manager import DBManager
from src.utils.therapist_report import generate_therapist_report, _EXPORTED_SESSIONS

db = DBManager("data/therapist.db")
_EXPORTED_SESSIONS.clear()   # bypass the in-memory run-once guard
path = generate_therapist_report(session_id=42, db=db)
print(path)                  # data/clinical/clinical_report_{SUBJECT_ID}_42.csv
```

### Mental model — concentric rings

- **Ring 1 (fastest clinical signal)**: `data/clinical/clinical_report_*.csv` → crisis + scored dims + interventions, one row per event.
- **Ring 2 (structured query)**: `data/therapist.db` tables → `turns`, `clinical_scores`, `intervention_logs`, `safety_deliveries`. Anything you can't see in Ring 1, you query here.
- **Ring 3 (forensic debug)**: `data/sessions/*.json` dossier + `data/logs/<subject>/*.json` NDJSON event stream + `data/logs/<subject>/*.log` CSV transcript. For intra-turn debugging (LLM latency, emotion tagging, RL state at decision time).

Ring 1 is what you hand a therapist. Ring 2 is what you query during research. Ring 3 is what you open when something broke.

---

## Configuration

### `config.yaml` (structural)

```yaml
app:
  subject_id: "8080"          # boot-time fallback only — the real subject id is
                              # captured at onboarding ("Who am I speaking with
                              # today?") and composed into <name>_<YYYYMMDD_HHMMSS>
                              # in io_record.init_record().

paths:
  # ${subject_id} in these templates is filled with the *composed* per-session id
  # (<onboarded_name>_<timestamp>) so every session is its own file entry.
  report_file: "data/results/Report_${subject_id}.csv"
  notes_file:  "data/results/Notes_${subject_id}.csv"

rl:
  item_n_states: 38            # 1 INIT + 37 dims (paper-equivalent to the 39-state grid)

  # Legacy-prototype / demo values in use (clinically-tested).
  # Paper §5.1 states ε=0.9 / α=0.1; legacy prototype shipped ε=1 / α=0.5
  # and that is what the published demo recording runs on.
  epsilon: 1                   # legacy prototype (demo)
  alpha: 0.5                   # legacy prototype (demo)
  gamma: 0.9                   # paper + legacy (agree)
  item_importance: [0, 5, 98, 99, 5, 4, 4, 4, 2, 2, 5, 97, 5, ...]
                               # legacy sentinel pins: mood=98, medication=99, eat=97
                               # pin the top-3 screening triage order (see demo)

  rephrase_at_runtime: true    # D5/D6 — legacy-parity: legacy prototype calls
                               # generate_synonymous_sentences() on ~95% of turns
  rephrase_probability: 0.20   # Tuned down from legacy 0.95 to cut ~1.5 s/turn on GPU.
                               # At 0.20 the Rephraser fires ~20% of the time, so most
                               # turns speak the therapist-authored variant verbatim.
                               # Set to 0.95 to restore legacy wording frequency.
  reward_mode: mean            # paper §5.1 + legacy prototype (arithmetic mean)

  # ── Feature gates (see "Feature Gates & Future Development" below) ──
  # All default false for legacy-parity EXCEPT soap_report_enabled + G8
  # multi-dim back-fill which are silent clinician/paper-research extensions.
  reask_dimension_n: false            # G5  — paper §4.2 re-ask
  multi_dim_backfill_enabled: true    # G8  — paper §4.1 multi-dim back-fill (GPU-cheap now)
  reflective_summarizer_enabled: false # G9 — paper §5.2 MI reflective summarizer
  warm_start_enabled: false           # G11 — returning-user Q nudge + recall greeting
  session_analysis_enabled: false     # G12 — post-session SUMMARY+prefs+safety LLM
  soap_report_enabled: true           # G13 — SOAP clinician note (silent, DB only)
  dimension_optouts_enabled: false    # G14 — per-user disabled_dim masking
  session_cap_enabled: false          # G15 — 60-minute hard cap (re-enable for trials)

audio:
  sample_rate: 16000
  chunk_size: 480
  vad_aggressiveness: 2

stt:
  model_path: "base.en"
  device: "cpu"
  compute_type: "int8"
  beam_size: 2

tts:
  model_path: "en_US-amy-medium.onnx"
  executable_path: "piper"

database:
  db_path: "data/therapist.db"
```

### `.env` (runtime / host-specific)

```env
# LiteRT model + backend
LLM_MODEL=gemma-4-E2B-it
LITERT_MODEL_PATH=./models/litert/gemma-4-E2B-it.litertlm
LITERT_BACKEND=gpu                 # gpu (ML Drift, ~4-8x faster on Jetson Orin Nano) | cpu (XNNPack fallback)
LITERT_CONTEXT_LENGTH=3072         # paragraph-length R-V outputs need headroom; 3072
                                   # covers RV_VALIDATOR worst-case (~2.2k tokens total)
                                   # while cutting KV-cache vs the old 4096 default.
LITERT_MAX_TOKENS=3072             # Engine-level ceiling on prompt + output. Lifted from
                                   # 512 so Gemma can emit the demo's 4-7 sentence MI
                                   # Validator without mid-word truncation; tightened to
                                   # 3072 from 4096 to save KV-cache on the Jetson GPU's
                                   # unified memory pool.

# Speech I/O
STT_MODEL=base.en
STT_COMPUTE_TYPE=int8
STT_BEAM_SIZE=2
TTS_MODEL_PATH=./models/piper/en_US-amy-medium.onnx
TTS_LENGTH_SCALE=0.8
TTS_SENTENCE_SILENCE=1.5

# Hardware (Jetson)
PIN_LISTENING_LED=18
PIN_BTN_START=11
PIN_BTN_END=13
PIN_BTN_OPT_OUT=15
PIN_BTN_4=16
PIN_BUTTONS_ACTIVE_LOW=1
PIN_LISTENING_LED_ACTIVE_LOW=0

# Clinical-trial safeguards (see "Clinical-Trial Hardening" section)
CLINICAL_MODE=0                    # 1 = fail-fast boot if any critical subsystem is unhealthy
REDACT_PII=0                       # 1 = redact user transcripts in logger output (DB/JSON logs untouched)
MIN_FREE_DISK_MB=500               # H6 pre-flight disk threshold (raise to 1000 for trials)
SESSION_MAX_SECONDS=3600           # M6 max session length before graceful termination

# Optional
DISABLE_INTERNAL_SPEECH=0          # 1 = run FastAPI only (no mic/speaker stack)
RESOURCE_AUDIT_ENABLED=1
```

`DISABLE_CONTEXT_HISTORY` has been removed (there's no auto-injected context any more).

**Recommended env for clinical trials:**

```env
CLINICAL_MODE=1
REDACT_PII=1
MIN_FREE_DISK_MB=1000
SESSION_MAX_SECONDS=3600
RESOURCE_AUDIT_ENABLED=1
```

---

## Feature Gates & Future Development

The current deployment is calibrated for **legacy / demo parity** — the runtime flow, prompts, and hyperparameters match the Flutter-demo recording on the legacy prototype exactly, so participants hear the clinically-tested CaiTI experience the paper describes.

Along the way we built **eleven extension features** that either refine paper-research behaviour (§4.1 multi-dim back-fill, §4.2 Dim-N re-ask, §5.2 reflective summarizer) or add new clinical-trial infrastructure (SOAP reports, longitudinal warm-start, dimension opt-outs, session-length caps, crisis override, CBT escalation referral). **Each of these is code-resident but gated OFF by default** so it can be re-enabled for future iterations without reopening the codebase.

Every gate is an independent toggle: flip one in `config.yaml` (or `src/core/therapy_content.py` for the two defined there) and restart — no code changes needed.

### Gate index

**Currently ON** (all legacy-parity safe): `MULTI_DIM_BACKFILL_ENABLED` (G8), `SOAP_REPORT_ENABLED` (G13), `REPHRASE_AT_RUNTIME` (D5/D6).
**Currently OFF** (future / off-demo): G5, G7, G9, G11, G12, G14, G15, D3.

| # | Gate | Default | Config key | Subsystem | Adds |
|---|---|---|---|---|---|
| G5 | `REASK_DIMENSION_N` | `false` | `rl.reask_dimension_n` | Questioner | Paper §4.2 Dim-N re-ask before `retry_guide`. |
| G7 | `CBT_ESCALATION_ENABLED` | `False` | `therapy_content.py` | CBT | 988/SAMHSA hotline referral after 3 failed CBT stage attempts. |
| G8 | `MULTI_DIM_BACKFILL_ENABLED` | **`true`** | `rl.multi_dim_backfill_enabled` | Questioner | Paper §4.1 multi-dimension back-fill (segment + joined LLM passes). **ON after session 11 (John, 2026-04-25):** the Analyzer classified "my sadness takes over my life" as `('emo', 2)` while the asked dim was `mood`; without back-fill the score was dropped and CBT never triggered. The segment-level pass is a free reuse of the classifier output; only the joined-utterance pass is LLM-gated. |
| G9 | `REFLECTIVE_SUMMARIZER_ENABLED` | `false` | `rl.reflective_summarizer_enabled` | Questioner | Paper §5.2 MI reflective summarizer LLM call for the "You mentioned that X…" follow-up. |
| G11 | `WARM_START_ENABLED` | `false` | `rl.warm_start_enabled` | Session lifecycle | Returning-user Q-table nudge (+0.3 on prior Score-2 dims) + recall-and-resume greeting. |
| G12 | `SESSION_ANALYSIS_ENABLED` | `false` | `rl.session_analysis_enabled` | Session lifecycle | Post-session LLM pass producing warm SUMMARY (spoken) + preferences (silent DB) + safety flags (silent DB). |
| G13 | `SOAP_REPORT_ENABLED` | **`true`** | `rl.soap_report_enabled` | Session lifecycle | Silent SOAP clinical note (Subjective/Objective/Assessment/Intervention) written to DB for clinician handoff. **ON — legacy-parity safe (never spoken).** |
| G14 | `DIMENSION_OPTOUTS_ENABLED` | `false` | `rl.dimension_optouts_enabled` | RL | Per-user `disabled_dim:<label>` preference masking at RL mask-init time. |
| G15 | `SESSION_CAP_ENABLED` | `false` | `rl.session_cap_enabled` | Session lifecycle | 60-minute hard session length cap (tunable via `SESSION_MAX_SECONDS` env) with graceful termination. |
| D3 | `CRISIS_OVERRIDE_ENABLED` | `False` | `therapy_content.py` | Safety | Score-2 on `CRITICAL_DIMS` (`sib`, `safe`, `risk`, `drug`, `alcohol`) short-circuits the RL loop and speaks `SAFETY_RESOURCES_MESSAGE` (988, SAMHSA, 911). Also runs between every CBT stage. |
| D5/D6 | `REPHRASE_AT_RUNTIME` | **`true`** | `rl.rephrase_at_runtime` | Questioner | Paper §5.1 / legacy runtime Rephraser — LLM rewrites ~95% of picked question variants before speaking. **ON — matches legacy's `generate_synonymous_sentences()` behaviour.** |

### How each gate is wired

All gates are consumed via `src.utils.config_loader` module constants (except `CRISIS_OVERRIDE_ENABLED` and `CBT_ESCALATION_ENABLED` which live in `src/core/therapy_content.py` for locality with the safety content they guard). Every call site has a short `if FLAG:` guard and the extension implementation is preserved in full — flipping the flag to `true` re-enables the feature with no other edits.

Grep points for each gate in source:

```bash
# Pipeline-shape gates (questioner)
grep -n "REASK_DIMENSION_N\|MULTI_DIM_BACKFILL_ENABLED\|REFLECTIVE_SUMMARIZER_ENABLED" src/core/questioner.py

# Safety / CBT content gates
grep -n "CRISIS_OVERRIDE_ENABLED\|CBT_ESCALATION_ENABLED" src/core/CBT.py src/core/handler_rl.py

# Session-lifecycle gates
grep -n "WARM_START_ENABLED\|SESSION_ANALYSIS_ENABLED\|SOAP_REPORT_ENABLED\|DIMENSION_OPTOUTS_ENABLED\|SESSION_CAP_ENABLED" src/core/handler_rl.py
```

### Hyperparameters (values, not flags)

These are single-value choices, not toggles — flip between legacy and paper by editing `config.yaml`:

| Knob | Current value | Legacy / Paper reference | Notes |
|---|---|---|---|
| `rl.epsilon` | `1` | legacy prototype (paper: 0.9) | Legacy pure exploitation vs. paper's 10% random exploration. |
| `rl.alpha` | `0.5` | legacy prototype (paper: 0.1) | Legacy aggressive Q-updates vs. paper's conservative learning rate. |
| `rl.gamma` | `0.9` | paper + legacy | Both agree. |
| `rl.item_importance` | legacy `[0, 5, 98, 99, 5, ..., 97, ...]` with 97/98/99 pins on mood/medication/eat | legacy prototype (demo) | Legacy sentinel pins force the top-3 screening triage order (demo shape). A therapist-rebalanced 1-10 alternate is kept commented in `config.yaml`. |
| `rl.reward_mode` | `mean` | paper §5.1 + legacy | `hybrid = (max+mean)/2` is a safer-revisit research alternate. |
| `rl.rephrase_probability` | `0.20` | legacy prototype: 0.95 | Tuned down for GPU-backend latency. Each Rephraser call costs ~1.6 s on Jetson GPU; at 0.20 the impact is ~0.3 s/turn averaged across the session. Set to 0.95 to restore legacy wording frequency. |
| `LITERT_MAX_TOKENS` | `3072` | n/a | Lifted from 512 so Gemma can emit the demo's paragraph-length Validator and Stage-1 Guide outputs; tightened from an earlier 4096 to save KV-cache on the Jetson GPU's unified memory pool (RV_VALIDATOR worst-case lands at ~2.2 k tokens). |
| `LITERT_BACKEND` | `gpu` | n/a | ML Drift GPU path on Jetson Orin Nano is 4–8× faster per LLM call than the CPU (XNNPack) fallback. Whisper stays on CPU to avoid contention for the same unified-memory pool. |

### Permanent fixes (not gated)

These are pure bug fixes or legacy restorations — no flag, no alternate code path:

| ID | File | What it does |
|---|---|---|
| G3 | `src/core/handler_rl.py` | Greeting raw seed restored to legacy's three-sentence text so the LLM rewrite produces the demo's warm opening ("Hi, I'm Caiti, and I'm here to support you…"). |
| G4 | `src/core/questioner.py` | On RV decision=0 the Validator's output returned by `rv_consolidated()` is reused directly, eliminating the duplicate `rv_validator_mi()` call (halves RV hot-path LLM latency). |
| G6 | `src/core/CBT.py` | CBT Stage 0 spoken wording restored verbatim from legacy ("you have issue in:", "Which dimension would you like to work on today?", "Tell me the dimension number. For example: 1"). |
| G10 | `src/core/handler_rl.py` | When `SESSION_ANALYSIS_ENABLED=true`, SUMMARY is written by the LLM as 1–2 warm spoken sentences for the user, while PREFERENCES and SAFETY_FLAGS remain structured clinician artifacts. |
| D1 | `src/core/CBT.py` | `_parse_decision` uses legacy substring semantics (`"0" if "0" in raw else "1"`) so Gemma replies without the literal `DECISION:` header still parse correctly. |
| D2 | `src/core/handler_rl.py` | PHQ-4 pre-screen removed from the handler. The 4 PHQ/GAD items are delivered by `SpeechInteractionService`'s intermission ladder as latency fillers (each fires at most once per session). |
| D4 | `src/core/handler_rl.py` | `_load_longitudinal_state` boost reduced from 3.0 → 0.3 and the "first 2 turns force-target prior Score-2 dim" bypass of ε-greedy removed. (Only active when `WARM_START_ENABLED=true`.) |
| D7 | multiple | Log taxonomy matches the demo: heartbeat, per-call LLM request logs, and `log_reasoning` INFO lines demoted to DEBUG. |

### Re-enablement checklists

Before flipping a gate on for a live session or trial, follow the checklist for that feature:

**G5 — `reask_dimension_n`** — Demo never re-asks verbatim; enabling this adds a user-visible "I missed that, let me ask again" moment for off-topic answers. No safety implication, purely UX.

**G7 — `CBT_ESCALATION_ENABLED`** — Must coordinate with the study PI. The 988/SAMHSA script is spoken *only* on 3 failed CBT attempts, so false-positive exposure is low, but the recording is a regulated clinical script and the PI should approve its exact wording.

**G8 — `multi_dim_backfill_enabled`** — Already ON (session-11 fix). Silent DB-level change — the user hears the same screening questions; the only effect is that the CBT Stage 0 candidate list may include dims the user mentioned incidentally (e.g. scoring `emo` Score-2 from a long answer to a `mood` question). To tighten for strict demo parity (single asked-dim-only scoring) flip to `false`. Current trade-off: the `_apply_multi_dim_updates` LLM pass is gated on utterance length (≥20 tokens, ≥2 segments) so the extra inference cost only fires on substantive replies.

**G9 — `reflective_summarizer_enabled`** — Adds one LLM call per Score-2 RV follow-up. Verify latency budget on the Jetson before enabling; the demo's "You mentioned that X. Can you tell me more?" wording is preserved by the legacy `generate_change()` regex path at zero LLM cost.

**G11 — `warm_start_enabled`** — Requires DB-backed `rl_state` rows. Safe to enable once at least one completed session exists for the subject; first-session users get a no-op. Check the greeting prompt still speaks naturally when the recall hint is auto-filled.

**G12 — `session_analysis_enabled`** — Adds one post-session LLM call (warm SUMMARY, preferences, safety flags). Gemma output must be validated to match the spoken-SUMMARY prompt's warm tone — the system prompt is explicit but Gemma-4-E2B's adherence is weaker than GPT-4's.

**G13 — `soap_report_enabled`** — Already ON. DB-only (never spoken to user). Ensure the SQLite DB path (`data/therapist.db` by default) lives on a HIPAA-compliant filesystem for the trial. The SOAP note is stored in the `summaries` table keyed by `session_id`.

**G14 — `dimension_optouts_enabled`** — Requires the `disabled_dim:<label>` preferences to already exist in the DB for the subject. Safe to enable any time; becomes a no-op for any subject without opt-outs.

**G15 — `session_cap_enabled`** — **Strongly recommended for clinical trials.** Protects against a stuck LLM holding a participant on-device indefinitely. Re-enable along with setting `SESSION_MAX_SECONDS` to the trial's per-visit time budget.

**D3 — `CRISIS_OVERRIDE_ENABLED`** — Higher bar. Before enabling:
1. FP rate of the DLA Analyzer on `CRITICAL_DIMS` (`sib`, `safe`, `risk`, `drug`, `alcohol`) must be measured < 2% on a held-out set.
2. `SAFETY_RESOURCES_MESSAGE` (988, SAMHSA, 911) content must be reviewed by the study PI.
3. The safety delivery audit trail (DB `safety_deliveries` + file fallback at `data/safety/`) must be verified to actually land in the trial deployment environment.
4. No double-delivery: mid-CBT crisis fires the callback exactly once per `_crisis_dims_handled` entry.

**D5/D6 — `rephrase_at_runtime`** — Already ON (legacy parity). Legacy prototype calls `generate_synonymous_sentences()` on ~95% of screening turns, so the demo recording varies wording from session to session. If disabling for a latency-sensitive deploy, validate that the pre-generated `question_synthetic[]` pool covers the turns you'd otherwise rewrite.

### Future development — beyond what is gated

Things that would require new code, not just a flag flip:

- **Per-role model routing.** The `ROLE_MODEL_MAP` in `src/models/llm_client.py` currently points every paper role (ANALYZER, REPHRASER, RV_REASONER, RV_VALIDATOR, CBT_REASONER, CBT_GUIDE, GENERAL, etc.) at the same on-device Gemma instance. When a larger model (local or cloud) is available for the heavy reasoning roles, splitting the map is the single change required — every call site already passes `role=LLMRole.X`.
- **PHQ-9 / GAD-7 extensions.** The intermission ladder today ships only the PHQ-4 / GAD-2 subset. Full-length instruments would live in `src/core/therapy_content.py::CLINICAL_SCREENING` and the ladder selector would pick them on sessions where the clinician has enabled the extended screen.
- **Offline LLM fallback.** When `litert_lm.Engine` throws repeatedly (see `engine_is_healthy()` in `src/models/llm_client.py`), the current behaviour is to raise `LLMError` and mark the turn SKIPPED. A future fallback path would route to a smaller cached model or a scripted decision tree so a hardware hiccup doesn't end the session.
- **Multi-language.** Every clinical prompt is currently English-only. Translation would change both the STT/TTS models (Whisper supports many; Piper has several voices) and the question/prompt libraries.
- **Clinician dashboard.** SOAP notes land in `DB.summaries` today but there is no browse UI. A read-only FastAPI endpoint plus a static page would close that loop.

---

## FastAPI Endpoints

Runs on `:8000` in parallel with the speech loop. Useful for remote monitoring, a web UI, or CI integration tests.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | `{status, subject_id, session_id}` |
| `GET` | `/api/turns` | Full dialogue history for the active session |
| `GET` | `/api/output` | Non-blocking poll of the next agent utterance |
| `POST` | `/api/login` | `{"user_id": "new_user"\|"test_user"}` — creates session, sets START event |
| `POST` | `/api/input` | `{"text": "..."}` — pushes text onto INPUT_QUEUE (same slot STT uses) |
| `POST` | `/api/intent` | `{"text": "..."}` — one-word START/END/NONE intent (LLM router) |
| `POST` | `/api/action` | `{"type": "stop"\|"start_listening"\|"set_mode", ...}` |
| `POST` | `/api/pause` / `/api/resume` | Pause / resume the speech loop |
| `POST` | `/api/end_session` | Immediate termination |

The legacy Flask SERVER mode (CSV-polling `/gpt` endpoint) has been removed — it duplicated the FastAPI path and was unused in practice.

---

## Running the System

### Prerequisites

- **Hardware:** NVIDIA Jetson Orin Nano (or any Linux PC with mic + speakers for dev).
- **OS packages:** `portaudio19-dev`, `python3-venv`, `piper-tts` binary (or espeak-ng fallback).
- **Python:** 3.10+.

### Local install

```bash
sudo apt-get install portaudio19-dev python3-venv espeak-ng

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fetch Gemma 4 E2B LiteRT model (~2.6 GB)
python scripts/model_fetch.py
```

### Ambient-music asset (staged manually)

The long-form ambient track is **not** in git (git-ignored to keep the repo small). Drop a licensed/royalty-free ambient track at `assets/audio/ambient_therapy.mp3` — anything from a few minutes to several hours works; duration is probed automatically. If the file is absent, `BackgroundMusicThread` falls back to the 2 min `assets/audio/waiting_music.wav` clip already in the repo. Random-segment playback still works on the fallback but is less varied because the track is short.

### One-command start (laptop or Jetson)

```bash
./start_therapist.sh
```

`start_therapist.sh` is the **single launcher** used for every real-session run. It auto-detects the host and behaves differently on each:

- **On a laptop** (no `/etc/nv_tegra_release` file): rsyncs the code to the Jetson, ssh'es in, kills any stale process, and re-invokes itself on the Jetson in Jetson mode. Ctrl-C here detaches the laptop terminal; the remote process keeps running.
- **On the Jetson**: activates `.venv`, sources `.env`, exports the clinician-view log env + native-library noise suppressors (`TF_CPP_MIN_LOG_LEVEL=3`, `GLOG_minloglevel=2`, `GRPC_VERBOSITY=ERROR`, `ABSL_LOG_LEVEL=ERROR`), and launches `main.py` with **split stream routing**:
  - **stdout** → console + `data/logs/therapist_<ts>.log` (clean clinician trace)
  - **stderr** → `data/logs/therapist_<ts>.stderr.log` only (native C++ noise from LiteRT / XNNPack / TensorFlow Lite, LLM prompt dumps, absl `I0000 ...` rows)

The clinician console therefore shows only clinical events (RL weights, module I/O, screening turns, CBT stage progression, every spoken utterance, warnings and above). The `.log` file captures the same clean trace with full DEBUG detail from Python; the `.stderr.log` file captures every byte the native runtime produced so forensic debugging still has the raw engine logs.

Force a specific mode with `THERAPIST_MODE=jetson ./start_therapist.sh` (for a non-Tegra dev host that you want to run locally) or `THERAPIST_MODE=laptop` (for an SSH-over-VPN loop from a Jetson dev kit).

Flags mirrored from `laptop_deploy.sh`:
- `SKIP_SYNC=1 ./start_therapist.sh` — no code changed, skip rsync
- `SKIP_KILL=1 ./start_therapist.sh` — Jetson is idle, skip the pre-launch kill
- `CONSOLE_LOG_LEVEL=DEBUG ./start_therapist.sh` — verbose debug on console (propagates over the ssh hop)
- `CLINICIAN_LOG_MODE=0 ./start_therapist.sh` — turn the clinician filter off (every logger lands on console; noisy). Also restores legacy stderr→console routing so native C++ logs return to the terminal.
- `CLINICIAN_LOG_TAGS="[DOSSIER],[LITERT]" ./start_therapist.sh` — extend the accepted tag set for a specific trial day without editing code.
- `tail -f data/logs/therapist_<ts>.stderr.log` — inspect native LiteRT/TensorFlow/XNNPack output if the clinician console shows something odd and the forensic file is ambiguous.

### Run locally on any dev machine (no Jetson, pure local)

```bash
# Force Jetson-mode on a laptop to run main.py directly
THERAPIST_MODE=jetson ./start_therapist.sh
```

Or raw (no banner, no clinician view):

```bash
python main.py
```

### Clinician-view console (tag-based filter)

When `CLINICIAN_LOG_MODE=1` (the default under `start_therapist.sh`), the console filter accepts an `INFO` line **only** if its message begins with one of these clinical-event tags. Anything else is still written to the file log.

| Tag | Subsystem | What it marks |
|---|---|---|
| `[SESSION]` | lifecycle | Session start/end, subject identity binding, resume semantics |
| `[GREETING]` | handler | Opening greeting rewrite + first-question handoff |
| `[PIPELINE]` | cross-cutting | Explicit stage entry (Response Analyzer, RV, CBT, etc.) |
| `[RL]` | Q-learning | State, action chosen, Q-value, Q-update, reward |
| `[DLA]` | Response Analyzer | (Dim, Score) classification of a user utterance |
| `[RV]` | Reflection-Validation | Reasoner decision, Guide redirect, Validator output |
| `[CBT]` | CBT protocol | Dimension pick, stage progression, success/failure |
| `[QUESTIONER]` | questioner | `ask_question` / `evaluate_result` milestones |
| `[INTERMISSION]` | speech service | Screening/breathing/music activity transitions |
| `[USER]` | I/O | User transcript (redacted if `REDACT_PII=1`) |
| `[AGENT]` | I/O | Handler-driven clinical turn queued to speech (via `log_question`) |
| `[TTS]` | speech service | Every utterance that actually becomes audio — onboarding greetings, bridge phrases, breathing scripts, goodbye lines, fallback beeps. Deduped against the immediately preceding `[AGENT]` line so handler-driven turns aren't logged twice. |
| `[SAFETY]` | crisis layer | `SAFETY_RESOURCES_MESSAGE` delivery (only if `CRISIS_OVERRIDE_ENABLED`) |
| `[PHQ4]` | screening | PHQ-4 / GAD-2 answer, scoring, clinical flag |
| `[SCORE]` | questioner | Per-dimension score write to `question_lib` |
| `[CLOSING]` | handler | Session-end farewell + SOAP persistence event |

**`WARNING` and above always pass** regardless of tag — clinicians still see `ResourceAudit` pressure warnings, `DBManager` write failures, `GPIOManager` hardware-missing warnings, etc. Warnings appear prefixed with `! WARNING:` so they stand out in an otherwise clean trace.

**Section banners** are auto-emitted when the active tag transitions into a new clinical stage — e.g. `[SESSION START]`, `[INITIALIZATION & RL SEEDING]`, `[SCREENING LOOP]`, `[NEXT TURN SELECTION]`, `[INTERMISSION & PHQ-4 GATING]`, `[RESPONSE ANALYSIS & SCORING]`, `[CBT STAGE 1: RECOGNIZE]`, `[SESSION CLOSING]`. Banners are global to the console handler so cross-logger transitions (HandlerRL → CBT → RL) still get the right header. The banner logic lives in `src/utils/log_util.py::_section_for`.

**Format** is minimal — no timestamp, hostname, logger name, or PID in the clinician console; the file log retains the full `YYYY-MM-DD HH:MM:SS host name[pid] LEVEL ...` format for forensic replay. Implemented by `_ClinicianConsoleFormatter` in `src/utils/log_util.py`.

**Native C++ noise separation.** Gemma runs through the compiled LiteRT-LM engine, which writes its own glog-style lines (`I0000 ...`, `INFO: [...]`, the tokenised prompt dumps, XNNPack warnings) directly to the process's file descriptor 2 (stderr) — Python's logger can't filter those. `start_therapist.sh` handles this by:
1. Setting `TF_CPP_MIN_LOG_LEVEL=3`, `GLOG_minloglevel=2`, `GRPC_VERBOSITY=ERROR`, `ABSL_LOG_LEVEL=ERROR` to silence most of it at the source.
2. Redirecting whatever still prints to `data/logs/therapist_<ts>.stderr.log` via `2> >(tee -a stderr.log >/dev/null)`, so the clinician console never sees it. The stderr log stays on disk beside the main log for any forensic debug that needs the raw runtime view.

Extend the tag vocabulary for a specific trial via `CLINICIAN_LOG_TAGS="[CUSTOM], [EXTRA]"` (comma-separated; brackets and uppercasing are normalised for you).

### Boot-time checklist

`main._startup_checklist()` prints a colour-coded audit on startup:
- GPIO init state + pin numbers
- LiteRT model file exists + size
- Faster-Whisper configured
- Piper model present
- SQLite connectivity

A `_ghost_hunt(50MB)` sweep flags any child process eating > 50 MB RSS.

### Latency audit harness

When you want to measure real per-module latency on whatever box you're on (laptop dev or Jetson live):

```bash
.venv/bin/python scripts/_audit_latency.py
```

Prints warm-call timings for STT, TTS, LLM (short prompt), and each **real production prompt** (REPHRASER, ANALYZER, MULTI-DIM, RV_REASONER, RV_VALIDATOR, RV_GUIDE) — plus simulated Score-0/1 and Score-2 turn totals with sum-of-parts vs. end-to-end overhead. On the live Jetson with `LITERT_BACKEND=gpu`, a Score-0/1 turn currently lands at ~10 s and a Score-2 RV turn at ~14 s. Re-run any time you change `.env` / `config.yaml` to see the latency impact before a trial session.

The `scripts/_build_latency_report_pdf.py` helper renders a paginated per-module + per-session latency PDF (Max session walkthrough, critical-path breakdown, before/after config table) to `~/Downloads/CaiTI_Latency_Report_Max.pdf`. Requires `reportlab` (dev-only, install with `pip install reportlab` — not a runtime dependency).

---

## Jetson Deployment

### Scripts at a glance

| Script | Runs on | One-line purpose |
|---|---|---|
| **`start_therapist.sh`** | laptop **or** Jetson | ⭐ Everyday launcher — dual-mode, auto-detects host. On laptop: sync → remote kill → remote Jetson launch. On Jetson: banner + clinician-view console + tee'd logs. |
| `laptop_sync.sh` | laptop | `rsync` code + assets to Jetson (preserves models, DB, logs, venv) |
| `laptop_pull.sh` | laptop | Pull therapist.db + sessions/ + logs/ + results/ + q_tables/ Jetson → `pulled_data/latest/` (single mirror; non-destructive on Jetson) |
| `laptop_deploy.sh` | laptop | *(legacy)* sync → remote kill → remote `jetson_run.sh`. Superseded by `start_therapist.sh` which also gives you the clinician-view console. Still shipped for non-CaiTI scripted automation. |
| `jetson_setup.sh` | Jetson | One-time env bootstrap — apt deps, venv, pip, Piper voice, LiteRT fetch |
| `jetson_kill.sh` | Jetson | Hard kill all CaiTI processes + free ports 8000/8001/8080; detects D-state survivors |
| `jetson_run.sh` | Jetson | *(legacy)* Launch `main.py` directly with file-level logs. Kept because `start_therapist.sh`'s Jetson branch still SSHes through it when invoked with `THERAPIST_MODE=legacy`, and for CI. |

Required env (from `.env`, read automatically):

```env
JETSON_HOST=user@1.2.3.4              # required for laptop-mode start_therapist.sh
JETSON_PROJECT_DIR=~/project          # default
JETSON_PASSWORD=...                   # optional: used only by jetson_setup for sudo apt
```

### First-time bring-up (fresh Jetson)

```bash
# On the laptop — push the code over
./laptop_sync.sh

# On the Jetson — one-time environment setup
ssh $JETSON_HOST
cd ~/project
./jetson_setup.sh
exit

# Back on the laptop — launch (dual-mode; auto-detects laptop here)
./start_therapist.sh
```

### Everyday dev loop

```bash
./start_therapist.sh        # sync + remote kill + remote launch + clinician view
```

Flags:
- `SKIP_SYNC=1 ./start_therapist.sh` — no code changed, skip rsync
- `SKIP_KILL=1 ./start_therapist.sh` — Jetson is idle, skip the pre-launch kill
- `CONSOLE_LOG_LEVEL=DEBUG ./start_therapist.sh` — verbose debug on console
- `CLINICIAN_LOG_MODE=0 ./start_therapist.sh` — turn the clinician filter off

Ctrl-C detaches from the laptop terminal; the remote Jetson process keeps running. Reattach with `ssh $JETSON_HOST 'tail -f ~/project/data/logs/therapist_*.log'`.

### Running directly on the Jetson

```bash
ssh $JETSON_HOST
cd ~/project
./jetson_kill.sh            # hard stop any active session
./start_therapist.sh        # launches in Jetson mode (auto-detected)
```

### Harvesting clinical data

```bash
./laptop_pull.sh            # single-mirror: rsyncs into ./pulled_data/latest/ in place
```

Always non-destructive on the Jetson — it remains the source of truth, and can keep running a session while the pull rsync completes. On the laptop side the script maintains **one folder** (`pulled_data/latest/`) that always reflects the Jetson's current state; per-session filenames (`session_<subject>_<ts>.json`, `Report_<subject>_<ts>.csv`, etc.) guarantee dossiers and CSVs accumulate without collision across pulls. `therapist.db` and per-subject Q-tables are overwritten in place with each pull. First-run migration: if an older `pulled_data/latest` symlink is found, the script seeds the new real `latest/` directory from the symlink's target and replaces the symlink — no re-download required.

### One-command kill

```bash
ssh $JETSON_HOST ~/project/jetson_kill.sh
```

Exit code 0 = clean, 1 = D-state processes survived (reboot required).

---

## Testing

```bash
# Question library invariants (paper: 7-11 variants per dim, 60/40 weighting, Yes/No preserved)
.venv/bin/python -m pytest tests/test_question_lib.py -q

# Full dev suite (pipeline, DB persistence, parsers, RV, CBT decision parsing)
.venv/bin/python -m pytest dev/tests tests -q
```

**Current baseline: 53 tests, green on Jetson (~8 s total).**

Key test files:

- `tests/test_question_lib.py` — every dim has ≥7 variants; synthetic entries have unreviewed provenance; Yes/No polarity preserved; legacy pool keeps 60 % selection probability.
- `tests/test_clinical_safety.py` — **H7 clinical-safety regression suite** (9 tests). Covers crisis-scan triggering, re-entrance, safety message delivery across TTS + file fallback + `safety_deliveries` audit, C5 sentinels, M2 crash recovery, M3 clinical_flags round-trip, H3 bounded queue, H4 atomic turn index.
- `dev/tests/test_clinical_export.py` — **Phase B regression suite** (5 tests). Covers `record_clinical_score` final/attempt propagation, sentinel-skip (-1/-2 stay out of `clinical_scores`), `record_intervention_log` roundtrip, `generate_therapist_report` CSV correctness, report-generator idempotency.
- `dev/tests/test_pipeline.py` — end-to-end screening flow smoke test (includes the Phase A `END_SESSION_EVENT` teardown so the handler daemon doesn't bleed into later tests).
- `dev/tests/test_rv_pipeline.py` — R-V Reasoner → Guide / Validator dispatch.
- `dev/tests/test_cbt_decision_parser.py` — strict `_parse_decision` (no stray-zero false-pass).
- `dev/tests/test_db_persistence.py` + `test_db_extensions.py` — SQLite schema + longitudinal `persistent_rl_state`.
- `dev/tests/test_multi_dim_classifier.py` — strict-JSON multi-dim output parsing.
- `scripts/_test_intermission_cut.py` — covers the output-ready watcher in isolation: fires `llm_done` + `stop_playback_event` within ~100 ms of the output queue becoming non-empty, respects the 6 s minimum-engagement floor, and exits cleanly when signalled to stop. Runs without loading any models.
- `scripts/_test_parallel_turn.py` — covers `_run_parallel_turn` orchestration: happy-path (long utterance → parallel STT + intermission + transcript queued mid-intermission), silence (no frames → no intermission), END command (worker does NOT queue sentinel → handler never sees it), and short-utterance serial fallback (confirms intermission runs AFTER transcription, not during). Both scripts use fake recorder / fake STT stubs so they execute anywhere in ~10 s.

**Before every trial day**, also run the end-to-end dry-run harness:

```bash
.venv/bin/python scripts/dry_run.py
```

It stubs out the real LLM engine, drives scripted user replies through the full pipeline against a scratch SQLite, and asserts every clinical checkpoint (safety delivery, clinical flag, PHQ-4 persistence, turn count, Q-table CSV, crisis checkpoint file). Exit 0 = green; non-zero exits list each failed assertion.

### Demo-parity smoke suite

```bash
dev/smoke_tests/run_all.sh
```

5 test files covering static invariants (flag values, legacy wordings, prompt structure), RL math (Q-table init, `choose_action`, `get_env_feedback`, Q-update formula), questioner pipeline + G4 double-Validator regression, CBT flow with legacy Stage 0 wording, and a full end-to-end `HandlerRL.run()` demo replay with scripted user inputs. **~35 s, 119 assertions, all tests gate-value aware.** See [`dev/smoke_tests/REPORT.md`](dev/smoke_tests/REPORT.md) for detail on what is and is not covered.

---

## Clinical-Trial Operations Playbook

### Pre-trial checklist

1. **Hardware.** Jetson Orin Nano reachable, Piper + Whisper models present, GPIO wiring verified.
2. **Software.** `.venv` healthy, `requirements.txt` satisfied, LiteRT model file ≥ 100 MB at configured path.
3. **Env.** Apply the clinical-trial env block shown in the Configuration section (`CLINICAL_MODE=1`, `REDACT_PII=1`, etc.).
4. **Disk.** At least 1 GB free on the data volume. Clean `data/logs/` and `data/sessions/` retention if approaching quota.
5. **Dry run.** `.venv/bin/python scripts/dry_run.py` — must print `✅ PASSED`.
6. **Smoke test.** One 30-second interactive session with the clinician as test user to confirm TTS, mic, buttons.
7. **DB snapshot.** `cp data/therapist.db data/therapist.db.pre-trial-$(date +%Y%m%d).bak` — provides a known-good rollback point.

### Between participants (Phase C — non-destructive)

The Jetson keeps all historical data; the laptop gets a current archival snapshot after every session.

```bash
# 1) Pull clinical artefacts Jetson → laptop (runs from laptop):
./laptop_pull.sh

# 2) Prepare the Jetson for the next participant (run on Jetson):
ssh $JETSON_HOST ~/project/scripts/new_participant_init.sh 9001    # optional: switch subject_id
# or keep the same subject_id and just reset their Q-table:
ssh $JETSON_HOST ~/project/scripts/new_participant_init.sh
```

`laptop_pull.sh` rsyncs `therapist.db`, `data/sessions/`, `data/logs/`, `data/results/`, and `data/q_tables/` from the Jetson into `pulled_data/latest/` on the laptop — a **single mirror** that accumulates per-session files (each filename already carries the onboarded subject + session timestamp, so no collisions) and overwrites mutating files (`therapist.db`, Q-tables) in place. Non-destructive on the Jetson: nothing there is modified or purged (per operator policy — laptop is the analyst view, Jetson is the source of truth). Keep a Jetson-side DB backup if you require point-in-time DB rollback.

`new_participant_init.sh` archives the per-subject Q-table CSV with a timestamped suffix so the next boot rebuilds from `config.yaml → rl.item_importance` (the therapist-authored empirical priors — matching paper §5.1 "initial Q-values"). `therapist.db` / `sessions/` / `results/` / `logs/` are all preserved for longitudinal analysis.

### During the session

- If the clinician sees the Green LED stuck ON for > 30 s, the LLM is stuck — use the End button (Pin 13) to terminate safely; the `atexit` hook will close the DB session row cleanly.
- If the device plays the tts_fallback.wav alert beep sequence, both TTS engines have failed — end the session, restart, and check Piper + espeak-ng installs.
- If the user says "skip", "stop", or "opt out" during PHQ-4 and the system delivers the 988/SAMHSA message, that is M7 firing on elevated partial scores — treat as a clinical event and follow your IRB protocol.

### Post-session auditor queries

Every clinical event the system generated is inspectable via these SQL queries:

```sql
-- Were all sessions closed cleanly?
SELECT end_reason, COUNT(*) FROM sessions GROUP BY end_reason;

-- Every CRITICAL_DIM flag MUST have a matching successful delivery
SELECT cf.session_id, cf.flag_type, sd.method, sd.success, sd.created_at
FROM clinical_flags cf
LEFT JOIN safety_deliveries sd ON cf.session_id = sd.session_id
WHERE cf.flag_type = 'CRITICAL_DIM_SCORE_2';

-- Any safety deliveries that FAILED on any method
SELECT * FROM safety_deliveries WHERE success = 0;

-- Any PHQ-4 questions the STT couldn't resolve (Phase A: status is now UNRESOLVED,
-- distinct from user-initiated SKIPPED):
SELECT session_id, question_id, status, reason, response_text
FROM intermission_screening
WHERE (status = 'UNRESOLVED' AND reason = 'stt_unresolved')
   OR (status = 'SKIPPED'    AND reason = 'stt_unresolved');

-- Phase B: final scored dimensions per session (crisis = 2):
SELECT s.id AS session_id, u.subject_id, cs.dim_label, cs.score, cs.evidence_text
FROM clinical_scores cs
JOIN sessions s ON cs.session_id = s.id
JOIN users u ON s.user_id = u.id
WHERE cs.score = 2
ORDER BY s.id DESC;

-- Phase B: full MI + CBT timeline for one session:
SELECT kind, stage, technique, outcome, dim_label, created_at
FROM intervention_logs
WHERE session_id = :session_id
ORDER BY id;

-- PHQ-4 / GAD-2 longitudinal trends per participant
SELECT u.subject_id, cs.anxiety_score, cs.depression_score, cs.phq4_total,
       cs.gad2_positive, cs.phq4_high_risk, cs.created_at
FROM clinical_screening cs
JOIN sessions s ON cs.session_id = s.id
JOIN users u ON s.user_id = u.id
ORDER BY u.subject_id, cs.created_at DESC;

-- Full clinical-flag audit for one subject across all sessions
SELECT cf.flag_type, cf.details_json, cf.created_at, s.end_reason
FROM clinical_flags cf
JOIN sessions s ON cf.session_id = s.id
JOIN users u ON s.user_id = u.id
WHERE u.subject_id = :subject_id
ORDER BY cf.created_at DESC;
```

### Therapist-facing export (Phase B)

For a session-level summary without writing SQL, open `data/clinical/clinical_report_{SUBJECT}_{SESSION}.csv`. Auto-generated on every clean session close. Structure:

1. Crisis rows (Score = 2 dimensions) at the top so heightened-attention items land first.
2. One `ScoredDim` row per dimension with the final score + attempt trail.
3. MI + CBT `Intervention` rows (stage, technique, outcome) in chronological order.
4. `SafetyDelivery` audit rows proving each crisis-message broadcast.
5. Raw `Turn` dialogue last.

To regenerate for an existing session (e.g. after importing a DB):

```python
from src.drivers.db_manager import DBManager
from src.utils.therapist_report import generate_therapist_report, _EXPORTED_SESSIONS

db = DBManager("data/therapist.db")
_EXPORTED_SESSIONS.clear()   # bypass the in-memory run-once guard
generate_therapist_report(session_id=42, db=db)
```

### Crisis file-fallback artefacts

Inspect `data/safety/crisis_<subject>_<session>_<dim>_<timestamp>.txt` — each file contains the exact safety message delivered, the triggering critical dim, and any TTS errors at the time of delivery. These are plain-text and can be shared with on-call clinicians directly.

### Incident response

| Symptom | First action | Where to look |
|---|---|---|
| Session didn't close cleanly | Next boot auto-recovers via M2 (`end_reason='crash_recovery'`) | `clinical_flags WHERE flag_type='CRASH_RECOVERY'` |
| No TTS heard by user | Check for tts_fallback.wav being played | `backend_session.log` for `[TTS Failure]` lines |
| Crisis resources not audibly delivered | **File fallback is authoritative** — inspect `data/safety/crisis_*.txt` | `safety_deliveries` rows for the session |
| LLM stuck | H2 auto-invalidates after 3 failures; handler raises `LLMError` | `backend_session.log` for `[LiteRT]` errors |
| Participant session exceeded 60 min | Auto-terminated with `SESSION_CAP_REACHED` flag | `clinical_flags WHERE flag_type='SESSION_CAP_REACHED'` |

---

## Key Technologies

| Component | Implementation | Purpose |
|---|---|---|
| STT | `faster-whisper` (base.en, int8, beam=2) | Local low-memory transcription |
| TTS | `piper` (en_US-amy-medium) + `espeak-ng` fallback | Neural / fallback speech synthesis |
| LLM | Gemma 4 E2B via `litert-lm-api` | In-process per-task inference on Jetson |
| VAD | `webrtcvad` | Voice activity detection |
| RL | Q-learning in `pandas.DataFrame` | Clinical dimension selection |
| Screening | PHQ-4 + GAD-2 | Standardised anxiety + depression triage |
| Intermission | `IntermissionLadderManager` | PHQ / breathing / music during LLM waits |
| Audio bed | `pygame.mixer` | Ambient music with auto-duck on speech |
| API | FastAPI + Uvicorn | Remote control / monitoring |
| Persistence | SQLite + CSV + NDJSON | Paper-compatible Q-table + rich session dossier |
| Hardware | `Jetson.GPIO` (stub on dev machines) | Buttons + LED |

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{10.1145/3712299,
  author  = {Nie, Jingping and Shao, Hanya (Vera) and Fan, Yuang and Shao, Qijia
             and You, Haoxuan and Preindl, Matthias and Jiang, Xiaofan},
  title   = {LLM-based Conversational AI Therapist for Daily Functioning Screening
             and Psychotherapeutic Intervention via Everyday Smart Devices},
  year    = {2025},
  journal = {ACM Trans. Comput. Healthcare},
  doi     = {10.1145/3712299},
  url     = {https://doi.org/10.1145/3712299}
}
```

---

## License

Research / educational use. Please check with the repository owner before commercial use. Clinical deployment requires licensed clinician oversight — this is a research prototype, not a replacement for professional mental-health care.
