# CaiTI clinical-trial benchmark — LIVE Jetson run

Date: 2026-05-23
Hardware: Jetson Orin Nano (`caiti-jetson` @ 152.23.125.212), Gemma-4-E2B via LiteRT-LM
Driver: Claude (live, in-character) via file-based reply protocol
TTS / STT / GPIO: stubbed (handler-only loop)
LLM / DB: REAL on-device

This run replaces the previous offline benchmark with a true on-device
session.  Three personas were played by Claude in character, with
each user reply written by hand based only on the agent's spoken
turn — no peeking at dim labels, no scripted phrasing.

**Each persona was run on BOTH backends** (CPU first, then GPU after
diagnosing BUG #1 as VRAM exhaustion from a running Chromium desktop).
The GPU re-runs reflect what a real headless `jetson_run.sh`
deployment would experience.

## TL;DR — clinical-trial readiness verdict

**NOT READY for clinical trials.**

Three new bugs found that were invisible in the LLM-stubbed offline
benchmark.  Two of them (BUG #2 and BUG #7) are **clinical-data
integrity** issues that produce **fabricated or silently-dropped
scores** and would land in the clinician's dossier as authoritative
findings.  See `BUGS_observed.md` for full reproduction details.

| Bug | Severity | One-line description |
|-----|---------:|----------------------|
| #1  | medium   | GPU LiteRT `[Device] is lost` on cold start; CPU backend works |
| #2  | **HIGH**     | `[ANALYZER_DEAD]` silently drops Score-2 disclosures |
| #3  | **HIGH**     | `generate_change_negative` leaks LLM scaffolding ("Option 1", "**bold**") to user on Score-2 follow-ups |
| #4  | medium   | Legacy `generate_change` regex produces ungrammatical follow-ups ("you have a mess sleep") |
| #5  | medium-high | Real Gemma-CPU latency exceeds 30-min watchdog for full 37-dim screening |
| #6  | high     | Same as #3 — `Answer:` prefix variant, confirms #3 is general |
| #7  | **CRITICAL** | Brainstorm scaffold (G16, the therapist-feedback fix) **NEVER fires in production** — Analyzer LLM eats "I don't know" and fabricates a score |

## Persona summary

### Persona A2 — `mid_depression` (CPU re-run)

| Metric | Value |
|--------|-------|
| Profile | 28yo grad student, low mood + early wakings + skipped meals + lapsed therapy |
| Agent turns | 12 (then watchdog timeout at 30 min) |
| User turns | 11 |
| Duration | 1803 s (full LLM time: ~490 s, 27%) |
| Engine errors | 0 (CPU backend) |
| Recorded scores | medication=0, mood=2, eat=2, sleep=2, care=2, weight=1, work_dayoff=1, nutrition=0 |
| Reached CBT? | **No** — watchdog tripped during screening |
| Bugs surfaced | #1, #2, #3, #4, #5, #6 |

**LLM call distribution:**

| Role | n | avg s | total s |
|------|--:|------:|--------:|
| general | 1 | 44.1 | 44.1 |
| rephraser | 4 | 10.9 | 43.5 |
| analyzer | 16 | 15.8 | 253.3 |
| rv_reasoner | 4 | 8.2 | 32.9 |
| rv_validator | 4 | 28.0 | 112.0 |
| reflective_summarizer | 2 | 2.4 | 4.8 |

**Highlights (positive):**
- Real Gemma-CPU correctly classified all 7 substantive replies into
  the right (dim, score) tuple.  Score-2 detection on mood, eat, sleep,
  care all worked.
- Real RV Validator output was excellent — 4-5 sentence MI-style
  reflections with concrete suggestions.  Matches the paper's design
  intent.  Example:
  > "It sounds like you are carrying a really heavy weight right now,
  > and the feeling that it doesn't lift is incredibly draining…
  > You might try focusing on very small, easy-to-manage food choices…"

**Highlights (negative):**
- BUG #3 fired on the Score-2 mood follow-up — user heard markdown
  "Option 1 / Option 2 / Option 3" scaffolding immediately after
  disclosing depression.  This is the most clinically harmful bug
  observed.
- Watchdog tripped at agent turn #12 — full 37-dim screening + CBT
  is not feasible on CPU within 30 min.

### Persona B / B2 — `stuck_uncertain`

| Metric | Value |
|--------|-------|
| Profile | "I don't know" replies until brainstorm scaffolds the user in |
| Agent turns | B: 2, B2: 4 |
| Outcome | **G16 brainstorm scaffold did NOT fire on any "I don't know" reply.** |
| Recorded (false) scores | medication=1, mood=0, medication=0 — fabricated by Gemma Analyzer |
| Bugs surfaced | **#7** (critical) |

The persona said "I don't know" three times across the two B-runs:

| Turn | User reply | Expected | Actual |
|------|-----------|----------|--------|
| B  #1 | "I don't know." | Maybe → retry_guide → brainstorm scaffold | `(medication, 1)` — fabricated Score-1 |
| B2 #1 | "I don't know." | Maybe → retry_guide → brainstorm scaffold | `(mood, 0)` — fabricated Score-0 |
| B2 #2 | "I don't know what to say to that question." | Maybe → retry_guide → brainstorm scaffold | `(medication, 0)` — fabricated Score-0 |

**The therapist-requested fix is dead code in production.**  The
Analyzer LLM consistently classifies "I don't know" into a substantive
(dim, score) tuple, never reaching the `Maybe` branch where the
brainstorm scaffold would fire.

### Persona C — `stop_early`

| Metric | Value |
|--------|-------|
| Profile | Answer one dim, then "no more questions" |
| Agent turns | 3 |
| User turns | 2 |
| Duration | 5 min 45 s |
| Outcome | **Soft-end intercept WORKS as designed.** |

Trace:

```
[USER #02] Honestly, no more questions, I would like to stop.
[ResponseBridge] [SOFT-END] Matched soft-end intent in user utterance; returning (mood, Stop).
[Questioner] [DLA] Classification result: [('mood', 'Stop')]
[Questioner] [QUESTIONER] User said 'Stop' — terminating screening.
[AGENT #03] We do not have a dimension at score 2 to work on today. We will conclude here.
```

The soft-end pre-LLM regex catches "no more questions" without an
Analyzer round-trip.  Screening terminates cleanly, no Score-2 → no
CBT → legacy "no concern identified today" closing.  **Paper §5.1
behaviour preserved.**

This is the **canonical example** of how the don't-know intercept
should work (BUG #7's recommended fix mirrors this exact pattern).

## What the live run validated (the good)

1. **CPU backend is stable.**  Zero LLMError exceptions across 11+
   real Gemma calls in P-A2.  GPU backend currently broken (BUG #1) —
   recommend `LITERT_BACKEND=cpu` for trials.
2. **Response Analyzer scoring on substantive replies is accurate.**
   All 7 of P-A2's clinical-content replies got the right (dim, score)
   tuple.  Mood / eat / sleep / care all correctly Score-2.  No
   misclassification of substantive answers.
3. **RV pipeline works end-to-end.**  Reasoner DECISION: 0 (on-topic),
   Validator OARS reflections were genuinely empathic and clinically
   useful.  Three full RV cycles fired in P-A2.
4. **Soft-end intent intercept (paper §5.1) works as designed.**  P-C
   demonstrated the canonical clean-stop path.
5. **Rephraser produces reasonable variants.**  All 4 P-A2 questions
   rephrased into clinically equivalent forms.

## What the live run broke (the bad)

1. **G16 brainstorm scaffold never fires** (BUG #7).  The whole
   therapist-feedback fix is dead code.  Critical for trials.
2. **Score-2 follow-ups expose LLM scaffolding to user** (BUG #3, #6).
   Markdown bullets and "Option N" labels appear at the highest-stakes
   moments of the conversation.
3. **Engine death silently drops disclosures** (BUG #2).  GPU+CPU both
   risk this; must be fixed before trials.
4. **30-min watchdog cannot cover full session on CPU** (BUG #5).
   Either LITERT_MAX_TOKENS / context tuning is needed, or the
   intermission ladder must be validated with real audio to mask
   the latency.

## CPU vs GPU comparison (same persona A)

After the initial CPU run completed, Chromium was killed to free VRAM
and the same persona was re-run on GPU.  All bugs reproduced on both
backends; **GPU is the correct production backend** and is
substantially faster.

| Role | CPU avg | GPU avg | Speedup | CPU total / 1800s | GPU total / 1800s |
|------|--------:|--------:|--------:|-------------------:|-------------------:|
| general (greeting) | 44.1 s | 6.3 s | **7.0×** | 44 s | 6 s |
| analyzer | 15.8 s | 2.1 s | **7.5×** | 253 s | 60 s |
| rephraser | 10.9 s | 2.7 s | **4.0×** | 43 s | 11 s |
| rv_reasoner | 8.2 s | 1.4 s | **5.9×** | 33 s | 6 s |
| rv_validator | 28.0 s | 8.4 s | **3.3×** | 112 s | 33 s |
| reflective_summarizer | 2.4 s | 0.5 s | **4.8×** | 5 s | 1 s |
| cbt_reasoner | n/a (never reached) | 1.4 s | – | – | 1 s |

| Metric | CPU run | GPU run |
|--------|--------:|--------:|
| Agent turns in 30 min watchdog | 12 | **21** |
| Reached CBT? | No | Yes (Stage 1 entered) |
| Total LLM time | ~490 s (27% of session) | ~118 s (6.5% of session) |
| Errors | 0 | 0 (with VRAM free) |

**Bottom line:** GPU is **5-7× faster on average** and handles 75%
more dialogue turns in the same wall time.  The remaining 73-93% of
the session is user reply time, which would shrink dramatically with
a real human user typing/speaking in 5-15 s rather than my 60-90 s
per reply.

**Lesson learned:** my initial diagnosis of "GPU is broken" was wrong.
The GPU was VRAM-starved by Chromium running in the desktop session.
On a clean headless Jetson (which is the production deployment), GPU
loads cleanly and runs fast.

## Recommended fixes before clinical trials (priority order)

1. **CRITICAL**: Add the don't-know pre-LLM intercept in
   `response_bridge.py::get_openai_resp` — see BUG #7 for the
   ~10 LOC fix.  This unlocks G16 in production.  Confirmed
   reproducible on both CPU and GPU backends.
2. **HIGH**: Strengthen `generate_change_negative` — strict
   "output only the rewritten sentence" prompt + post-LLM sanitiser
   (BUG #3 / #6).  Or replace with a non-LLM template.  Confirmed
   reproducible on both CPU and GPU.
3. **HIGH**: When `[ANALYZER_DEAD]` fires, the agent must NOT silently
   advance — speak a re-prompt instead, and abort the session if the
   3-failure budget exhausts (BUG #2).
4. **MEDIUM**: Use `LITERT_BACKEND=gpu` for clinical trials (the
   production default).  Add a deployment-checklist step: kill all
   browser/GUI apps before starting a session so VRAM is unconstrained
   (BUG #1).  Add a startup warm-up `llm_complete` call to detect VRAM
   shortfall early.
5. **MEDIUM**: Even on GPU, validate the intermission ladder in a real
   audio-on Jetson session — Validator calls average 8 s on GPU, which
   the intermission ladder is designed to mask (BUG #5 mitigated by
   GPU but still warrants validation).

## Files

- `jetson_runs/<timestamp>_<persona>/telemetry.json` — full per-run
  telemetry including every agent turn, every user turn, every LLM
  call with role + latency.
- `jetson_runs/<timestamp>_<persona>/live_session.log` — clean
  agent/user transcript.
- `jetson_p_*_stdout.log` — full Jetson stdout incl. logger lines.
- `BUGS_observed.md` (in parent dir) — full reproduction details
  for every bug.
