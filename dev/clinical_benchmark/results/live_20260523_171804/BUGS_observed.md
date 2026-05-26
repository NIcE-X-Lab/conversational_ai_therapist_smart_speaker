# Bugs / quirks observed during live Jetson runs

Captured incrementally as Claude (acting as live user) observes them
during the three-persona benchmark.

---

## #1 — GPU LiteRT engine: `VK_ERROR_OUT_OF_DEVICE_MEMORY` when VRAM contended

**ROOT CAUSE (corrected after re-investigation):** The `[Device] is lost`
error was a downstream symptom.  The actual root cause was Vulkan VRAM
exhaustion at model load:

```
E0000 environment.cc:108] Out of memory: vkAllocateMemory failed with
VK_ERROR_OUT_OF_DEVICE_MEMORY - While calling [Device].CreateTexture(...)
```

The Jetson Orin Nano has 8GB unified memory shared between CPU and GPU.
At the time of the failed runs, **Chromium had 7+ renderer/GPU
processes resident** consuming ~700MB+ of memory and competing with
the LiteRT WebGPU allocator for VRAM.  Gemma-4-E2B needs roughly 3GB
of GPU memory for weight textures; with the desktop running, that
budget was unavailable.

**Verification:** killed Chromium (`pkill -9 -f chromium`), freed
~600MB → 5.7GB available → re-ran `LITERT_BACKEND=gpu` → **GPU loaded
cleanly, zero errors, model latency dropped 7-9× compared to CPU.**

| Role | CPU (P-A2) | GPU (retry) | Speedup |
|------|-----------:|------------:|--------:|
| general (greeting) | 44.1 s | 4.7 s | **9.4×** |
| analyzer | 15.8 s | 2.3 s | **6.9×** |

**Severity:** medium for clinical trials.  GPU is the correct backend
on Jetson and is **substantially faster than CPU**.  But the failure
mode is silent — without the warm-up call, a clinical session starts,
loads the model with insufficient VRAM, then dies on first
`send_message` and silently degrades into fallback strings.

**Recommended action:**
1. **For clinical trials, use `LITERT_BACKEND=gpu` with the desktop
   killed.**  The official `jetson_run.sh` already runs in headless
   mode; that's the correct deployment.  The CPU fallback I used during
   live testing was a misdiagnosis — GPU is preferred.
2. Add a startup warm-up `llm_complete` call before the user wakes the
   device.  Detect VRAM OOM and `[Device] is lost` early.  Surface a
   clinician-visible error rather than silently degrading.
3. Document explicitly in the deployment checklist: kill all browser
   tabs / GUI apps before starting a trial session.  An auto-check
   that flags resident Chromium / Firefox / GNOME processes at boot
   would prevent this.
4. Consider raising swap / setting `LITERT_CONTEXT_LENGTH` lower
   (currently 3072) only if VRAM remains tight after killing the
   desktop — but with the desktop dead, the model fits with headroom.

---

## #2 — Score-2 mood reply silently DROPPED on `[ANALYZER_DEAD]`

**Where:** `src/core/questioner.py::ask_question` → `classify_segments` → `LLMError`
caught and turn returned with `0.0, 0, ""`.

**Symptom:** During Persona A live run, agent #02 asked "Have your moods
been something you can manage lately?" The user (Claude in character)
replied with a clearly Score-2 phrasing — "really low most days for the
last couple of months. It feels heavy and I can't shake it."

The Analyzer LLM crashed with `Device is lost` after 2.17s, so the
handler took the `[ANALYZER_DEAD]` branch and **moved on to the next
dimension as if the user said nothing**. No score recorded for mood,
no RV "tell me more" follow-up, no Validator empathic reflection.

This is the "C6 typed exception → SKIPPED" path the codebase intends
— so it's correct that nothing was scored 0 by accident — BUT:

  • From the participant's perspective the AI just IGNORED a heavy
    disclosure and pivoted to "Are you eating at fairly steady times?"
    That is a serious clinical-safety UX issue.
  • The session's Q-table now has a stale 0.0 reward for mood, which
    biases the next session's exploration AWAY from re-asking the
    very dimension that the participant flagged as concerning.

After this failure, the engine refused further attempts (`H2: bounded
re-init attempts`, max 3 failures).  The next agent question
("Are you eating at fairly steady times each day?") was the
**unrephrased canonical** because the Rephraser also failed silently
with `LiteRT engine has failed 3 times this session; refusing further
attempts`.  From this point onward, EVERY LLM-driven turn in the run
will either skip or use a fallback, while the user is unaware.

**Severity:** HIGH for clinical trials — the participant believes
they shared a meaningful answer that was never recorded, and a
clinician reviewing the dossier sees "no mood concern recorded today"
when the opposite is true.

**Recommended action (urgent for trials):**
1. When `[ANALYZER_DEAD]` fires, the agent should NOT silently advance
   to the next dim.  Speak something like "I'm having trouble
   processing that — could you say it again in a different way?"
   so the user's signal is not lost.
2. If 3 LLM failures bound is reached, abort the session with a
   clinician-visible error rather than continuing in a degraded state
   that produces an incorrect-but-plausible dossier.
3. Switch `LITERT_BACKEND` to CPU for trials until the GPU `[Device]
   is lost` is reproducible+fixable.

---

## #3 — `generate_change_negative` leaks LLM meta-reasoning to user

**Where:** `src/utils/text_generators.py::generate_change_negative` →
called from `src/core/questioner.py::_if_valid_response` when a user
"No" answer scores >1 (i.e. clinical Score-2).

**Symptom (Persona A2 turn 3):** Mood was correctly classified as
`(mood, No)` keyword path, scored 2.  The follow-up template is
"It seems that ${negative-rewrite-of-question}. Can you tell me more
about it?"  `generate_change_negative` invoked Gemma to negate the
question — but Gemma returned a multi-option meta-explanation:

```
[AGENT #03] It seems that Here are a few ways to turn that question
into a negative declarative sentence, depending on the nuance you
want to emphasize:

**Option 1 (Focusing on the inability to manage):**

> You haven't been able to manage your moods.

**Option 2 (More direct negation of the ability):**

> You have not been able to manage your moods.

**Option 3 (Slightly more conversational, mirroring the structure of the original question):**

> You haven't been able to manage your moods. (This is the most natural and common way to phrase it.) Can you tell me more about it?
```

The user, having just disclosed deep depression, hears markdown-formatted
LLM scaffolding ("**Option 1**", ">", "(This is the most natural and
common way to phrase it.)") immediately followed by "Can you tell me
more about it?".  This is jarring and breaks therapeutic rapport.

**Severity:** HIGH for clinical trials.  This is a Score-2 follow-up,
so it fires precisely on the highest-clinical-value turns.

**Recommended action:**
1. `text_generators` was clearly written for an older LLM that obeyed
   "give me only the rewritten sentence."  It needs:
   - A stricter system prompt ("Output ONLY the rewritten sentence,
     no preamble, no options, no markdown.")
   - A post-LLM sanitiser that strips markdown, "Option N", "**bold**",
     "> quote" prefixes, and parenthetical commentary.
   - A regex fallback that just adds "haven't"/"don't" to the question
     verb when the LLM output is unrecognisable.
2. As a stop-gap, replace the `generate_change_*` helpers with a
   non-LLM template (e.g. "It sounds like that's been hard. Can you
   tell me more about it?") for Score-2 follow-ups.

---

## #4 — Legacy `generate_change` regex produces ungrammatical follow-ups

**Where:** `src/utils/text_generators.py::generate_change` →
`src/core/questioner.py::_if_valid_response` (Score-2 follow-up,
Response Analyzer path).

**Symptom (Persona A2 turn 7):** User said "my sleep is a mess. I wake
up at three…" → Analyzer scored (sleep, 2) → handler built
"You mentioned that {generate_change(seg)}. Can you tell me more?"
which produced:

> "You mentioned that you have a mess sleep. Can you tell me more?"

The legacy regex inversion of "my sleep is a mess" is "you have a mess
sleep" — grammatically broken; "have a mess sleep" is not English.

**Severity:** medium for clinical trials.  Doesn't break flow but
sounds AI-glitchy and undermines trust.  Several other personas in
the offline benchmark already hit this — it's a known weakness of
the legacy regex.

**Recommended action:** When `REFLECTIVE_SUMMARIZER_ENABLED` is on,
the LLM-based ReflectiveSummarizer (paper §5.2) replaces this regex.
Recommend turning that flag ON for clinical trials; this would have
produced "You mentioned that your sleep has been a mess.  Can you
tell me more?" which is at least grammatical.  Cost: one extra
LLM round-trip per Score-2 follow-up.

---

## #5 — Real Gemma-CPU latency exceeds 30-min session watchdog

**Where:** observed Persona A2 live run; `_SESSION_MAX_SECONDS = 3600`
in handler_rl.py is gated off by `SESSION_CAP_ENABLED=False`, but the
runner-side `HARDER_TIMEOUT_S = 1800` watchdog tripped at 30 minutes.

**Symptom:** Persona A2 reached only 12 agent turns / 11 user turns
in 30 minutes before the runner's watchdog forced END_SESSION_EVENT.
Per-call latency on CPU backend:

| Role         |  n | avg s | total s |
|--------------|----|-------|---------|
| general      |  1 | 44.1  | 44.1    |
| rephraser    |  4 | 10.9  | 43.5    |
| analyzer     | 16 | 15.8  | 253.3   |
| rv_reasoner  |  4 |  8.2  | 32.9    |
| rv_validator |  4 | 28.0  | 112.0   |
| reflective_summarizer | 2 | 2.4 | 4.8 |

**Total LLM time per session at 12 turns: ~490 seconds (8 min).**

A full 37-dim screening + CBT requires roughly 40-50 turns × multiple
LLM calls per turn — projected total LLM time is 25-35 minutes of
inference alone, before any user thinking time.  On a 60-minute
real session that's borderline; with even brief user pauses it tips
over.

**Severity:** medium-to-high for clinical trials.  Real participants
will encounter this on long sessions.  At minimum this needs:
1. A clear UX "thinking…" indicator during the 15-30s gaps.
2. The intermission ladder (already in `speech_service.py`) is the
   designed mitigation — its timing has not been validated in a real
   end-to-end CPU-backend run on the Jetson.

**Recommended action:** before clinical trials, run one full session
(any persona) on the Jetson with TTS+STT enabled to validate the
intermission ladder masks the latency.  Consider a max-turn cap as
well — the handler already supports `SESSION_CAP_ENABLED` via
`SESSION_MAX_SECONDS` env var; turn it on for trials.

---

## #6 — `[Answer:]` prefix leaks into Score-2 follow-up (variant of #3)

**Where:** same as #3 (`generate_change_negative` → `_if_valid_response`).

**Symptom (Persona A2 turn 9):**

> "It seems that **Answer:** You haven't been seeing your doctor,
>  therapist, or case manager consistently. Can you tell me more about it?"

The Gemma model returned `Answer: <negated sentence>` and the handler
spliced the whole thing into the follow-up.  Same pattern as #3 just
with a different leading scaffolding token.

**Severity:** same as #3 (HIGH) — confirms the issue is general, not
a one-off.  Sanitiser must strip `Answer:` / `Option N:` / `Here are
a few...` / `**bold**` / `>` / parenthetical commentary.

---

## #7 — **CRITICAL: brainstorm scaffold NEVER fires in production**

**Where:** `src/services/response_bridge.py::get_openai_resp` (≤3 word
short-circuit) + `src/core/questioner.py::retry_guide` (where the
brainstorm scaffold lives).

**Symptom (Persona B turn 1):** User said `"I don't know."` —
expected: Analyzer returns `Maybe` keyword → questioner falls into
retry_guide → `_is_dont_know` matches → brainstorm scaffold spoken.
Actual: real Gemma Analyzer LLM classified it as `(medication, 1)` —
i.e. Score-1, indicating "some medication concerns".  The user's
literal "I don't know" was scored as a positive clinical signal.

The codepath that prevents this is the ≤3 word shortcut in
`response_bridge.py:get_openai_resp`, but that block has no entry for
`"I don't know"` / `"no idea"` / etc. — it only handles `Yes`, `No`,
`Maybe`, `Question`, `Stop` literals.  The legacy ResponseAnalyzer
prompt's docstring claims `"I don't know"` should classify as `Maybe,
0`, but **the actual LLM (Gemma 4 E2B) does NOT obey that** — it
extracts a substantive Score-1 from "I don't know" instead.

**Impact:** the G16 don't-know brainstorm therapist-feedback fix is
**dead code in production** under real Gemma.  Every "I don't know"
gets a fabricated score and the session moves on, which is
clinically the same regression the therapist asked us to fix
("interaction may abruptly stop").

**Severity:** CRITICAL.  Two harms:
1. The therapist-requested fix doesn't actually work in production.
2. Every "I don't know" / "no idea" reply produces a false score
   (most often Score-1) on the asked dimension.  Scoring fabrication
   under uncertainty is a clinical-data integrity violation.

**Recommended action — REQUIRED for clinical trials:**
Add a pre-LLM short-circuit in `response_bridge.get_openai_resp` for
the don't-know patterns, mirroring the soft-end intercept already
present:

```python
_DONT_KNOW_PATTERNS = (
    re.compile(r"\bi\s*(?:don'?t|do\s+not|dont)\s+know\b", re.IGNORECASE),
    re.compile(r"\bno\s+idea\b", re.IGNORECASE),
    re.compile(r"\bi\s*(?:don'?t|do\s+not|dont)\s+want\s+to\s+answer\b", re.IGNORECASE),
    ...
)

def _matches_dont_know(text):
    return any(p.search(text or "") for p in _DONT_KNOW_PATTERNS)

# in get_openai_resp, AFTER soft-end check, BEFORE any LLM call:
if _matches_dont_know(_clean_input):
    logger.info(f"[DONT-KNOW] Pre-LLM intercept; returning ({dimension_label}, Maybe)")
    return dimension_label, "Maybe"
```

This will route the reply to `Maybe` → `_if_valid_response` →
`had_ambiguous=True` → `retry_guide(...)` → `_is_dont_know` →
brainstorm scaffold spoken.  Fix is ~10 LOC, mirrors the existing
soft-end pattern, and unlocks the entire G16 feature in production.
