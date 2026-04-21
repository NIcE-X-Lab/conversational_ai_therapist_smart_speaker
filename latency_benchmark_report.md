# Jetson Orin Nano Latency Benchmark Report

## 1. Overview

**Goal:** characterize end-to-end response latency on Jetson Orin Nano.

**Latency definition for this report:**
- Target end-to-end metric: **user stops speaking -> TTS starts playing**.
- Directly measured in this run: **LLM inference latency** (LLM request start -> response returned).

## 2. System Components & Their Latency Budgets

- **STT (faster-whisper):** transcription time.
- **LLM (Gemma 4 E2B via LiteRT):** inference time (current dominant cost).
- **TTS (Piper):** synthesis time.
- **Pipeline overhead:** VAD, queuing, RL/screening logic, and orchestration overhead.

Practical budget guidance:

| Component | Target Range |
|---|---:|
| STT | 0.3s to 1.5s |
| LLM | 2.0s to 8.0s typical (tails higher) |
| TTS start delay | 0.2s to 1.0s |
| Pipeline overhead | 0.1s to 0.5s |
| End-to-end total | <4s ideal, <8s acceptable, >10s needs mitigation |

## 3. Measurement Methodology

### Timing capture

- LLM timing was captured using `time.monotonic()` in `src/models/llm_client.py`.
- Start timestamp: request dispatch setup (`started_at = time.monotonic()`).
- End timestamp: response receipt log (`Received response from LLM in {elapsed:.2f}s`).

### Test conditions

- Hardware: Jetson Orin Nano.
- Model: Gemma 4 E2B LiteRT (`gemma-4-E2B-it.litertlm`).
- Backend observed in runtime logs: CPU backend.
- Context length: `LITERT_CONTEXT_LENGTH=512`.
- Max tokens: `LITERT_MAX_TOKENS=80`.
- Prompt mode: short prompts with `inject_context=False` to isolate model latency.
- Procedure: 1 warmup call, then 8 measured prompts.

### Prompt profile

- Short therapy-style prompts (CBT, grounding, breathing, reframing, self-compassion).
- No large context-history injection in benchmark calls.

## 4. Results

### 4.1 Per-component latency summary

| Component | Mean | P50 | P95 | Max | Status |
|---|---:|---:|---:|---:|---|
| STT | N/A | N/A | N/A | N/A | Not instrumented in this benchmark pass |
| LLM | **7.047s** | **6.353s** | **15.119s** | **15.119s** | Measured (8 samples, warmup excluded) |
| TTS | N/A | N/A | N/A | N/A | Not instrumented in this benchmark pass |
| Pipeline overhead | N/A | N/A | N/A | N/A | Not instrumented in this benchmark pass |
| End-to-end (speech stop -> TTS start) | N/A | N/A | N/A | N/A | Definition established, measurement pending |

### 4.2 LLM sample latencies (seconds)

| Sample | Latency |
|---:|---:|
| 1 | 9.968 |
| 2 | 10.511 |
| 3 | 3.903 |
| 4 | 6.353 |
| 5 | 2.544 |
| 6 | 4.188 |
| 7 | 15.119 |
| 8 | 3.788 |

### 4.3 End-to-end breakdown as % of total

Not yet available for this run, because STT start/end and TTS start timestamps were not captured in the same benchmark trace. Current data supports that LLM dominates observed latency behavior.

### 4.4 Additional component tests to include

Yes, the results should also cover the other major pipeline components. A complete results section can include:

| Component | What to measure | Suggested test input |
|---|---|---|
| STT | Audio-to-text latency | 1s, 3s, and 8s WAV clips with silence + speech |
| LLM | Prompt-to-response latency | Short prompt, long context, and max-token prompts |
| TTS | Text-to-audio latency | Short sentence and long sentence synthesis |
| Pipeline overhead | Orchestration delay | Full turn loop with VAD, queueing, and intermission |
| End-to-end | User stops speaking -> TTS starts | Full conversational turn with microphone input |

### 4.5 Interactive case test

Yes, we can test the AI by interacting with it directly. A good case test is a real turn-based conversation that measures user-perceived latency, not just raw model inference.

**Case test example:**
1. Start the system on Jetson.
2. Speak a short prompt such as: "I feel overwhelmed today."
3. Measure:
   - when user speech ends,
   - when STT finishes,
   - when LLM response begins,
   - when TTS starts playing.
4. Record whether interstitial mitigation activates if the LLM exceeds the threshold.

**What this test validates:**
- Audio capture and transcription.
- Prompt assembly and LLM response time.
- TTS playback delay.
- Whether the interstitial ladder hides long LLM latency well enough to keep the interaction feeling responsive.

**Recommended test cases:**
- Short user utterance: "I need help calming down."
- Medium context utterance: "I had a hard day at work and I keep replaying it in my head."
- Stress-case utterance: a longer prompt with context history already loaded.

## 5. Interstitial Mitigation

### How latency is masked

The pipeline uses an interstitial ladder to keep user engagement while waiting on slow LLM responses.

- In `src/services/speech_service.py`, the code waits for LLM completion up to `_INTERMISSION_TRIGGER_SEC`.
- If exceeded, it enters intermission flow (screening prompts, breathing, then fallback behaviors).

### Trigger threshold currently in code

Two thresholds exist in different paths:

- `src/services/response_bridge.py`: default trigger threshold is **2.0s**.
- `src/services/speech_service.py`: `_INTERMISSION_TRIGGER_SEC` is **3.0s**.

Implication: behavior can vary depending on which intermission path is active.

## 6. Bottlenecks & Recommendations

### Dominant bottleneck

- LLM first-token and generation latency dominate this benchmark, especially tail latency (P95 ~15.1s).

### Tuning knobs

- Reduce `LITERT_CONTEXT_LENGTH` (currently 512).
- Reduce `LITERT_MAX_TOKENS` (currently 80).
- Evaluate GPU backend for LiteRT if stable in your runtime environment.
- Keep interstitial mitigation enabled to protect user experience during high-latency tails.

### Next actions (recommended)

1. Add full end-to-end timestamps in speech pipeline:
   - speech end detected
   - STT complete
   - LLM complete
   - TTS playback start
2. Re-run with 20 to 30 samples for stronger p95/p99 confidence.
3. Unify intermission threshold (2.0s vs 3.0s) for predictable behavior.
4. Export benchmark runs to CSV for configuration A/B comparisons.

## 7. Project Test Harness

The project now includes a runnable benchmark script:

`scripts/latency_benchmark.py`

### Component benchmark

Run component timing on a WAV file or with a fresh microphone capture:

```bash
source .venv/bin/activate
python scripts/latency_benchmark.py --audio-path path/to/sample.wav
python scripts/latency_benchmark.py --record-mic --samples 8 --playback
```

### Interactive case test

Run a mic-driven end-to-end case test that exercises STT -> LLM -> TTS:

```bash
source .venv/bin/activate
python scripts/latency_benchmark.py --record-mic --case
```

### What the script reports

- STT transcription latency
- LLM generation latency
- TTS synthesis latency
- Playback handoff / blocking time
- Full turn timing for the interactive case path
