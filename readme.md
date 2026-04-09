# CaiTI — Conversational AI Therapist Interface

## Overview

CaiTI is a fully local, privacy-preserving smart-speaker system that delivers **Motivational Interviewing (MI)** and **Cognitive Behavioral Therapy (CBT)** micro-interventions. Built for the **NVIDIA Jetson** platform (Orin Nano/NX/AGX), the system operates entirely offline as a headless audio assistant. A Python-based speech-client loop communicates with a local Dialogue Engine powered by a fine-tuned **Gemma 2B** model served through Ollama.

---

## Design Philosophy

The architecture is built on three pillars:

1. **Clinical Observability** — Every conversational turn records RL Q-values, semantic scores, and validation flags to a structured SQLite database and per-session JSON/CSV logs. Researchers can audit exactly *why* the AI chose a specific clinical dimension. Sessions are logged independently to `data/logs/[UserID]/[Timestamp].{log,json}`.

2. **Edge Privacy** — By using `faster-whisper`, lightweight local SER (MFCC-RF), `Piper`, and local `Ollama` models, sensitive patient data never leaves the device. The expensive LLM layer is triggered only upon confirmed speech intent.

3. **Sandboxed State Management** — Strict session lifecycle prevents "user-leakage." On session termination, RL memory (Pandas DataFrames) and screening state are explicitly purged, ensuring the next user starts with a clean slate seeded only from their historic database context.

---

## System Architecture

### High-Level Data Flow

```
                          +------------------+
                          |   GPIO Manager   |  Buttons: Start / End / Opt-Out
                          |  (gpio_manager)  |  LED: Listening indicator
                          +--------+---------+
                                   |
    Mic ──> AudioRecorder ──> STTGenerator ──────────+
            (PyAudio+VAD)   (Faster-Whisper + Light SER) |
                            [parallel threads]       |
                                   |                 |
                            { transcript,            |
                              emotion }              |
                                   |                 |
                    +--------------v--------------+  |
                    |  SpeechInteractionService    |  |
                    |  (speech_service.py)          |  |
                    |  Wake-word / Turn-taking      |  |
                    +--------------+--------------+  |
                                   |                 |
                    INPUT_QUEUE <---+  OUTPUT_QUEUE ---+---> TTSGenerator ──> AudioPlayer
                          |                |               (Piper TTS)     (aplay)
                          v                ^
                    +------+---------------+------+
                    |           main.py            |
                    |    FastAPI server (port 8000) |
                    |    Session lifecycle manager  |
                    +------+-----------------------+
                           |
                    +------v-----------------------+
                    |        HandlerRL              |
                    |   (RL Screening Orchestrator) |
                    +------+-----------------------+
                           |
          +----------------+------------------+
          |                |                  |
   +------v------+  +-----v-------+  +-------v--------+
   |  Questioner  |  | Response    |  |   CBT Module   |
   |  ask_question|  | Analyzer    |  | (3-stage proto)|
   +--------------+  | classify/   |  +----------------+
          |          | score       |          |
          |          +------+------+          |
          |                 |                 |
          +--------+--------+---------+-------+
                   |                  |
          +--------v-------+  +------v--------+
          | ResponseBridge |  | Reflection    |
          | (DLA parsing + |  | Validation    |
          |  Intermission) |  | (RV Module)   |
          +--------+-------+  +---------------+
                   |
          +--------v-------+
          |   LLM Client   |  ──> Ollama (Gemma 2B / llama3.2)
          |  (llm_client)  |      via OpenAI-compatible API
          +--------+-------+
                   |
          +--------v-------+
          |   DB Manager   |  ──> SQLite (therapist.db)
          |  (db_manager)  |
          +----------------+
```

### Dual Operating Modes

The system supports two modes controlled by `APP_MODE` env var:

| Mode | Server | Interface | Use Case |
|------|--------|-----------|----------|
| `local` (default) | FastAPI on `:8000` | Embedded `SpeechInteractionService` with mic/speaker | Standalone Jetson deployment |
| `server` | Flask on `:8080` | External client polls `/gpt` endpoint | Remote client / web frontend |

When `DISABLE_INTERNAL_SPEECH=1`, the local mode runs in **API-only** mode — no audio stack, all I/O through REST endpoints.

---

## Module Architecture

### `src/` Package Layout

```
src/
├── core/                      # Clinical domain logic
│   ├── handler_rl.py          # Top-level RL session orchestrator
│   ├── questioner.py          # Question selection & delivery engine
│   ├── response_analyzer.py   # LLM-based dimension/score classification
│   ├── reflection_validation.py  # Follow-up relevance validation (RV)
│   ├── CBT.py                 # 4-stage CBT protocol (Stage 0-3)
│   └── therapy_content.py     # PHQ-4 / GAD-2 screening & meditation content
│
├── models/                    # AI model wrappers
│   ├── llm_client.py          # Unified Ollama LLM caller + interstitial engine
│   ├── stt.py                 # Faster-Whisper STT + lightweight MFCC SER
│   ├── tts.py                 # Piper neural TTS subprocess wrapper
│   └── light_ser.py           # Lightweight MFCC random-forest SER classifier
│
├── services/                  # Integration & orchestration
│   ├── speech_service.py      # Real-time speech loop (mic, GPIO, turn-taking)
│   └── response_bridge.py     # DLA parsing + IntermissionManager
│
├── drivers/                   # Hardware & persistence
│   ├── audio.py               # PyAudio + WebRTC-VAD microphone driver
│   ├── player.py              # ALSA aplay subprocess for WAV playback
│   ├── gpio_manager.py        # Jetson GPIO buttons & LED (graceful degradation)
│   └── db_manager.py          # SQLite schema, CRUD, and clinical persistence
│
└── utils/                     # Shared utilities
    ├── config_loader.py       # YAML + .env config aggregator
    ├── inference_guard.py     # Heavy-inference serialization lock, VRAM cache cleanup, sysfs GPU telemetry
    ├── resource_audit.py      # Runtime resource forensics (RSS/VMS tracking, module init profiling, KV cache estimation)
    ├── io_record.py           # IPC queues, session state, CSV/JSON logging
    ├── rl_qtables.py          # Q-table init, action selection, env feedback
    ├── io_question_lib.py     # Question library JSON I/O
    ├── text_generators.py     # LLM-based text transformations (rephrase, etc.)
    └── log_util.py            # Colored logging configuration
```

---

## Core Components (Detailed)

### 1. Audio Perception Pipeline (`src/drivers/audio.py`, `src/models/stt.py`)

The `AudioRecorder` captures audio via PyAudio with WebRTC-VAD for voice activity detection:
- Configurable VAD aggressiveness (0-3), sample rate (16kHz), and chunk size (480 frames)
- ALSA warning suppression via ctypes for clean Jetson operation
- OS buffer flushing on stream start to avoid echo from recent TTS playback
- VAD callback hooks to the GPIO LED for visual feedback

When speech is captured, `STTGenerator` performs a memory-aware sequential pipeline serialized via `inference_guard.heavy_stage()`:
- **Faster-Whisper STT** (`faster-whisper`) produces the transcript with low-memory decode settings
- **Lightweight SER** (`light_ser.py` — MFCC random-forest-style voting) classifies vocal emotion
- Cache is explicitly cleared between STT and SER phases via `clear_inference_cache()`

Both results are packaged as `{"transcript": "...", "detected_emotion": "..."}` and pushed onto the `INPUT_QUEUE`. The SER backend is selectable via `SER_BACKEND` env var (defaults to `light_mfcc_rf`).

**VRAM Shielding**: On 8GB Jetson, Whisper and the LLM cannot coexist in VRAM. `STTGenerator` exposes `suspend_all()` / `resume_all()` methods that fully unload both the Whisper model and SER classifier before the LLM handshake, then reload them afterward. RSS deltas are logged at each transition.

**Environment Verification**: On import, `stt.py` checks for the heavy `openai-whisper` package (which pulls ~2GB PyTorch). If detected, the process exits immediately with `[ENVIRONMENT ERROR]` and instructions to uninstall.

### 2. Speech Interaction Service (`src/services/speech_service.py`)

The `SpeechInteractionService` orchestrates the real-time client loop:

- **Idle State**: Polls 1-second audio windows for wake-word detection ("Hello CaiTI", "Hey Katie", etc.)
- **Onboarding**: On wake-word or GPIO Button 1, asks for user name, initializes session via `io_record.reset_session()`. A multi-layer **Name Guard** rejects common fillers ("Of course", "Good morning", "I'm fine") using word and phrase blacklists, minimum character thresholds, and prefix stripping ("My name is..." → name only)
- **Active Session**: Turn-based loop — dequeue agent text from `OUTPUT_QUEUE`, TTS+play it, then listen for user response and push to `INPUT_QUEUE`
- **Sequential VRAM Handoff** ("Safe-Pass"): After the user speaks, the service (1) unloads both Whisper and SER via `stt.suspend_all()`, (2) waits 0.5s for OS memory reclamation, (3) lets the LLM handshake proceed, then (4) reloads both models via `stt.resume_all()`. Memory snapshots are logged at each phase
- **Intermission Watchdog**: If the LLM takes longer than 3 seconds, PHQ-4 screening questions and guided breathing exercises play automatically. When the LLM response arrives, the current TTS clip finishes naturally (no mid-sentence interruption), then a randomized **bridge phrase** (e.g., "Thank you for reflecting on that with me. Now, going back to what you shared...") transitions smoothly into the therapist's answer
- **Modes**: `hands_free` (auto-listen after agent speaks) or `manual` (wait for `manual_input_event`)
- **Voice Commands**: "end session" / "stop session" triggers session termination
- **Hardware Events**: GPIO buttons for Start/End/Opt-Out are polled each loop iteration

### 3. Reinforcement Learning Orchestrator (`src/core/handler_rl.py`)

`HandlerRL` is the main clinical pipeline coordinator:

1. **Setup**: Loads question library JSON, initializes Q-table from `ITEM_IMPORTANCE` weights, loads persistent Q-table if one exists for the subject
2. **Greeting**: LLM-generated personalized greeting incorporating user context from prior sessions
3. **RL Screening Loop**:
   - `choose_action()` selects the next clinical dimension (37 possible: mood, sleep, weight, etc.) using epsilon-greedy Q-learning
   - `ask_question()` delivers the question, collects response, classifies it
   - Q-table is updated via standard Q-learning: `Q(s,a) += alpha * (target - predict)`
   - RL decisions (state, action, Q-values, mask) are logged to the DB as reasoning events
4. **CBT Trigger**: After screening, dimensions scoring 2 (severe) are offered for CBT
5. **Post-Session**: Clinical summary generation, session analysis (safety flags, preferences), Q-table persistence, memory purge

### 4. Question Delivery & Classification (`src/core/questioner.py`, `src/core/response_analyzer.py`)

**Questioner**:
- Selects question variants from the library, optionally rephrases via LLM
- Classifies user response segments through `ResponseBridge` → `ResponseAnalyzer`
- Handles invalid responses with an LLM-generated retry guide (clarification, angle-shift, or restatement)
- Invokes `ReflectionValidation` for follow-up relevance checking

**ResponseAnalyzer** (`classify_dimension_and_score`):
- LLM-based classifier maps user text to one of 37 clinical dimensions + a severity score (0/1/2)
- Also handles general responses: Yes, No, Maybe, Question, Stop
- Includes `reflective_summarizer` and `rephrase_question` for natural conversation flow

### 5. Response Bridge & Intermission Engine (`src/services/response_bridge.py`)

**DLA Parsing**: Robust parser handling multiple LLM output formats — plain text (`talk, 1`), JSON (`{"res": "talk, 1"}`), prefixed (`DLA_3_talk, 1`) — with cascading fallback logic.

**IntermissionManager**: Non-blocking interstitial content during slow LLM inference:
1. If inference < 3s: wait silently (fast path)
2. If >= 3s: deliver **GAD-2/PHQ-2 clinical screening** questions
3. If screening complete/refused: cycle through **guided meditations** (5 breathing exercises)
4. If still waiting: play **ambient music** (`assets/waiting_music.wav`)
5. User can opt out at any time ("skip", "stop", "don't want to")

Screening scores are persisted to DB and trigger clinical flags:
- `GAD2_POSITIVE` when anxiety >= 3
- `PHQ4_HIGH_RISK` when total >= 6

### 6. Reflection Validation Module (`src/core/reflection_validation.py`)

Validates follow-up responses for topical relevance using a consolidated LLM prompt:
- **Decision 0 (Related)**: Returns empathetic **VALIDATION** text, prepended to next question
- **Decision 1 (Unrelated)**: Returns a **GUIDE** to steer user back on topic, then re-collects response
- Uses `llm_complete_with_interstitial` so the intermission engine covers latency

### 7. CBT Protocol (`src/core/CBT.py`)

A 4-stage Cognitive Behavioral Therapy protocol triggered when any dimension receives severity score 2:

| Stage | Name | Action |
|-------|------|--------|
| 0 | **Select** | User chooses which severe dimension to work on |
| 1 | **Identify** | User identifies unhelpful thoughts; LLM *reasons* + *guides* (up to 2 retries) |
| 2 | **Challenge** | User challenges those thoughts; LLM validates with cognitive distortion awareness |
| 3 | **Reframe** | User reframes thoughts; preceded by LLM-recap of their challenge |

Each stage uses a **Reasoner** (validates user input, returns DECISION 0/1) and a **Guide** (provides therapeutic direction on failure). The system tracks 13 cognitive distortions (filtering, catastrophizing, polarized thinking, etc.). All CBT progress is recorded in the question library's `notes` field.

### 8. LLM Client (`src/models/llm_client.py`)

Unified LLM interface with three calling modes:

| Function | Behavior |
|----------|----------|
| `llm_complete()` | Synchronous call to Ollama via OpenAI-compatible API |
| `llm_complete_async()` | Background thread pool execution |
| `llm_complete_with_interstitial()` | Async + intermission engine for latency coverage |

All calls automatically inject:
- **User context** (preferences + session summaries from DB)
- **Context pack** (latest transcript, emotion tag, RL state, screening scores)
- Context injection can be disabled via `DISABLE_CONTEXT_HISTORY=1` for diagnostic isolation of init-peak vs. context-bloat OOM crashes
- Supports both `client.responses.create` and `client.chat.completions.create` APIs (automatic fallback)
- **Model fallback chain**: If the primary `LLM_MODEL` fails, the client iterates through `LLM_FALLBACK_MODELS` before returning a graceful error message

**Strict VRAM Budget**: Each Ollama request enforces a hard memory profile via `extra_body`:
- `num_ctx=256` (default, overridable) — aggressive cap to fit KV cache into fragmented Jetson VRAM
- `num_batch=1` — single-batch processing to minimize peak allocation
- `f16_kv=false` — 8-bit KV cache to halve memory vs. FP16
- `num_gpu=1` — GPU execution enforced (CPU fallback causes 7-minute inference)
- `keep_alive=0` — model unloaded immediately after each request to free VRAM

**Auto-Scaling Context Window**: On cudaMalloc/OOM failure (HTTP 500, "runner terminated"), the client automatically decrements `num_ctx` by 64 and retries once. The floor is 128 tokens. This provides runtime recovery from VRAM fragmentation without manual intervention.

**VRAM monitoring**: Logs system memory snapshot (RAM, GPU sysfs, Ollama footprint) before and after each inference call. A heartbeat thread logs every 10s during inference so the terminal never appears dead.

### 8a. Inference Guard (`src/utils/inference_guard.py`)

Serializes heavy inference stages (STT, SER, LLM) to prevent concurrent GPU/memory contention on resource-constrained Jetson hardware:
- **`heavy_stage(name)`**: Context manager with a global threading lock ensuring only one heavy inference runs at a time
- **`clear_inference_cache(reason)`**: Best-effort cleanup between phases — runs `gc.collect()` and optionally `torch.cuda.empty_cache()` when PyTorch is available
- **`get_current_vram_usage()`**: Lightweight VRAM proxy via `ollama ps` subprocess
- **`get_system_memory_snapshot()`**: Clinical-grade memory transparency combining system RAM (`free -h`), GPU telemetry (priority: sysfs zero-cost reads → `nvidia-smi` → `tegrastats`), and Ollama model footprint. Sysfs reads GPU clock frequency and utilization directly from `/sys/class/devfreq/` on Jetson (Orin Nano, Orin NX, Xavier NX paths supported)

### 8b. Lightweight SER (`src/models/light_ser.py`)

Pure-NumPy MFCC-based emotion classifier designed for memory-constrained devices:
- Extracts 30-dimensional feature vector: 13 MFCC means + 13 MFCC stds + RMS mean/std + ZCR mean/std
- **RMS auto-gain normalization**: Input audio is normalized to a target RMS of 0.1 before feature extraction, preventing Jetson mic gain variance from inflating MFCC std features
- **Silence gate**: Audio with raw RMS below 0.005 is immediately tagged as "neutral" without classification, preventing noise-only buffers from triggering false anger detections
- Classifies into 4 emotions: neutral, happy, sad, angry via rule-based random-forest-style voting (15 decision stumps)
- **Adaptive anger calibration**: A rolling 10-sample window tracks classification history. If 7+ of the last 10 results are "angry", MFCC std thresholds are automatically raised by 10% to suppress persistent mic-gain-induced anger bias. Calibrations are logged as `[SER CALIBRATE]` warnings
- Zero external ML dependencies — uses only NumPy and soundfile
- Configurable via `SER_BACKEND` env var (enabled when set to `light_mfcc_rf`, `mfcc_rf`, or `lightweight`)

### 9. Database Layer (`src/drivers/db_manager.py`)

SQLite database (`data/therapist.db`) with 7 tables:

| Table | Purpose |
|-------|---------|
| `users` | Subject IDs with creation timestamps |
| `sessions` | Session start/end times linked to users |
| `turns` | Full dialogue transcript with speaker, text, and JSON metadata |
| `summaries` | LLM-generated session summaries |
| `user_preferences` | Extracted user preferences (key-value, upsert) |
| `safety_flags` | Risk flags with severity (1-5 scale) |
| `clinical_screening` | PHQ-4/GAD-2 sub-scores with automatic clinical flag columns |
| `feedback` | User feedback with optional turn-level granularity |

### 10. GPIO Hardware Interface (`src/drivers/gpio_manager.py`)

Singleton manager for Jetson GPIO with graceful degradation to no-op stubs on non-Jetson hardware:

| Pin (BOARD) | Function | Type |
|-------------|----------|------|
| 11 | Start Session | Input (FALLING edge, 300ms debounce) |
| 13 | End Session | Input (FALLING edge, 300ms debounce) |
| 15 | Opt-Out | Input (FALLING edge, 300ms debounce) |
| 16 | Button 4 (spare) | Input (FALLING/RISING edge, 300ms debounce) |
| 18 | Listening LED | Output (HIGH while recording) |

Supports **per-pin active-low/active-high** configuration (via `PIN_BTN_*_ACTIVE_LOW` env vars), interrupt-based detection with fallback level polling, and thread-safe event queue. The LED polarity is also configurable via `PIN_LISTENING_LED_ACTIVE_LOW`.

### 11. Configuration System (`src/utils/config_loader.py`)

Two-tier configuration:

- **`config.yaml`**: Audio parameters, STT/TTS model paths, RL hyperparameters (epsilon, alpha, gamma, item_importance), database path, VAD aggressiveness
- **`.env`**: Runtime secrets and host-specific overrides — Ollama URL, model name + fallback chain, request timeout, Ollama memory controls (`num_ctx`, `num_predict`, `num_gpu`), SER backend selection, TTS pacing (`length_scale`, `sentence_silence`), GPIO pin numbers, and per-pin active-low polarity flags

### 12. Session I/O & Logging (`src/utils/io_record.py`)

Central hub for session state and inter-process communication:

- **IPC Queues**: `INPUT_QUEUE` (user responses) and `OUTPUT_QUEUE` (agent questions) bridge the speech service and clinical pipeline
- **Session State**: `START_SESSION_EVENT` / `END_SESSION_EVENT` threading events, session ID, turn index
- **Context Pack**: Aggregates latest transcript, emotion, RL state, and screening scores for LLM injection
- **Multi-format Logging**: Simultaneous writes to SQLite DB, per-session CSV, per-session JSON (newline-delimited), and terminal output
- **Question Prefix**: Mechanism to prepend validation/recap text to the next agent question

---

## Project Structure

```
.
├── main.py                    # Application entrypoint (FastAPI + Flask dual-mode)
├── LLM_therapist_Application.py  # Legacy backward-compatible shim
├── config.yaml                # System configuration (audio, RL, models, DB)
├── Modelfile                  # Ollama Gemma 2B system prompt + inference params
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment specification
├── .env                       # Runtime secrets (Ollama URL, model, GPIO pins)
│
├── src/                       # Source package (see Module Architecture above)
│   ├── core/                  # Clinical domain logic
│   ├── models/                # AI model wrappers (LLM, STT, TTS)
│   ├── services/              # Orchestration services
│   ├── drivers/               # Hardware & persistence drivers
│   └── utils/                 # Shared utilities
│
├── data/                      # Runtime data directory
│   ├── therapist.db           # SQLite database (auto-created)
│   ├── libs/                  # Question library JSONs
│   ├── q_tables/              # Persisted RL Q-tables per subject
│   ├── results/               # Session reports and notes CSVs
│   └── logs/                  # Per-user session logs (CSV + JSON)
│
├── assets/                    # Static assets
│   ├── waiting_music.wav      # Ambient music for intermission fallback
│   └── audio/                 # Additional audio assets
│
├── dev/                       # Development tools and tests
│   ├── tests/                 # pytest test suite (API, pipeline, DB)
│   └── eric_test/             # Remote speech client experiments
│
├── scripts/                   # Deployment & management scripts
│   ├── start_headless.sh      # Launch Ollama + backend (headless mode)
│   ├── stop_system.sh         # Kill all CaiTI processes
│   ├── deploy_to_jetson.sh    # rsync code to Jetson device
│   ├── deploy_and_run.sh      # Deploy + start + tail logs (one command)
│   ├── board_scan.py          # Jetson board pin scanning utility
│   ├── find_pins.py           # GPIO pin discovery helper
│   └── gpio_probe.py          # GPIO probe/debug tool
│
├── start_caiti.sh             # One-command deploy + run (wraps start_headless.sh)
├── start_headless.sh          # Root-level convenience wrapper for scripts/start_headless.sh
├── stop_system.sh             # Root-level convenience wrapper for scripts/stop_system.sh
└── readme.md                  # This file
```

---

## Session Lifecycle

```
IDLE ──[wake-word / GPIO Button 1 / API /login]──> ONBOARDING
  │                                                      │
  │                                            Ask name, init session
  │                                                      │
  │                                                      v
  │                                               SESSION ACTIVE
  │                                                      │
  │                                         ┌────────────┴────────────┐
  │                                         v                         v
  │                                  RL Screening Loop          API endpoints
  │                                   (HandlerRL.run)           (/input, /output,
  │                                         │                    /turns, /status)
  │                                         v
  │                                  Q-learning selects
  │                                  clinical dimension
  │                                         │
  │                              ┌──────────┴──────────┐
  │                              v                     v
  │                       Ask Question            Score Response
  │                       (Questioner)            (ResponseAnalyzer)
  │                              │                     │
  │                              v                     v
  │                     Reflection Validation    Intermission Engine
  │                     (if follow-up needed)    (if LLM is slow)
  │                              │                     │
  │                              └──────────┬──────────┘
  │                                         v
  │                                  Score == 2?
  │                                   /        \
  │                                 Yes         No
  │                                  │           │
  │                                  v           v
  │                             CBT Protocol   Next dimension
  │                             (4 stages)     (loop continues)
  │                                  │
  │                                  v
  │                          Post-Session Analysis
  │                          - Clinical summary
  │                          - Safety flag extraction
  │                          - Preference mining
  │                          - Q-table persistence
  │                          - Memory purge
  │                                  │
  └──────────────────────────────────┘
                              IDLE (clean slate)
```

---

## API Reference (Local Mode — FastAPI on `:8000`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Current system state, subject ID, session ID |
| `GET` | `/api/turns` | Full dialogue history for active session |
| `GET` | `/api/output` | Poll next agent message (non-blocking) |
| `POST` | `/api/login` | Authenticate user, initialize session (`{"user_id": "..."}`) |
| `POST` | `/api/input` | Submit user text input (`{"text": "..."}`) |
| `POST` | `/api/intent` | LLM-based intent classification (START/END/NONE) |
| `POST` | `/api/action` | Control commands: stop, start_listening, set_mode |
| `POST` | `/api/pause` | Pause speech loop |
| `POST` | `/api/resume` | Resume speech loop |
| `POST` | `/api/end_session` | Terminate active session |

---

## Installation and Setup

### Prerequisites

* **Hardware**: NVIDIA Jetson (Orin series) or Linux PC with Mic/Speaker
* **Software**: Python 3.10+, `portaudio19-dev`, Ollama
* **Models**: Gemma 2B (or configured LLM), Piper TTS model, Whisper base.en

### Local Installation

```bash
# Install system dependencies
sudo apt-get install portaudio19-dev python3-venv

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

**`.env`** — Runtime configuration:
```bash
JETSON_HOST="user@192.168.x.x"
OLLAMA_BASE_URL="http://localhost:11434"
LLM_MODEL="llama3.2:3b"
LLM_FALLBACK_MODELS="gemma:2b"           # Comma-separated fallback model chain
LLM_REQUEST_TIMEOUT_SECONDS=90
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=400
OLLAMA_KEEP_ALIVE=0
OLLAMA_NUM_CTX=256                        # Context window (256 = safe for 8GB Jetson; raise once stable)
OLLAMA_NUM_PREDICT=256                    # Max tokens to predict (capped to num_ctx)
OLLAMA_NUM_GPU=1                          # GPU layers (1 = GPU; 0 = CPU-only, causes 7-min inference)
SER_BACKEND="light_mfcc_rf"              # SER engine: light_mfcc_rf | disabled
DISABLE_CONTEXT_HISTORY=0                # Set to 1 to strip context from LLM (diagnostic mode)
DISABLE_INTERNAL_SPEECH=0                # Set to 1 for API-only mode (no mic/speaker)
TTS_LENGTH_SCALE=0.8                     # Piper speech rate (lower = faster)
TTS_SENTENCE_SILENCE=1.5                 # Pause between sentences (seconds)
PIN_BTN_START=11
PIN_BTN_END=13
PIN_BTN_OPT_OUT=15
PIN_BTN_4=16
PIN_LISTENING_LED=18
PIN_LISTENING_LED_ACTIVE_LOW=0
PIN_BUTTONS_ACTIVE_LOW=1
PIN_BTN_START_ACTIVE_LOW=0
PIN_BTN_END_ACTIVE_LOW=0
PIN_BTN_OPT_OUT_ACTIVE_LOW=1
PIN_BTN_4_ACTIVE_LOW=1
```

**`config.yaml`** — System parameters:
```yaml
audio:
  sample_rate: 16000
  vad_aggressiveness: 2    # 0 (least aggressive) to 3 (most)

rl:
  epsilon: 1               # 1.0 = pure exploitation
  alpha: 0.5               # Learning rate
  gamma: 0.9               # Discount factor
  item_n_states: 38        # Number of clinical dimensions
```

**`Modelfile`** — Ollama model configuration with system prompt optimized for 8GB Jetson (2048 context, 250 max predict, temperature 0.7).

---

## Managing the System

| Command | Action | Description |
|---------|--------|-------------|
| `./start_caiti.sh` | **Deploy + Start** | One-command remote deployment: aggressive process sanitization (kills all project-associated Python, Ollama, llama-runner, ggml processes), dependency drift cleanup (auto-removes blocklisted packages like `openai-whisper`, `torch`, `mlc-ai-nightly`), filesystem cache drop, code sync via rsync, Piper voice repair, model provisioning, and launch |
| `./scripts/start_headless.sh` | **Start (local)** | Unlocks GPIO pinmux, aggressive sanitization (same 5-phase cleanup), starts Ollama, launches backend |
| `./stop_system.sh` | **Stop (local)** | Root-level convenience wrapper for `scripts/stop_system.sh` |
| `./scripts/stop_system.sh` | **Stop** | Kills all CaiTI processes and frees ports |
| `./scripts/deploy_to_jetson.sh` | **Deploy** | Syncs codebase, installs deps, downloads Piper model on Jetson |
| `./scripts/deploy_and_run.sh` | **Deploy + Run** | Syncs code to Jetson, starts services, tails live logs |
| `tail -f backend_session.log` | **Monitor** | View real-time AI reasoning and transcriptions |

### Startup Checklist

On boot, `main.py` runs a connectivity audit checking:
- GPIO availability and pin configuration
- Ollama connection, model availability, and keep-alive policy
- Faster-Whisper model configuration
- Piper TTS model file existence
- SQLite database connectivity

After all subsystems initialize, a **Ghost Hunt** scans for child processes consuming >50MB RSS and logs warnings (`[GHOST HUNT]`). **RSS Audit** checkpoints log process memory at key lifecycle points (before startup, after init, after SpeechService load, after thread start) to pinpoint memory culprits.

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| STT | Faster-Whisper (base.en) | Local speech-to-text transcription |
| SER | Lightweight MFCC-RF (NumPy + soundfile) | Low-memory vocal emotion recognition |
| TTS | Piper (en_US-amy-medium) | Neural text-to-speech with therapeutic pacing |
| LLM | Gemma 2B / LLaMA 3.2 via Ollama | Clinical dialogue, classification, CBT reasoning |
| VAD | WebRTC-VAD | Voice activity detection for recording triggers |
| RL | Q-Learning (Pandas DataFrames) | Clinical dimension selection policy |
| Screening | PHQ-4 / GAD-2 | Standardized anxiety/depression screening |
| API | FastAPI + Uvicorn / Flask | REST control and monitoring interface |
| Database | SQLite | Session persistence, clinical audit trail |
| Hardware | Jetson.GPIO | Physical button/LED interface |
| Audio | PyAudio + ALSA (aplay) | Microphone capture and speaker playback |
