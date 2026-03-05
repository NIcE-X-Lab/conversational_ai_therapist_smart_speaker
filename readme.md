# Smart-Speaker Micro-Intervention System (Local Jetson Deployment)

## 🌟 Overview

This project implements a fully local, privacy-preserving smart-speaker system—**CaiTI** (Conversational AI Therapist Interface)—designed to deliver **Motivational Interviewing (MI)** and **Cognitive Behavioral Therapy (CBT)** micro-interventions.

Built specifically for the **NVIDIA Jetson** platform (Orin Nano/NX/AGX), the system operates entirely offline. It functions as a headless audio assistant, leveraging a robust Python-based speech-client loop that communicates with a local Dialogue Engine.

---

## 🧠 Design Philosophy & Core Pillars

The CaiTI architecture is built on three non-negotiable pillars:

1. **Clinical Observability**: Standard LLM logs are insufficient for clinical research. This system implements a **"Reasoning Logic"** stream. Every conversational turn records the exact RL Q-values, Semantic Scores, and Validation Flags, allowing researchers to audit *why* the AI chose a specific clinical dimension.
2. **Edge Privacy**: By using `faster-whisper`, `Piper`, and local `Ollama` models, sensitive patient data never leaves the Jetson device. Ambient states are handled locally, only triggering the expensive LLM layer upon high-probability intent detection.
3. **Sandboxed State Management**: To prevent "user-leakage," the system utilizes a strict session lifecycle. When a session terminates, the Reinforcement Learning memory (Pandas DataFrames) is explicitly purged, ensuring the next user starts with a clean slate seeded only by their specific historic database context.

---

## 🏗️ System Architecture & Logic

The system is a multi-process application designed for high observability and low-latency response.

### 1. The Core State Machine: `AMBIENT_IDLE` vs `ACTIVE_SESSION`

* **Logic**: The system begins in `AMBIENT_IDLE`. The microphone is active, but only a lightweight background STT process monitors for wake words ("Hi CaiTI").
* **Transition**: Detection of the wake word triggers a move to `ACTIVE_SESSION`. The **`HandlerRL`** initiates, loading user context from SQLite to personalize the interaction.
* **Cleanup**: At session end, the system performs a "sandboxing" routine—it flushes RL Q-tables, commits logs, and garbage-collects live memory before returning to `AMBIENT_IDLE`.

### 2. Component Breakdown

* **Dialogue Engine (`LLM_therapist_Application.py`)**: The orchestration brain. It manages the Reinforcement Learning policy and hosts FastAPI endpoints for interaction logs and manual overrides.
* **Audio Pipeline ("Arth" Modules)**:
* **Input (`AudioRecorder`)**: Uses `pyaudio` and `webrtcvad` for dynamic Voice Activity Detection (VAD). It aggressively flushes the ALSA buffer to implement **Echo Cancellation** (preventing the mic from capturing its own TTS output).
* **STT**: `faster-whisper` decodes raw audio into transcripts locally.
* **TTS**: `piper-tts` generates synthesized voice responses.


* **Policy Engine**:
* **`handler_rl.py`**: Decisions are driven by **Reinforcement Learning** to explore clinical dimensions (mood, sleep, weight). It injects `USER_CONTEXT` into greetings for continuity.
* **Q-Tables**: Maintains a `pandas.DataFrame` to decide which dimension to ask about next, masking out topics already covered.


* **Semantic Processing & Validation**:
* **Response Analyzer**: Wraps the local Llama model (via Ollama) to parse unstructured input into `(Dimension, Score)` tuples.
* **Questioner**: Manages retry logic if answers are ambiguous.
* **Reflection & Validation (RV)**: Validates if follow-ups are on-topic and provides empathetic validation.


* **CBT Module (`src/CBT.py`)**: Triggered for dimensions receiving a severe score (Score 2). It executes a 3-stage protocol: **Identify** unhelpful thoughts, **Challenge** them, and **Reframe** into balanced thoughts.
* **Persistence Layer**: `db_manager.py` persists conversational turns, user preferences, and historical summaries. It acts as the clinical record of the AI's internal reasoning.

---

## 📂 Project Structure

```text
.
├── src/                        # Backend & Logic Core
│   ├── perception/             # Audio, VAD (webrtcvad), STT (Whisper)
│   ├── cognition/              # RL Logic, CBT Protocol, Response Analysis
│   ├── action/                 # TTS (Piper) and Audio Playback
│   ├── database/               # SQLite DB Manager & Schema logic
│   └── utils/                  # Config loader, AI Privacy Logging, IPC Queues
├── data/                       # Persistent SQLite DB, Q-Tables, and CSV transcripts
├── services/                   # Frontend Services (Speech Client)
├── LLM_therapist_Application.py  # Main API Server & Session Orchestrator
├── start_headless.sh           # Master launch script
├── stop_system.sh              # Master termination script
├── deploy_and_run.sh           # One-command Deployment + Execution
├── config.yaml                 # System sensitivity and model settings
└── readme.md                   # You are here!

```

---

## 🛠️ Installation and Setup

### 1. Prerequisites

* **Hardware**: NVIDIA Jetson (Orin series) or Linux PC with Mic/Speaker.
* **Software**: Python 3.10+, `portaudio19-dev`.
* **External**: Ollama (running Llama 3.1 8B).

### 2. Local Installation

```bash
# Install system dependencies
sudo apt-get install portaudio19-dev python3-venv

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

### 3. Configuration

Modify your environment settings in `.env` or `config.yaml`:

```bash
JETSON_HOST=""
OLLAMA_HOST=""

```

---

## 🚀 Managing the System

| Command | Action | Description |
| --- | --- | --- |
| `./start_headless.sh` | **Start** | Spawns Backend and Client processes in the background. |
| `./stop_system.sh` | **Stop** | Kills all CaiTI processes and frees ALSA audio buffers. |
| `./deploy_and_run.sh` | **Deploy** | Syncs code to Jetson, restarts services, and attaches to live logs. |
| `tail -f *.log` | **Monitor** | View real-time AI "reasoning" and transcriptions. |

---

## 🔄 Summary Workflow

1. **Wake**: User says "Hello CaiTI" $\rightarrow$ `SpeechInteractionLoop` detects wake word and identifies user.
2. **Personalize**: `HandlerRL` loads user context from SQLite and picks the highest-priority dimension from the Q-Table.
3. **Screen**: AI generates a naturally phrased question; `Piper` speaks it.
4. **Analyze**: User answers; `Whisper` transcribes; `ResponseAnalyzer` scores the answer (0, 1, or 2).
5. **Intervene**: If a score of 2 is detected, the **CBT Protocol** initiates (Identify $\rightarrow$ Challenge $\rightarrow$ Reframe).
6. **Persistence**: Entire turn, including internal RL values and semantic scores, is committed to `record.csv` and SQLite.
7. **Reset**: Session ends; memory is purged; system returns to `AMBIENT_IDLE`.

---

## 📚 Key Technologies

* **STT**: Faster-Whisper (Local OpenAI implementation)
* **TTS**: Piper (High-speed neural TTS)
* **LLM**: Llama 3.1 8B (via Ollama)
* **VAD**: WebRTC-VAD (Voice Activity Detection)
* **Logic**: Reinforcement Learning (Q-Learning) via Pandas
* **API**: FastAPI & Uvicorn
