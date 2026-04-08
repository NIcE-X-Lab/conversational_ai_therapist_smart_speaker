# Smart-Speaker Micro-Intervention System (Local Jetson Deployment)

## 🌟 Overview

This project implements a fully local, privacy-preserving smart-speaker system—**CaiTI** (Conversational AI Therapist Interface)—designed to deliver **Motivational Interviewing (MI)** and **Cognitive Behavioral Therapy (CBT)** micro-interventions.

Built specifically for the **NVIDIA Jetson** platform (Orin Nano/NX/AGX), the system operates entirely offline. It functions as a headless audio assistant, leveraging a robust Python-based speech-client loop that communicates with a local Dialogue Engine powered by a fine-tuned **Gemma 2B** model.

---

## 🧠 Design Philosophy & Core Pillars

The CaiTI architecture is built on three non-negotiable pillars:

1. **Clinical Observability**: Standard LLM logs are insufficient for clinical research. This system implements a **"Reasoning Logic"** stream. Every conversational turn records the exact RL Q-values, Semantic Scores, and Validation Flags, allowing researchers to audit *why* the AI chose a specific clinical dimension. All sessions are logged independently to `sessions/[UserName]/[Timestamp].log`.
2. **Edge Privacy**: By using `faster-whisper`, `SpeechBrain`, `Piper`, and local `Ollama` models, sensitive patient data never leaves the Jetson device. Ambient states are handled locally, only triggering the expensive LLM layer upon high-probability intent detection.
3. **Sandboxed State Management**: To prevent "user-leakage," the system utilizes a strict session lifecycle. When a session terminates, the Reinforcement Learning memory (Pandas DataFrames) is explicitly purged, ensuring the next user starts with a clean slate seeded only by their specific historic database context.

---

## 🏗️ System Architecture & Logic

The system is a multi-process application designed for high observability, multi-modal perception, and low-latency response.

### 1. Dual-Pipeline Audio Perception ("Arth" Modules)
The frontend utilizes a highly robust audio recording loop (`AudioRecorder`) initialized via PyAudio and WebRTC-VAD. When user speech is captured, two parallel thread pools execute on the raw `wav` chunk:
- **Speech-to-Text (STT)**: `faster-whisper` decodes text.
- **Speech Emotion Recognition (SER)**: `SpeechBrain` (Wav2Vec2) parses vocal tonality to label the emotion.
- The outcome is packaged dynamically into a single IPC queue payload: `{"transcript": "...", "detected_emotion": "..."}`.

### 2. Asynchronous "Latency Shield" & Interstitial Screening
CaiTI runs entirely locally, meaning Llama inference on Jetson can occasionally exceed acceptable auditory thresholds. To prevent unnatural silence, an asynchronous latency shield covers every LLM completion:
- A `concurrent.futures` watcher tracks generation latency. 
- If 3.0s is exceeded, CaiTI outputs an **Interstitial Exercise** (e.g., breath counting, nostril awareness) OR triggers a **GAD-2 Clinical Screening**.
- **Opt-Out Control**: While generating, `INPUT_QUEUE` listens for variations of "Stop, "Skip", or "I don't want to". If a user opts out over the mic, the system instantly bypasses processing and launches `waiting_music.wav` loop until the model finishes decoding.

### 3. State Management & Q-Learning Policy
- **RL Handler**: Decisions run on Reinforcement Learning to explore clinical dimensions (mood, sleep, weight). It injects `USER_CONTEXT` into greetings. The Q-Table maps the optimum dimension trajectory per individual session. 
- **CBT Module (`src/CBT.py`)**: Triggered when a dimension receives a severe categorization (Score 2). Executes a 3-stage protocol: **Identify** unhelpful thoughts, **Challenge** them, and **Reframe** into balanced thoughts.

### 4. Logging and Persistence
- `db_manager.py` dynamically pushes data to a unified SQLite format (`therapist.db`). 
- Interstitial fallback screenings mapping $\geq 2$ severity proactively flag `[HIGH-PRIORITY-REVIEW]` attributes into the permanent CSV trail for immediate psychotherapist oversight.

---

## 📂 Project Structure

```text
.
├── src/                        # Backend & Logic Core
│   ├── perception/             # Audio, VAD, STT (Whisper + SpeechBrain SER)
│   ├── cognition/              # RL Logic, CBT Protocol, Response Analysis
│   ├── action/                 # TTS (Piper) and Audio Playback
│   ├── database/               # SQLite DB Manager & Schema logic
│   └── utils/                  # Async LLM wrappers, IO Queues, Loggers
├── sessions/                   # Unique CSV logging chronologies per patient
├── data/                       # Persistent SQLite DB, Q-Tables
├── services/                   # Frontend Services (Speech Client)
├── LLM_therapist_Application.py  # Main API Server & Session Orchestrator
├── start_headless.sh           # Master launch script
├── stop_system.sh              # Master termination script
├── deploy_and_run.sh           # One-command Deployment + Execution
├── config.yaml                 # System sensitivity and model settings
├── Modelfile                   # System prompt + LLM configs for Ollama
└── readme.md                   # You are here!
```

---

## 🛠️ Installation and Setup

### 1. Prerequisites

* **Hardware**: NVIDIA Jetson (Orin series) or Linux PC with Mic/Speaker.
* **Software**: Python 3.10+, `portaudio19-dev`.
* **External**: Ollama (running the fine-tuned `gemma:2b-caiti` model).

### 2. Local Installation

```bash
# Install system dependencies
sudo apt-get install portaudio19-dev python3-venv

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install speechbrain torchaudio
```

### 3. Configuration

Modify your environment settings in `.env`. By default `start_headless.sh` auto-deploys `llama3.2-caiti` via the included Modelfile:

```bash
JETSON_HOST="arth@152.23.X.X"
OPENAI_BASE_URL="http://localhost:11434/v1"
OPENAI_MODEL="gemma:2b-caiti"
```

---

## 🚀 Managing the System

| Command | Action | Description |
| --- | --- | --- |
| `./start_headless.sh` | **Start** | Spawns Backend and Client processes. Auto-builds Gemma Modelfile. |
| `./stop_system.sh` | **Stop** | Kills all CaiTI processes and frees ALSA audio buffers. |
| `./deploy_and_run.sh` | **Deploy** | Syncs code to Jetson, restarts services, and attaches to live logs. |
| `tail -f *.log` | **Monitor** | View real-time AI "reasoning", async latency hits, and transcriptions. |

---

## 🔄 Summary Workflow

1. **Wake**: User says "Hello CaiTI" $\rightarrow$ `SpeechInteractionLoop` detects wake word and identifies user.
2. **Personalize**: `HandlerRL` loads user context from SQLite and picks the highest-priority dimension from the Q-Table.
3. **Screen**: AI generates a naturally phrased question; `Piper` speaks it using `--length_scale 0.8` param for therapeutic, calm pacing.
4. **Analyze**: User answers; `Whisper` transcribes & `SpeechBrain` detects emotion; `ResponseAnalyzer` scores the answer (0, 1, or 2) while interpreting the physiological emotion.
5. **Latency Shield**: If LLM parsing surpasses 3.0 seconds, ambient interstitial routing fires off CBT grounding exercises or GAD-2 fallback screenings.
6. **Intervene**: If a score of 2 is detected, the **CBT Protocol** initiates (Identify $\rightarrow$ Challenge $\rightarrow$ Reframe).
7. **Persistence**: Entire turn is logged to `sessions/USER/TIMESTAMP.log`.
8. **Reset**: Session ends; memory is purged; system returns to `AMBIENT_IDLE`.

---

## 📚 Key Technologies

* **STT**: Faster-Whisper (Local OpenAI implementation)
* **SER**: SpeechBrain (wav2vec2-IEMOCAP)
* **TTS**: Piper (High-speed neural TTS)
* **LLM**: Gemma 2B (fine-tuned: `gemma:2b-caiti`)
* **VAD**: WebRTC-VAD (Voice Activity Detection)
* **Logic**: Reinforcement Learning (Q-Learning) via Pandas
* **API**: FastAPI & Uvicorn
