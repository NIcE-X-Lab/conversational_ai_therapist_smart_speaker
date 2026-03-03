# Smart-Speaker Micro-Intervention System (Local Jetson Deployment)

## 🌟 Overview
This project implements a fully local, privacy-preserving smart-speaker system designed to deliver **Motivational Interviewing (MI)** and **Cognitive Behavioral Therapy (CBT)** micro-interventions. 

Built for the NVIDIA Jetson platform, the system operates entirely offline to ensure maximum privacy and low latency. It functions as a headless audio assistant, relying on a robust Python-based speech-client loop that communicates with a local Dialogue Engine.

---

## 🏗️ Architecture

The conversational AI therapist smart speaker is designed to provide localized, low-latency micro-interventions (MI and CBT). It leverages local models (Whisper for STT, Piper for TTS, and Llama 3.1 via Ollama for LLM logic) deployed on an NVIDIA Jetson device.

Below is a breakdown of the core components and how they interact.

### 1. Application Entry Point (`LLM_therapist_Application.py` & `LLM_therapist_Application_server.py`)
- **FastAPI Backend (`_server.py`)**: Hosts the web API for incoming interaction logs and manual overrides.
- **Main App (`LLM_therapist_Application.py`)**: Orchestrates the entire flow. It starts the `SpeechInteractionLoop` in a background thread to continuously listen for wake words and handles the primary Reinforcement Learning (RL) loop (`HandlerRL`).

### 2. Audio Pipeline ("Arth" Modules: `src/perception/audio.py` & `src/action/tts.py` & `src/perception/stt.py`)
- **Input (AudioRecorder)**: Uses `pyaudio` to stream bytes from the local microphone. It integrates `webrtcvad` for Voice Activity Detection (VAD) to dynamically detect speech (Audio Event Detection). When speech starts, it records until a predefined silence timeout is reached. It also aggressively flushes the ALSA buffer during initialization to prevent the mic from capturing the device's own TTS output (Echo Cancellation).
- **Processing (STT & TTS)**:
  - **STT**: `faster-whisper` decodes the recorded raw audio into transcripts locally.
  - **TTS**: `piper-tts` generates synthesized voice responses locally.
- **SpeechInteractionLoop**: The background worker that bridges the audio hardware to the backend. It continuously listens for wake words ("Hello", "Start", etc.) while idle, initiates a session when triggered, asks for user identity, and then relays questions and answers to the `io_record` queues.

### 3. Policy & Handler ("Eric" Modules: `src/handler_rl.py` & `src/utils/rl_qtables.py`)
- **HandlerRL**: The primary logic engine driving the screening session. It iterates through different conversational dimensions (topics like mood, sleep, weight) using an RL approach.
- **Q-Tables**: The agent maintains a state-action Q-table (`pandas.DataFrame`) to decide which dimension to ask about next. Actions (dimensions) are masked out once they are exhaustively discussed.
- **Evaluation Loop**: For each topic, it asks a question, gets the user's transcript, and uses the `response_analyzer` to classify the user's answer into a score (0: fine, 1: minor issue, 2: severe issue).

### 4. Semantic Processing & Validation (`src/questioner.py` & `src/response_analyzer.py` & `src/reflection_validation.py`)
- **Response Analyzer**: Wraps the local Llama model (via Ollama) to parse unstructured user input into discrete `(Dimension, Score)` tuples, and generates third-person reflective summaries.
- **Questioner**: Manages the retry logic if a user's answer is ambiguous (e.g., "I don't know"), asking the question from a different angle.
- **Reflection & Validation (RV)**: Validates if the user's follow-up response is on-topic. If related, it produces an empathetic validation. If unrelated, it generates a guide to steer the user back.

### 5. Cognitive Behavioral Therapy Module (`src/CBT.py`)
- Triggered at the end of the session, the CBT module zeroes in on dimensions that received a critical score of 2.
- It walks the user through a 3-stage protocol:
  - **Stage 1 (Identify)**: Asking the user to identify unhelpful thoughts related to their statement.
  - **Stage 2 (Challenge)**: Prompting the user to challenge those negative thoughts.
  - **Stage 3 (Reframe)**: Guiding the user to reframe their thoughts into balanced, constructive ones.
- Specialized LLM prompters validate the user's progress at each stage and provide guidance if they get stuck or exhibit cognitive distortions.

### 6. Persistence & IO (`src/database/db_manager.py` & `src/utils/io_record.py`)
- **SQLite Database**: Persists conversational turns, user preferences, historical summaries, and safety flags across sessions. Provides context to seed the LLM.
- **CSV Logging**: `io_record.py` handles writing the full conversational transcript (`Timestamp, Type, Speaker, Text`) incrementally to `data/record.csv` for analytics, decoupling the system state from arbitrary IPC locks.

### Summary Workflow
1. User says "Hello CaiTI".
2. `SpeechInteractionLoop` detects wake word, asks "Who is the user?", confirms identity, and signals session start.
3. `HandlerRL` loads user context from DB and uses Q-tables to pick the first dimension.
4. `LLM` generates a naturally phrased question, which `piper` speaks out loud.
5. User answers. `whisper` transcribes the audio.
6. `response_analyzer` scores the answer. If the answer needs clarification, `questioner.py` retries.
7. Once all dimensions are checked, `CBT.py` initiates an intervention on high-score items.
8. Entire log is appended to `record.csv` and SQLite.

---

## 📂 Project Structure

```text
.
├── src/                        # Backend & Logic Core
│   ├── perception/             # Audio, VAD, STT modules
│   ├── cognition/              # RL Logic & CBT frameworks
│   ├── action/                 # TTS and Audio Playback
│   ├── database/               # SQLite DB Manager & Schema
│   └── utils/                  # Config, Logging, IPC Queues
├── data/                       # Persistent SQLite DB and File logs
├── services/                   # Microservices (e.g., speech client)
├── LLM_therapist_Application.py  # Main Entry Point (Backend + API)
├── start_headless.sh           # Script to launch backend and speech service
├── config.yaml                 # System Configuration
├── deploy_to_jetson.sh         # Automated Jetson deployment script
├── requirements.txt            # Python Dependencies
└── readme.md                   # You are here!
```

---

## 🛠️ Setup and Installation

### 1. Prerequisites
*   **Hardware**: NVIDIA Jetson (Orin Nano/NX/AGX) or a Linux PC with a Microphone/Speaker.
*   **Software**: Python 3.8+

### 2. Local Setup (Linux PC)
```bash
# Install system dependencies
sudo apt-get install portaudio19-dev python3-venv

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Automated Jetson Deployment
The `deploy_to_jetson.sh` script automates the process of syncing code, installing dependencies, and running a sanity check on a remote Jetson device.
```bash
# Make the script executable
chmod +x deploy_to_jetson.sh

# Run the deployment (update JETSON_IP in the script first)
./deploy_to_jetson.sh
```

---

## 🚀 Usage

### ⚙️ Configuration
Edit `config.yaml` to configure:
*   VAD sensitivity and audio sample rates.
*   STT model sizes (e.g., `tiny.en`, `base.en`).
*   TTS voice paths and executable locations.

### ▶️ Running the System
```bash
chmod +x start_headless.sh
./start_headless.sh
```
*This starts both the Dialogue Engine (FastAPI backend) and the Speech Service in the background.*
*Check logs using `tail -f backend_session.log speech_service.log`.*

---

## 🎮 Controls & Interaction

| Action | Method | Description |
| :--- | :--- | :--- |
| **Start / Interaction** | Voice | Speak wake words to the system (e.g., "Hello CaiTI"). |
| **End Session** | Voice | Speak concluding remarks to end the interaction. |

---

## ✅ Verification
Run the integrated sanity check to verify the audio pipeline and database connection:
```bash
python sanity_check.py
```

---

## 📚 Key Technologies
*   **Speech**: Faster-Whisper, Piper, WebRTC-VAD.
*   **Cognition**: Python, Contextual Bandits, OpenAI-compatible local APIs.
*   **Database**: SQLite3.
