# Smart-Speaker Micro-Intervention System (Local Jetson Deployment)

## 🌟 Overview
This project implements a fully local, privacy-preserving smart-speaker system designed to deliver **Motivational Interviewing (MI)** and **Cognitive Behavioral Therapy (CBT)** micro-interventions. 

Built for the NVIDIA Jetson platform, the system operates entirely offline to ensure maximum privacy and low latency. It bridges a robust Python-based speech-cognition loop with a modern React-based frontend dashboard for real-time monitoring and control.

---

## 🏗️ Architecture

The system follows a modular three-layer architecture with an integrated **Governance Layer** and **FastAPI Bridge**.

### 1. Perception Layer ("The Ears")
*   **Audio Capture**: High-fidelity capture using `pyaudio`.
*   **VAD (Voice Activity Detection)**: Uses `webrtcvad` for silence filtering and turn-taking.
*   **STT (Speech-to-Text)**: Local transcription using `faster-whisper` (Defaults to `base.en`).

### 2. Cognition Layer ("The Brain")
*   **Orchestration**: `HandlerRL` manages therapeutic logic using Reinforcement Learning (Contextual Bandits) to select optimal interventions.
*   **AI Governance Layer**: A reflection-validation loop (`reflection_validation.py`) that:
    *   Evaluates if user responses are on-topic.
    *   Guides users back to the therapeutic focus if they drift.
    *   Provides empathic validation for shared experiences.
*   **Memory & Persistence**: SQLite (`data/therapist.db`) manages:
    *   **User Profiles & Preferences**: Long-term memory of user needs.
    *   **Session Summaries**: Compressed history of past interactions.
    *   **Safety Flags**: Detecting and logging high-severity concerns.
*   **FastAPI Bridge**: Synchronizes the internal speech loop with the external frontend via REST API.

### 3. Action Layer ("The Voice")
*   **TTS (Text-to-Speech)**: Local speech synthesis using `piper` (Default: `en_US-amy-medium`).
*   **Audio Playback**: Low-latency synthesis and playback via `pyaudio`.

---

## 📂 Project Structure

```text
.
├── LLM_therapist_prototype/    # Backend & Logic Core
│   ├── src/
│   │   ├── perception/         # Audio, VAD, STT modules
│   │   ├── cognition/          # RL Logic & CBT frameworks
│   │   ├── action/             # TTS and Audio Playback
│   │   ├── database/           # SQLite DB Manager & Schema
│   │   └── utils/              # Config, Logging, IPC Queues
│   ├── data/                   # Persistent SQLite DB and File logs
│   ├── LLM_therapist_Application.py  # Main Entry Point (Backend + API)
│   └── config.yaml             # System Configuration
├── frontend/                   # React + Vite Monitoring Dashboard
├── deploy_to_jetson.sh         # Automated Jetson deployment script
├── requirements.txt            # Python Dependencies
└── readme.md                   # You are here!
```

---

## 🛠️ Setup and Installation

### 1. Prerequisites
*   **Hardware**: NVIDIA Jetson (Orin Nano/NX/AGX) or a Linux PC with a Microphone/Speaker.
*   **Software**: Python 3.8+, Node.js (for manual frontend setup).

### 2. Local Setup (Linux PC)
```bash
# Install system dependencies
sudo apt-get install portaudio19-dev python3-venv

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r LLM_therapist_prototype/requirements.txt

# Install Frontend dependencies
cd frontend && npm install && cd ..
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
Edit `LLM_therapist_prototype/config.yaml` to configure:
*   VAD sensitivity and audio sample rates.
*   STT model sizes (e.g., `tiny.en`, `base.en`).
*   TTS voice paths and executable locations.

### ▶️ Running the System
#### 1. Start the Backend
```bash
cd LLM_therapist_prototype
python LLM_therapist_Application.py
```
*The backend exposes a FastAPI server at `http://localhost:8000`.*

#### 2. Start the Frontend Dashboard
In a separate terminal:
```bash
cd frontend
npm run dev -- --host
```
*Access the UI at `http://<machine-ip>:5173`. The dashboard allows for session login, pausing, and real-time transcript viewing.*

---

## 🎮 Controls & Interaction

| Action | Method | Description |
| :--- | :--- | :--- |
| **Login** | UI | Select "New User" or "Test User" to start a session. |
| **Pause/Resume** | UI | Temporarily halt the speech loop. |
| **End Session** | UI / Voice | Say "End Session" or click the End button. |
| **Hands-Free** | UI Toggle | Toggle between auto-VAD and manual trigger mode. |

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
*   **Web**: React, Vite, FastAPI, Uvicorn.
*   **Database**: SQLite3.

