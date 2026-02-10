# Smart-Speaker Micro-Intervention System (Local Jetson Deployment)

## Overview
This project implements a fully local smart-speaker system on the NVIDIA Jetson platform. The system is designed to screen everyday functioning and deliver Motivational Interviewing (MI) and Cognitive Behavioral Therapy (CBT) micro-interventions. It operates entirely offline to ensure privacy and low latency, utilizing local models for speech processing and cognition.

## Architecture

The system follows a modular three-layer architecture:

### 1. Perception Layer ("The Ears")
-   **Audio Capture**: Captures raw audio from the microphone using `pyaudio`.
-   **VAD (Voice Activity Detection)**: Uses `webrtcvad` to detect speech and filter silence.
    -   **Timeout**: Automatically stops recording after 15 seconds of silence (configurable).
    -   **Wake Word**: Implicitly supports "start speaking to activate" (session-based).
-   **STT (Speech-to-Text)**: Local transcription using `faster-whisper`.
    -   **Model**: Defaults to `base.en` (configurable in `config.yaml`).

### 2. Cognition Layer ("The Brain")
-   **Orchestration**: `HandlerRL` manages the therapeutic logic, selecting questions and interventions based on Reinforcement Learning (Contextual Bandits).
-   **LLM (Large Language Model)**: Generates natural language responses. Supports local LLMs (e.g., Llama 3 via Ollama) or OpenAI-compatible APIs.
-   **Database**: SQLite (`data/therapist.db`) stores all session data, including:
    -   `users`: User profiles.
    -   `sessions`: Interaction sessions.
    -   `turns`: Individual conversation turns (User input, Agent output, Metadata).
-   **IPC (Inter-Process Communication)**: Uses thread-safe Queues for low-latency communication between the Speech Loop and the Logic Core.
    -   **Legacy Sync**: Also writes transcripts to `data/record.csv` for compatibility with external frontend tools.

### 3. Action Layer ("The Voice")
-   **TTS (Text-to-Speech)**: Local speech synthesis using `piper`.
    -   **Voice**: Configurable (default: `en_US-amy-medium`).
-   **Playback**: Low-latency audio playback using `pyaudio`.

## Project Structure
```
LLM_therapist_prototype/
├── data/                   # Database and CSV logs
├── src/
│   ├── perception/         # Audio, VAD, STT modules
│   │   ├── audio.py
│   │   └── stt.py
│   ├── cognition/          # (Logic resides in handler_rl.py)
│   ├── action/             # TTS, Player modules
│   │   ├── tts.py
│   │   └── player.py
│   ├── database/           # SQLite manager
│   │   └── db_manager.py
│   └── utils/              # Config, Logging, IPC
├── LLM_therapist_Application.py  # Main Entry Point
├── config.yaml             # Configuration
└── environment.yml         # Dependencies
```

## Setup and Installation

### Prerequisites
-   NVIDIA Jetson (Orin Nano/NX/AGX) or Linux PC.
-   Python 3.8+.
-   Audio Hardware: Microphone and Speaker.

### Dependencies
Install system dependencies (for PyAudio):
```bash
sudo apt-get install portaudio19-dev
```

Install Python dependencies:
```bash
pip install -r requirements.txt
# OR
conda env update -f environment.yml
```

### Models
1.  **Whisper**: The `faster-whisper` model will be downloaded automatically on first run.
2.  **Piper**: Ensure the `piper` executable is in your PATH or configured in `config.yaml`.
    -   Download voices (onnx + json) to `models/piper/`.

## Usage

### 1. Configuration
Edit `config.yaml` to set your preferences:
```yaml
audio:
  sample_rate: 16000
  vad_aggressiveness: 3

stt:
  model_path: "base.en"

tts:
  executable_path: "/path/to/piper"
```

### 2. Running the System
To start the full smart speaker application (Backend + API):
```bash
python LLM_therapist_Application.py
```
-   The backend also starts a FastAPI server at `http://localhost:8000/api`.

### 3. Running the Frontend UI
To launch the web interface (desktop/mobile view):
1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies (first time only):
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
4.  Open your browser to the URL shown (e.g., `http://localhost:5173`).

The UI will automatically connect to the running Python backend and display the conversation in real-time.

## Verification
Run the integration tests to verify the pipeline:
```bash
python -m unittest tests/test_refinement.py
```

## Abbreviations Glossary
-   **VAD**: Voice Activity Detection
-   **STT/ASR**: Speech-to-Text / Automatic Speech Recognition
-   **TTS**: Text-to-Speech
-   **LLM**: Large Language Model
-   **RL**: Reinforcement Learning
-   **CBT**: Cognitive Behavioral Therapy
-   **MI**: Motivational Interviewing
