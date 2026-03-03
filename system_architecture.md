# CaiTI Smart Speaker: System Architecture

The conversational AI therapist smart speaker is designed to provide localized, low-latency micro-interventions (MI and CBT). It leverages local models (Whisper for STT, Piper for TTS, and Llama 3.1 via Ollama for LLM logic) deployed on an NVIDIA Jetson device.

Below is a breakdown of the core components and how they interact.

## 1. Application Entry Point ([LLM_therapist_Application.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/LLM_therapist_Application.py) & [LLM_therapist_Application_server.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/LLM_therapist_Application_server.py))
- **FastAPI Backend ([_server.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/LLM_therapist_Application_server.py))**: Hosts the web API for incoming interaction logs and manual overrides.
- **Main App ([LLM_therapist_Application.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/LLM_therapist_Application.py))**: Orchestrates the entire flow. It starts the [SpeechInteractionLoop](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/LLM_therapist_Application.py#157-340) in a background thread to continuously listen for wake words and handles the primary Reinforcement Learning (RL) loop ([HandlerRL](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/handler_rl.py#32-341)).

## 2. Audio Pipeline ("Arth" Modules: [src/perception/audio.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/perception/audio.py) & [src/action/tts.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/action/tts.py) & [src/perception/stt.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/perception/stt.py))
- **Input (AudioRecorder)**: Uses `pyaudio` to stream bytes from the local microphone. It integrates `webrtcvad` for Voice Activity Detection (VAD) to dynamically detect speech (Audio Event Detection). When speech starts, it records until a predefined silence timeout is reached. It also aggressively flushes the ALSA buffer during initialization to prevent the mic from capturing the device's own TTS output (Echo Cancellation).
- **Processing (STT & TTS)**:
  - **STT**: `faster-whisper` decodes the recorded raw audio into transcripts locally.
  - **TTS**: `piper-tts` generates synthesized voice responses locally.
- **SpeechInteractionLoop**: The background worker that bridges the audio hardware to the backend. It continuously listens for wake words ("Hello", "Start", etc.) while idle, initiates a session when triggered, asks for user identity, and then relays questions and answers to the `io_record` queues.

## 3. Policy & Handler ("Eric" Modules: [src/handler_rl.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/handler_rl.py) & [src/utils/rl_qtables.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/utils/rl_qtables.py))
- **HandlerRL**: The primary logic engine driving the screening session. It iterates through different conversational dimensions (topics like mood, sleep, weight) using an RL approach.
- **Q-Tables**: The agent maintains a state-action Q-table (`pandas.DataFrame`) to decide which dimension to ask about next. Actions (dimensions) are masked out once they are exhaustively discussed.
- **Evaluation Loop**: For each topic, it asks a question, gets the user's transcript, and uses the `response_analyzer` to classify the user's answer into a score (0: fine, 1: minor issue, 2: severe issue).

## 4. Semantic Processing & Validation ([src/questioner.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/questioner.py) & [src/response_analyzer.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/response_analyzer.py) & [src/reflection_validation.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/reflection_validation.py))
- **Response Analyzer**: Wraps the local Llama model (via Ollama) to parse unstructured user input into discrete [(Dimension, Score)](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/handler_rl.py#83-237) tuples, and generates third-person reflective summaries.
- **Questioner**: Manages the retry logic if a user's answer is ambiguous (e.g., "I don't know"), asking the question from a different angle.
- **Reflection & Validation (RV)**: Validates if the user's follow-up response is on-topic. If related, it produces an empathetic validation. If unrelated, it generates a guide to steer the user back.

## 5. Cognitive Behavioral Therapy Module ([src/CBT.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/CBT.py))
- Triggered at the end of the session, the CBT module zeroes in on dimensions that received a critical score of 2.
- It walks the user through a 3-stage protocol:
  - **Stage 1 (Identify)**: Asking the user to identify unhelpful thoughts related to their statement.
  - **Stage 2 (Challenge)**: Prompting the user to challenge those negative thoughts.
  - **Stage 3 (Reframe)**: Guiding the user to reframe their thoughts into balanced, constructive ones.
- Specialized LLM prompters validate the user's progress at each stage and provide guidance if they get stuck or exhibit cognitive distortions.

## 6. Persistence & IO ([src/database/db_manager.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/database/db_manager.py) & [src/utils/io_record.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/utils/io_record.py))
- **SQLite Database**: Persists conversational turns, user preferences, historical summaries, and safety flags across sessions. Provides context to seed the LLM.
- **CSV Logging**: [io_record.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/utils/io_record.py) handles writing the full conversational transcript (`Timestamp, Type, Speaker, Text`) incrementally to [data/record.csv](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/data/record.csv) for analytics, decoupling the system state from arbitrary IPC locks.

## Summary Workflow
1. User says "Hello CaiTI".
2. [SpeechInteractionLoop](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/LLM_therapist_Application.py#157-340) detects wake word, asks "Who is the user?", confirms identity, and signals session start.
3. [HandlerRL](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/handler_rl.py#32-341) loads user context from DB and uses Q-tables to pick the first dimension.
4. `LLM` generates a naturally phrased question, which `piper` speaks out loud.
5. User answers. `whisper` transcribes the audio.
6. `response_analyzer` scores the answer. If the answer needs clarification, [questioner.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/questioner.py) retries.
7. Once all dimensions are checked, [CBT.py](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/src/CBT.py) initiates an intervention on high-score items.
8. Entire log is appended to [record.csv](file:///home/arthv/projects/unc_lab/conversational_ai_therapist_smart_speaker/data/record.csv) and SQLite.
