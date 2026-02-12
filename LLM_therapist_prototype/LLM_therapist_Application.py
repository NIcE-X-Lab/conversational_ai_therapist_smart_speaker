import os
import time
import threading
import queue
import logging

from src.handler_rl import HandlerRL
from src.utils.io_record import init_record, OUTPUT_QUEUE, INPUT_QUEUE
from src.utils.log_util import get_logger

# Import Perception and Action modules
from src.perception.audio import AudioRecorder
from src.perception.stt import STTGenerator
from src.perception.stt import STTGenerator
from src.action.tts import TTSGenerator
from src.action.player import AudioPlayer

# FastAPI Bridge
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio

logger = get_logger("MainApp")

# FastAPI App
app = FastAPI()

# CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.utils import io_record
from src.utils.config_loader import SUBJECT_ID

@app.get("/api/status")
def get_status():
    current_status = "ready"
    if hasattr(app.state, 'speech_loop'):
        current_status = app.state.speech_loop.state

    return {
        "status": current_status, 
        "subject_id": SUBJECT_ID,
        "session_id": io_record.SESSION_ID
    }

@app.get("/api/turns")
def get_turns():
    if io_record.DB and io_record.SESSION_ID:
        # get_session_history returns list of dicts: {speaker, text, meta_data}
        history = io_record.DB.get_session_history(io_record.SESSION_ID)
        return history
    return []

@app.post("/api/action")
def post_action(action: dict):
    # Handle actions
    action_type = action.get("type")
    
    if action_type == "stop":
        # Stop current audio / reset session logic if needed
        # For now, just logging. Detailed stop logic would require interrupting Action/Player.
        logger.info("Received STOP command")
        if hasattr(app.state, 'speech_loop'):
             app.state.speech_loop.stop_audio()
        return {"status": "stopped"}
        
    elif action_type == "start_listening":
        logger.info("Received START_LISTENING command")
        if hasattr(app.state, 'speech_loop'):
             app.state.speech_loop.manual_input_event.set()
        return {"status": "listening_triggered"}
        
    elif action_type == "set_mode":
        mode = action.get("mode") # "hands_free" or "manual"
        logger.info(f"Setting mode to {mode}")
        if hasattr(app.state, 'speech_loop'):
            app.state.speech_loop.is_hands_free = (mode == "hands_free")
        return {"status": "mode_set", "mode": mode}

@app.post("/api/pause")
def pause_session():
    if hasattr(app.state, 'speech_loop'):
        app.state.speech_loop.set_paused(True)
    return {"status": "paused"}

@app.post("/api/resume")
def resume_session():
    if hasattr(app.state, 'speech_loop'):
        app.state.speech_loop.set_paused(False)
    return {"status": "resumed"}

@app.post("/api/end_session")
def end_session():
    logger.info("Ending session via API.")
    if hasattr(app.state, 'speech_loop'):
        app.state.speech_loop.stop_audio()
    
    # Signal HandlerRL to stop
    # Signal HandlerRL to stop
    io_record.END_SESSION_EVENT.set()
    io_record.START_SESSION_EVENT.clear() # Block future sessions until login
    # Unblock any waiting input
    io_record.INPUT_QUEUE.put("SESSION_END")
    
    return {"status": "session_ended"}

@app.post("/api/login")
def login_user(data: dict):
    user_type = data.get("user_id", "test_user")
    logger.info(f"Logging in user: {user_type}")
    
    # Reset session with new user
    if user_type == "new_user":
        import uuid
        uid = f"user_{str(uuid.uuid4())[:8]}"
    else:
        uid = "test_user"
        
    io_record.reset_session(uid)

    # Signal session start
    io_record.END_SESSION_EVENT.clear()
    io_record.START_SESSION_EVENT.set() 
    
    return {"status": "logged_in", "user_id": uid, "session_id": io_record.SESSION_ID}
    return {"status": "logged_in", "user_id": uid, "session_id": io_record.SESSION_ID}

@app.post("/api/input")
def receive_input(data: dict):
    """Bridge for Speech Service to push user text."""
    text = data.get("text")
    if text:
        logger.info(f"API Input received: {text}")
        io_record.INPUT_QUEUE.put(text)
    return {"status": "received"}

@app.get("/api/output")
def get_output():
    """Bridge for Speech Service to poll agent text."""
    try:
        text = str(OUTPUT_QUEUE.get_nowait())
        logger.info(f"API Output served: {text}")
        return {"text": text}
    except queue.Empty:
        return {"text": None}
def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

class SpeechInteractionLoop:
    """
    Handles the Audio Inputs/Outputs:
    Agent Question -> TTS -> Speaker
    Microphone -> VAD -> STT -> Agent Response
    """
    def __init__(self):
        logger.info("Initializing Speech Interaction Modules...")
        self.recorder = AudioRecorder()
        self.stt = STTGenerator()
        self.tts = TTSGenerator()
        self.player = AudioPlayer()
        self.running = True
        
        # Controls
        self.manual_input_event = threading.Event()
        self.stop_playback_event = threading.Event()
        self.is_hands_free = True
        self.paused = False
        self.state = "idle"  # idle, playing, listening, processing, paused

    def set_paused(self, paused: bool):
        self.paused = paused
        if paused:
            self.state = "paused"
            self.stop_audio()
        else:
            self.state = "idle"

    def run(self):
        logger.info("Starting Speech Interaction Loop.")
        while self.running:
            try:
                if self.paused:
                    time.sleep(0.5)
                    continue

                # 1. Check for Agent Output (Question)
                try:
                    text_to_speak = OUTPUT_QUEUE.get(timeout=0.1)
                    if text_to_speak:
                        self.process_agent_turn(text_to_speak)
                except queue.Empty:
                    pass

                # The loop continues to poll/wait. 
                # Note: In a turn-taking system, we usually wait for Agent to speak first (or Greeting),
                # providing the floor to the user only after Agent speaks.
                # However, HandlerRL initiates with a greeting (via set_question_prefix/log_question).
                # So we just wait for OUTPUT_QUEUE.
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in speech loop: {e}")
                time.sleep(1)

    def process_agent_turn(self, text):
        """
        Agent has spoken. Synthesize speech, play it, then listen for user response.
        """
        logger.info(f"Agent says: {text}")
        
        # 1. TTS
        self.state = "processing"
        wav_file = "response_temp.wav"
        if self.tts.generate(text, wav_file):
            # 2. Play Audio (Interruptible)
            self.state = "playing"
            self.stop_playback_event.clear()
            self.player.play(wav_file, stop_event=self.stop_playback_event)
        
        # Check if stopped during playback
        if self.stop_playback_event.is_set():
            logger.info("Turn interrupted. Skipping listening.")
            self.state = "idle"
            return

        # 3. Listen for User Response
        self.state = "processing" # Waiting/Transition
        time.sleep(1.5) # Increased wait to avoid echo/self-transcription
        
        # Clear Input Queue to remove stale data/echoes
        with INPUT_QUEUE.mutex:
            INPUT_QUEUE.queue.clear()
            
        logger.info("Listening for user response (Max 15s wait)...")
        
        # Logic: Hands-Free vs Manual
        if not self.is_hands_free:
            self.state = "idle" # Waiting for trigger
            logger.info("Manual Mode: Waiting for trigger...")
            self.manual_input_event.wait()
            self.manual_input_event.clear()
            logger.info("Trigger received. Listening...")
        
        self.state = "listening"
        audio_frames = self.recorder.record_until_silence(max_duration=15.0)
        
        if not audio_frames:
            logger.warning("No audio recorded (Timeout or Silence).")
            user_text = "" # Send empty to indicate silence/no-response
        else:
            # 4. Save and STT
            self.state = "processing"
            user_wav = "user_input_temp.wav"
            self.recorder.save_wav(audio_frames, user_wav)
            user_text = self.stt.transcribe(user_wav)
        
        logger.info(f"User Transcribed: {user_text}")
        
        # Voice Command Check
        lower_text = user_text.lower().strip()
        if "end session" in lower_text or "stop session" in lower_text:
            logger.info("Voice Command: End Session detected.")
            end_session() # Call API function directly
            return

        # 5. Send to Agent (Back to Cognition)
        # Note: io_record will also log this to record.csv (Frontend Requirement)
        INPUT_QUEUE.put(user_text)
        self.state = "idle"

    def stop_audio(self):
        """External hook to stop audio."""
        logger.info("Stopping audio playback...")
        self.stop_playback_event.set()
        # Also interrupt manual wait if stuck there
        if not self.is_hands_free:
             self.manual_input_event.set()

    def cleanup(self):
        self.running = False
        if self.recorder: self.recorder.terminate()
        if self.player: self.player.terminate()


def main():
    """
    Main entry point for the application.
    """
    # Initialize the record system (queues + DB)
    init_record()

    # Start API Server in separate thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    logger.info("API Server started on port 8000")

    # Remove SpeechInteractionLoop from Main Logic
    # It is now a separate service.
    
    # Start the main RL workflow (this will drive the therapy session)
    try:
        while True:
            # Wait for valid session via Login OR Voice Command
            if not io_record.START_SESSION_EVENT.is_set():
                # Listen for "Start Session" command while idle?
                # This requires SpeechLoop to be listening.
                # Currently SpeechLoop only listens when triggered by Agent or Manual.
                # To support "Start Session", we'd need a background listening loop.
                # For now, we rely on the Login UI button as the primary trigger.
                time.sleep(1)
                continue
                
            io_record.END_SESSION_EVENT.clear()
            logger.info(f"Starting HandlerRL for Session {io_record.SESSION_ID}")
            HandlerRL().run()
            
            # If run() returns, session ended
            logger.info("HandlerRL finished. Waiting for next login.")
            io_record.START_SESSION_EVENT.clear()
            
    except KeyboardInterrupt:
        logger.info("Application interrupted.")
    finally:

        # cleanup if needed
        pass
        # Give cleanup time
        time.sleep(0.5)

if __name__ == "__main__":
    main()