import os
import time
import requests
import threading
import logging
import queue
import numpy as np
import sys
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.perception.audio import AudioRecorder
from src.perception.stt import STTGenerator
from src.action.tts import TTSGenerator
from src.action.player import AudioPlayer
from src.utils.log_util import get_logger

logger = get_logger("RemoteSpeechClient")

# Configuration
STARTUP_GREETING = "Hello, I'm CaiTI. Who am I speaking with?"

class SpeechClient:
    def __init__(self):
        print("\n--- Eric's Remote Speech Client ---")
        jetson_ip = input("Enter the IP Address of the Jetson Board: ").strip()
        if not jetson_ip:
            logger.error("No IP address provided. Exiting.")
            sys.exit(1)
        
        self.api_url = f"http://{jetson_ip}:8000/api"
        logger.info(f"Targeting Jetson API at: {self.api_url}")
        
        logger.info("Initializing Audio Hardware...")
        self.recorder = AudioRecorder()
        self.stt = STTGenerator()
        self.tts = TTSGenerator()
        self.player = AudioPlayer()
        self.running = True
        self.paused = False
        self.session_active = False
        self.stop_playback_event = threading.Event()
        
        # Stream logs from Jetson Backend
        logger.info("Initializing remote log stream...")
        self.log_process = subprocess.Popen(
            ["ssh", f"{os.environ.get('JETSON_HOST', 'arth@152.23.12.147')}", "tail", "-f", "~/project/backend_session.log"],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
    def say(self, text):
        """Synthesize and play text."""
        logger.info(f"Speaking: {text}")
        wav_file = "service_response_temp.wav"
        if self.tts.generate(text, wav_file):
            self.stop_playback_event.clear()
            self.player.play(wav_file, stop_event=self.stop_playback_event)

    def listen(self, timeout=15.0):
        """Record and transcribe."""
        logger.info("Listening...")
        audio_frames = self.recorder.record_until_silence(max_duration=timeout)
        if not audio_frames:
            return ""
            
        user_wav = "service_input_temp.wav"
        self.recorder.save_wav(audio_frames, user_wav)
        text = self.stt.transcribe(user_wav)
        logger.info(f"Heard: {text}")
        return text

    def handle_voice_login(self):
        """Startup flow: Identify user."""
        self.say(STARTUP_GREETING)
        
        while True:
            name = self.listen()
            if not name:
                continue
                
            name_lower = name.lower()
            user_id = None
            
            if "test" in name_lower:
                user_id = "test_user"
                self.say("Welcome back, Test User. Starting session.")
            else:
                user_id = f"user_{name.replace(' ', '_')}"
                self.say(f"Hello {name}. Starting new session.")
                
            if user_id:
                # Login via API
                try:
                    res = requests.post(f"{self.api_url}/login", json={"user_id": user_id})
                    if res.status_code == 200:
                        self.session_active = True
                        return
                except Exception as e:
                    logger.error(f"Login failed: {e}")
                    self.say("I had trouble logging you in. Please try again.")

    def check_commands(self, text):
        """Local command handling."""
        lower = text.lower()
        if "end session" in lower or "stop session" in lower:
            self.say("Ending session.")
            try:
                requests.post(f"{self.api_url}/end_session")
            except:
                pass
            self.session_active = False
            return True
        elif "pause" in lower:
            self.paused = True
            self.say("Paused.")
            return True
        elif "resume" in lower or "continue" in lower:
            self.paused = False
            self.say("Resuming.")
            return True
        return False

    def run(self):
        # 1. Wait for Server
        logger.info("Waiting for Dialogue Engine...")
        while True:
            try:
                requests.get(f"{self.api_url}/status")
                break
            except:
                time.sleep(2)
        
        # 2. Voice Login
        self.handle_voice_login()
        
        # 3. Main Loop
        while self.running:
            try:
                if self.paused:
                    time.sleep(0.5)
                    continue
                    
                if not self.session_active:
                    # If session ended, go back to login? 
                    # For now, just wait or re-login.
                    # Let's loop back to login for continuous kiosk mode.
                    time.sleep(1)
                    self.handle_voice_login()
                    continue

                # A. Poll for Output (Agent Turn)
                try:
                    res = requests.get(f"{self.api_url}/output")
                    if res.status_code == 200:
                        data = res.json()
                        agent_text = data.get("text")
                        if agent_text and agent_text != "None":
                            self.say(agent_text)
                            # After agent speaks, immediately listen (Turn-taking)
                            # Unless it was a goodbye?
                            # We assume standard turn-taking for now.
                            
                            # Small delay to prevent echo
                            time.sleep(0.5)
                            
                            # Listen
                            user_text = self.listen()
                            if user_text:
                                # Check Commands
                                if self.check_commands(user_text):
                                    continue
                                    
                                # Push to Backend
                                requests.post(f"{self.api_url}/input", json={"text": user_text})
                                
                except Exception as e:
                    logger.error(f"Loop error: {e}")
                    time.sleep(1)
                    
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
        
        self.cleanup()

    def cleanup(self):
        self.recorder.terminate()
        self.player.terminate()
        if hasattr(self, 'log_process'):
            self.log_process.terminate()

if __name__ == "__main__":
    client = SpeechClient()
    client.run()
