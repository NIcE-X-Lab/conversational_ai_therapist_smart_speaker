import os
import time
import requests
import threading
import logging
import queue
import numpy as np
from src.perception.audio import AudioRecorder
from src.perception.stt import STTGenerator
from src.action.tts import TTSGenerator
from src.action.player import AudioPlayer
from src.utils.log_util import get_logger

logger = get_logger("SpeechService")

# Configuration
API_URL = "http://localhost:8000/api"
STARTUP_GREETING = "Hello, I'm CaiTI. Who am I speaking with?"

class SpeechClient:
    def __init__(self):
        logger.info("Initializing Speech Service Client...")
        self.recorder = AudioRecorder()
        self.stt = STTGenerator()
        self.tts = TTSGenerator()
        self.player = AudioPlayer()
        
        self.running = True
        self.paused = False
        self.session_active = False
        self.stop_playback_event = threading.Event()
        
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
                    res = requests.post(f"{API_URL}/login", json={"user_id": user_id})
                    if res.status_code == 200:
                        self.session_active = True
                        return
                except Exception as e:
                    logger.error(f"Login failed: {e}")
                    self.say("I had trouble logging you in. Please try again.")

    def _wait_for_summary_and_end(self):
        """Wait for the backend to generate the historical summary, speak it, and end the session."""
        self.say("Ending session. Give me a moment to generate your summary.")
        try:
            requests.post(f"{API_URL}/end_session")
        except:
            pass
        
        # Block until we receive the summary from the backend queue
        max_waits = 10
        while max_waits > 0:
            try:
                res = requests.get(f"{API_URL}/output")
                if res.status_code == 200:
                    data = res.json()
                    agent_text = data.get("text")
                    if agent_text and agent_text != "None":
                        self.say(agent_text)
                        break
            except Exception:
                pass
            time.sleep(1)
            max_waits -= 1
            
        self.session_active = False

    def check_intent(self, text):
        """Use the backend LLM to evaluate complex start/end commands dynamically."""
        try:
            res = requests.post(f"{API_URL}/intent", json={"text": text}, timeout=10.0)
            if res.status_code == 200:
                data = res.json()
                return data.get("intent", "none")
        except Exception as e:
            logger.error(f"Intent check failed: {e}")
        return "none"

    def run(self):
        # 1. Wait for Server
        logger.info("Waiting for Dialogue Engine...")
        while True:
            try:
                requests.get(f"{API_URL}/status")
                break
            except:
                time.sleep(2)
        
        # 2. Inform user how to wake up the system
        logger.info("Ready. Awaiting wake word.")
        
        # 3. Main Loop
        while self.running:
            try:
                if self.paused:
                    time.sleep(0.5)
                    continue
                    
                if not self.session_active:
                    # Idle Mode - Waiting for wake word
                    idle_text = self.listen()
                    if idle_text:
                        idle_lower = idle_text.lower()
                        # Fast Path
                        if "start session" in idle_lower or "hi caiti" in idle_lower:
                            self.handle_voice_login()
                        else:
                            # Dynamic Intent Checking
                            intent = self.check_intent(idle_text)
                            if intent == "start":
                                self.handle_voice_login()
                    continue

                # A. Active Session Mode - Poll for Agent Output
                try:
                    res = requests.get(f"{API_URL}/output")
                    if res.status_code == 200:
                        data = res.json()
                        agent_text = data.get("text")
                        if agent_text and agent_text != "None":
                            self.say(agent_text)
                            
                            # Small delay to prevent echo
                            time.sleep(0.5)
                            
                            # Listen with retries for blank responses
                            retries = 0
                            while retries < 3 and self.session_active:
                                user_text = self.listen()
                                
                                if user_text:
                                    lower = user_text.lower()
                                    if "end session" in lower or "stop session" in lower:
                                        self._wait_for_summary_and_end()
                                        break
                                    elif "pause" in lower:
                                        self.paused = True
                                        self.say("Paused.")
                                        break
                                    elif "resume" in lower or "continue" in lower:
                                        self.paused = False
                                        self.say("Resuming.")
                                        break
                                        
                                    # Intent LLM path
                                    intent = self.check_intent(user_text)
                                    if intent == "end":
                                        self._wait_for_summary_and_end()
                                        break
                                    else:
                                        # Push standard response to Backend
                                        requests.post(f"{API_URL}/input", json={"text": user_text})
                                        break
                                else:
                                    # Blank response -> Ask again
                                    retries += 1
                                    if retries < 3:
                                        self.say(agent_text)
                                    else:
                                        self.say("I am not hearing any response. I will end the session for now.")
                                        self._wait_for_summary_and_end()
                                        break
                                        
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

if __name__ == "__main__":
    client = SpeechClient()
    client.run()
