"""
Service orchestrating the real-time client-side speech loop for CaiTI.
Handles voice wake-word, hardware buttons (GPIO), and turn-taking logic.
"""

import os
import time
import json
import threading
import queue
import string
from src.drivers.audio import AudioRecorder
from src.models.stt import STTGenerator
from src.models.tts import TTSGenerator
from src.drivers.player import AudioPlayer
from src.drivers.gpio_manager import GPIOManager, EVENT_START, EVENT_END, EVENT_OPT_OUT
from src.utils.log_util import get_logger
import src.utils.io_record as io_record

logger = get_logger("SpeechService")

class SpeechInteractionService:
    """
    Unified speech orchestration service.
    Can be run as an integrated loop within the main process.
    """

    def __init__(self, input_queue, output_queue, is_hands_free=True):
        logger.info("Initializing Unified Speech Interaction Service...")
        self.recorder = AudioRecorder()
        self.stt = STTGenerator()
        self.tts = TTSGenerator()
        self.player = AudioPlayer()
        
        # GPIO — singleton, used for LED sync and button polling
        self.gpio = GPIOManager()

        self.input_queue = input_queue
        self.output_queue = output_queue
        
        self.running = True
        self.paused = False
        self.is_hands_free = is_hands_free
        self.state = "idle"  # idle, listening, processing, playing
        
        self.manual_input_event = threading.Event()
        self.stop_playback_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Hardware / LED Helpers                                               #
    # ------------------------------------------------------------------ #

    def _led_on(self):
        self.gpio.set_led(True)

    def _led_off(self):
        self.gpio.set_led(False)

    def _poll_gpio(self):
        """Consume events from the hardware queue."""
        try:
            return self.gpio.gpio_event_queue.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------ #
    # Core Actions                                                         #
    # ------------------------------------------------------------------ #

    def say(self, text):
        """Speak text and handle interruptions."""
        logger.info(f"Agent Action: {text}")
        if not text: return

        if text.startswith("[PLAY_MUSIC]"):
            music_file = text.split(" ", 1)[1] if " " in text else "assets/waiting_music.wav"
            logger.info(f"Playing background music: {music_file}")
            self.state = "playing"
            self.stop_playback_event.clear()
            self.player.play(music_file, stop_event=self.stop_playback_event)
            self.state = "idle"
            return
        
        self.state = "processing"
        wav_file = "service_response_temp.wav"
        if self.tts.generate(text, wav_file):
            self.state = "playing"
            self._led_off() # Ensure LED is off during speech
            self.stop_playback_event.clear()
            self.player.play(wav_file, stop_event=self.stop_playback_event)
            self.state = "idle"

    def listen(self, timeout=15.0):
        """Record and transcribe with LED feedback."""
        self.state = "listening"
        self._led_on()
        audio_frames = self.recorder.record_until_silence(max_duration=timeout)
        self._led_off()
        
        if not audio_frames:
            self.state = "idle"
            return ""

        self.state = "processing"
        user_wav = "service_input_temp.wav"
        self.recorder.save_wav(audio_frames, user_wav)
        stt_payload = self.stt.transcribe(user_wav)
        
        try:
            text = json.loads(stt_payload).get("transcript", "").strip()
        except Exception:
            text = stt_payload.strip()

        logger.info(f"User heard: {text}")
        self.state = "idle"
        return text

    # ------------------------------------------------------------------ #
    # Session Flows                                                        #
    # ------------------------------------------------------------------ #

    def handle_onboarding(self):
        """Triggered by voice or Button 1: Ask for user name and init session."""
        logger.info("Starting Onboarding flow...")
        self.say("Hello, I'm CaiTI. Who am I speaking with today?")
        
        name = self.listen(timeout=10.0)
        if name:
            uid = name.lower().replace(" ", "_").translate(str.maketrans('', '', string.punctuation))
            logger.info(f"Initializing session for user: {uid}")
            io_record.reset_session(uid)
            io_record.END_SESSION_EVENT.clear()
            io_record.START_SESSION_EVENT.set()
        else:
            self.say("I didn't catch that. Please try again or press the start button.")

    def handle_end_session(self):
        """Triggered by voice or Button 2: End current session."""
        logger.info("Ending session via hardware/voice command.")
        io_record.END_SESSION_EVENT.set()
        io_record.START_SESSION_EVENT.clear()
        self.input_queue.put("SESSION_END")
        self.say("Ending our session now. Goodbye.")

    # ------------------------------------------------------------------ #
    # Main Loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self):
        """Main service loop."""
        logger.info("Speech Interaction Service started.")
        
        while self.running:
            try:
                # 1. Check for Hardware Button Events
                gpio_ev = self._poll_gpio()
                if gpio_ev == EVENT_START:
                    if not io_record.START_SESSION_EVENT.is_set():
                        self.handle_onboarding()
                elif gpio_ev == EVENT_END:
                    if io_record.START_SESSION_EVENT.is_set():
                        self.handle_end_session()
                elif gpio_ev == EVENT_OPT_OUT:
                    logger.info("[GPIO] Opt-out button pressed.")
                    self.stop_playback_event.set()
                    self.input_queue.put("[OPT_OUT]")

                if self.paused:
                    time.sleep(0.5)
                    continue

                # 2. Handle Idle State (Waiting for Voice Wake-up)
                if not io_record.START_SESSION_EVENT.is_set():
                    # Simplified wake-up listener
                    audio_frames = self.recorder.record_until_silence(max_duration=5.0)
                    if audio_frames:
                        temp_wav = "wake_temp.wav"
                        self.recorder.save_wav(audio_frames, temp_wav)
                        stt_json = self.stt.transcribe(temp_wav)
                        try:
                            text = json.loads(stt_json).get("transcript", "").lower().strip()
                        except:
                            text = stt_json.lower().strip()
                        
                        if any(kw in text for kw in ("hello caiti", "hi caiti", "hey caiti")):
                            self.handle_onboarding()
                    continue

                # 3. Active Session turned based logic
                try:
                    # Check for outgoing agent speech
                    text_to_speak = self.output_queue.get(timeout=0.2)
                    if text_to_speak:
                        self.say(text_to_speak)
                        
                        # Wait for user response
                        if not self.is_hands_free:
                            self.manual_input_event.wait()
                            self.manual_input_event.clear()
                        
                        user_response = self.listen(timeout=15.0)
                        
                        # Check for voice-triggered end session
                        lower = user_response.lower()
                        if "end session" in lower or "stop session" in lower:
                            self.handle_end_session()
                        else:
                            self.input_queue.put(user_response)

                except queue.Empty:
                    pass

            except Exception as e:
                logger.error(f"Error in speech service loop: {e}")
                time.sleep(1)

    def stop(self):
        self.running = False
        self.gpio.cleanup()
        self.recorder.terminate()
        self.player.terminate()
