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
import re
import itertools
from src.drivers.audio import AudioRecorder
from src.models.stt import STTGenerator
from src.models.tts import TTSGenerator
from src.drivers.player import AudioPlayer
from src.drivers.gpio_manager import GPIOManager, EVENT_START, EVENT_END, EVENT_OPT_OUT
from src.utils.log_util import get_logger
from src.utils.resource_audit import get_resource_audit
from src.utils.inference_guard import get_system_memory_snapshot
import src.utils.io_record as io_record

logger = get_logger("SpeechService")
RESOURCE_AUDIT = get_resource_audit()

WAKE_TRIGGERS = ("hello", "hi", "hey", "start", "wake")
WAKE_NAME = "katie"

# ── Intermission watchdog constants ────────────────────────────────────────
_INTERMISSION_TRIGGER_SEC = 3.0   # seconds before triggering exercises
_INTERMISSION_HEARTBEAT_SEC = 10.0

# Bridge phrases spoken after intermission exercises, before the LLM response.
# Randomized to feel natural, not robotic.  Includes both "gratitude" and
# "transition" phrases so the handoff from exercise → therapist answer
# feels like a real conversation beat rather than a system glitch.
import random as _random
_BRIDGE_PHRASES = [
    "Thank you for reflecting on that with me. Now, going back to what you shared...",
    "I appreciate you sharing that. I've been thinking about what you said...",
    "Thank you for being open with me. Let me respond to what's on your mind.",
    "I appreciate your patience while I gathered my thoughts.",
    "Thank you for staying with me through that. Here's what I'd like to say...",
    "That was a nice moment of stillness. Now, about what you mentioned...",
]

# Common adjectives/fillers that are NOT names — for the onboarding guard.
_NOT_A_NAME_WORDS = frozenset({
    "good", "fine", "okay", "ok", "great", "well", "nice", "yes", "no",
    "hello", "hi", "hey", "sure", "thanks", "thank", "right", "yeah",
    "alright", "cool", "um", "uh", "hmm", "hm", "ah", "oh",
    "nothing", "none", "nobody", "bye", "stop", "maybe",
    "of", "course", "morning", "evening", "afternoon", "please",
    "i", "im", "am", "my", "name", "is", "its", "me", "the", "a",
    "doing", "just", "really", "very", "so", "pretty",
})
# Multi-word phrases that should be rejected wholesale.
_NOT_A_NAME_PHRASES = frozenset({
    "of course", "good morning", "good evening", "good afternoon",
    "im fine", "i am fine", "im good", "i am good", "im okay", "i am okay",
    "im doing well", "im doing good", "im doing fine",
    "not sure", "i dont know", "i dunno", "no idea",
    "thank you", "thanks a lot", "yes please", "no thanks",
})


def _normalize_transcript(text: str) -> str:
    """Lowercase and strip punctuation so wake matching is robust."""
    if not text:
        return ""
    table = str.maketrans("", "", string.punctuation)
    return str(text).lower().translate(table).strip()

class SpeechInteractionService:
    """
    Unified speech orchestration service.
    Can be run as an integrated loop within the main process.
    """

    def __init__(self, input_queue, output_queue, is_hands_free=True):
        logger.info("Initializing Unified Speech Interaction Service...")
        with RESOURCE_AUDIT.track_module_init("SpeechService/AudioRecorder"):
            self.recorder = AudioRecorder()
        with RESOURCE_AUDIT.track_module_init("SpeechService/STTGenerator"):
            self.stt = STTGenerator()
        with RESOURCE_AUDIT.track_module_init("SpeechService/TTSGenerator"):
            self.tts = TTSGenerator()
        with RESOURCE_AUDIT.track_module_init("SpeechService/AudioPlayer"):
            self.player = AudioPlayer()
        
        # GPIO — singleton, used for LED sync and button polling
        with RESOURCE_AUDIT.track_module_init("SpeechService/GPIOManager"):
            self.gpio = GPIOManager()
        self.recorder.set_vad_active_callback(self.gpio.set_led)

        self.input_queue = input_queue
        self.output_queue = output_queue
        
        self.running = True
        self.paused = False
        self.is_hands_free = is_hands_free
        self.state = "idle"  # idle, listening, processing, playing
        
        self.manual_input_event = threading.Event()
        self.stop_playback_event = threading.Event()

        RESOURCE_AUDIT.capture_process_inventory("speech_service_init_complete")

    # ------------------------------------------------------------------ #
    # Hardware / LED Helpers                                               #
    # ------------------------------------------------------------------ #

    def _led_on(self):
        self.gpio.set_led(True)

    def _led_off(self):
        self.gpio.set_led(False)

    def _poll_gpio(self):
        """Consume events from the hardware queue."""
        return self.gpio.poll_event()

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

    @staticmethod
    def _is_valid_name(raw: str) -> bool:
        """Return True if the transcript looks like an actual name, not a
        common adjective/filler like 'Good', 'Fine', 'Of course'."""
        clean = re.sub(r"[^A-Za-z ]", "", raw).strip().lower()
        if not clean:
            return False

        # Reject known non-name phrases (e.g. "of course", "good morning")
        if clean in _NOT_A_NAME_PHRASES:
            return False

        # Reject very short transcripts (< 2 alphabetic chars stripped of filler)
        alpha_only = re.sub(r"[^a-z]", "", clean)
        if len(alpha_only) < 2:
            return False

        # If every word is a common filler, reject it.
        words = clean.split()
        return not all(w in _NOT_A_NAME_WORDS for w in words)

    def handle_onboarding(self):
        """Triggered by voice or Button 1: Ask for user name and init session."""
        self.state = "IDENTITY_PROMPT"
        logger.info("Starting Onboarding flow...")
        self.say("Hello, I'm CaiTI. Who am I speaking with today?")

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            name = self.listen(timeout=10.0)
            if not name:
                if attempt < max_attempts:
                    self.say("I didn't catch that. Could you tell me your name?")
                    continue
                else:
                    self.say("I didn't catch that. Please try again or press the start button.")
                    self.state = "idle"
                    return

            if not self._is_valid_name(name):
                logger.warning(
                    f"[NAME GUARD] Invalid name '{name}' detected (attempt {attempt}/{max_attempts}). "
                    f"Re-prompting for identity."
                )
                if attempt < max_attempts:
                    self.say("I'm sorry, I missed that. What was your name again?")
                    continue
                else:
                    self.say("Let me just call you 'User' for now. We can change that later.")
                    name = "User"

            logger.info(f"[NAME GUARD] Accepted raw transcript '{name}' as valid name.")
            # Strip common prefixes like "My name is", "I'm", "It's", "I am"
            stripped = re.sub(
                r"^(?:my\s+name\s+is|i\s*(?:am|'m)\s|it'?s\s|they\s+call\s+me\s)",
                "", name, flags=re.IGNORECASE,
            ).strip()
            clean = re.sub(r"[^A-Za-z0-9 _-]", "", stripped or name).strip()
            uid = clean.replace(" ", "_") or "User"
            logger.info(f"Initializing session for user: {uid}")
            io_record.reset_session(uid)
            io_record.END_SESSION_EVENT.clear()

            # ── Pre-session VRAM handoff ───────────────────────────────
            # The RL pipeline will call llm_complete() for the opening
            # greeting immediately after START_SESSION_EVENT is set.
            # Suspend STT/SER now so the LLM has enough GPU memory.
            # The main loop will resume STT after the first output is spoken.
            logger.info("[VRAM HANDOFF] Pre-session: suspending STT before pipeline starts.")
            try:
                self.stt.suspend_all()
            except Exception as e:
                logger.warning(f"[VRAM HANDOFF] Pre-session STT suspend failed: {e}")

            io_record.START_SESSION_EVENT.set()
            break

        self.state = "idle"

    def handle_end_session(self):
        """Triggered by voice or Button 2: End current session immediately."""
        logger.info("Ending session via hardware/voice command.")
        # Signal termination first so all blocking queues unblock immediately
        io_record.END_SESSION_EVENT.set()
        io_record.START_SESSION_EVENT.clear()
        # Stop any ongoing playback before speaking goodbye
        self.stop_audio()
        # Drain any pending items from the output queue so they don't replay
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        self.input_queue.put("SESSION_END")
        self.say("Ending our session now. Goodbye.")

    def stop_audio(self):
        """Stop any ongoing playback immediately."""
        self.stop_playback_event.set()
        self.player.stop_playback()
        self.state = "idle"

    def set_paused(self, is_paused: bool):
        """Pause/resume mic loop and current playback."""
        self.paused = bool(is_paused)
        if self.paused:
            self.stop_audio()
            logger.info("Speech loop paused.")
        else:
            logger.info("Speech loop resumed.")

    # ------------------------------------------------------------------ #
    # Intermission Watchdog (non-blocking LLM wait)                        #
    # ------------------------------------------------------------------ #

    def _wait_for_output_with_intermission(self):
        """
        Wait for the core pipeline to put the next agent utterance on
        `self.output_queue`.  If it takes longer than _INTERMISSION_TRIGGER_SEC,
        play PHQ-4 / breathing exercises so the user isn't sitting in silence.

        A threading.Event is used to signal the watchdog thread when the LLM
        response arrives so exercises stop gracefully.

        Smooth Handoff: When the LLM finishes mid-exercise, the current TTS
        clip is allowed to complete naturally (no interruption).  A randomized
        bridge phrase is then spoken before delivering the LLM response so the
        transition feels conversational rather than jarring.
        """
        llm_done_event = threading.Event()
        start_time = time.monotonic()
        last_heartbeat = start_time
        response_text = [None]  # mutable container for thread result
        intermission_played = False  # track whether exercises were triggered

        def _watch_output_queue():
            """Background thread: blocks on output_queue and signals when ready."""
            try:
                # Block up to the full timeout waiting for the pipeline response
                resp = self.output_queue.get(timeout=120.0)
                response_text[0] = resp
            except queue.Empty:
                logger.warning("[INTERMISSION] Output queue timed out after 120s.")
            finally:
                llm_done_event.set()

        watcher = threading.Thread(target=_watch_output_queue, daemon=True)
        watcher.start()

        # ── Phase 1: Silent fast-path wait ─────────────────────────────────
        if llm_done_event.wait(timeout=_INTERMISSION_TRIGGER_SEC):
            # LLM responded within the trigger threshold — speak immediately
            if response_text[0]:
                self.say(response_text[0])
            return

        # ── Phase 2: Intermission — LLM is slow, engage the user ──────────
        logger.info(
            f"[INTERMISSION] LLM latency detected ({_INTERMISSION_TRIGGER_SEC}s). "
            "Triggering exercises."
        )

        try:
            from src.core.therapy_content import MEDITATIONS, CLINICAL_SCREENING
            meditation_cycle = itertools.cycle(MEDITATIONS)
            screening_idx = 0
        except ImportError:
            logger.warning("[INTERMISSION] therapy_content unavailable; waiting silently.")
            llm_done_event.wait()
            if response_text[0]:
                self.say(response_text[0])
            return

        while not llm_done_event.is_set():
            # Bail out immediately if session was ended
            if io_record.END_SESSION_EVENT.is_set():
                logger.info("[INTERMISSION] Session ended. Stopping intermission.")
                break

            now = time.monotonic()
            elapsed = now - start_time

            # Heartbeat
            if now - last_heartbeat >= _INTERMISSION_HEARTBEAT_SEC:
                last_heartbeat = now
                logger.info(
                    f"[Heartbeat] LLM still thinking... "
                    f"Time elapsed: {elapsed:.0f}s. Intermission active."
                )

            # Pick content: PHQ-4 screening first, then breathing exercises
            if screening_idx < len(CLINICAL_SCREENING):
                q = CLINICAL_SCREENING[screening_idx]
                play_text = q["text"]
                logger.info(
                    f"[INTERMISSION] LLM Latency detected. "
                    f"Triggering PHQ-4 Question #{screening_idx + 1}."
                )
                screening_idx += 1
            else:
                play_text = next(meditation_cycle)
                logger.info("[INTERMISSION] Triggering breathing exercise.")

            # NOTE: self.say() blocks until TTS playback finishes.
            # This is intentional — the current exercise clip completes
            # naturally even if the LLM response arrives mid-sentence.
            intermission_played = True
            self.say(play_text)

            # After speaking the exercise, give the user a moment to respond
            # or check if LLM is done.
            if llm_done_event.wait(timeout=5.0):
                break

        # ── Phase 3: LLM response ready — speak it ────────────────────────
        watcher.join(timeout=2.0)
        if response_text[0]:
            elapsed_total = time.monotonic() - start_time
            logger.info(
                f"[INTERMISSION] Complete. LLM responded after "
                f"{elapsed_total:.1f}s total."
            )

            # Smooth handoff: if exercises were played, speak a brief bridge
            # phrase so the user feels a natural transition rather than an
            # abrupt jump from "breathe in..." to the therapist's answer.
            if intermission_played:
                bridge = _random.choice(_BRIDGE_PHRASES)
                logger.info(f"[HANDOFF] Bridge phrase: '{bridge}'")
                self.say(bridge)

            self.say(response_text[0])

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
                    preferred_music = "assets/audio/waiting_music.wav"
                    fallback_music = "assets/waiting_music.wav"
                    music_path = preferred_music if os.path.isfile(preferred_music) else fallback_music
                    self.say(f"[PLAY_MUSIC] {music_path}")

                if self.paused:
                    time.sleep(0.5)
                    continue

                # 2. Handle Idle State (Waiting for Voice Wake-up)
                if not io_record.START_SESSION_EVENT.is_set():
                    # Polling wake-up (1s windows for responsiveness)
                    audio_frames = self.recorder.record_until_silence(max_duration=1.0)
                    if audio_frames:
                        temp_wav = "wake_temp.wav"
                        self.recorder.save_wav(audio_frames, temp_wav)
                        stt_json = self.stt.transcribe(temp_wav)
                        try:
                            raw_text = json.loads(stt_json).get("transcript", "")
                        except Exception:
                            raw_text = stt_json

                        text = _normalize_transcript(raw_text)
                        words = set(text.split())
                        has_trigger = any(trigger in words for trigger in WAKE_TRIGGERS)
                        has_name = WAKE_NAME in words
                        logger.info(f"Wake transcription: {text}")

                        if has_trigger and has_name:
                            logger.info("Wake phrase accepted. Transitioning to IDENTITY_PROMPT.")
                            self.handle_onboarding()
                    continue

                # 3. Active Session turn-based logic with intermission watchdog
                try:
                    # Check for outgoing agent speech
                    text_to_speak = self.output_queue.get(timeout=0.2)
                    if text_to_speak:
                        self.say(text_to_speak)

                        # Ensure STT is loaded before listening (may have been
                        # suspended by pre-session handoff or previous turn).
                        try:
                            self.stt.resume_all()
                        except Exception as e:
                            logger.warning(f"[VRAM HANDOFF] STT resume before listen failed: {e}")

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
                            # Send response to the core pipeline
                            self.input_queue.put(user_response)

                            # ── Sequential VRAM Handoff ("Safe-Pass") ──────────
                            # 1. Unload BOTH Whisper and SER to maximize free VRAM
                            logger.info(f"[VRAM HANDOFF] Phase 1 — Pre-unload: {get_system_memory_snapshot()}")
                            try:
                                self.stt.suspend_all()
                            except Exception as e:
                                logger.warning(f"[VRAM HANDOFF] STT suspend failed (non-fatal): {e}")
                            logger.info(f"[VRAM HANDOFF] Phase 1 — Post-unload: {get_system_memory_snapshot()}")

                            # 2. Wait 0.5s for OS to reclaim memory pages
                            time.sleep(0.5)
                            logger.info(f"[VRAM HANDOFF] Phase 2 — After 0.5s reclaim: {get_system_memory_snapshot()}")

                            # 3. LLM handshake proceeds (via intermission watchdog)
                            self._led_off()
                            self._wait_for_output_with_intermission()

                            # 4. Reload Whisper + SER for next listen cycle
                            logger.info(f"[VRAM HANDOFF] Phase 4 — Pre-reload: {get_system_memory_snapshot()}")
                            try:
                                self.stt.resume_all()
                            except Exception as e:
                                logger.warning(f"[VRAM HANDOFF] STT resume failed (non-fatal): {e}")
                            logger.info(f"[VRAM HANDOFF] Phase 4 — Post-reload: {get_system_memory_snapshot()}")

                except queue.Empty:
                    pass

            except Exception as e:
                logger.error(f"Error in speech service loop: {e}")
                time.sleep(1)

    def stop(self):
        self.running = False
        self.stop_audio()
        self.gpio.cleanup()
        self.recorder.terminate()
        self.player.terminate()
