"""
Service orchestrating the real-time client-side speech loop for CaiTI.
Handles voice wake-word, hardware buttons (GPIO), turn-taking logic,
and a full intermission state machine (screening -> exercises -> music).
"""

import os
import time
import json
import threading
import queue
import string
import re
import random as _random
from difflib import SequenceMatcher
from src.drivers.audio import AudioRecorder, BackgroundMusicThread
from src.models.stt import STTGenerator
from src.models.tts import TTSGenerator
from src.drivers.player import AudioPlayer
from src.drivers.gpio_manager import GPIOManager, EVENT_START, EVENT_END, EVENT_OPT_OUT
from src.core.intermission_manager import IntermissionLadderManager, IntermissionStage
from src.core.therapy_content import score_response
from src.utils.log_util import get_logger
from src.utils.resource_audit import get_resource_audit
from src.utils.inference_guard import get_system_memory_snapshot
import src.utils.io_record as io_record

logger = get_logger("SpeechService")
RESOURCE_AUDIT = get_resource_audit()

# ── Constants ─────────────────────────────────────────────────────────────────

WAKE_TRIGGERS = ("hello", "hi", "hey", "start", "wake")
WAKE_NAME = "katie"

# Intermission watchdog timing
_INTERMISSION_TRIGGER_SEC = 3.0     # wait this long before engaging user
_INTERMISSION_HEARTBEAT_SEC = 10.0  # heartbeat log interval
_SCREENING_LISTEN_TIMEOUT = 20.0    # max seconds to wait for screening answer
_SCREENING_MIN_LISTEN_WINDOW_SEC = 5.0
_EXERCISE_HOLD_SEC = 30.0           # time given per breathing exercise
_SILENCE_REPROMPT_SEC = 12.0        # seconds of silence before gentle re-prompt
_TRANSITION_PAUSE = 1.0             # pause after user answers (seconds)

# Bridge phrases spoken after intermission, before LLM response.
_BRIDGE_PHRASES = [
    "Thank you for reflecting on that with me. Now, going back to what you shared...",
    "I appreciate you sharing that. I've been thinking about what you said...",
    "Thank you for being open with me. Let me respond to what's on your mind.",
    "I appreciate your patience while I gathered my thoughts.",
    "Thank you for staying with me through that. Here's what I'd like to say...",
    "That was a nice moment of stillness. Now, about what you mentioned...",
]

# Filler intros for screening questions (avoid abrupt "question" delivery)
_SCREENING_INTROS = [
    "While I'm processing that, I'd like to ask you something.",
    "It's taking me a moment to reflect. Let me ask you this in the meantime.",
    "While I work through your response, let me check in with you.",
    "Give me just a moment. In the meantime, I'd like to ask...",
    "Let me ask you this while I gather my thoughts.",
]

# Keywords that trigger opt-out from intermission
_OPT_OUT_KEYWORDS = ("skip", "don't want", "opt out", "no thanks",
                     "just music", "play music", "i'd rather not")

# Keywords that trigger a "repeat" request
_REPEAT_KEYWORDS = ("repeat", "again", "say that again", "what was that",
                    "pardon", "sorry what", "one more time", "can you repeat")

# Background music paths
_MUSIC_PATH_PREFERRED = "assets/audio/ambient_therapy.mp3"
_MUSIC_PATH_FALLBACK = "assets/audio/waiting_music.wav"
_SCREENING_OPTIONS_HINT = (
    "You can answer: Not at all, Several days, More than half the days, or Nearly every day."
)

_SKIP_QUESTION_PATTERNS = (
    r"\bi\s+do\s+not\s+want\s+to\s+answer\b",
    r"\bi\s+don't\s+want\s+to\s+answer\b",
    r"\bi\s+dont\s+want\s+to\s+answer\b",
    r"\brather\s+not\s+answer\b",
    r"\bskip\s+this\s+question\b",
    r"\bpass\b",
)

# Name guard sets
_NOT_A_NAME_WORDS = frozenset({
    "good", "fine", "okay", "ok", "great", "well", "nice", "yes", "no",
    "hello", "hi", "hey", "sure", "thanks", "thank", "right", "yeah",
    "alright", "cool", "um", "uh", "hmm", "hm", "ah", "oh",
    "nothing", "none", "nobody", "bye", "stop", "maybe",
    "of", "course", "morning", "evening", "afternoon", "please",
    "i", "im", "am", "my", "name", "is", "its", "me", "the", "a",
    "doing", "just", "really", "very", "so", "pretty",
})
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


def _get_music_path() -> str:
    if os.path.isfile(_MUSIC_PATH_PREFERRED):
        return _MUSIC_PATH_PREFERRED
    return _MUSIC_PATH_FALLBACK


def _is_opt_out(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _OPT_OUT_KEYWORDS)


def _is_repeat_request(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _REPEAT_KEYWORDS)


def _is_skip_question_request(text: str) -> bool:
    low = text.lower().strip()
    return any(re.search(pattern, low) for pattern in _SKIP_QUESTION_PATTERNS)


class GlobalCommandMatcher:
    """Regex + fuzzy priority matcher for start/end global voice commands."""

    FUZZY_THRESHOLD = 0.80

    _END_CANONICAL = (
        "end session",
        "and session",
        "end the session",
        "and the session",
        "and this session",
        "end dis session",
        "stop session",
        "stop the session",
        "finish session",
        "finish the session",
        "close session",
        "close the session",
        "goodbye",
        "good bye",
    )
    _START_CANONICAL = (
        "start session",
        "begin session",
        "hello session",
        "hi session",
        "lets go",
    )

    _START_PATTERNS = (
        re.compile(r"\b(?:start|begin|hello|hi|let'?s\s+go)\b(?:.*\bsession\b)?", re.IGNORECASE),
        re.compile(r"\bhi\s+katie\b", re.IGNORECASE),
    )
    _END_PATTERNS = (
        re.compile(r"\b(?:end\s+session|stop\s+session|finish\s+session|goodbye)\b", re.IGNORECASE),
        re.compile(r"\b(?:end|and|stop|finish|goodbye)\b(?:\s+(?:the|this|dis|da))?\s+session\b", re.IGNORECASE),
        re.compile(r"^\s*(?:end|and|stop|finish)\s*(?:the|this|dis|da)?\s*(?:session)?\s*(?:please)?\s*[.!?]*\s*$", re.IGNORECASE),
        # Aggressive catch-all: any phrase containing an end-word near "session"
        re.compile(r"(?:end|and|stop|finish|goodbye|close).*session", re.IGNORECASE),
    )

    @staticmethod
    def _normalize(text: str) -> str:
        t = text.lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _token_hit(tokens: set[str], refs: tuple[str, ...], threshold: float) -> bool:
        return any(
            SequenceMatcher(None, tok, ref).ratio() >= threshold
            for tok in tokens
            for ref in refs
        )

    def _fuzzy_end(self, text: str) -> bool:
        tokens = set(text.split())
        if not tokens:
            return False

        # "goodbye" / "bye" alone is sufficient
        if self._token_hit(tokens, ("goodbye", "bye"), self.FUZZY_THRESHOLD):
            return True

        # Require "session" (or close fuzzy match) to be present
        has_session = "session" in tokens or self._token_hit(tokens, ("session",), 0.75)
        if not has_session:
            return False

        # "end" is commonly mis-heard as "and" — use a lower threshold
        # for these short words to catch typos / STT errors.
        has_end_token = self._token_hit(tokens, ("end", "and", "stop", "finish", "close"), 0.75)
        has_start_token = self._token_hit(tokens, ("start", "begin", "hello", "hi"), self.FUZZY_THRESHOLD)
        return has_end_token and not has_start_token

    def _fuzzy_start(self, text: str) -> bool:
        tokens = set(text.split())
        if not tokens:
            return False

        has_start_token = self._token_hit(tokens, ("start", "begin", "hello", "hi"), self.FUZZY_THRESHOLD)
        has_end_token = self._token_hit(tokens, ("end", "and", "stop", "finish", "goodbye", "bye", "close"), self.FUZZY_THRESHOLD)
        has_session_or_name = (
            "session" in tokens
            or "katie" in tokens
            or self._token_hit(tokens, ("session", "katie"), self.FUZZY_THRESHOLD)
        )
        return has_start_token and has_session_or_name and not has_end_token

    def match(self, transcript: str) -> str | None:
        text = self._normalize(str(transcript or ""))
        if not text:
            return None

        if any(p.search(text) for p in self._END_PATTERNS):
            return "END"
        if self._fuzzy_end(text):
            return "END"
        if any(p.search(text) for p in self._START_PATTERNS):
            return "START"
        if self._fuzzy_start(text):
            return "START"
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Main Service
# ══════════════════════════════════════════════════════════════════════════════

class SpeechInteractionService:
    """
    Unified speech orchestration service.
    Can be run as an integrated loop within the main process.

    State tracking (self.state):
        idle              — no session, waiting for wake word / button
        onboarding        — asking for user name
        main_listen       — listening for user response to main question
        main_process      — LLM is generating / pipeline processing
        intermission_screening  — asking a PHQ-4 screening question
        intermission_exercise   — guiding a breathing exercise
        music_fallback          — playing ambient music during wait
        speaking          — TTS is playing
    """

    def __init__(self, input_queue, output_queue, is_hands_free=True):
        logger.info("Initializing Unified Speech Interaction Service...")
        with RESOURCE_AUDIT.track_module_init("SpeechService/AudioRecorder"):
            self.recorder = AudioRecorder()
        with RESOURCE_AUDIT.track_module_init("SpeechService/STTGenerator"):
            self.stt = STTGenerator()
        with RESOURCE_AUDIT.track_module_init("SpeechService/TTSGenerator"):
            self.tts = TTSGenerator()
        with RESOURCE_AUDIT.track_module_init("SpeechService/BackgroundMusic"):
            self.music_service = BackgroundMusicThread()
        with RESOURCE_AUDIT.track_module_init("SpeechService/AudioPlayer"):
            self.player = AudioPlayer(playback_signal_handler=self.music_service.handle_signal)

        with RESOURCE_AUDIT.track_module_init("SpeechService/GPIOManager"):
            self.gpio = GPIOManager()
        self.recorder.set_vad_active_callback(self.gpio.set_led)

        self.input_queue = input_queue
        self.output_queue = output_queue

        self.running = True
        self.paused = False
        self.is_hands_free = is_hands_free
        self.state = "idle"
        self.global_command_matcher = GlobalCommandMatcher()
        self.intermission_ladder = IntermissionLadderManager()
        self._music_announced_for_turn = False

        self.manual_input_event = threading.Event()
        self.stop_playback_event = threading.Event()
        self._consecutive_silence_count = 0

        RESOURCE_AUDIT.capture_process_inventory("speech_service_init_complete")

        # Zero-silence therapeutic bed starts at boot and ducks while AI speaks.
        self.music_service.start(_get_music_path())

    # ------------------------------------------------------------------ #
    # Hardware / LED Helpers                                               #
    # ------------------------------------------------------------------ #

    def _led_on(self):
        self.gpio.set_led(True)

    def _led_off(self):
        self.gpio.set_led(False)

    def _poll_gpio(self):
        return self.gpio.poll_event()

    # ------------------------------------------------------------------ #
    # Core Actions                                                         #
    # ------------------------------------------------------------------ #

    def say(self, text):
        """Speak text via TTS or play music.  Blocks until playback finishes."""
        if not text:
            return
        logger.info(f"Agent Action: {text[:120]}{'...' if len(text) > 120 else ''}")

        if text.startswith("[PLAY_MUSIC]"):
            music_file = text.split(" ", 1)[1] if " " in text else _get_music_path()
            logger.info(f"Starting background music loop: {music_file}")
            prev_state = self.state
            self.state = "music_fallback"
            self.music_service.start(music_file)
            self.state = prev_state
            return

        prev_state = self.state
        self.state = "speaking"
        wav_file = "active_ai_response.wav"
        if self.tts.generate(text, wav_file):
            self._led_off()
            self.stop_playback_event.clear()
            self.player.play(wav_file, stop_event=self.stop_playback_event)
        else:
            # TTS completely failed (Piper + espeak both down).
            # Bump the music so the user hears *something* rather than silence.
            logger.error("[TTS FAILURE] Both engines failed. Raising music to cover silence gap.")
            self.music_service.set_base_volume(0.15)
        self.state = prev_state

    def _persist_intermission_status(self, question_id: str, status: str, score=None, response_text="", reason=""):
        if not io_record.DB or not io_record.SESSION_ID:
            return
        try:
            io_record.DB.upsert_intermission_screening_status(
                session_id=io_record.SESSION_ID,
                question_id=question_id,
                status=status,
                score=score,
                response_text=response_text,
                reason=reason,
            )
        except Exception as e:
            logger.warning(f"Failed to persist intermission status for {question_id}: {e}")

    def _apply_global_command_priority(self, transcript: str) -> str | None:
        """Apply START/END priority matching before queueing input to the NLP stack."""
        command = self.global_command_matcher.match(transcript)
        if command == "END":
            logger.info("[COMMAND_GATE] END command detected. Bypassing analyzer pipeline.")
            self.handle_exit()
            return "END"
        return command

    def _sync_intermission_state_from_db(self):
        if not io_record.DB or not io_record.SESSION_ID:
            return
        try:
            statuses = io_record.DB.get_intermission_screening_statuses(io_record.SESSION_ID)
            self.intermission_ladder.load_checkpoint(statuses)
        except Exception as e:
            logger.warning(f"Failed restoring intermission checkpoints from DB: {e}")

    def transcribe(self, wav_path: str, apply_priority_gate: bool = False) -> str:
        """Transcribe audio and optionally intercept global start/end commands first."""
        stt_payload = self.stt.transcribe(wav_path)

        try:
            text = json.loads(stt_payload).get("transcript", "").strip()
        except Exception:
            text = stt_payload.strip()

        if apply_priority_gate:
            command = self._apply_global_command_priority(text)
            if command == "END":
                return "__CMD_END__"
            if command == "START":
                return "__CMD_START__"
        return text

    def _listen_for_intermission_answer(self, timeout: float, min_window: float = _SCREENING_MIN_LISTEN_WINDOW_SEC) -> str:
        """No-interrupt listener lock for PHQ intermission answers."""
        started = time.monotonic()
        deadline = started + timeout

        while time.monotonic() < deadline:
            elapsed = time.monotonic() - started
            remaining = max(0.1, deadline - time.monotonic())
            per_try_timeout = max(2.0, min(remaining, min_window))
            heard = self.listen(timeout=per_try_timeout, apply_priority_gate=True)
            if heard:
                return heard
            if elapsed >= min_window:
                break
        return ""

    def listen(self, timeout=15.0, apply_priority_gate: bool = False):
        """Record and transcribe with LED feedback."""
        self.state = "main_listen"
        self._led_on()
        audio_frames = self.recorder.record_until_silence(max_duration=timeout)
        self._led_off()

        if not audio_frames:
            self.state = "idle"
            return ""

        self.state = "main_process"
        user_wav = "active_user_input.wav"
        rms = self.recorder.compute_rms(audio_frames)
        if rms < 0.005:
            logger.info(f"[AUDIO HYGIENE] RMS {rms:.5f} below threshold — skipping disk write.")
            self.state = "idle"
            return ""
        self.recorder.save_wav(audio_frames, user_wav)
        text = self.transcribe(user_wav, apply_priority_gate=apply_priority_gate)

        logger.info(f"User heard: {text}")
        self.state = "idle"
        return text

    def _listen_with_retry(self, timeout=15.0, confirm_threshold=2, apply_priority_gate: bool = False):
        """Listen with low-confidence fallback.

        If the transcript is very short (< confirm_threshold words) and
        looks like a fragment, do one more short listen to see if the user
        continues.  This merges split utterances.
        """
        text = self.listen(timeout=timeout, apply_priority_gate=apply_priority_gate)
        if not text:
            return ""

        if text in {"__CMD_END__", "__CMD_START__"}:
            return text

        words = text.split()
        # If very short and ends with a trailing word, try to capture more
        if len(words) <= confirm_threshold and not text.rstrip().endswith((".", "!", "?")):
            logger.info(f"[STT] Short transcript ({len(words)} words). Checking for continuation...")
            extra = self.listen(timeout=4.0, apply_priority_gate=apply_priority_gate)
            if extra:
                if extra in {"__CMD_END__", "__CMD_START__"}:
                    return extra
                merged = f"{text} {extra}"
                logger.info(f"[STT] Merged fragments: '{merged}'")
                return merged
        return text

    # ------------------------------------------------------------------ #
    # Session Flows                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_valid_name(raw: str) -> bool:
        clean = re.sub(r"[^A-Za-z ]", "", raw).strip().lower()
        if not clean:
            return False
        if clean in _NOT_A_NAME_PHRASES:
            return False
        alpha_only = re.sub(r"[^a-z]", "", clean)
        if len(alpha_only) < 2:
            return False
        words = clean.split()
        return not all(w in _NOT_A_NAME_WORDS for w in words)

    # Keyword buffer: if the user says any of these during onboarding,
    # skip the name loop and start the session immediately as "User".
    _ONBOARD_BYPASS_KEYWORDS = frozenset({
        "start", "hello", "begin", "ready", "hey", "katie", "let's go",
        "lets go", "go", "session", "hi",
    })

    def _is_onboard_bypass(self, text: str) -> bool:
        """Return True if the transcript is a session-trigger phrase, not a name."""
        words = set(_normalize_transcript(text).split())
        return bool(words & self._ONBOARD_BYPASS_KEYWORDS)

    def handle_onboarding(self):
        """Triggered by voice or Button 1: Ask for user name and init session."""
        self.state = "onboarding"
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

            # Keyword buffer: if the user said "start", "hello", "ready",
            # "hey Katie" etc., skip the name loop — start as "User".
            if self._is_onboard_bypass(name):
                logger.info(f"[ONBOARD BYPASS] Trigger phrase detected in '{name}'. Starting as 'User'.")
                name = "User"
            elif not self._is_valid_name(name):
                logger.warning(f"[NAME GUARD] Invalid name '{name}' (attempt {attempt}/{max_attempts}).")
                if attempt < max_attempts:
                    self.say("I'm sorry, I missed that. What was your name again?")
                    continue
                else:
                    self.say("Let me just call you 'User' for now. We can change that later.")
                    name = "User"

            logger.info(f"[NAME GUARD] Accepted raw transcript '{name}' as valid name.")
            stripped = re.sub(
                r"^(?:my\s+name\s+is|i\s*(?:am|'m)\s|it'?s\s|they\s+call\s+me\s)",
                "", name, flags=re.IGNORECASE,
            ).strip()
            clean = re.sub(r"[^A-Za-z0-9 _-]", "", stripped or name).strip()
            uid = clean.replace(" ", "_") or "User"
            logger.info(f"Initializing session for user: {uid}")
            io_record.reset_session(uid)
            io_record.END_SESSION_EVENT.clear()

            # Suspend STT before starting session — LLM needs memory for greeting.
            logger.info("[VRAM HANDOFF] Pre-session: suspending STT before pipeline starts.")
            try:
                self.stt.suspend_all()
            except Exception as e:
                logger.warning(f"[VRAM HANDOFF] Pre-session STT suspend failed: {e}")

            io_record.START_SESSION_EVENT.set()
            self.intermission_ladder.reset()
            self.music_service.start(_get_music_path())
            break

        self.state = "idle"

    def handle_end_session(self):
        """Triggered by voice or Button 2: Clean, immediate shutdown.

        Exit sequence:
        1. Signal session end to pipeline
        2. Generate spoken closing reflection from clinical context
        3. Speak reflection via Piper TTS
        4. Save session dossier to disk
        5. Play goodbye music (or ambient fallback)
        6. Return to idle state
        """
        logger.info("Ending session via hardware/voice command.")
        io_record.END_SESSION_EVENT.set()
        io_record.START_SESSION_EVENT.clear()
        self.intermission_ladder.reset()
        self._music_announced_for_turn = False
        # Stop any ongoing playback instantly
        self.stop_audio()
        self.music_service.stop()
        self.post_turn_cleanup()
        # Drain stale output so it doesn't replay on next session
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        self.input_queue.put("SESSION_END")

        # ── Step 2: Generate closing reflection ──────────────────────
        reflection = ""
        try:
            from src.core.context_manager import get_context_manager
            reflection = get_context_manager().generate_closing_reflection()
        except Exception as e:
            logger.warning(f"[EXIT] Closing reflection generation failed: {e}")

        # ── Step 3: Speak reflection (or fallback goodbye) ──────────
        if reflection:
            self.say(reflection)
        self.say("Ending our session now. Goodbye.")

        # ── Step 4: Save session dossier ─────────────────────────────
        try:
            dossier = io_record.get_dossier()
            if dossier:
                # Record the closing reflection as the final dossier interaction
                if reflection:
                    dossier.record_interaction(
                        llm_response=reflection,
                        ser_metrics={"event": "closing_reflection"},
                    )
                dossier.save_and_close()
        except Exception as e:
            logger.warning(f"[EXIT] Dossier save failed: {e}")

        # ── Step 5: Goodbye music → idle ambient bed ─────────────────
        _GOODBYE_MUSIC = "assets/audio/goodbye_music.mp3"
        if os.path.isfile(_GOODBYE_MUSIC):
            self.music_service.start(_GOODBYE_MUSIC)
        else:
            self.music_service.set_base_volume(0.15)
            self.music_service.start(_get_music_path())

        self.state = "idle"

    @staticmethod
    def post_turn_cleanup():
        """Delete transient .wav files to prevent disk bloat on embedded storage."""
        for fname in ("active_user_input.wav", "active_ai_response.wav", "wake_temp.wav"):
            try:
                if os.path.exists(fname):
                    os.remove(fname)
            except OSError as e:
                logger.debug(f"[AUDIO HYGIENE] Could not remove {fname}: {e}")

    def stop_audio(self):
        """Stop any ongoing playback immediately."""
        self.stop_playback_event.set()
        self.player.stop_playback()

    def end_session(self):
        """Public alias used by command-priority gate."""
        self.handle_end_session()

    def handle_exit(self):
        """Explicit exit handler used by transcription priority gate."""
        self.handle_end_session()

    def initialize_session(self):
        """Explicit session starter used by transcription priority gate."""
        if not io_record.START_SESSION_EVENT.is_set():
            self.handle_onboarding()

    def set_paused(self, is_paused: bool):
        self.paused = bool(is_paused)
        if self.paused:
            self.stop_audio()
            logger.info("Speech loop paused.")
        else:
            logger.info("Speech loop resumed.")

    # ------------------------------------------------------------------ #
    # Intermission State Machine                                           #
    # ------------------------------------------------------------------ #

    def _wait_for_output_with_intermission(self):
        """
                Wait for the next pipeline utterance and engage strict intermission ladder:
                        Stage 1: next unanswered PHQ/GAD screening item.
                        Stage 2: one random breathing exercise once screening is complete/skipped.
                        Stage 3: calming music while waiting.
        """
        llm_done = threading.Event()
        response_text = [None]
        intermission_was_active = False
        # Listener lock: held True while a screening question has been asked
        # and we are waiting for the user's answer.  The main loop must NOT
        # break out (even if llm_done fires) until the listener completes —
        # otherwise the user's clinical answer is lost mid-sentence.
        _listener_active = threading.Event()

        def _watcher():
            try:
                resp = self.output_queue.get(timeout=120.0)
                response_text[0] = resp
            except queue.Empty:
                logger.warning("[INTERMISSION] Output queue timed out after 120s.")
            finally:
                llm_done.set()

        watcher = threading.Thread(target=_watcher, daemon=True)
        watcher.start()

        # Fast path for short LLM latency
        if llm_done.wait(timeout=_INTERMISSION_TRIGGER_SEC):
            if response_text[0]:
                self.say(response_text[0])
            return

        logger.info(
            f"[INTERMISSION] LLM latency > {_INTERMISSION_TRIGGER_SEC}s. "
            "Entering intermission ladder."
        )
        intermission_was_active = True

        # Sync ladder state from DB once at entry — merges without
        # regressing already-answered/skipped questions in memory.
        self._sync_intermission_state_from_db()

        self._music_announced_for_turn = False
        last_heartbeat = time.monotonic()

        while not llm_done.is_set() or _listener_active.is_set():
            if io_record.END_SESSION_EVENT.is_set():
                logger.info("[INTERMISSION] Session ended. Aborting.")
                break

            # If LLM is done but listener is still active, wait for the
            # user to finish answering before breaking out.
            if llm_done.is_set() and _listener_active.is_set():
                logger.info("[INTERMISSION] LLM ready, but listener lock held. Waiting for user to finish.")
                time.sleep(0.3)
                continue

            now = time.monotonic()
            if now - last_heartbeat >= _INTERMISSION_HEARTBEAT_SEC:
                last_heartbeat = now
                logger.info(
                    f"[Heartbeat] Intermission stage={self.intermission_ladder.current_stage().value}; "
                    "still waiting for LLM output."
                )

            current_stage = self.intermission_ladder.current_stage()

            if current_stage == IntermissionStage.SCREENING:
                question = self.intermission_ladder.next_screening_question()
                if question is None:
                    # All screening questions answered/skipped but tracker
                    # still reports SCREENING — force completion so the
                    # ladder advances to BREATHING / MUSIC instead of
                    # spinning here forever (the "ladder deadlock").
                    logger.info("[INTERMISSION] No unanswered screening questions remain. Advancing ladder.")
                    continue

                self.state = "intermission_screening"

                intro = _random.choice(_SCREENING_INTROS)
                full_prompt = f"{intro} {question.text}\n{_SCREENING_OPTIONS_HINT}"
                logger.info(f"[INTERMISSION] Screening question: {question.question_id}")
                self.say(full_prompt)

                # ── Listener Lock ON ─────────────────────────────
                # Prevents the loop from yielding to the LLM response
                # while we are capturing the user's clinical answer.
                _listener_active.set()

                # NOTE: DB sync removed here — in-memory tracker is the
                # authoritative source during a turn.  The old call reset
                # state mid-cycle, causing gad_1 to repeat every turn.

                try:
                    self.stt.resume_all()
                except Exception as e:
                    logger.warning(f"[INTERMISSION] STT resume for screening failed: {e}")

                response = self._listen_for_intermission_answer(
                    timeout=_SCREENING_LISTEN_TIMEOUT,
                    min_window=_SCREENING_MIN_LISTEN_WINDOW_SEC,
                )

                try:
                    self.stt.suspend_all()
                except Exception:
                    pass

                # ── Listener Lock OFF ────────────────────────────
                _listener_active.clear()

                clean = response.lower().strip()
                if clean == "__cmd_end__":
                    break
                if clean == "__cmd_start__":
                    self.say("We're already in session, and I'm listening.")
                    continue

                if not clean or len(clean) < 2:
                    logger.info("[INTERMISSION] No response to screening. Re-prompting.")
                    self.say("I didn't catch that. Could you try again?")

                    _listener_active.set()
                    try:
                        self.stt.resume_all()
                    except Exception:
                        pass
                    response = self._listen_for_intermission_answer(
                        timeout=_SILENCE_REPROMPT_SEC,
                        min_window=min(_SCREENING_MIN_LISTEN_WINDOW_SEC, _SILENCE_REPROMPT_SEC),
                    )
                    try:
                        self.stt.suspend_all()
                    except Exception:
                        pass
                    _listener_active.clear()
                    clean = response.lower().strip()
                    if clean == "__cmd_end__":
                        break
                    if clean == "__cmd_start__":
                        self.say("We're already in session, and I'm listening.")
                        continue

                    if not clean:
                        logger.info("[INTERMISSION] Silence timeout; marking question skipped.")
                        self.intermission_ladder.skip_screening_question(
                            question.question_id,
                            reason="silence_timeout",
                        )
                        self._persist_intermission_status(
                            question_id=question.question_id,
                            status="SKIPPED",
                            reason="silence_timeout",
                        )
                        continue

                if _is_repeat_request(clean):
                    logger.info("[INTERMISSION] Repeat requested.")
                    continue

                if _is_skip_question_request(clean):
                    logger.info("[INTERMISSION] User skipped current screening question.")
                    self.intermission_ladder.skip_screening_question(
                        question.question_id,
                        reason="user_skip_phrase",
                    )
                    self._persist_intermission_status(
                        question_id=question.question_id,
                        status="SKIPPED",
                        response_text=clean,
                        reason="user_skip_phrase",
                    )
                    continue

                if _is_opt_out(clean):
                    logger.info("[INTERMISSION] User opted out of this screening question.")
                    self.intermission_ladder.skip_screening_question(
                        question.question_id,
                        reason="opt_out",
                    )
                    self._persist_intermission_status(
                        question_id=question.question_id,
                        status="SKIPPED",
                        response_text=clean,
                        reason="opt_out",
                    )
                    continue

                score = score_response(clean)
                if score < 0:
                    self.intermission_ladder.skip_screening_question(
                        question.question_id,
                        reason="declined",
                    )
                    self._persist_intermission_status(
                        question_id=question.question_id,
                        status="SKIPPED",
                        response_text=clean,
                        reason="declined",
                    )
                else:
                    self.intermission_ladder.record_screening_answer(
                        question.question_id,
                        score=score,
                        response=clean,
                    )
                    self._persist_intermission_status(
                        question_id=question.question_id,
                        status="ANSWERED",
                        score=score,
                        response_text=clean,
                    )
                    logger.info(
                        f"[INTERMISSION] {question.question_id}: '{clean}' -> score={score}"
                    )

                time.sleep(_TRANSITION_PAUSE)
                continue

            if current_stage == IntermissionStage.BREATHING_EXERCISE:
                self.state = "intermission_exercise"
                exercise_text = self.intermission_ladder.next_breathing_exercise()
                logger.info("[INTERMISSION] Stage 2: breathing exercise.")
                self.say(exercise_text)

                # Brief listen for opt-out ("no", "skip", "just music")
                try:
                    self.stt.resume_all()
                except Exception:
                    pass
                opt_response = self._listen_for_intermission_answer(
                    timeout=6.0, min_window=3.0,
                )
                try:
                    self.stt.suspend_all()
                except Exception:
                    pass

                opt_clean = opt_response.lower().strip()
                if opt_clean and (_is_opt_out(opt_clean) or opt_clean in ("no", "nah", "nope", "no thanks")):
                    logger.info("[INTERMISSION] User declined breathing exercise. Switching to music bed.")
                    self.say("I'll let the music play while I finish my thoughts.")
                    self.music_service.set_base_volume(0.20)
                    self.music_service.start(_get_music_path())
                    self.intermission_ladder.mark_breathing_complete()
                    llm_done.wait(timeout=120.0)
                    break

                self.intermission_ladder.mark_breathing_complete()

                if llm_done.wait(timeout=_EXERCISE_HOLD_SEC):
                    break
                continue

            if current_stage == IntermissionStage.MUSIC:
                self.state = "music_fallback"
                if not self._music_announced_for_turn:
                    self.say("I'm still thinking, enjoy the music while I continue.")
                    self._music_announced_for_turn = True
                self.music_service.set_base_volume(0.15)
                self.music_service.start(_get_music_path())
                llm_done.wait(timeout=120.0)
                break

        watcher.join(timeout=2.0)
        self.music_service.set_base_volume(0.15)
        if io_record.END_SESSION_EVENT.is_set():
            self.state = "main_process"
            return

        if response_text[0]:
            logger.info("[INTERMISSION] Complete. Delivering LLM response.")

            if intermission_was_active:
                bridge = _random.choice(_BRIDGE_PHRASES)
                logger.info(f"[HANDOFF] Bridge phrase: '{bridge}'")
                self.say(bridge)

            time.sleep(0.5)
            self.say(response_text[0])
        else:
            # LLM timed out or produced no response — never leave silence.
            # Fall back to a breathing exercise so the user stays engaged.
            logger.warning("[INTERMISSION] LLM produced no response. Delivering therapeutic fallback.")
            fallback = self.intermission_ladder.next_breathing_exercise()
            self.say(fallback)
            self.say("I'm having a little trouble with my thoughts right now. Let me try again shortly.")

        self.state = "main_process"

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
                    self.say(f"[PLAY_MUSIC] {_get_music_path()}")

                if self.paused:
                    time.sleep(0.5)
                    continue

                # 2. Handle Idle State (Waiting for Voice Wake-up)
                if not io_record.START_SESSION_EVENT.is_set():
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
                        has_trigger = any(t in words for t in WAKE_TRIGGERS)
                        has_name = WAKE_NAME in words
                        command = self.global_command_matcher.match(text)
                        logger.info(f"Wake transcription: {text}")

                        if command == "START":
                            logger.info("Global START command detected in idle state.")
                            self.initialize_session()
                        elif has_trigger and has_name:
                            logger.info("Wake phrase accepted. Transitioning to onboarding.")
                            self.initialize_session()
                    continue

                # 3. Active Session — turn-taking loop
                #
                # Flow:  get first output -> speak -> [loop: listen -> send
                #        -> suspend STT -> intermission(speaks next output)
                #        -> loop back to listen]
                #
                # The intermission watchdog consumes the next agent utterance
                # from output_queue AND speaks it.  We loop back to listen()
                # directly — NOT to output_queue.get() — because the handler
                # is already waiting for the user's answer.
                try:
                    text_to_speak = self.output_queue.get(timeout=0.2)
                    if text_to_speak:
                        self.say(text_to_speak)

                        while (self.running
                               and io_record.START_SESSION_EVENT.is_set()
                               and not io_record.END_SESSION_EVENT.is_set()):

                            # Resume STT for listening
                            try:
                                self.stt.resume_all()
                            except Exception as e:
                                logger.warning(f"[VRAM HANDOFF] STT resume failed: {e}")

                            if not self.is_hands_free:
                                self.manual_input_event.wait()
                                self.manual_input_event.clear()

                            user_response = self._listen_with_retry(timeout=15.0, apply_priority_gate=True)
                            if not user_response:
                                self._consecutive_silence_count += 1
                                # After 2 consecutive silence rounds, reassure
                                # the user so the device never feels "broken".
                                if self._consecutive_silence_count >= 2:
                                    self.say("I'm still here, just listening to the music with you. Take your time.")
                                    self._consecutive_silence_count = 0
                                continue

                            self._consecutive_silence_count = 0

                            if user_response == "__CMD_END__":
                                break
                            if user_response == "__CMD_START__":
                                self.say("We're already in session, and I'm listening.")
                                continue

                            command = self._apply_global_command_priority(user_response)
                            if command == "END":
                                break
                            if command == "START":
                                self.say("We're already in session, and I'm listening.")
                                continue

                            # Global command gate is evaluated above before queueing text
                            # into the downstream response-analyzer path.
                            self.input_queue.put(user_response)

                            # ── VRAM Handoff: suspend STT for LLM ────────
                            logger.info(f"[VRAM HANDOFF] Pre-unload: {get_system_memory_snapshot()}")
                            try:
                                self.stt.suspend_all()
                            except Exception as e:
                                logger.warning(f"[VRAM HANDOFF] STT suspend failed: {e}")

                            time.sleep(0.3)

                            # Wait for next response with intermission
                            self._led_off()
                            self.state = "main_process"
                            self._wait_for_output_with_intermission()

                            self.post_turn_cleanup()
                            logger.info(f"[VRAM HANDOFF] Post-intermission: {get_system_memory_snapshot()}")

                except queue.Empty:
                    pass

            except Exception as e:
                logger.error(f"Error in speech service loop: {e}")
                time.sleep(1)

    def stop(self):
        self.running = False
        self.stop_audio()
        self.music_service.stop()
        self.gpio.cleanup()
        self.recorder.terminate()
        self.player.terminate()
