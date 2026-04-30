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
from src.core.therapy_content import SCORE_OPT_OUT, SCORE_UNRESOLVED, score_response
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

# Minimum engagement before an intermission can be cut short by an
# llm_done signal. If the LLM answers within the first few seconds of a
# meditation, we still let the user land at least one breath cycle (or
# hear a whole line of the music announcement) so the cut doesn't feel
# jarring. Anything above this threshold is safe to interrupt.
_INTERMISSION_MIN_ENGAGEMENT_SEC = 6.0

# Fast output-queue poll interval used by the proactive-activity's
# llm_done watcher. 100 ms is plenty — the activity's natural cadence
# is sentence-length (~3-5 s), so sub-second poll granularity is free.
_OUTPUT_READY_POLL_SEC = 0.1

# Bridge phrases spoken after intermission, before LLM response.
_BRIDGE_PHRASES = [
    "Thank you for reflecting on that with me. Now, going back to what you shared...",
    "I appreciate you sharing that. I've been thinking about what you said...",
    "Thank you for being open with me. Let me respond to what's on your mind.",
    "I appreciate your patience while I gathered my thoughts.",
    "Thank you for staying with me through that. Here's what I'd like to say...",
    "That was a nice moment of stillness. Now, about what you mentioned...",
]

# Intermission framing — spoken as a clear signpost so the user knows the
# next beat is SEPARATE from the therapy questioning.  A short lead-in
# plays before ANY intermission activity (screening / breathing / music)
# and a matching outro (close to, but clearly different from, the bridge)
# plays on the way back to CaiTI's next reply.  Keep both sides friendly
# and un-clinical — the goal is "we're stepping aside for a brief pause"
# rather than "we're starting a new clinical task".
_INTERMISSION_LEAD_INS = [
    "Let's take a brief intermission together while I gather my thoughts. "
    "This is separate from our main conversation — just a short pause.",
    "I'd like to step aside for a quick intermission. "
    "This is a little break, not part of the main questions we've been working through.",
    "Let's pause for a short intermission. "
    "This is apart from our main session — think of it as a gentle side-beat.",
    "Before I respond, let's take a brief intermission. "
    "This part is separate from the questions we've been exploring together.",
]

# Signposted end-of-intermission phrase. Distinct from `_BRIDGE_PHRASES`
# so the user can clearly tell the intermission has ended and we're
# returning to therapy.
_INTERMISSION_OUTROS = [
    "That wraps up our little intermission. Coming back to our session now...",
    "That's the end of this short intermission. Let's return to what we were exploring together.",
    "And with that, our intermission is complete. Returning to our main conversation now...",
    "Our brief intermission is over. Let's pick back up with our session.",
]

# Filler intros for screening questions (avoid abrupt "question" delivery).
# Used for the FIRST question in a paired screening block; the second
# question uses `_SCREENING_FOLLOWUPS` below so the transition sounds
# natural rather than like two unrelated prompts stapled together.
_SCREENING_INTROS = [
    "While I'm processing that, I'd like to ask you something.",
    "It's taking me a moment to reflect. Let me ask you this in the meantime.",
    "While I work through your response, let me check in with you.",
    "Give me just a moment. In the meantime, I'd like to ask...",
    "Let me ask you this while I gather my thoughts.",
]

# Softer connector used between the two questions in a paired screening
# block — keeps the pair feeling like a single gentle check-in rather
# than two abrupt clinical prompts.
_SCREENING_FOLLOWUPS = [
    "And one more quick check-in.",
    "While we're here, one more short question.",
    "And just one more along the same lines.",
    "One more brief question before we move on.",
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

# Background-music loudness contract
# ----------------------------------
# Two states, audibly distinct:
#
#   LOUD  (0.85, foreground listen level, comparable to Piper TTS):
#     - idle between turns
#     - in any intermission "hold" window (screening gap before the user
#       speaks, breathing gap between guidance phrases, music block)
#     - during the proactive activity's `never_done.wait()` hold
#
#   DUCKED (0.02, whisper, auto-applied by `_target_volume()`):
#     - whenever `_AI_IS_SPEAKING` is set (Piper TTS playing) or
#       `_USER_IS_SPEAKING` is set (mic listening, from the moment the
#       stream opens — not just after VAD detects voice).
#
# The auto-ducker in `src/drivers/audio.py::BackgroundMusicThread._target_volume`
# short-circuits any base_volume we set here as long as either flag is
# on, so raising these constants cannot make TTS less intelligible or
# leak music over the mic.  All four levels below therefore only take
# effect when NO ducking is active.
#
# Why we keep separate names for the four code-paths even though three
# of them use the same loud value: the call sites still semantically
# differ (handoff dip before TTS, breathing hold, music-block peak,
# resting ambient) and this lets a deployment tune one independently.
_MUSIC_BED_AMBIENT = 0.85          # idle / post-turn resting level (LOUD)
_MUSIC_BED_BREATHING = 0.85        # between meditation guidance phrases (LOUD)
_MUSIC_BED_INTERMISSION = 0.85     # MUSIC intermission block (LOUD, TTS-match)
_MUSIC_BED_HANDOFF = 0.05          # deep dip ~1 s before LLM response TTS
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
    """Regex + fuzzy priority matcher for start/end global voice commands.

    Bug-2 fix — end commands are split into two categories:

    * SOFT_END: phrases signalling "I'm done with screening questions"
      but not necessarily "quit the whole app". Examples:
      "that's enough for today", "no more questions", "I don't want
      to answer any more questions", "I'm done with questions".
      Pre-CBT these route to the Response Analyzer as the paper's
      `Stop` keyword so the screening loop terminates and CBT still
      runs (paper §5.1). Once CBT has started, SOFT_END is treated
      as HARD_END — the user wants to leave therapy.

    * HARD_END: explicit "end/stop/finish/close session", "goodbye",
      "bye". Always terminates the whole session immediately, skips
      CBT if not yet started, runs the goodbye/closing-summary path.

    START commands are unchanged.
    """

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
    # HARD_END: user explicitly wants to end the session (kill switch).
    # Short-utterance + explicit-session-word bias keeps "I don't want to
    # answer any more questions. Let's end the session." (long, mixed) OUT
    # of this category — that sentence carries the softer "finish
    # screening" intent and should route through SOFT_END instead.
    _HARD_END_PATTERNS = (
        # Whole-utterance end command (≤ ~5 words): "end session", "stop
        # the session please", "finish session".
        re.compile(
            r"^\s*(?:end|stop|finish|close)\s+(?:the|this|dis|da)?\s*session\s*(?:please|now)?\s*[.!?]*\s*$",
            re.IGNORECASE,
        ),
        # Whole-utterance goodbye: "goodbye", "bye", "goodbye now"
        re.compile(r"^\s*(?:good\s?bye|bye)\s*(?:now|please)?\s*[.!?]*\s*$", re.IGNORECASE),
    )
    # SOFT_END: "finish screening, move to CBT" intent. Matches any
    # phrasing that signals the user is done with screening QUESTIONS
    # specifically, not the whole session. Pre-CBT these become `Stop`
    # via the Response Analyzer; post-CBT-start they upgrade to HARD_END.
    #
    # NOTE: patterns are matched AFTER `_normalize()` strips punctuation
    # (so "don't" → "don t", "that's" → "that s"). Patterns below use
    # optional-space forms to match both normalized and raw shapes.
    _SOFT_END_PATTERNS = (
        re.compile(r"\bno\s+more\s+questions?\b", re.IGNORECASE),
        # "I don't/dont/do not want to answer (any) (more) questions"
        re.compile(
            r"\b(?:i\s+)?(?:don\s*t|dont|do\s+not)\s+want\s+to\s+answer\s+(?:any\s+)?(?:more\s+)?questions?\b",
            re.IGNORECASE,
        ),
        # "that's / thats / that is / that s enough for today/now"
        re.compile(
            r"\b(?:that\s*s|thats|that\s+is)\s+enough\s+(?:for\s+today|for\s+now)\b",
            re.IGNORECASE,
        ),
        # "I'm / I am / im done with (the) questions"
        re.compile(
            r"\b(?:i\s*m|im|i\s+am)\s+done\s+(?:with\s+)?(?:the\s+)?questions?\b",
            re.IGNORECASE,
        ),
        re.compile(r"\benough\s+questions?\b", re.IGNORECASE),
        re.compile(r"\bstop\s+(?:the\s+)?questions?\b", re.IGNORECASE),
        # Longer "let's end the session" phrasings that mix end-intent
        # with extra clauses route softly; short "end session" alone
        # matches _HARD_END_PATTERNS above and is unaffected.
        re.compile(
            r"\blet\s*s\s+end\s+(?:the|this)\s+session\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bi\s+want\s+to\s+end\s+(?:the|this)\s+session\b",
            re.IGNORECASE,
        ),
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

    def _fuzzy_hard_end(self, text: str) -> bool:
        """Only short, unambiguous utterances qualify for HARD_END.

        Previous implementation matched ANY utterance containing an end-
        word near "session", which greedily swallowed mixed-intent lines
        like "I don't want to answer any more questions. Let's end the
        session." — that's now handled by SOFT_END instead.
        """
        tokens = text.split()
        if not tokens:
            return False

        # "goodbye" / "bye" alone is sufficient — but require EXACT match.
        # SequenceMatcher ratios are unreliable for short tokens: "be"
        # ↔ "bye" scores 0.80 (triggering at the old default threshold),
        # and any utterance containing the common word "be" (e.g.
        # "need to be healthy") would falsely END the session (observed
        # mid-CBT Stage 2 for Kyle, 2026-04-26 — user said "I need to be
        # healthy and have green food" while answering the CHALLENGE
        # prompt and the session terminated).  A strict exact-match is
        # the only safe policy for these 3-letter tokens.
        tok_set = set(tokens)
        if len(tokens) <= 3 and ("goodbye" in tok_set or "bye" in tok_set):
            return True

        # Short utterance (≤ 4 tokens) containing "session" + an end-word.
        # Prevents the old catch-all from swallowing long mixed-intent
        # sentences; short utterances like "end the session please" still
        # qualify.
        if len(tokens) > 5:
            return False
        has_session = "session" in tok_set or self._token_hit(tok_set, ("session",), 0.75)
        if not has_session:
            return False

        has_end_token = self._token_hit(tok_set, ("end", "stop", "finish", "close"), 0.80)
        has_start_token = self._token_hit(tok_set, ("start", "begin", "hello", "hi"), self.FUZZY_THRESHOLD)
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
        """Classify utterance as HARD_END / SOFT_END / START / None.

        Returns:
            "HARD_END" — explicit quit-the-whole-session command.
            "SOFT_END" — finish-screening-questions intent (pre-CBT
                         routes to Response Analyzer's Stop keyword;
                         post-CBT-start upgrades to HARD_END).
            "START"    — session-start wake command.
            None       — no global command detected; normal clinical
                         utterance.

        Matching order: SOFT_END is checked BEFORE HARD_END because
        SOFT_END phrases are more specific ("let's end the session"
        with context "screening questions") and should not be swallowed
        by a broad HARD_END match. Only utterances that fail SOFT_END
        are considered for HARD_END.
        """
        text = self._normalize(str(transcript or ""))
        if not text:
            return None

        # SOFT_END first: more specific "finish screening" intent.
        if any(p.search(text) for p in self._SOFT_END_PATTERNS):
            return "SOFT_END"

        # HARD_END: short, unambiguous kill-switch phrases. We now only
        # reach this branch if no SOFT_END pattern matched.
        if any(p.search(text) for p in self._HARD_END_PATTERNS):
            return "HARD_END"
        if self._fuzzy_hard_end(text):
            return "HARD_END"

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
        logger.debug("Initializing Unified Speech Interaction Service...")
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
        # Set at the end of onboarding; consumed by the main loop to route
        # the first LLM utterance through the intermission pipeline so the
        # post-greeting / pre-first-question gap is never silent.
        self._first_output_pending = False

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

    def say(self, text, voice: str = "primary"):
        """Speak text via TTS or play music.  Blocks until playback finishes.

        Every utterance that actually becomes audio is logged as [TTS] so
        the clinician log is a complete record of what the device said,
        including onboarding greetings, bridge phrases, breathing scripts
        and goodbye lines that never route through log_question's [AGENT]
        tag. Dedup guard below drops the log line when the last [AGENT]
        event already carried the same text (handler-driven clinical turns).

        ``voice="primary"`` (default) uses the CaiTI voice; callers can
        pass ``voice="intermission"`` to route this utterance through the
        second Piper voice configured in config.yaml.  When the second
        voice isn't configured or its files are missing, TTSGenerator
        silently falls back to the primary voice so the user never
        experiences a dropout.
        """
        if not text:
            return
        logger.debug(f"Agent Action [{voice}]: {text[:120]}{'...' if len(text) > 120 else ''}")

        if text.startswith("[PLAY_MUSIC]"):
            music_file = text.split(" ", 1)[1] if " " in text else _get_music_path()
            logger.debug(f"Starting background music loop: {music_file}")
            prev_state = self.state
            self.state = "music_fallback"
            self.music_service.start(music_file)
            self.state = prev_state
            return

        # Clinician-facing record of the actual spoken audio. Dedup against
        # the last [AGENT] line so handler-driven questions (already logged
        # by log_question) don't appear twice on the console.
        last_agent = str(getattr(io_record, "_LAST_AGENT_LOGGED", "") or "")
        if text.strip() and text.strip() != last_agent.strip():
            logger.info(f"[TTS/{voice}] {text}")

        prev_state = self.state
        self.state = "speaking"
        wav_file = "active_ai_response.wav"
        if self.tts.generate(text, wav_file, voice=voice):
            self._led_off()
            self.stop_playback_event.clear()
            self.player.play(wav_file, stop_event=self.stop_playback_event)
        else:
            # TTS completely failed (Piper + espeak both down).
            # Bump the music so the user hears *something* rather than silence.
            logger.error("[TTS FAILURE] Both engines failed. Raising music to cover silence gap.")
            self.music_service.set_base_volume(_MUSIC_BED_AMBIENT)
        self.state = prev_state

    def say_intermission(self, text):
        """Convenience wrapper: speak through the intermission voice.

        Routes the utterance through the second Piper voice (Alan by
        default) so the user hears a clearly different speaker during
        intermission beats vs. therapy turns.  Falls back silently to
        the primary voice when the second voice isn't configured —
        clinical data must still land audibly.
        """
        self.say(text, voice="intermission")

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
        """Apply START/END priority matching before queueing input to the NLP stack.

        Bug-2 fix — routing depends on whether CBT has started:

        * HARD_END → always close session immediately (paper §5.1 is
          unaffected; this is a kill-switch, not a clinical signal).
        * SOFT_END → pre-CBT: return None so the transcript flows to the
          Response Analyzer; the analyzer will classify as `Stop`,
          which terminates screening and still routes into CBT
          (handler_rl → run_cbt). Post-CBT-start: upgrade to HARD_END
          so the user can leave therapy.
        * START → pass through.
        """
        command = self.global_command_matcher.match(transcript)
        if command == "HARD_END":
            logger.info("[SESSION] Hard end command heard — closing session.")
            self.handle_exit()
            return "END"
        if command == "SOFT_END":
            if io_record.CBT_STARTED_EVENT.is_set():
                # Mid/post-CBT: user wants out. Escalate to hard end.
                logger.info(
                    "[SESSION] Soft end command heard during CBT — "
                    "escalating to hard end."
                )
                self.handle_exit()
                return "END"
            # Pre-CBT: let the transcript reach the Response Analyzer.
            # It will classify the phrase (e.g. "no more questions",
            # "that's enough for today") as `Stop`, which the paper's
            # screening loop treats as "terminate screening, proceed
            # to CBT" (paper §5.1 / §5.3). Return None so the caller
            # does NOT substitute a __CMD_END__ sentinel.
            logger.info(
                "[SESSION] Soft end command heard pre-CBT — "
                "routing to screening Stop keyword → CBT."
            )
            return None
        if command == "START":
            return "START"
        return None

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
        """Record and transcribe with LED feedback.

        Kept for call sites (wake listen, intermission screening) that
        want the whole record-and-transcribe step to run on the calling
        thread.  The main turn loop now uses `record_utterance_to_wav`
        + a background STT worker so the intermission can start speaking
        the moment the user stops speaking instead of waiting on STT.
        """
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
            logger.debug(f"[AUDIO HYGIENE] RMS {rms:.5f} below threshold — skipping disk write.")
            self.state = "idle"
            return ""
        self.recorder.save_wav(audio_frames, user_wav)
        text = self.transcribe(user_wav, apply_priority_gate=apply_priority_gate)

        logger.debug(f"User heard: {text}")
        self.state = "idle"
        return text

    # Audio-frames duration threshold below which we fall back to the
    # serial path (listen() + fragment merge).  Short utterances risk
    # being partial — the current retry path listens a second time, so
    # we must not commit to an intermission until we know the full thing.
    # 3 s of VAD-gated audio is plenty to distinguish "yes" (~0.7s) from
    # "I have been feeling pretty down lately" (~3-4s).
    _PARALLEL_MIC_WINDOW_MIN_SEC = 3.0

    def _record_utterance_to_wav(self, timeout: float) -> tuple[str, float] | None:
        """Record until silence and save to disk.  Returns (wav_path, duration_sec)
        or None if the mic window produced nothing usable.

        This is the mic-bound half of the old `listen()`.  STT is NOT run
        here — the caller decides whether to transcribe inline or fire a
        background worker while it does something else (e.g., intermission).
        """
        self.state = "main_listen"
        self._led_on()
        audio_frames = self.recorder.record_until_silence(max_duration=timeout)
        self._led_off()

        if not audio_frames:
            self.state = "idle"
            return None

        rms = self.recorder.compute_rms(audio_frames)
        if rms < 0.005:
            logger.debug(f"[AUDIO HYGIENE] RMS {rms:.5f} below threshold — skipping disk write.")
            self.state = "idle"
            return None

        # Per-turn unique filename so a SCREENING block running in
        # parallel (which writes to "active_user_input.wav" via listen()
        # → save_wav) can't overwrite this turn's audio while the STT
        # worker is still reading it.
        user_wav = f"active_user_input_{int(time.monotonic() * 1000)}.wav"
        self.recorder.save_wav(audio_frames, user_wav)
        # Duration = n_frames * chunk_size / sample_rate (each frame is
        # one chunk from the PyAudio read loop).
        chunk = getattr(self.recorder, "chunk", 480)
        rate = getattr(self.recorder, "rate", 16000)
        duration = (len(audio_frames) * chunk) / float(rate) if rate else 0.0
        self.state = "main_process"
        return user_wav, duration

    def _start_transcription_worker(
        self,
        wav_path: str,
        apply_priority_gate: bool,
    ) -> tuple[threading.Thread, queue.Queue, threading.Event]:
        """Transcribe `wav_path` in a daemon thread, hand-off to handler.

        The worker does three things in order:

        1. Transcribe the saved WAV and apply the global command gate.
        2. On the HAPPY path (non-empty, non-command transcript) push the
           text onto ``self.input_queue`` so the handler can start its
           LLM call immediately — BEFORE the intermission finishes.
           This is what actually makes the pipeline parallel; without
           it, the handler would sit idle until the main thread reads
           from `result_q` and puts to input_queue, which only happens
           AFTER intermission returns.
        3. Suspend Whisper so Gemma has the memory budget it needs.

        The returned `result_q` carries the transcript (or sentinel)
        so the main thread can decide whether to deliver the LLM
        response (happy path) or short-circuit (silence / END / START).
        On sentinel outcomes the worker does NOT push to input_queue —
        the handler must never see an empty / command-masquerading-as-
        answer input.

        `abort_event` is set alongside sentinel outcomes so a main-
        thread caller (e.g. one driving the intermission) can stop
        playback promptly without polling the queue.  The VRAM handoff
        (`stt.suspend_all`) MUST happen inside this worker — if the
        main thread suspended Whisper before decode completed, the
        model reference would vanish mid-inference.
        """
        result_q: queue.Queue = queue.Queue(maxsize=1)
        abort_event = threading.Event()

        def _worker():
            try:
                text = self.transcribe(wav_path, apply_priority_gate=apply_priority_gate)
            except Exception as e:
                logger.error(f"[PARALLEL_STT] Transcription raised: {e}")
                text = ""

            is_sentinel = text in {"__CMD_END__", "__CMD_START__"} or not text
            if is_sentinel:
                abort_event.set()
            else:
                # HAPPY PATH: queue the transcript for the handler RIGHT
                # NOW so its LLM call runs in parallel with the ongoing
                # intermission. Without this the handler is blocked on
                # `input_queue.get()` for the full duration of the
                # intermission block and latency-masking is defeated.
                try:
                    self.input_queue.put_nowait(text)
                except queue.Full:
                    # input_queue is bounded (see io_record.py); if full,
                    # fall back to a blocking put so we don't drop the
                    # user's utterance.
                    logger.warning("[PARALLEL_STT] input_queue full; blocking put.")
                    self.input_queue.put(text)

            try:
                result_q.put_nowait(text)
            except queue.Full:
                # Shouldn't happen (maxsize=1 and we only put once), but
                # defend against a double-put race on reentrancy.
                logger.warning("[PARALLEL_STT] result_q unexpectedly full; dropping.")

            try:
                self.stt.suspend_all()
            except Exception as e:
                logger.warning(f"[VRAM HANDOFF] STT suspend (worker) failed: {e}")

        thread = threading.Thread(target=_worker, daemon=True,
                                  name="ParallelSTTWorker")
        thread.start()
        return thread, result_q, abort_event

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
            logger.debug(f"[STT] Short transcript ({len(words)} words). Checking for continuation...")
            extra = self.listen(timeout=4.0, apply_priority_gate=apply_priority_gate)
            if extra:
                if extra in {"__CMD_END__", "__CMD_START__"}:
                    return extra
                merged = f"{text} {extra}"
                logger.debug(f"[STT] Merged fragments: '{merged}'")
                return merged
        return text

    def _run_parallel_turn(self) -> str:
        """One turn with STT running in parallel with the intermission.

        Orchestration outline:

        1. Record mic until silence (synchronous — mic is a single lock).
        2. If the captured audio was short (< _PARALLEL_MIC_WINDOW_MIN_SEC)
           OR we heard nothing, fall back to the old serial path that
           supports fragment-merge. Short/fragmentary utterances are rare
           but clinically important; we pay the old ~3 s serialisation
           cost only on them.
        3. Otherwise: spawn a STT worker on the saved WAV, and
           immediately call `_run_one_intermission_activity()` on this
           thread. Once intermission returns (either naturally or
           interrupted by `_start_output_ready_watcher` firing), wait
           briefly for the STT worker so we can deal with empty / END /
           START results before handing off to the handler.
        4. If STT outcome was OK, queue the transcript on `input_queue`
           (if it wasn't queued already by the parallel worker — we
           handle that inside the worker to avoid another round of
           gating on the main thread) and run the delivery phase.

        Returns one of:
          ``"delivered"``   — happy path, LLM output delivered to user.
          ``"silence"``     — mic silent / sub-RMS / STT empty.
          ``"session_end"`` — user said an END command.
          ``"start_echo"``  — user said a START command (already in session).
        """
        # Resume STT if a prior turn suspended it; otherwise the upcoming
        # record→save is fine but the transcription worker would fail.
        try:
            self.stt.resume_all()
        except Exception as e:
            logger.warning(f"[VRAM HANDOFF] STT resume failed: {e}")

        rec = self._record_utterance_to_wav(timeout=15.0)
        if rec is None:
            logger.debug("[TURN] Mic window produced no usable audio.")
            return "silence"
        wav_path, duration = rec
        logger.debug(
            f"[TURN] Captured {duration:.1f}s of audio -> {wav_path}. "
            f"{'Parallel' if duration >= self._PARALLEL_MIC_WINDOW_MIN_SEC else 'Serial'} path."
        )

        try:
            # ── Short-utterance serial fallback ────────────────────────────
            # Fragment-merge retry requires reopening the mic for a second
            # listen, which is incompatible with starting an intermission
            # (intermission talks on the same audio device and would need to
            # be torn down to re-listen). Use the old path for these rare
            # cases.
            if duration < self._PARALLEL_MIC_WINDOW_MIN_SEC:
                user_response = self.transcribe(wav_path, apply_priority_gate=True)
                # Fragment merge if short + no punctuation.
                if user_response not in ("", "__CMD_END__", "__CMD_START__"):
                    words = user_response.split()
                    if len(words) <= 2 and not user_response.rstrip().endswith((".", "!", "?")):
                        extra = self.listen(timeout=4.0, apply_priority_gate=True)
                        if extra and extra not in ("__CMD_END__", "__CMD_START__"):
                            user_response = f"{user_response} {extra}"
                            logger.debug(f"[STT] Serial-path merged fragments: '{user_response}'")
                        elif extra in ("__CMD_END__", "__CMD_START__"):
                            user_response = extra
                try:
                    self.stt.suspend_all()
                except Exception as e:
                    logger.warning(f"[VRAM HANDOFF] STT suspend (serial) failed: {e}")

                if not user_response:
                    return "silence"
                if user_response == "__CMD_END__":
                    return "session_end"
                if user_response == "__CMD_START__":
                    return "start_echo"
                command = self._apply_global_command_priority(user_response)
                if command == "END":
                    return "session_end"
                if command == "START":
                    return "start_echo"
                self.input_queue.put(user_response)

                self._led_off()
                self.state = "main_process"
                self._run_one_intermission_activity()
                self._wait_for_output_with_intermission()
                return "delivered"

            # ── Parallel path ─────────────────────────────────────────────
            # Kick off STT decode in the worker immediately, then drop into
            # the intermission on this thread so the user hears "While I'm
            # thinking..." / breathing / music starting within ~0.1 s of the
            # mic closing.
            stt_thread, result_q, abort_event = self._start_transcription_worker(
                wav_path, apply_priority_gate=True,
            )

            self._led_off()
            self.state = "main_process"

            # Drive the intermission. The output-ready watcher inside will
            # trip stop_playback_event as soon as the handler produces its
            # response; the STT-side abort_event handles the rare case
            # where the user's utterance was empty / a global command.
            self._run_one_intermission_activity()

            # Give STT a moment to finish if it hasn't already (intermission
            # may be over; Whisper typically wraps up in under 2 s for a
            # 3-10 s utterance on CPU). We bound the wait at ~8 s so a
            # degenerate stuck decode can't block the whole session.
            try:
                user_response = result_q.get(timeout=8.0)
            except queue.Empty:
                logger.error("[PARALLEL_STT] Worker did not report result within 8s; treating as silence.")
                try:
                    self.stt.suspend_all()
                except Exception:
                    pass
                return "silence"
            finally:
                # Daemon thread cleanup — don't join hard so a wedged decode
                # can't block shutdown.
                stt_thread.join(timeout=0.5)

            # Handle STT outcomes. The worker has already suspended Whisper.
            if not user_response:
                logger.info("[TURN] Parallel STT returned empty — silence path.")
                return "silence"
            if user_response == "__CMD_END__":
                logger.info("[TURN] Parallel STT detected END command.")
                # Cut any still-playing intermission audio.
                self.stop_playback_event.set()
                return "session_end"
            if user_response == "__CMD_START__":
                logger.info("[TURN] Parallel STT detected START command.")
                self.stop_playback_event.set()
                return "start_echo"

            # Normal path: the STT worker has ALREADY queued the
            # transcript on `input_queue` (see
            # `_start_transcription_worker`) so the handler is
            # already busy running its LLM call in parallel with the
            # intermission we just ran. All we need to do here is run
            # the delivery phase, whose watcher will short-circuit if
            # the handler's output is already in output_queue.
            # (We do NOT re-apply the priority gate here — the worker
            # already ran it via `transcribe(apply_priority_gate=True)`.)
            self._wait_for_output_with_intermission()
            return "delivered"
        finally:
            # Per-turn WAV cleanup — best-effort, don't let a missing file
            # or permission error fail the turn. The unique filename means
            # leftover WAVs are harmless even if cleanup misses one; this
            # loop just keeps the working directory tidy over long sessions.
            try:
                if os.path.isfile(wav_path):
                    os.remove(wav_path)
            except OSError:
                pass

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
        """Triggered by voice or Button 1: Ask for user name and init session.

        The greeting is split into two TTS utterances with a short music
        beat between them so the bed swells briefly before the name
        prompt, giving the opener an "arrival" feel rather than sounding
        like a single fast sentence.
        """
        self.state = "onboarding"
        logger.info("[SESSION] Onboarding — capturing subject name.")
        self.say("Hello, I'm CaiTI.")
        # Short music beat: hold music at the foreground bed level between
        # the introduction and the name prompt so the opener has an
        # "arrival" feel.  Auto-duck re-engages as soon as `say()` fires
        # the next TTS, so this doesn't compete with the name question.
        self.music_service.fade_to(_MUSIC_BED_AMBIENT, duration=0.8)
        time.sleep(1.2)
        self.say("Who am I speaking with today?")

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
                logger.debug(f"[ONBOARD BYPASS] Trigger phrase detected in '{name}'. Starting as 'User'.")
                name = "User"
            elif not self._is_valid_name(name):
                logger.warning(f"[NAME GUARD] Invalid name '{name}' (attempt {attempt}/{max_attempts}).")
                if attempt < max_attempts:
                    self.say("I'm sorry, I missed that. What was your name again?")
                    continue
                else:
                    self.say("Let me just call you 'User' for now. We can change that later.")
                    name = "User"

            logger.debug(f"[NAME GUARD] Accepted raw transcript '{name}' as valid name.")
            stripped = re.sub(
                r"^(?:my\s+name\s+is|i\s*(?:am|'m)\s|it'?s\s|they\s+call\s+me\s)",
                "", name, flags=re.IGNORECASE,
            ).strip()
            clean = re.sub(r"[^A-Za-z0-9 _-]", "", stripped or name).strip()
            # Canonicalise to lowercase so "Alice" and "alice" resolve to the
            # same longitudinal Q-table + DB user row across sessions. Without
            # this, a returning user spoken as "Alice" one day and "alice" the
            # next would start with a fresh Q-table each time.
            uid = (clean.replace(" ", "_") or "User").lower()
            logger.info(f"[SESSION] Initializing session for subject: {uid}")
            io_record.reset_session(uid)
            io_record.END_SESSION_EVENT.clear()

            # Suspend STT before starting session — LLM needs memory for greeting.
            logger.debug("[VRAM HANDOFF] Pre-session: suspending STT before pipeline starts.")
            try:
                self.stt.suspend_all()
            except Exception as e:
                logger.warning(f"[VRAM HANDOFF] Pre-session STT suspend failed: {e}")

            io_record.START_SESSION_EVENT.set()
            self.intermission_ladder.reset()
            self.music_service.start(_get_music_path())

            # Personalised handshake greeting (protocol stage 2).  "User" is
            # the fallback uid when the name loop fails — drop it to keep
            # the opener natural ("Hello, I'm CaiTI..." rather than
            # "Hello, User, I'm CaiTI...").
            spoken_name = clean if clean and clean.lower() != "user" else ""
            if spoken_name:
                self.say(
                    f"Hello, {spoken_name}. I'm CaiTI, your intelligent "
                    "therapist. Thank you for joining me today."
                )
            else:
                self.say(
                    "I'm CaiTI, your intelligent therapist. "
                    "Thank you for joining me today."
                )

            # Arm the main loop to route the *first* LLM utterance through
            # the intermission pipeline — PHQ/breathing/music will fill the
            # pre-first-question gap instead of the user hearing silence
            # while the LLM generates the opening dimension question.
            self._first_output_pending = True
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
        logger.info("[SESSION] Ending via hardware button or voice command.")
        io_record.END_SESSION_EVENT.set()
        io_record.START_SESSION_EVENT.clear()
        self.intermission_ladder.reset()
        self._music_announced_for_turn = False
        # If the session ended before the first LLM output arrived, clear
        # the flag so the next wake starts cleanly instead of routing
        # a non-existent "first" turn through intermission.
        self._first_output_pending = False
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
            self.music_service.set_base_volume(_MUSIC_BED_AMBIENT)
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
            logger.debug("Speech loop paused.")
        else:
            logger.debug("Speech loop resumed.")

    # ------------------------------------------------------------------ #
    # Intermission State Machine                                           #
    # ------------------------------------------------------------------ #

    # ── Intermission helpers ───────────────────────────────────────────
    #
    # Each _run_*_block returns a dict describing the outcome so the
    # parent loop can decide whether to loop, fall through to another
    # activity within the same turn, or break out and deliver the LLM.
    #
    # Outcome keys used across blocks:
    #   "end"         — END_SESSION signal received; caller should break.
    #   "llm_ready"   — LLM output arrived during the block; caller should
    #                   break and deliver it (after music fade-down).
    #   "declined"    — user asked to skip this activity; caller should
    #                   fall through to the declared fallback for this turn.
    #   "completed"   — activity finished; caller continues normal cycling.

    def _speak_intermission_lead_in(self, stage: IntermissionStage) -> None:
        """Speak a short "this is an intermission" signpost before an activity.

        Plays before every proactive intermission block so the user can
        clearly distinguish the intermission from CaiTI's therapy
        questioning.  Lead-in wording is stage-agnostic by design — the
        point is to mark the separation, not preview the activity.

        Routed through the intermission voice so the voice change itself
        is the first sensory signal that the intermission has started.
        """
        lead_in = _random.choice(_INTERMISSION_LEAD_INS)
        logger.debug(f"[INTERMISSION] Lead-in ({stage.value}): {lead_in}")
        self.say_intermission(lead_in)
        # Small beat so the lead-in lands before the activity starts.
        time.sleep(0.4)

    def _run_paired_screening_block(self, llm_done, listener_active):
        """Ask up to two screening questions together, preferring same-scale pairs.

        Groups GAD-1+GAD-2 (anxiety) and PHQ-1+PHQ-2 (depression) so the
        user experiences a single gentle check-in rather than one abrupt
        question at a time.  Falls back to one question when only one
        remains, or when the next scale-matched question is missing.
        Outcome propagation:
          - The first question's outcome dominates (end / declined /
            completed).  A declined first question falls through without
            asking the second; the caller will move on to another
            activity within the same turn per the existing design.
          - After a successful first answer, the second (if available)
            is introduced via `_SCREENING_FOLLOWUPS` so the transition
            reads naturally.
        """
        first = self.intermission_ladder.next_screening_question()
        if first is None:
            return {"outcome": "completed"}

        # Skip the "while I'm processing that..." intro on the first
        # question — the lead-in speech that just played already handled
        # framing, so we go straight into the question text.
        first_result = self._run_screening_block(
            first, llm_done, listener_active, skip_intro=True,
        )
        first_outcome = first_result.get("outcome", "completed")
        if first_outcome in ("end", "declined"):
            # Honour the user's signal — don't push a second question on
            # top of a decline / end command.
            return first_result

        # Look for a scale-matched partner; else any remaining question.
        partner = self._pick_scale_paired_question(first.question_id)
        if partner is None:
            return first_result

        # Softer connector so the pair feels like one gentle check-in
        # instead of two stapled prompts.  Uses its own intros pool so
        # the wording doesn't collide with the primary intro.  Spoken in
        # the intermission voice for continuity with the rest of the
        # screening block.
        connector = _random.choice(_SCREENING_FOLLOWUPS)
        logger.debug(f"[PHQ4] Paired follow-up connector: {connector}")
        self.say_intermission(connector)
        time.sleep(0.3)

        second_result = self._run_screening_block(
            partner, llm_done, listener_active,
            # Skip the random intro on the second question so we don't
            # double up on "while I'm processing that..." phrasing after
            # the connector we just spoke.
            skip_intro=True,
        )
        return second_result

    def _pick_scale_paired_question(self, first_question_id: str):
        """Return the next unanswered question whose scale matches `first_question_id`.

        Falls back to any remaining unanswered question if the matching
        scale's partner is already resolved.  Returns None when no
        screening question is available at all.
        """
        from src.core.therapy_content import CLINICAL_SCREENING
        scale_by_id = {q["id"]: q["scale"] for q in CLINICAL_SCREENING}
        target_scale = scale_by_id.get(first_question_id)
        next_q = self.intermission_ladder.next_screening_question()
        if next_q is None:
            return None
        if target_scale and scale_by_id.get(next_q.question_id) != target_scale:
            # The next pending question is a different scale.  Prefer it
            # only as a fallback — if there's no same-scale partner
            # pending, we still ask the cross-scale one so the user
            # gets two check-ins per intermission instead of one.
            pass
        return next_q

    def _run_screening_block(self, question, llm_done, listener_active, skip_intro: bool = False):
        """Ask one PHQ/GAD question and record the result.

        All intermission-domain utterances (intro, question text, silence
        re-prompt) use the intermission voice so the PHQ/GAD questions
        don't sound like CaiTI's therapy prompts.  System-level lines
        like "we're already in session" stay on the primary voice — they
        are CaiTI speaking a correction, not part of the intermission.
        """
        self.state = "intermission_screening"
        if skip_intro:
            full_prompt = f"{question.text}\n{_SCREENING_OPTIONS_HINT}"
        else:
            intro = _random.choice(_SCREENING_INTROS)
            full_prompt = f"{intro} {question.text}\n{_SCREENING_OPTIONS_HINT}"
        logger.info(f"[PHQ4] Asking screening question: {question.question_id}")
        self.say_intermission(full_prompt)

        listener_active.set()
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
        listener_active.clear()

        clean = response.lower().strip()
        if clean == "__cmd_end__":
            return {"outcome": "end"}
        if clean == "__cmd_start__":
            self.say("We're already in session, and I'm listening.")
            return {"outcome": "completed"}

        # Empty / very short transcript — one silent re-prompt before skip.
        if not clean or len(clean) < 2:
            logger.info("[PHQ4] No response heard — re-prompting.")
            self.say_intermission("I didn't catch that. Could you try again?")
            listener_active.set()
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
            listener_active.clear()
            clean = response.lower().strip()
            if clean == "__cmd_end__":
                return {"outcome": "end"}
            if clean == "__cmd_start__":
                self.say("We're already in session, and I'm listening.")
                return {"outcome": "completed"}
            if not clean:
                logger.info("[PHQ4] Silence timeout — marking question SKIPPED.")
                self.intermission_ladder.skip_screening_question(
                    question.question_id, reason="silence_timeout",
                )
                self._persist_intermission_status(
                    question_id=question.question_id,
                    status="SKIPPED", reason="silence_timeout",
                )
                return {"outcome": "completed"}

        if _is_repeat_request(clean):
            logger.info("[INTERMISSION] Repeat requested.")
            return {"outcome": "completed"}

        # Phase A: distinguish the three outcomes so the audit trail
        # records what actually happened:
        #   - Explicit skip / opt-out  -> SKIPPED (reason captured)
        #   - SCORE_UNRESOLVED         -> UNRESOLVED (non-empty utterance
        #                                 with no Likert anchor; NOT a
        #                                 false-negative on PHQ-4)
        #   - Valid 0-3 score          -> ANSWERED
        user_declined = False
        skip_reason = None
        if _is_skip_question_request(clean):
            user_declined, skip_reason = True, "user_skip_phrase"
        elif _is_opt_out(clean):
            user_declined, skip_reason = True, "opt_out"
        else:
            score = score_response(clean)
            if score == SCORE_OPT_OUT:
                user_declined, skip_reason = True, "opt_out_keyword"
            elif score == SCORE_UNRESOLVED:
                # A1/A2: non-empty but not a Likert anchor. Record as
                # UNRESOLVED so a clinician reviewing the export can
                # distinguish "user mumbled" from "user opted out" and
                # from "user scored 0". The reason tag is `stt_unresolved`
                # so the forensic audit query groups it under an STT-layer
                # issue rather than a user-initiated skip.
                logger.info(
                    f"[PHQ4] {question.question_id} UNRESOLVED: '{clean}' (STT couldn't map to a Likert anchor)"
                )
                self.intermission_ladder.skip_screening_question(
                    question.question_id, reason="stt_unresolved",
                )
                self._persist_intermission_status(
                    question_id=question.question_id,
                    status="UNRESOLVED", response_text=clean,
                    reason="stt_unresolved",
                )
                return {"outcome": "completed"}
            else:
                self.intermission_ladder.record_screening_answer(
                    question.question_id, score=score, response=clean,
                )
                self._persist_intermission_status(
                    question_id=question.question_id,
                    status="ANSWERED", score=score, response_text=clean,
                )
                logger.info(
                    f"[PHQ4] {question.question_id} ANSWERED: '{clean}' -> score={score}"
                )
                time.sleep(_TRANSITION_PAUSE)
                return {"outcome": "completed"}

        if user_declined:
            logger.info(f"[PHQ4] Declined by user ({skip_reason}); falling through to breathing.")
            self.intermission_ladder.skip_screening_question(
                question.question_id, reason=skip_reason,
            )
            self._persist_intermission_status(
                question_id=question.question_id,
                status="SKIPPED", response_text=clean, reason=skip_reason,
            )
            return {"outcome": "declined"}

        return {"outcome": "completed"}

    def _run_breathing_block(self, llm_done):
        """Guide one random breathing exercise; no listen step, no mid-cut.

        The meditation / breathing exercise is a passive guided activity —
        the user does the breathing, they don't respond to it.  Cutting
        the script mid-breath was confusing UX ("it told me to breathe,
        then disappeared mid-sentence"), so the guidance now plays to
        completion even if the LLM answer arrives early.  The LLM
        response simply waits in the output queue until the exercise
        finishes.  `llm_done` is accepted for signature compatibility
        with the other blocks but is intentionally NOT polled.

        Because the caller (`_run_one_intermission_activity`) no longer
        attaches an output-ready watcher for BREATHING, `stop_playback_event`
        will not be set mid-script by the watcher.  We still clear it
        defensively at entry so a prior turn's leftover signal can't
        truncate the new meditation.
        """
        self.state = "intermission_exercise"
        # Clear any stale stop signal from a prior turn's watcher so the
        # full meditation script plays through.  BREATHING runs with no
        # watcher of its own — this is belt-and-braces for the case where
        # a prior MUSIC block's watcher raced past its stop_event.set().
        self.stop_playback_event.clear()
        exercise_text = self.intermission_ladder.next_breathing_exercise()
        logger.info("[INTERMISSION] Guiding a breathing exercise while LLM generates.")
        # Gentle lift so the bed rides *with* the exercise instead of
        # dropping to a whisper between MUSIC blocks.  Ducking kicks in
        # automatically while the guidance TTS plays (set_ai_speaking).
        self.music_service.fade_to(_MUSIC_BED_BREATHING, duration=1.2)
        # Meditation is spoken in the intermission voice so the guided
        # script sounds clearly different from CaiTI's therapy prompts.
        self.say_intermission(exercise_text)
        # The meditation has played in full at this point.  Return
        # `completed` so the delivery pipeline can transition back to
        # the LLM response cleanly via the usual outro + bridge.
        return {"outcome": "completed"}

    def _start_output_ready_watcher(self, llm_done: threading.Event) -> threading.Event:
        """Fire `llm_done` as soon as the handler queues a response, and
        cut any in-flight TTS playback short.

        Called from `_run_one_intermission_activity` so the proactive
        intermission exits as soon as the next LLM answer is ready —
        instead of holding the user through the remainder of a breathing
        or music block after the answer is already waiting in the
        output queue.

        We intentionally only *peek* (via `Queue.empty()`), we do NOT
        consume the item.  The downstream
        :meth:`_wait_for_output_with_intermission` is still responsible
        for pulling the response off the queue and speaking it, so this
        watcher must be side-effect-free on the queue itself.

        Returns a `stop_event` the caller can set to shut the watcher
        down cleanly (e.g. on end-of-session) before the queue delivers.
        """
        stop_event = threading.Event()

        def _poll():
            # A minimum-engagement floor protects very-fast GPU turns
            # from chopping the activity mid-first-phrase.  A cut that
            # lands 1 s into a meditation feels broken; a cut that lands
            # after one full breath cycle feels natural.
            started_at = time.monotonic()
            while not stop_event.is_set():
                if not self.output_queue.empty():
                    engagement = time.monotonic() - started_at
                    if engagement < _INTERMISSION_MIN_ENGAGEMENT_SEC:
                        # LLM already ready but we haven't given the
                        # user their full beat yet — wait out the floor
                        # before triggering, then trip both events.
                        stop_event.wait(
                            timeout=_INTERMISSION_MIN_ENGAGEMENT_SEC - engagement,
                        )
                        if stop_event.is_set():
                            return
                    logger.info(
                        "[INTERMISSION] Output ready — interrupting activity "
                        f"(engagement={time.monotonic() - started_at:.1f}s)."
                    )
                    # Cut any playing TTS first, then trip the hold-timer
                    # event so the block's wait() returns immediately.
                    self.stop_playback_event.set()
                    llm_done.set()
                    return
                stop_event.wait(timeout=_OUTPUT_READY_POLL_SEC)

        thread = threading.Thread(target=_poll, daemon=True,
                                  name="IntermissionOutputWatcher")
        thread.start()
        return stop_event

    def _run_one_intermission_activity(self):
        """Run a single intermission activity proactively while handler works.

        Called from the main loop AFTER queuing user input but BEFORE
        waiting on the LLM output.  This is the load-bearing piece of
        latency masking on Jetson:  the activity's TTS + listen blocks
        run inside pygame / PyAudio C extensions that release the
        Python GIL, so while the handler is mid-inference (holding the
        GIL in LiteRT-LM), the user is already hearing SCREENING /
        BREATHING / MUSIC as intended by the intermission protocol.

        SCREENING is picked first when PHQ-4 / GAD-2 questions remain,
        so the paper's clinical instrument leads every turn.  Declines
        and silence timeouts fall through to BREATHING (and then MUSIC)
        within the same call — the user never leaves this function
        without some activity having happened.  The LLM's response
        itself is delivered by the subsequent
        :meth:`_wait_for_output_with_intermission` call, which by then
        is essentially a no-wait delivery path.

        Framing.  Before the first audible beat of the activity plays
        we speak a short lead-in that explicitly tells the user "this
        is an intermission, separate from the main session", so it
        does not sound like another therapy question.  A matching
        outro is spoken on the way out of the intermission (in
        `_wait_for_output_with_intermission`), replacing the older
        bridge phrase when an intermission was actually active.

        Early-exit on LLM ready (MUSIC only).  MUSIC is interruptible
        via the output-ready watcher — it's just an ambient bed, the
        user is not being asked to engage with it.  BREATHING is NOT
        interruptible any more: the meditation script is a guided
        experience and cutting it mid-breath was unsettling.  The LLM
        response simply waits for the full meditation to finish.
        """
        logger.debug("[INTERMISSION] Proactive pre-wait activity starting.")
        self._sync_intermission_state_from_db()
        self._music_announced_for_turn = False

        # Pick SCREENING when a PHQ/GAD question is still pending,
        # else let the ladder pick between BREATHING and MUSIC.
        if self.intermission_ladder.screening_available():
            stage = IntermissionStage.SCREENING
        else:
            stage = self.intermission_ladder.next_activity()
        logger.info(f"[INTERMISSION] Pre-wait activity: {stage.value}")  # user-engaging activity while LLM thinks

        # Signpost the intermission so the user hears a clear separation
        # from the main therapy thread before the activity itself starts.
        # We do this here, centrally, so every stage benefits without the
        # individual block functions having to duplicate the wording.
        self._speak_intermission_lead_in(stage)

        # Only MUSIC is safe to interrupt proactively now. SCREENING
        # collects a PHQ/GAD answer from the user — that's clinical
        # data, and cutting the user mid-answer would lose it. BREATHING
        # is a guided meditation; cutting it mid-script was confusing
        # UX.  Both now play to completion; the LLM response simply
        # waits in the output queue until the activity finishes.
        llm_done = threading.Event()
        watcher_stop: threading.Event | None = None
        if stage == IntermissionStage.MUSIC:
            watcher_stop = self._start_output_ready_watcher(llm_done)

        try:
            listener_active = threading.Event()
            if stage == IntermissionStage.SCREENING:
                result = self._run_paired_screening_block(llm_done, listener_active)
                self.intermission_ladder.mark_activity(stage)
                if result.get("outcome") in ("declined", "completed", "end"):
                    # Whether answered / skipped / declined, the user
                    # has had their activity beat; return without
                    # chaining further.
                    return
                # Defensive: screening picked but no question available — fall through.
                stage = IntermissionStage.BREATHING_EXERCISE

            if stage == IntermissionStage.BREATHING_EXERCISE:
                # Breathing is passive — user doesn't respond mid-meditation.
                # The block speaks the full guided script and returns.
                # No interrupt watcher: the meditation plays through to
                # completion so the user experiences the whole exercise.
                self._run_breathing_block(llm_done)
                self.intermission_ladder.mark_activity(stage)
                return

            if stage == IntermissionStage.MUSIC:
                self._run_music_block(llm_done)
                self.intermission_ladder.mark_activity(stage)
        finally:
            # Always stop the watcher (if we started one) so it can't
            # leak across turns and accidentally trip stop_playback_event
            # on the next one.
            if watcher_stop is not None:
                watcher_stop.set()

    def _run_music_block(self, llm_done):
        """Raise music bed, wait for LLM output or a hold interval.

        Interruptible — if the caller's output-ready watcher fires
        `llm_done`, we return `llm_ready` immediately rather than
        sitting on the music until the hold-timer expires.  This removes
        the "~30 s of music after the LLM already answered" tail that
        used to pad every GPU turn.  A short minimum engagement is
        still respected so the music announcement has time to finish
        before we hand back.  BREATHING, by contrast, is NOT interrupted
        mid-script (see `_run_breathing_block`) so the user always
        experiences the full guided meditation.
        """
        self.state = "music_fallback"
        if not self._music_announced_for_turn:
            # Music announcement is an intermission utterance.
            self.say_intermission("I'm still thinking, enjoy the music while I continue.")
            self._music_announced_for_turn = True
        # Fade up so the music bed becomes the foreground while the LLM
        # is thinking — the user should feel like they're being *given*
        # music to sit with, not listening to incidental background.
        self.music_service.fade_to(_MUSIC_BED_INTERMISSION, duration=2.0)
        self.music_service.start(_get_music_path())
        # Jump to a fresh random segment so each MUSIC intermission sounds
        # different — avoids the "same opening bars every time" feel on
        # long ambient tracks.  No-op if the worker isn't running yet.
        self.music_service.jump_to_random_segment()
        # Give the music a 30 s window before we consider cycling back
        # (prevents frantic activity churn on long LLM stalls), but wake
        # early once the LLM response is ready so we don't eat 20+ s of
        # post-answer music.
        if llm_done.wait(timeout=_EXERCISE_HOLD_SEC):
            return {"outcome": "llm_ready"}
        return {"outcome": "completed"}

    def _wait_for_output_with_intermission(self, is_session_start: bool = False):
        """Wait for the next pipeline utterance, engaging the intermission pipeline.

        The three activities (screening, breathing, music) are picked
        randomly with last-activity deprioritisation — NOT strictly
        ordered.  If the user declines an activity, we fall through to
        another within the same turn.  Music always wins any tie and is
        the guaranteed fallback so the user never hears silence.

        Music fades softly up during the MUSIC block and fades softly
        back down before the LLM response is delivered, so the handoff
        never feels like a jump cut.

        ``is_session_start=True`` suppresses the "going back to what you
        shared" bridge phrase — the user hasn't shared anything yet on
        the very first turn, so that bridge would be incoherent.  The
        post-greeting gap still gets the full intermission treatment so
        PHQ-4 / breathing / music fills the silence until the first
        dimension question is ready.
        """
        llm_done = threading.Event()
        response_text = [None]
        intermission_was_active = False
        # Listener lock: held True while a screening question has been asked
        # and we are waiting for the user's answer.  The main loop must NOT
        # break out (even if llm_done fires) until the listener completes —
        # otherwise the user's clinical answer is lost mid-sentence.
        listener_active = threading.Event()

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

        # This pipeline is the "delivery + overflow" phase of the
        # intermission protocol.  The main loop has already run one
        # proactive activity via `_run_one_intermission_activity()`
        # before calling us, so the user has heard their SCREENING /
        # BREATHING / MUSIC beat for this turn.  From here we either:
        #   - Deliver the LLM response immediately (llm_done already
        #     set by the time the proactive activity returned), or
        #   - Cycle additional activities while the LLM is still
        #     thinking and then deliver when it's ready.
        # Either path goes through the same music-fade / bridge-phrase
        # handoff below.
        logger.debug(
            f"[INTERMISSION] Delivery pipeline entered "
            f"(session_start={is_session_start}, llm_done={llm_done.is_set()})."
        )
        intermission_was_active = True

        # Sync ladder state from DB once at entry — merges without
        # regressing already-answered/skipped questions in memory.
        self._sync_intermission_state_from_db()

        self._music_announced_for_turn = False
        last_heartbeat = time.monotonic()
        # Per-iteration fallback exclusions: resets each time we pick
        # fresh (i.e., after a completed activity).  Within one iteration,
        # a declined activity is added here so the re-pick chooses an
        # alternative for the *same* turn.
        turn_exclude: set[IntermissionStage] = set()

        # NOTE: the proactive `_run_one_intermission_activity()` call in
        # the main loop has already delivered one activity by the time
        # we get here — this pipeline exists to (a) absorb the rare case
        # where the handler's LLM chain is still running after the
        # proactive activity, and (b) deliver the output once ready.
        # We therefore do NOT force an extra activity on entry; if the
        # LLM is already done, fall straight through to delivery.
        force_first_stage: IntermissionStage | None = None

        while (
            not llm_done.is_set()
            or listener_active.is_set()
        ):
            if io_record.END_SESSION_EVENT.is_set():
                logger.info("[SESSION] End received during intermission — aborting.")
                break

            # If LLM is done but listener is still active, wait for the
            # user to finish answering before breaking out.
            if llm_done.is_set() and listener_active.is_set():
                logger.debug("[INTERMISSION] LLM ready, listener lock held. Waiting for user.")
                time.sleep(0.3)
                continue

            now = time.monotonic()
            if now - last_heartbeat >= _INTERMISSION_HEARTBEAT_SEC:
                last_heartbeat = now
                logger.debug("[Heartbeat] Still waiting for LLM output; cycling intermission activities.")

            if force_first_stage is not None:
                stage = force_first_stage
                force_first_stage = None
                logger.info(
                    f"[INTERMISSION] Session-start forced first activity: {stage.value}"
                )
            else:
                stage = self.intermission_ladder.next_activity(
                    exclude=frozenset(turn_exclude) if turn_exclude else None,
                )
                logger.info(f"[INTERMISSION] Selected activity: {stage.value}")

            if stage == IntermissionStage.SCREENING:
                if self.intermission_ladder.next_screening_question() is None:
                    # Defensive: ladder said SCREENING but no question is
                    # available — skip without marking (shouldn't happen
                    # since next_activity() gates on screening_available).
                    turn_exclude.add(IntermissionStage.SCREENING)
                    continue
                # Lead-in so this extra activity is still framed clearly
                # as a separate intermission, not another therapy question.
                self._speak_intermission_lead_in(stage)
                result = self._run_paired_screening_block(llm_done, listener_active)
                self.intermission_ladder.mark_activity(stage)
                if result["outcome"] == "end":
                    break
                if result["outcome"] == "declined":
                    # Paper-aligned fallback: if user declines screening,
                    # don't re-ask screening this turn — fall through to
                    # a random pick of {BREATHING, MUSIC}.
                    turn_exclude.add(IntermissionStage.SCREENING)
                    continue
                # Completed (answered, silence-skip, repeat, etc.) — reset
                # the exclude set so next cycle can pick any activity.
                turn_exclude.clear()
                continue

            if stage == IntermissionStage.BREATHING_EXERCISE:
                # Lead-in for the additional breathing beat during the
                # delivery-phase cycle.
                self._speak_intermission_lead_in(stage)
                result = self._run_breathing_block(llm_done)
                self.intermission_ladder.mark_activity(stage)
                # Breathing now always plays to completion (see
                # _run_breathing_block docstring) — no llm_ready mid-cut.
                turn_exclude.clear()
                continue

            if stage == IntermissionStage.MUSIC:
                # Lead-in for the music beat only on first entry this
                # turn (`_music_announced_for_turn` guards repeats).
                if not self._music_announced_for_turn:
                    self._speak_intermission_lead_in(stage)
                result = self._run_music_block(llm_done)
                self.intermission_ladder.mark_activity(stage)
                if result["outcome"] == "llm_ready":
                    break
                # Music held its full interval and LLM is still thinking —
                # cycle back and let another activity take the next beat.
                turn_exclude.clear()
                continue

        watcher.join(timeout=2.0)

        if io_record.END_SESSION_EVENT.is_set():
            # On forced end, let the existing end-session path manage audio.
            self.state = "main_process"
            return

        if response_text[0]:
            logger.info("[INTERMISSION] Complete — delivering agent response.")

            # Soft handoff: fade music down over ~1.2 s before we speak.
            # Brief sleep so the fade is audibly underway before TTS plays;
            # once TTS starts, audio.set_ai_speaking() will further duck
            # the bed to speaking_volume automatically.
            self.music_service.fade_to(_MUSIC_BED_HANDOFF, duration=1.2)
            time.sleep(0.9)

            # Intermission outro — a clearly distinct "that was the
            # intermission; now back to our session" signpost. Played
            # before the standard bridge so the user hears: (1) end of
            # intermission (intermission voice), (2) bridge back into
            # therapy (CaiTI voice), (3) the LLM reply itself (CaiTI
            # voice).  The voice flip between outro and bridge is itself
            # an audible "CaiTI is back" cue.
            if intermission_was_active and not is_session_start:
                outro = _random.choice(_INTERMISSION_OUTROS)
                logger.debug(f"[HANDOFF] Intermission outro: '{outro}'")
                self.say_intermission(outro)
                time.sleep(0.3)

            # The standard bridge phrases all reference something the user
            # "shared" — on the very first turn they haven't said anything
            # yet, so the bridge would be nonsensical.  Skip it for
            # session start; the LLM's opening dimension question is a
            # clean lead-in on its own.
            if intermission_was_active and not is_session_start:
                bridge = _random.choice(_BRIDGE_PHRASES)
                logger.debug(f"[HANDOFF] Bridge phrase: '{bridge}'")
                self.say(bridge)

            time.sleep(0.3)
            self.say(response_text[0])

            # Defensive drain: after the primary LLM utterance finishes
            # speaking, give the handler a short window to enqueue any
            # immediately-adjacent follow-up utterances (e.g. a CBT
            # Guide example + re-ask pair, or a handler-side closing
            # message on the last turn). This keeps paper §5.3's
            # "one user-facing beat" semantics intact even when the
            # handler emits multiple log_question calls per turn.
            # 400 ms window per drain, extended whenever something is
            # actually spoken, so a real trailing utterance from the
            # handler is never orphaned on the queue.
            drain_deadline = time.monotonic() + 0.4
            while time.monotonic() < drain_deadline:
                try:
                    more = self.output_queue.get(timeout=0.1)
                except queue.Empty:
                    break
                if more:
                    logger.debug("[INTERMISSION] Draining trailing agent utterance.")
                    self.say(more)
                    drain_deadline = time.monotonic() + 0.4

            # Restore the ambient base volume after the reply so the bed
            # sits at a calm level for the next listen.
            self.music_service.fade_to(_MUSIC_BED_AMBIENT, duration=1.5)
        else:
            # LLM timed out or produced no response — never leave silence.
            # Fall back to a breathing exercise so the user stays engaged.
            logger.warning("[INTERMISSION] LLM produced no response. Delivering therapeutic fallback.")
            self.music_service.fade_to(_MUSIC_BED_HANDOFF, duration=1.0)
            time.sleep(0.7)
            fallback = self.intermission_ladder.next_breathing_exercise()
            # Meditation fallback rides on the intermission voice; the
            # apology line (CaiTI explaining the hiccup) stays on the
            # primary voice so the user hears the therapist acknowledge
            # the issue directly.
            self.say_intermission(fallback)
            self.say("I'm having a little trouble with my thoughts right now. Let me try again shortly.")
            self.music_service.fade_to(_MUSIC_BED_AMBIENT, duration=1.5)

        self.state = "main_process"

    # ------------------------------------------------------------------ #
    # Main Loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self):
        """Main service loop."""
        logger.debug("Speech Interaction Service started.")

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
                    logger.info("[SESSION] GPIO opt-out button pressed — user requesting silence.")
                    self.stop_playback_event.set()
                    self.input_queue.put("[OPT_OUT]")
                    self.say(f"[PLAY_MUSIC] {_get_music_path()}")

                if self.paused:
                    time.sleep(0.5)
                    continue

                # 2. Handle Idle State (Waiting for Voice Wake-up)
                if not io_record.START_SESSION_EVENT.is_set():
                    # The last LLM hand-off suspended STT for VRAM headroom;
                    # once a session ends (either user-initiated or after CBT
                    # completes / fails), main.py clears START_SESSION_EVENT
                    # and control falls through here.  STT is still suspended
                    # at that point, so every wake-word transcribe silently
                    # errors with "Model not loaded, cannot transcribe."
                    # Observed in Hudson's session 16 (2026-04-26) — after
                    # CBT Stage 2 failed, the idle loop spent ~45 s firing
                    # STT errors because the model was never reloaded.
                    # Idempotent: resume_all() returns immediately if the
                    # model is already loaded.
                    try:
                        self.stt.resume_all()
                    except Exception as e:
                        logger.warning(f"[IDLE] STT resume for wake-detect failed: {e}")
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
                        logger.debug(f"Wake transcription: {text}")

                        if command == "START":
                            logger.info("[SESSION] Wake command heard — starting new session.")
                            self.initialize_session()
                        elif has_trigger and has_name:
                            logger.info("[SESSION] Wake phrase 'Hey CaiTI' detected — starting onboarding.")
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
                    if self._first_output_pending:
                        # Post-greeting gap: route the first LLM utterance
                        # (the opening dimension question) through the
                        # intermission pipeline so PHQ-4 / breathing / music
                        # fills the pre-first-question silence.  The
                        # watchdog is the one that actually speaks the
                        # LLM output on its way out, so we don't call
                        # self.say() here.
                        self._first_output_pending = False
                        logger.info("[SESSION] Routing first agent utterance through intermission pipeline.")
                        # Proactive activity during the post-greeting gap
                        # (handler is loading LiteRT + generating the first
                        # dimension question — both hold the GIL, so we
                        # must start an activity BEFORE waiting on the
                        # output queue for the same reason described in
                        # the per-turn path below).
                        self._run_one_intermission_activity()
                        self._wait_for_output_with_intermission(is_session_start=True)
                        self.post_turn_cleanup()
                    else:
                        text_to_speak = self.output_queue.get(timeout=0.2)
                        if not text_to_speak:
                            raise queue.Empty
                        self.say(text_to_speak)

                    while (self.running
                           and io_record.START_SESSION_EVENT.is_set()
                           and not io_record.END_SESSION_EVENT.is_set()):

                        # STT resume is now handled inside _run_parallel_turn
                        # (it needs to run AFTER any prior intermission
                        # completes) so we don't double-resume here.

                        if not self.is_hands_free:
                            self.manual_input_event.wait()
                            self.manual_input_event.clear()

                        # ── Parallel STT + intermission ─────────────────
                        # Previously: record → STT → suspend → queue →
                        # intermission, all serial. That serialisation left
                        # ~3-4 s of dead air between "mic closes" and
                        # "intermission starts speaking" because STT +
                        # suspend take time.
                        #
                        # Now: record → save WAV (mic idle) →
                        #      BRANCH:
                        #        Thread A (this thread):
                        #          run intermission immediately; meanwhile
                        #          keep peeking output_queue so we can exit
                        #          early when the LLM response is ready.
                        #        Thread B (parallel_stt worker):
                        #          transcribe saved WAV, run command gate,
                        #          put transcript on input_queue, suspend
                        #          Whisper. Sets abort_event on empty /
                        #          END / START.
                        #
                        # For very short mic windows (< 3 s) we fall back
                        # to the serial path so the fragment-merge retry
                        # in _listen_with_retry still works — short
                        # utterances are the ones most likely to be
                        # fragments, and we shouldn't commit to an
                        # intermission for a 1-word clarification.
                        outcome = self._run_parallel_turn()

                        if outcome == "session_end":
                            break
                        if outcome == "start_echo":
                            self.say("We're already in session, and I'm listening.")
                            continue
                        if outcome == "silence":
                            self._consecutive_silence_count += 1
                            if self._consecutive_silence_count >= 2:
                                self.say("I'm still here, just listening to the music with you. Take your time.")
                                self._consecutive_silence_count = 0
                            continue
                        if outcome == "delivered":
                            self._consecutive_silence_count = 0
                            self.post_turn_cleanup()
                            logger.debug(f"[VRAM HANDOFF] Post-intermission: {get_system_memory_snapshot()}")
                            continue
                        # Defensive: unknown outcome — loop back.
                        logger.warning(f"[TURN] Unknown parallel-turn outcome: {outcome!r}")
                        continue

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
