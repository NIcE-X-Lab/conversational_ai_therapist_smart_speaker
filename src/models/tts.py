"""AI model wrapper abstracting text-to-speech engine generation."""
import json
import os
import shutil
import subprocess
from src.utils.log_util import get_logger
from src.utils.config_loader import (
    TTS_MODEL_PATH,
    TTS_EXECUTABLE,
    TTS_LENGTH_SCALE,
    TTS_SENTENCE_SILENCE,
    TTS_INTERMISSION_MODEL_PATH,
)

logger = get_logger("TTSGenerator")

# espeak-ng is a lightweight fallback TTS available on most Linux/Jetson systems.
_ESPEAK_FALLBACK = "espeak-ng"

# M5: pre-generated cached WAV shipped with the repo. Played as a LAST-RESORT
# when both Piper and espeak-ng fail — ensures the device is never silent
# during a clinical trial, especially for safety messages. Expected at
# `assets/audio/tts_fallback.wav`. Content is a generic "I'm having trouble
# speaking, please check the device" prompt that alerts the participant.
_CACHED_FALLBACK_WAV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "audio", "tts_fallback.wav")
)


class TTSGenerator:
    """Dual-voice Piper wrapper.

    Primary voice (CaiTI) is always loaded and vetted at startup.  A
    secondary intermission voice is optional — when configured (and
    present on disk) it is used by callers passing `voice="intermission"`
    to `generate()`.  Piper is subprocess-invoked per utterance, so
    neither voice is resident in memory between calls; the only cost of
    carrying two voices is disk space (~63 MB per Piper medium model).

    Memory contract:
      - No long-lived model handles inside this Python process.
      - Each `generate()` spawns one `piper` subprocess with the
        `--model <path>` argument, which loads the ONNX weights into
        the subprocess's memory, synthesises the WAV, then exits.
        Weights are released back to the OS as soon as the subprocess
        terminates (C-extension memory, not Python-heap).
      - Two concurrent calls would briefly double the footprint, but
        our call sites are serial (one `say()` at a time via
        `self.player.play(..., stop_event=...)` in SpeechService), so
        only ONE Piper subprocess is ever alive at once.
    """

    def __init__(self):
        self.executable = TTS_EXECUTABLE
        # Primary voice — always required for the main therapy channel.
        self.model_path = TTS_MODEL_PATH
        self.model_config_path = f"{self.model_path}.json"
        # Secondary voice — optional.  Empty string means "no second
        # voice configured, fall back to primary for intermission TTS".
        self.intermission_model_path = TTS_INTERMISSION_MODEL_PATH or ""
        self.intermission_model_config_path = (
            f"{self.intermission_model_path}.json"
            if self.intermission_model_path else ""
        )
        self._piper_available = self._check_deps(self.model_path)
        self._intermission_available = bool(self.intermission_model_path) and \
            self._check_deps(self.intermission_model_path)
        self._espeak_available = shutil.which(_ESPEAK_FALLBACK) is not None
        if not self._piper_available and self._espeak_available:
            logger.warning(
                "Piper TTS unavailable. espeak-ng will be used as fallback."
            )
        elif not self._piper_available:
            logger.error(
                "TTS completely unavailable: neither Piper nor espeak-ng found."
            )
        if self.intermission_model_path and not self._intermission_available:
            logger.warning(
                f"[TTS] Intermission voice configured but unavailable at "
                f"{self.intermission_model_path}. Intermissions will use the "
                "primary voice as a fallback."
            )
        elif self._intermission_available:
            logger.info(
                f"[TTS] Intermission voice ready: {self.intermission_model_path}"
            )

    @property
    def is_ready(self) -> bool:
        return self._piper_available or self._espeak_available

    @property
    def intermission_voice_ready(self) -> bool:
        """True when the second voice is configured and passed dep checks."""
        return self._intermission_available

    def _check_deps(self, model_path: str) -> bool:
        """Validate a Piper voice (executable + model + .onnx.json config)."""
        if not self.executable:
            logger.error("TTS unavailable: Piper executable path is empty.")
            return False

        if not model_path:
            return False

        if not os.path.exists(model_path):
            logger.error(f"TTS unavailable: Piper model not found: {model_path}")
            return False

        if os.path.getsize(model_path) == 0:
            logger.error(f"TTS unavailable: Piper model is empty: {model_path}")
            return False

        model_config_path = f"{model_path}.json"
        if not os.path.exists(model_config_path):
            logger.error(f"TTS unavailable: Piper model config not found: {model_config_path}")
            return False

        if os.path.getsize(model_config_path) == 0:
            logger.error(f"TTS unavailable: Piper model config is empty: {model_config_path}")
            return False

        try:
            with open(model_config_path, "r", encoding="utf-8") as cfg_file:
                json.load(cfg_file)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.error(
                f"TTS unavailable: Piper model config is invalid JSON: "
                f"{model_config_path} ({e}). "
                "Session will continue with espeak-ng fallback if available."
            )
            return False
        except Exception as e:
            logger.error(f"TTS unavailable: unexpected error reading config: {e}")
            return False

        return True

    def _resolve_voice_path(self, voice: str) -> str:
        """Pick the on-disk model path for the requested voice role.

        voice="intermission" prefers the second voice when available,
        silently falling back to the primary voice if not configured.
        Any other value (including None / "primary") returns the primary.
        """
        if voice == "intermission" and self._intermission_available:
            return self.intermission_model_path
        return self.model_path

    def _generate_espeak(self, text: str, output_file: str):
        """Fallback TTS using espeak-ng when Piper is unavailable."""
        cmd = [
            _ESPEAK_FALLBACK,
            "-v", "en-us",
            "-s", "140",       # words per minute
            "-w", output_file,
            text,
        ]
        logger.info(f"[FALLBACK] espeak-ng TTS: '{text[:60]}...' -> {output_file}")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                logger.error(f"espeak-ng failed: {proc.stderr}")
                return None
            if not os.path.exists(output_file) or os.path.getsize(output_file) <= 44:
                logger.error("espeak-ng produced empty/invalid WAV.")
                return None
            return output_file
        except FileNotFoundError:
            logger.error("espeak-ng binary not found on PATH.")
            return None
        except Exception as e:
            logger.error(f"espeak-ng error: {e}")
            return None

    def generate(self, text, output_file, voice: str = "primary"):
        """
        Generate audio from text using Piper.  Falls back to espeak-ng if
        Piper is not operational (corrupt config, missing model, etc.).
        Args:
            text: Text to synthesize.
            output_file: Path to save the .wav file.
            voice: "primary" (default, CaiTI voice) or "intermission"
                   (second Piper voice, if configured).  Unknown values
                   are treated as "primary".  When "intermission" is
                   requested but the second voice is unavailable, we
                   silently fall back to the primary voice so the user
                   still hears the message.
        """
        if not text:
            return None

        # Re-check Piper deps on each call so a runtime config repair is picked up.
        if not self._piper_available:
            self._piper_available = self._check_deps(self.model_path)

        if not self._piper_available:
            logger.warning("[TTS Failure] Piper not available. Attempting espeak-ng fallback.")
            if self._espeak_available:
                espeak_result = self._generate_espeak(text, output_file)
                if espeak_result:
                    return espeak_result
            # M5: both engines dead — copy the cached fallback WAV so at
            # least SOMETHING is audible (clinical-trial safety rule).
            if os.path.isfile(_CACHED_FALLBACK_WAV):
                try:
                    shutil.copyfile(_CACHED_FALLBACK_WAV, output_file)
                    logger.error(
                        "[TTS Failure] Both engines unavailable. "
                        f"Using cached fallback WAV: {_CACHED_FALLBACK_WAV}"
                    )
                    return output_file
                except Exception as e:
                    logger.error(f"[TTS Failure] Cached fallback copy failed: {e}")
            logger.error("[TTS Failure] No TTS engine or cached fallback available.")
            return None

        # Resolve which voice to use for this call.  Intermission falls
        # back to primary silently if the second voice isn't configured.
        voice_model_path = self._resolve_voice_path(voice)

        cmd = [
            self.executable,
            "--model", voice_model_path,
            "--length_scale", str(TTS_LENGTH_SCALE),
            "--sentence_silence", str(TTS_SENTENCE_SILENCE),
            "--noise_scale", "0.4",
            "--noise_w_scale", "0.8",
            "--output_file", output_file
        ]

        logger.info(
            f"Generating TTS [{voice}={os.path.basename(voice_model_path)}] "
            f"for: '{text[:60]}{'...' if len(text) > 60 else ''}' -> {output_file}"
        )

        try:
            # Piper accepts text from stdin
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(input=text.encode('utf-8'))

            if process.returncode != 0:
                logger.error(f"Piper failed (rc={process.returncode}): {stderr.decode()}")
                # Fall back to espeak on Piper runtime failure
                if self._espeak_available:
                    logger.info("Attempting espeak-ng fallback after Piper failure.")
                    return self._generate_espeak(text, output_file)
                return None

            # Guard against silent/invalid synthesis output.
            if not os.path.exists(output_file):
                logger.error(f"TTS failed: output file not created: {output_file}")
                return None

            # A valid WAV should at least include a header and audio frames.
            if os.path.getsize(output_file) <= 44:
                logger.error(f"TTS failed: output WAV appears empty: {output_file}")
                return None

            logger.info("TTS generation successful.")
            return output_file

        except FileNotFoundError:
            logger.error(f"Piper executable not found at {self.executable}")
            self._piper_available = False
            if self._espeak_available:
                r = self._generate_espeak(text, output_file)
                if r:
                    return r
            return self._use_cached_fallback(output_file)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            if self._espeak_available:
                r = self._generate_espeak(text, output_file)
                if r:
                    return r
            return self._use_cached_fallback(output_file)

    def _use_cached_fallback(self, output_file: str):
        """M5: copy cached WAV so the device is never silent during a trial."""
        if os.path.isfile(_CACHED_FALLBACK_WAV):
            try:
                shutil.copyfile(_CACHED_FALLBACK_WAV, output_file)
                logger.error(
                    "[TTS Failure] Used cached fallback WAV "
                    f"({_CACHED_FALLBACK_WAV})"
                )
                return output_file
            except Exception as e:
                logger.error(f"[TTS Failure] Cached fallback copy failed: {e}")
        return None

if __name__ == "__main__":
    tts = TTSGenerator()
    # tts.generate("Hello, how are you today?", "test_tts.wav")
