"""AI model wrapper abstracting text-to-speech engine generation."""
import json
import os
import shutil
import subprocess
from src.utils.log_util import get_logger
from src.utils.config_loader import TTS_MODEL_PATH, TTS_EXECUTABLE, TTS_LENGTH_SCALE, TTS_SENTENCE_SILENCE

logger = get_logger("TTSGenerator")

# espeak-ng is a lightweight fallback TTS available on most Linux/Jetson systems.
_ESPEAK_FALLBACK = "espeak-ng"


class TTSGenerator:
    def __init__(self):
        self.executable = TTS_EXECUTABLE
        self.model_path = TTS_MODEL_PATH
        self.model_config_path = f"{self.model_path}.json"
        self._piper_available = self._check_deps()
        self._espeak_available = shutil.which(_ESPEAK_FALLBACK) is not None
        if not self._piper_available and self._espeak_available:
            logger.warning(
                "Piper TTS unavailable. espeak-ng will be used as fallback."
            )
        elif not self._piper_available:
            logger.error(
                "TTS completely unavailable: neither Piper nor espeak-ng found."
            )

    @property
    def is_ready(self) -> bool:
        return self._piper_available or self._espeak_available

    def _check_deps(self):
        """Check if Piper executable, model, and model config are valid."""
        if not self.executable:
            logger.error("TTS unavailable: Piper executable path is empty.")
            return False

        if not os.path.exists(self.model_path):
            logger.error(f"TTS unavailable: Piper model not found: {self.model_path}")
            return False

        if os.path.getsize(self.model_path) == 0:
            logger.error(f"TTS unavailable: Piper model is empty: {self.model_path}")
            return False

        if not os.path.exists(self.model_config_path):
            logger.error(f"TTS unavailable: Piper model config not found: {self.model_config_path}")
            return False

        if os.path.getsize(self.model_config_path) == 0:
            logger.error(f"TTS unavailable: Piper model config is empty: {self.model_config_path}")
            return False

        try:
            with open(self.model_config_path, "r", encoding="utf-8") as cfg_file:
                json.load(cfg_file)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.error(
                f"TTS unavailable: Piper model config is invalid JSON: "
                f"{self.model_config_path} ({e}). "
                "Session will continue with espeak-ng fallback if available."
            )
            return False
        except Exception as e:
            logger.error(f"TTS unavailable: unexpected error reading config: {e}")
            return False

        return True

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

    def generate(self, text, output_file):
        """
        Generate audio from text using Piper.  Falls back to espeak-ng if
        Piper is not operational (corrupt config, missing model, etc.).
        Args:
            text: Text to synthesize.
            output_file: Path to save the .wav file.
        """
        if not text:
            return None

        # Re-check Piper deps on each call so a runtime config repair is picked up.
        if not self._piper_available:
            self._piper_available = self._check_deps()

        if not self._piper_available:
            logger.warning("[TTS Failure] Piper not available. Attempting espeak-ng fallback.")
            if self._espeak_available:
                return self._generate_espeak(text, output_file)
            logger.error("[TTS Failure] No TTS engine available. Skipping audio generation.")
            return None

        cmd = [
            self.executable,
            "--model", self.model_path,
            "--length_scale", str(TTS_LENGTH_SCALE),
            "--sentence_silence", str(TTS_SENTENCE_SILENCE),
            "--noise_scale", "0.4",
            "--noise_w_scale", "0.8",
            "--output_file", output_file
        ]

        logger.info(f"Generating TTS for: '{text}' -> {output_file}")

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
                return self._generate_espeak(text, output_file)
            return None
        except Exception as e:
            logger.error(f"TTS error: {e}")
            if self._espeak_available:
                return self._generate_espeak(text, output_file)
            return None

if __name__ == "__main__":
    tts = TTSGenerator()
    # tts.generate("Hello, how are you today?", "test_tts.wav")
