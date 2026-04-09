"""AI model wrapper abstracting Speech-to-Text and lightweight emotion parsing."""

import json
import os
import sys

# ── Environment verification: reject heavy openai-whisper ──────────────────
# The `openai-whisper` package pulls in full PyTorch (~2GB) and is incompatible
# with the 8GB Jetson memory budget.  Detect it early and fail fast.
try:
    import whisper as _heavy_whisper  # noqa: F401
    # If this succeeds, the heavy package is installed
    _has_heavy_whisper = hasattr(_heavy_whisper, "load_model")
except ImportError:
    _has_heavy_whisper = False

if _has_heavy_whisper:
    print(
        "[ENVIRONMENT ERROR] Heavy 'openai-whisper' package detected. "
        "This pulls ~2GB of PyTorch and will OOM on Jetson. "
        "Uninstall it: pip uninstall openai-whisper whisper && pip install faster-whisper",
        file=sys.stderr,
    )
    sys.exit(1)

from faster_whisper import WhisperModel

from src.models.light_ser import LightweightRandomForestSER
from src.utils.config_loader import (
    SER_BACKEND,
    STT_BEST_OF,
    STT_BEAM_SIZE,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_MODEL_PATH,
    STT_WITHOUT_TIMESTAMPS,
)
from src.utils.inference_guard import clear_inference_cache, heavy_stage
from src.utils.log_util import get_logger
from src.utils.resource_audit import get_resource_audit

logger = get_logger("STTGenerator")
RESOURCE_AUDIT = get_resource_audit()


def _rss_mb() -> float:
    """Return current process RSS in MB (best-effort)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


class STTGenerator:
    def __init__(self):
        # STT_MODEL_PATH can be "base.en", "small.en", etc. or a local path
        logger.info(
            f"Loading Faster-Whisper model: {STT_MODEL_PATH} on {STT_DEVICE} "
            f"(compute_type={STT_COMPUTE_TYPE}, beam_size={STT_BEAM_SIZE}, best_of={STT_BEST_OF})"
        )
        rss_before = _rss_mb()
        try:
            with RESOURCE_AUDIT.track_module_init("STT/FasterWhisperModelInit"):
                self.model = WhisperModel(
                    STT_MODEL_PATH,
                    device=STT_DEVICE,
                    compute_type=STT_COMPUTE_TYPE,
                    num_workers=1,
                )
            rss_after = _rss_mb()
            logger.info(
                f"Faster-Whisper model loaded successfully. "
                f"RSS delta: +{rss_after - rss_before:.1f}MB (now {rss_after:.1f}MB)"
            )
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}")
            self.model = None

        # Lightweight SER model (MFCC-based random-forest-style voting).
        self.emotion_model = None
        try:
            if SER_BACKEND in {"light_mfcc_rf", "mfcc_rf", "lightweight"}:
                rss_pre_ser = _rss_mb()
                with RESOURCE_AUDIT.track_module_init("SER/LightMFCCRFInit"):
                    self.emotion_model = LightweightRandomForestSER()
                rss_post_ser = _rss_mb()
                logger.info(
                    f"Lightweight SER model loaded successfully. "
                    f"RSS delta: +{rss_post_ser - rss_pre_ser:.1f}MB (now {rss_post_ser:.1f}MB)"
                )
            else:
                logger.info(f"SER backend '{SER_BACKEND}' disabled; using neutral fallback.")
        except Exception as e:
            logger.info(f"Lightweight SER unavailable; using neutral fallback. Reason: {e}")

    def suspend(self):
        """Release the Whisper model from memory to free VRAM for the LLM.
        Call resume() before next transcription to re-load."""
        if self.model is not None:
            rss_pre = _rss_mb()
            del self.model
            self.model = None
            clear_inference_cache("STT model suspended for LLM VRAM headroom")
            rss_post = _rss_mb()
            logger.info(f"STT model suspended. RSS freed: {rss_pre - rss_post:.1f}MB")

    def suspend_ser(self):
        """Release the SER model from memory alongside Whisper.
        The lightweight MFCC-RF model is small but every MB counts on 8GB Jetson."""
        if self.emotion_model is not None:
            rss_pre = _rss_mb()
            del self.emotion_model
            self.emotion_model = None
            clear_inference_cache("SER model suspended for LLM VRAM headroom")
            rss_post = _rss_mb()
            logger.info(f"SER model suspended. RSS freed: {rss_pre - rss_post:.1f}MB")

    def suspend_all(self):
        """Full sequential handoff: unload both Whisper and SER."""
        self.suspend()
        self.suspend_ser()

    def resume(self):
        """Re-load the Whisper model after an LLM call."""
        if self.model is None:
            rss_pre = _rss_mb()
            try:
                self.model = WhisperModel(
                    STT_MODEL_PATH,
                    device=STT_DEVICE,
                    compute_type=STT_COMPUTE_TYPE,
                    num_workers=1,
                )
                rss_post = _rss_mb()
                logger.info(f"STT model resumed. RSS delta: +{rss_post - rss_pre:.1f}MB")
            except Exception as e:
                logger.error(f"Failed to resume STT model: {e}")

    def resume_ser(self):
        """Re-load the SER model after an LLM call."""
        if self.emotion_model is None:
            try:
                if SER_BACKEND in {"light_mfcc_rf", "mfcc_rf", "lightweight"}:
                    rss_pre = _rss_mb()
                    self.emotion_model = LightweightRandomForestSER()
                    rss_post = _rss_mb()
                    logger.info(f"SER model resumed. RSS delta: +{rss_post - rss_pre:.1f}MB")
            except Exception as e:
                logger.info(f"SER resume failed; using neutral fallback. Reason: {e}")

    def resume_all(self):
        """Full sequential handoff: reload both Whisper and SER."""
        self.resume()
        self.resume_ser()

    def _classify_emotion(self, audio_path: str) -> str:
        if not self.emotion_model:
            return "neu"
        try:
            with RESOURCE_AUDIT.track_peak("SER/MFCC-RF"):
                with heavy_stage("SER/MFCC-RF"):
                    return self.emotion_model.classify_file(audio_path)
        except Exception as e:
            logger.error(f"SER Error: {e}")
            return "neu"

    def transcribe(self, audio_path):
        """
        Transcribe audio file to text.
        Args:
            audio_path: Path to the .wav file.
        Returns:
            text: Transcribed text string.
        """
        if not self.model:
            logger.error("Model not loaded, cannot transcribe.")
            return ""

        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return ""

        try:
            with RESOURCE_AUDIT.track_peak("STT/FasterWhisper"):
                with heavy_stage("STT/FasterWhisper"):
                    segments, info = self.model.transcribe(
                        audio_path,
                        beam_size=STT_BEAM_SIZE,
                        best_of=STT_BEST_OF,
                        condition_on_previous_text=False,
                        without_timestamps=STT_WITHOUT_TIMESTAMPS,
                    )
                    text = " ".join([segment.text for segment in segments]).strip()

            clear_inference_cache("After STT phase")
            emotion = self._classify_emotion(audio_path)
            clear_inference_cache("After SER phase")

            logger.info(f"Transcription: {text} | Emotion: {emotion}")
            return json.dumps({"transcript": text, "detected_emotion": emotion})
        except Exception as e:
            logger.error(f"Transcription/SER error: {e}")
            return json.dumps({"transcript": "", "detected_emotion": "neu"})

if __name__ == "__main__":
    stt = STTGenerator()
    # Test transcription
    # print(stt.transcribe("test_recording.wav"))
