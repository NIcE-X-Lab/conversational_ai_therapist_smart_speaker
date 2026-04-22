"""AI model wrapper for Speech-to-Text (SER removed per research alignment)."""

import gc
import json
import os
import sys

# ── Environment verification: reject heavy openai-whisper ──────────────────
# The `openai-whisper` package pulls in full PyTorch (~2GB) and is incompatible
# with the 8GB Jetson memory budget.  Detect it early and fail fast.
try:
    import whisper as _heavy_whisper  # noqa: F401
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

from src.utils.config_loader import (
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

    def suspend(self):
        """Release the Whisper model from memory to free VRAM for the LLM.
        Call resume() before next transcription to re-load."""
        if self.model is not None:
            rss_pre = _rss_mb()
            del self.model
            self.model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("torch.cuda.empty_cache() called after STT suspend.")
            except ImportError:
                pass
            clear_inference_cache("STT model suspended for LLM VRAM headroom")
            rss_post = _rss_mb()
            logger.info(f"STT model suspended. RSS freed: {rss_pre - rss_post:.1f}MB")

    def suspend_all(self):
        """Compat alias — SER removed, so this just suspends STT."""
        self.suspend()

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

    def resume_all(self):
        """Compat alias — SER removed, so this just resumes STT."""
        self.resume()

    def transcribe(self, audio_path):
        """
        Transcribe audio file to text.
        Args:
            audio_path: Path to the .wav file.
        Returns:
            JSON string: {"transcript": str, "detected_emotion": "neu"}.
            The emotion field is retained as "neu" for backwards compatibility
            with downstream consumers but is no longer classified.
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

            logger.info(f"Transcription: {text}")
            return json.dumps({"transcript": text, "detected_emotion": "neu"})
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return json.dumps({"transcript": "", "detected_emotion": "neu"})

if __name__ == "__main__":
    stt = STTGenerator()
