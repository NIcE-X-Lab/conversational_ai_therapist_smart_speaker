from faster_whisper import WhisperModel
import os
from src.utils.log_util import get_logger
from src.utils.config_loader import STT_MODEL_PATH, STT_DEVICE

logger = get_logger("STTGenerator")

class STTGenerator:
    def __init__(self):
        # STT_MODEL_PATH can be "base.en", "small.en", etc. or a local path
        logger.info(f"Loading Whisper model: {STT_MODEL_PATH} on {STT_DEVICE}")
        try:
            self.model = WhisperModel(STT_MODEL_PATH, device=STT_DEVICE, compute_type="int8")
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None

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
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            text = " ".join([segment.text for segment in segments]).strip()
            logger.info(f"Transcription: {text}")
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

if __name__ == "__main__":
    stt = STTGenerator()
    # Test transcription
    # print(stt.transcribe("test_recording.wav"))
