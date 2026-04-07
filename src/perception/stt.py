from faster_whisper import WhisperModel
import os
from src.utils.log_util import get_logger
from src.utils.config_loader import STT_MODEL_PATH, STT_DEVICE
import json
import concurrent.futures

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
            
        # Add SpeechBrain SER Model
        self.emotion_model = None
        try:
            logger.info("Loading SpeechBrain Emotion Model: speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
            from speechbrain.inference.interfaces import foreign_class
            self.emotion_model = foreign_class(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
                pymodule_file="custom_interface.py", 
                classname="CustomEncoderWav2vec2Classifier"
            )
            logger.info("Emotion model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load SpeechBrain model: {e}")

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
            def run_whisper():
                segments, info = self.model.transcribe(audio_path, beam_size=5)
                return " ".join([segment.text for segment in segments]).strip()
                
            def run_ser():
                if self.emotion_model:
                    try:
                        out_prob, score, index, text_lab = self.emotion_model.classify_file(audio_path)
                        return text_lab[0]
                    except Exception as e:
                        logger.error(f"SER Error: {e}")
                return "Neutral"

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_stt = executor.submit(run_whisper)
                future_ser = executor.submit(run_ser)
                
                text = future_stt.result()
                emotion = future_ser.result()

            logger.info(f"Transcription: {text} | Emotion: {emotion}")
            return json.dumps({"transcript": text, "detected_emotion": emotion})
        except Exception as e:
            logger.error(f"Transcription/SER error: {e}")
            return json.dumps({"transcript": "", "detected_emotion": "Neutral"})

if __name__ == "__main__":
    stt = STTGenerator()
    # Test transcription
    # print(stt.transcribe("test_recording.wav"))
