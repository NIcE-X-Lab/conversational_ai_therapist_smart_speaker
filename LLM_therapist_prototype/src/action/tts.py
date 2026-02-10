import subprocess
import os
import wave
from src.utils.log_util import get_logger
from src.utils.config_loader import TTS_MODEL_PATH, TTS_EXECUTABLE

logger = get_logger("TTSGenerator")

class TTSGenerator:
    def __init__(self):
        self.executable = TTS_EXECUTABLE
        self.model_path = TTS_MODEL_PATH
        self._check_deps()

    def _check_deps(self):
        """Check if piper executable and model exist."""
        # Check executable
        # This is a basic check, might need full path verification
        pass

    def generate(self, text, output_file):
        """
        Generate audio from text using Piper.
        Args:
            text: Text to synthesize.
            output_file: Path to save the .wav file.
        """
        if not text:
            return None

        cmd = [
            self.executable,
            "--model", self.model_path,
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
                logger.error(f"Piper validation failed: {stderr.decode()}")
                return None
            
            logger.info("TTS generation successful.")
            return output_file
            
        except FileNotFoundError:
            logger.error(f"Piper executable not found at {self.executable}")
            return None
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

if __name__ == "__main__":
    tts = TTSGenerator()
    # tts.generate("Hello, how are you today?", "test_tts.wav")
