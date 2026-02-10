import pyaudio
import wave
import time
from src.utils.log_util import get_logger

logger = get_logger("AudioPlayer")

class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()

    def play(self, filename):
        """Play a WAV file."""
        try:
            wf = wave.open(filename, 'rb')
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            return

        stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                             channels=wf.getnchannels(),
                             rate=wf.getframerate(),
                             output=True)

        logger.info(f"Playing {filename}...")
        
        chunk = 1024
        data = wf.readframes(chunk)
        
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        wf.close()
        logger.info("Playback finished.")

    def terminate(self):
        self.p.terminate()

if __name__ == "__main__":
    player = AudioPlayer()
    # player.play("test_tts.wav")
