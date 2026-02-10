import pyaudio
import wave
import time
from src.utils.log_util import get_logger

logger = get_logger("AudioPlayer")

class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self._stop_event = None

    def _get_device_index(self):
        """Find the index of the USB Audio device."""
        count = self.p.get_device_count()
        for i in range(count):
            try:
                info = self.p.get_device_info_by_index(i)
                name = info.get("name", "")
                # Look for USB or UAC devices (typical for USB audio)
                if "USB" in name or "UAC" in name or "Jabra" in name:
                    logger.info(f"Found USB Audio Device: {name} at index {i}")
                    return i
            except Exception:
                continue
        logger.warning("No USB Audio Device found. Using default.")
        return None

    def play(self, filename, stop_event=None):
        """
        Play a WAV file.
        If stop_event is provided, playback checks it to interrupt.
        """
        try:
            wf = wave.open(filename, 'rb')
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            return

        self._stop_event = stop_event
        device_index = self._get_device_index()
        
        stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                             channels=wf.getnchannels(),
                             rate=wf.getframerate(),
                             output=True,
                             output_device_index=device_index)

        logger.info(f"Playing {filename}...")
        
        chunk = 1024
        data = wf.readframes(chunk)
        
        while len(data) > 0:
            if self._stop_event and self._stop_event.is_set():
                logger.info("Playback interrupted by user.")
                break
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        wf.close()
        logger.info("Playback finished.")
        self._stop_event = None

    def stop_playback(self):
        """Signal the current playback to stop."""
        if self._stop_event:
            self._stop_event.set()

    def terminate(self):
        self.p.terminate()

if __name__ == "__main__":
    player = AudioPlayer()
    # player.play("test_tts.wav")
