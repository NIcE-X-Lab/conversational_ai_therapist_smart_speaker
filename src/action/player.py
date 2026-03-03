import subprocess
import time
from src.utils.log_util import get_logger

logger = get_logger("AudioPlayer")

class AudioPlayer:
    def __init__(self):
        self.process = None

    def _get_device_string(self):
        """
        Find the ALSA device string for the USB Audio device.
        We invoke 'aplay -l' and parse the output to find card number.
        Returns 'plughw:<card>,0' or 'default'.
        """
        try:
            result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                # Look for lines like "card 1: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]"
                if "USB" in line or "UAC" in line or "Jabra" in line:
                    # Extract card number
                    parts = line.split(':')
                    if len(parts) > 1 and "card" in parts[0]:
                        card_part = parts[0].strip() # "card 1"
                        card_num = card_part.split(' ')[1]
                        device_str = f"plughw:{card_num},0"
                        logger.info(f"Found USB Audio Device: {line.strip()} -> {device_str}")
                        return device_str
        except Exception as e:
            logger.warning(f"Error finding audio device: {e}")
        
        logger.warning("No USB Audio Device found. Using default.")
        return "default"

    def play(self, filename, stop_event=None):
        """
        Play a WAV file using aplay.
        If stop_event is provided, we monitor it to terminate the process.
        """
        device = self._get_device_string()
        cmd = ['aplay', '-D', device, filename]
        
        logger.info(f"Playing {filename} on {device}...")
        
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Monitor loop
            while self.process.poll() is None:
                if stop_event and stop_event.is_set():
                    logger.info("Playback interrupted by user.")
                    self.process.terminate()
                    break
                time.sleep(0.1)
                
            self.process.wait()
            logger.info("Playback finished.")
            
        except Exception as e:
            logger.error(f"Playback error: {e}")
        finally:
            self.process = None

    def stop_playback(self):
        """Signal the current playback to stop."""
        if self.process:
            self.process.terminate()

    def terminate(self):
        self.stop_playback()

if __name__ == "__main__":
    player = AudioPlayer()
    # player.play("test_tts.wav")
