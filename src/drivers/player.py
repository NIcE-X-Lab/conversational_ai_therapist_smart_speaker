"""Unified Pygame-based audio player for TTS voice playback.

Uses pygame.mixer.Sound instead of aplay subprocess to avoid ALSA hardware
lock conflicts.  Both the BackgroundMusicThread (pygame.mixer.music) and this
player (pygame.mixer.Sound) share the same mixer, allowing voice and music to
overlap without "Device Busy" errors.
"""

import os
import time
import threading
from src.utils.log_util import get_logger
from src.drivers.audio import set_ai_speaking

logger = get_logger("AudioPlayer")

_pygame = None
_mixer_ready = False
_mixer_lock = threading.Lock()


def _ensure_mixer() -> bool:
    """Initialise the pygame mixer once (thread-safe).

    The BackgroundMusicThread may have already initialised the mixer —
    this is safe to call again (pygame.mixer.init is a no-op if already
    initialised).
    """
    global _pygame, _mixer_ready
    if _mixer_ready:
        return True
    with _mixer_lock:
        if _mixer_ready:
            return True
        try:
            import pygame
            _pygame = pygame
            if not pygame.mixer.get_init():
                # 22050 Hz stereo, 1024-sample buffer: matches BackgroundMusicThread
                # config so whichever initialises first sets the same parameters.
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
            _mixer_ready = True
            logger.info(f"Pygame mixer ready: {pygame.mixer.get_init()}")
            return True
        except Exception as e:
            logger.error(f"Pygame mixer init failed: {e}")
            return False


class AudioPlayer:
    """Play WAV files through the unified pygame mixer.

    Replaces the old aplay-subprocess approach to eliminate ALSA device
    contention between music and voice playback.
    """

    def __init__(self, playback_signal_handler=None):
        self._channel = None
        self.playback_signal_handler = playback_signal_handler

    def _emit_playback_signal(self, signal: str):
        if not self.playback_signal_handler:
            return
        try:
            self.playback_signal_handler(signal)
        except Exception as e:
            logger.debug(f"Playback signal handler failed ({signal}): {e}")

    def play(self, filename, stop_event=None, duck=True):
        """Play a WAV file through the pygame mixer.

        Blocks until playback finishes or *stop_event* is set.
        """
        if not os.path.isfile(filename):
            logger.warning(f"Audio file not found: {filename}")
            return

        if not _ensure_mixer():
            logger.error("Cannot play — pygame mixer not available.")
            return

        try:
            sound = _pygame.mixer.Sound(filename)
        except Exception as e:
            logger.error(f"Failed to load sound {filename}: {e}")
            return

        logger.info(f"Playing {filename} via pygame mixer ({sound.get_length():.1f}s)")

        try:
            if duck:
                set_ai_speaking(True)
                self._emit_playback_signal("DUCK")

            self._channel = sound.play()

            # Poll until playback ends or stop_event fires
            while self._channel and self._channel.get_busy():
                if stop_event and stop_event.is_set():
                    logger.info("Playback interrupted by stop_event.")
                    self._channel.stop()
                    break
                time.sleep(0.05)

            logger.info("Playback finished.")

        except Exception as e:
            logger.error(f"Playback error: {e}")
        finally:
            self._channel = None
            if duck:
                self._emit_playback_signal("RESTORE")
                set_ai_speaking(False)

    def stop_playback(self):
        """Signal the current playback to stop."""
        if self._channel:
            try:
                self._channel.stop()
            except Exception:
                pass

    def terminate(self):
        self.stop_playback()


if __name__ == "__main__":
    player = AudioPlayer()
    # player.play("test_tts.wav")
