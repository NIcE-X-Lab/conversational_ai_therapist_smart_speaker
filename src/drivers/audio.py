"""Low-level driver handling microphone streams and Voice Activity Detection."""
import ctypes
import pyaudio
import webrtcvad
import collections
import sys
import signal
import time
import wave
import os
import threading
import numpy as np
from src.utils.log_util import get_logger
from src.utils.config_loader import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_CHUNK_SIZE, AUDIO_VAD_AGGRESSIVENESS


logger = get_logger("AudioRecorder")

_ALSA_ERR_HANDLER_REF = None
_AI_IS_SPEAKING = threading.Event()


def set_ai_speaking(is_speaking: bool):
    """Global speaking flag used by background music ducking."""
    if is_speaking:
        _AI_IS_SPEAKING.set()
    else:
        _AI_IS_SPEAKING.clear()


def is_ai_speaking() -> bool:
    return _AI_IS_SPEAKING.is_set()


class BackgroundMusicThread:
    """Always-on non-blocking ambient music loop with speaking ducking."""

    SIGNAL_DUCK = "DUCK"
    SIGNAL_RESTORE = "RESTORE"

    def __init__(
        self,
        track_path: str = "assets/audio/ambient_music.mp3",
        base_volume: float = 0.10,
        speaking_volume: float = 0.02,
    ):
        self.track_path = track_path
        self.base_volume = float(base_volume)
        self.speaking_volume = float(speaking_volume)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._worker = None
        self._running = False
        self._duck_override = False
        self._pygame = None
        self._mixer_ready = False

    def _resolve_track(self) -> str | None:
        candidates = [
            self.track_path,
            "assets/audio/ambient_therapy.mp3",
            "assets/audio/waiting_music.wav",
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return None

    def _ensure_mixer(self) -> bool:
        if self._mixer_ready and self._pygame is not None:
            return True
        try:
            import pygame
            self._pygame = pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._mixer_ready = True
            return True
        except Exception as e:
            logger.warning(f"Background music disabled: mixer init failed: {e}")
            return False

    def _target_volume(self) -> float:
        if self._duck_override or is_ai_speaking():
            return self.speaking_volume
        return self.base_volume

    def _run(self):
        if not self._ensure_mixer():
            with self._lock:
                self._running = False
            return

        track = self._resolve_track()
        if not track:
            logger.warning("Background music file not found; skipping ambient bed.")
            with self._lock:
                self._running = False
            return

        try:
            self._pygame.mixer.music.load(track)
            self._pygame.mixer.music.set_volume(self._target_volume())
            self._pygame.mixer.music.play(loops=-1)
            logger.info(f"Background music started: {track}")
        except Exception as e:
            logger.warning(f"Failed starting background music: {e}")
            with self._lock:
                self._running = False
            return

        while not self._stop_event.wait(0.2):
            try:
                if not self._pygame.mixer.music.get_busy():
                    self._pygame.mixer.music.play(loops=-1)
                self._pygame.mixer.music.set_volume(self._target_volume())
            except Exception as e:
                logger.warning(f"Background music loop stopped unexpectedly: {e}")
                break

        try:
            self._pygame.mixer.music.stop()
        except Exception:
            pass

        with self._lock:
            self._running = False
        logger.info("Background music stopped.")

    def start(self, track_path: str | None = None):
        with self._lock:
            if track_path:
                self.track_path = track_path
            if self._running:
                return
            self._stop_event.clear()
            self._running = True
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    def stop(self):
        with self._lock:
            worker = self._worker
            self._stop_event.set()
        if worker:
            worker.join(timeout=1.0)

    def handle_signal(self, signal: str):
        if signal == self.SIGNAL_DUCK:
            self._duck_override = True
        elif signal == self.SIGNAL_RESTORE:
            self._duck_override = False

    def set_base_volume(self, volume: float):
        with self._lock:
            self.base_volume = max(0.0, min(1.0, float(volume)))


def _suppress_alsa_warnings():
    """Mute low-level ALSA library stderr spam from device probing."""
    global _ALSA_ERR_HANDLER_REF
    if _ALSA_ERR_HANDLER_REF is not None:
        return
    try:
        asound = ctypes.cdll.LoadLibrary("libasound.so")
        err_cb_type = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
        )

        def _err_handler(filename, line, function, err, fmt):
            return

        _ALSA_ERR_HANDLER_REF = err_cb_type(_err_handler)
        asound.snd_lib_error_set_handler(_ALSA_ERR_HANDLER_REF)
    except Exception:
        # If unavailable, proceed normally.
        pass

class AudioRecorder:
    def __init__(self):
        _suppress_alsa_warnings()
        self.format = pyaudio.paInt16
        self.channels = AUDIO_CHANNELS
        self.rate = AUDIO_SAMPLE_RATE
        self.chunk = AUDIO_CHUNK_SIZE
        self._audio = pyaudio.PyAudio()
        self.stream = None
        self.vad = webrtcvad.Vad(AUDIO_VAD_AGGRESSIVENESS)
        self._vad_active_callback = None

    def set_vad_active_callback(self, callback):
        """Set callback(state: bool) to mirror active VAD capture state."""
        self._vad_active_callback = callback

    def _set_vad_state(self, state: bool):
        try:
            if self._vad_active_callback:
                self._vad_active_callback(state)
        except Exception as e:
            logger.debug(f"VAD callback failed: {e}")

    def start_stream(self):
        """Start the audio input stream."""
        if self.stream is None:
            self.stream = self._audio.open(format=self.format,
                                           channels=self.channels,
                                           rate=self.rate,
                                           input=True,
                                           frames_per_buffer=self.chunk)
            
            # Clear OS buffer (Echo Cancellation/Mic-Mute during recent TTS)
            # Read and discard ~0.5s of audio to flush any lingering data
            flush_chunks = int(0.5 * self.rate / self.chunk)
            for _ in range(flush_chunks):
                try:
                    self.stream.read(self.chunk, exception_on_overflow=False)
                except IOError:
                    pass
            
            logger.info("Audio stream started and OS buffer flushed.")

    def stop_stream(self):
        """Stop and close the audio input stream."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            logger.info("Audio stream stopped.")

    def terminate(self):
        """Terminate PyAudio."""
        self.stop_stream()
        self._audio.terminate()

    def is_speech(self, frame_bytes):
        """Check if a frame contains speech using WebRTC VAD."""
        try:
            return self.vad.is_speech(frame_bytes, self.rate)
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False

    def record_until_silence(self, silence_duration=2.0, max_duration=15.0,
                             trailing_pad=0.4, min_speech_sec=0.3):
        """
        Record audio dynamically: wait until speech starts, then record until
        sustained silence or max limit reached.

        Args:
            silence_duration: Seconds of continuous silence before stopping.
            max_duration:     Hard cap on total recording time.
            trailing_pad:     Extra seconds to capture after initial silence
                              trigger, catching trailing words/breaths.
            min_speech_sec:   Minimum speech duration to accept (filters noise
                              bursts and VAD false positives).

        Returns:
            frames: List of audio frames (bytes).  Empty if no speech detected.
        """
        self.start_stream()

        frames = []
        silence_chunks = int(silence_duration * self.rate / self.chunk)
        trailing_chunks = int(trailing_pad * self.rate / self.chunk)
        min_speech_chunks = int(min_speech_sec * self.rate / self.chunk)
        max_chunks = int(max_duration * self.rate / self.chunk)
        wait_chunks = int(5.0 * self.rate / self.chunk)  # Max 5s wait for speech start

        silent_count = 0
        speech_chunk_count = 0
        has_speech = False

        logger.info("Listening (waiting for speech)...")
        self._set_vad_state(True)

        # Phase 1: Wait for voice activity
        for _ in range(wait_chunks):
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                if self.is_speech(data):
                    has_speech = True
                    frames.append(data)
                    speech_chunk_count += 1
                    logger.info("Speech detected, recording started.")
                    break
            except IOError as e:
                logger.warning(f"Audio read error during wait: {e}")
                time.sleep(0.01)

        if not has_speech:
            logger.info("No speech detected, stopping recording window.")
            self.stop_stream()
            self._set_vad_state(False)
            return []

        # Phase 2: Record until sustained silence
        for _ in range(max_chunks):
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)

                if self.is_speech(data):
                    silent_count = 0
                    speech_chunk_count += 1
                else:
                    silent_count += 1

                if silent_count > silence_chunks:
                    break

            except IOError as e:
                logger.warning(f"Audio read error: {e}")

        # Phase 3: Trailing pad — capture a few extra chunks to catch
        # trailing words, breaths, or sentence-ending sounds.
        for _ in range(trailing_chunks):
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
                if self.is_speech(data):
                    # User resumed speaking — reset silence and keep recording
                    silent_count = 0
                    speech_chunk_count += 1
                    logger.info("Trailing speech detected, extending recording.")
                    # Continue main recording for more data
                    for _ in range(max_chunks - len(frames)):
                        try:
                            data = self.stream.read(self.chunk, exception_on_overflow=False)
                            frames.append(data)
                            if self.is_speech(data):
                                silent_count = 0
                                speech_chunk_count += 1
                            else:
                                silent_count += 1
                            if silent_count > silence_chunks:
                                break
                        except IOError:
                            break
                    break
            except IOError:
                break

        self.stop_stream()
        self._set_vad_state(False)

        # Phase 4: Reject fragments — if total speech was shorter than
        # min_speech_sec, treat it as noise/false positive.
        if speech_chunk_count < min_speech_chunks:
            logger.info(
                f"Speech too short ({speech_chunk_count} chunks < "
                f"{min_speech_chunks} min). Discarding as noise."
            )
            return []

        logger.info(
            f"Recording complete: {len(frames)} chunks, "
            f"{speech_chunk_count} speech chunks."
        )
        return frames

    @staticmethod
    def compute_rms(frames) -> float:
        """Compute RMS energy of raw PCM-16 frames (list of bytes).
        Returns a float in [0.0, 1.0] range (normalised by int16 max)."""
        if not frames:
            return 0.0
        raw = b''.join(frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(samples ** 2) + 1e-9))

    def save_wav(self, frames, filename):
        """Save recorded frames to a WAV file."""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self._audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        rms = self.compute_rms(frames)
        logger.info(f"Saved audio to {filename} (RMS={rms:.4f})")

if __name__ == "__main__":
    # Test recording
    logging.basicConfig(level=logging.INFO)
    recorder = AudioRecorder()
    try:
        frames = recorder.record_until_silence()
        recorder.save_wav(frames, "test_recording.wav")
    finally:
        recorder.terminate()
