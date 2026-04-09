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
import numpy as np
from src.utils.log_util import get_logger
from src.utils.config_loader import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_CHUNK_SIZE, AUDIO_VAD_AGGRESSIVENESS


logger = get_logger("AudioRecorder")

_ALSA_ERR_HANDLER_REF = None


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

    def record_until_silence(self, silence_duration=1.5, max_duration=15.0):
        """
        Record audio dynamically: wait until speech starts, then record until
        silence or max limit reached.
        Returns:
            frames: List of audio frames (bytes).
        """
        self.start_stream()
        
        frames = []
        silence_chunks = int(silence_duration * self.rate / self.chunk)
        max_chunks = int(max_duration * self.rate / self.chunk)
        wait_chunks = int(5.0 * self.rate / self.chunk) # Max 5s wait for speech start
        
        silent_count = 0
        has_speech = False
        
        logger.info("Listening (waiting for speech)...")
        self._set_vad_state(True)
        
        # Audio Event Detection: wait for voice activity
        for _ in range(wait_chunks):
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                if self.is_speech(data):
                    has_speech = True
                    frames.append(data)
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

        # Now record up to max_chunks
        for _ in range(max_chunks):
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
                
                if self.is_speech(data):
                    silent_count = 0
                else:
                    silent_count += 1
                
                if silent_count > silence_chunks:
                    logger.info("Silence detected, stopping recording.")
                    self._set_vad_state(False)
                    break
                    
            except IOError as e:
                logger.warning(f"Audio read error: {e}")
                
        self.stop_stream()
        self._set_vad_state(False)
        
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
