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
import random
import struct
import threading
import numpy as np
from src.utils.log_util import get_logger
from src.utils.config_loader import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_CHUNK_SIZE, AUDIO_VAD_AGGRESSIVENESS


# MPEG Audio Layer III header parsing tables — used to estimate a track's
# duration without pulling in mutagen/ffprobe.  Indexing matches the bit
# fields defined in ISO/IEC 11172-3.
_MPEG_BITRATES_V1_L3 = (
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, None,
)  # kbps
_MPEG_BITRATES_V2_L3 = (
    0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, None,
)  # kbps (V2 / V2.5 Layer III)
_MPEG_SAMPLERATES = {
    3: (44100, 48000, 32000),  # MPEG-1
    2: (22050, 24000, 16000),  # MPEG-2
    0: (11025, 12000, 8000),   # MPEG-2.5
}


def _probe_audio_duration(path: str) -> float:
    """Best-effort duration probe for WAV and MP3 files.

    Zero external dependencies — avoids mutagen/ffprobe.  Accurate for
    WAV and CBR MP3; for VBR MP3 the result is an estimate based on the
    first frame's bitrate (usually within a few percent, good enough for
    picking a random start offset).

    Returns 0.0 on any parse failure; callers should treat 0.0 as
    "unknown duration → start from 0."
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".wav":
            with wave.open(path, "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate() or 1
                return frames / float(rate)
        if ext in (".mp3", ".mpeg"):
            return _probe_mp3_duration(path)
    except Exception as e:  # pragma: no cover — diagnostic only
        logger.debug(f"Duration probe failed for {path}: {e}")
    return 0.0


def _probe_mp3_duration(path: str) -> float:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(10)
        audio_start = 0
        if len(head) >= 10 and head[:3] == b"ID3":
            # Syncsafe size: 4 bytes, top bit of each byte is 0.
            tag_size = (
                (head[6] & 0x7F) << 21
                | (head[7] & 0x7F) << 14
                | (head[8] & 0x7F) << 7
                | (head[9] & 0x7F)
            )
            audio_start = 10 + tag_size
            f.seek(audio_start)

        # Scan forward a bit for the first valid MPEG frame sync.
        buf = f.read(8192)
        for i in range(len(buf) - 4):
            if buf[i] != 0xFF or (buf[i + 1] & 0xE0) != 0xE0:
                continue
            b1, b2 = buf[i + 1], buf[i + 2]
            version_id = (b1 >> 3) & 0x03
            layer = (b1 >> 1) & 0x03
            bitrate_idx = (b2 >> 4) & 0x0F
            sr_idx = (b2 >> 2) & 0x03
            if layer != 1 or bitrate_idx in (0, 15) or sr_idx == 3:
                continue
            # Layer III only (matches _MPEG_BITRATES tables above).
            if version_id == 3:
                bitrate_kbps = _MPEG_BITRATES_V1_L3[bitrate_idx]
            elif version_id in (2, 0):
                bitrate_kbps = _MPEG_BITRATES_V2_L3[bitrate_idx]
            else:
                continue
            if not bitrate_kbps:
                continue
            # Audio byte stream length / bytes-per-second.
            audio_bytes = max(0, size - audio_start)
            return audio_bytes / (bitrate_kbps * 1000.0 / 8.0)
    return 0.0


logger = get_logger("AudioRecorder")

_ALSA_ERR_HANDLER_REF = None
_AI_IS_SPEAKING = threading.Event()
_USER_IS_SPEAKING = threading.Event()


def set_ai_speaking(is_speaking: bool):
    """Global speaking flag used by background music ducking."""
    if is_speaking:
        _AI_IS_SPEAKING.set()
    else:
        _AI_IS_SPEAKING.clear()


def is_ai_speaking() -> bool:
    return _AI_IS_SPEAKING.is_set()


def set_user_speaking(is_speaking: bool):
    """Global flag: duck music while the user is being recorded."""
    if is_speaking:
        _USER_IS_SPEAKING.set()
    else:
        _USER_IS_SPEAKING.clear()


def is_user_speaking() -> bool:
    return _USER_IS_SPEAKING.is_set()


class BackgroundMusicThread:
    """Always-on non-blocking ambient music loop with speaking ducking."""

    SIGNAL_DUCK = "DUCK"
    SIGNAL_RESTORE = "RESTORE"

    def __init__(
        self,
        track_path: str = "assets/audio/ambient_music.mp3",
        base_volume: float = 0.15,
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
        # Fade ramping: when active, _target_volume() interpolates between
        # _fade_start and _fade_target over _fade_duration seconds.
        self._fade_active = False
        self._fade_start_volume = self.base_volume
        self._fade_target_volume = self.base_volume
        self._fade_started_at = 0.0
        self._fade_duration = 0.0
        # Random-segment playback: probed once per resolved track so long
        # ambient mixes never replay from the top.  Duration of 0 means
        # the probe failed / wasn't applicable — we then fall back to
        # start=0 and rely on loops=-1 to keep audio going.
        self._resolved_track: str | None = None
        self._track_duration: float = 0.0
        self._segment_change_requested = threading.Event()
        # Leave a safety margin so the chosen start offset never lands in
        # the final seconds of the track (avoids near-instant loop-back).
        # SDL2's MP3 decoder silently fails on seeks that land in the
        # last few percent of very long tracks, so we also cap the
        # maximum usable offset at 90% of duration — see
        # _seekable_upper_bound().  Playback at deeper offsets returns
        # get_busy() == False with get_pos() == -1 and NO audio.
        self._segment_tail_margin: float = 30.0
        self._segment_safe_fraction: float = 0.90
        # Click-killer: fade-in/out applied to every play()/stop() so
        # segment jumps and startup transitions never snap to silence
        # mid-waveform.  USB DAC chains (UACDemo) have an audible ramp
        # transient well beyond 80 ms, so we use 400 ms by default —
        # short enough to still feel like an immediate seek, long
        # enough to fully mask the click on every tested device.
        self._segment_fade_ms: int = 400

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

    def _seekable_upper_bound(self) -> float:
        """Highest offset we'll attempt to seek to without risking a silent
        failure.

        SDL2's MP3 decoder silently bails out on very deep seeks into very
        long tracks (observed empirically at ~5 h files: a seek to
        duration-120 s returned get_busy()=False and NO audio).  Capping
        at ``_segment_safe_fraction`` of duration keeps us well inside
        the region the decoder actually handles.
        """
        if self._track_duration <= 0:
            return 0.0
        return max(
            0.0,
            min(
                self._track_duration - self._segment_tail_margin,
                self._track_duration * self._segment_safe_fraction,
            ),
        )

    def _pick_random_offset(self) -> float:
        """Return a random start offset within the track.

        Zero when duration is unknown or the track is shorter than the
        tail margin — in that case the caller just plays from the
        beginning, which is still correct behaviour.
        """
        upper = self._seekable_upper_bound()
        if upper <= 0:
            return 0.0
        return random.uniform(0.0, upper)

    def _ensure_mixer(self) -> bool:
        if self._mixer_ready and self._pygame is not None:
            return True
        try:
            import pygame
            self._pygame = pygame
            if not pygame.mixer.get_init():
                # 22050 Hz stereo, 1024-sample buffer: low latency on Jetson
                # USB audio while supporting both 16kHz WAVs and MP3 music.
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
            self._mixer_ready = True
            return True
        except Exception as e:
            logger.warning(f"Background music disabled: mixer init failed: {e}")
            return False

    def _target_volume(self) -> float:
        if self._duck_override or is_ai_speaking() or is_user_speaking():
            return self.speaking_volume
        if self._fade_active:
            elapsed = time.monotonic() - self._fade_started_at
            if elapsed >= self._fade_duration or self._fade_duration <= 0:
                self.base_volume = self._fade_target_volume
                self._fade_active = False
                return self.base_volume
            progress = elapsed / self._fade_duration
            return (
                self._fade_start_volume
                + (self._fade_target_volume - self._fade_start_volume) * progress
            )
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

        # Probe duration once per resolved track so long ambient mixes
        # can start at a random offset on each (re)start — avoids users
        # always hearing the opening seconds of the track.
        if self._resolved_track != track:
            self._resolved_track = track
            self._track_duration = _probe_audio_duration(track)
            logger.info(
                f"Background music duration probe: {track} = "
                f"{self._track_duration:.1f}s"
            )

        def _play_verified(start_offset: float, cross_fade: bool) -> bool:
            """Play from ``start_offset`` with fade-in; verify audio is
            actually running, and if not, retry with a shallower offset
            or from 0 as a last resort.

            SDL2's MP3 decoder sometimes silently fails on deep seeks
            into long tracks — get_busy() reports False and get_pos()
            returns -1.  We detect that within ~150 ms and fall back.

            ``cross_fade`` requests a short fadeout() on the current
            playback before starting the new position, so segment jumps
            don't click/snap.
            """
            if cross_fade:
                try:
                    if self._pygame.mixer.music.get_busy():
                        self._pygame.mixer.music.fadeout(self._segment_fade_ms)
                        # Wait for the full fadeout to finish before we
                        # overwrite the stream — pygame.mixer.music only
                        # supports ONE stream at a time, so a real
                        # cross-fade (two streams overlapping) isn't
                        # possible here.  A full wait guarantees the
                        # ramp-down completes instead of being clipped
                        # by the incoming play(), which was the source
                        # of the residual click.
                        deadline = time.monotonic() + (self._segment_fade_ms / 1000.0) + 0.05
                        while time.monotonic() < deadline:
                            if not self._pygame.mixer.music.get_busy():
                                break
                            time.sleep(0.02)
                except Exception:
                    pass

            for attempt_offset in (start_offset, start_offset * 0.5, 0.0):
                attempt_offset = max(0.0, float(attempt_offset))
                try:
                    self._pygame.mixer.music.play(
                        loops=-1,
                        start=attempt_offset,
                        fade_ms=self._segment_fade_ms,
                    )
                except Exception as e:
                    logger.warning(
                        f"Background music play() raised at offset={attempt_offset:.1f}: {e}"
                    )
                    continue

                # Verify the decoder actually started producing audio.
                # get_pos() returns -1 until playback starts; up to ~200 ms
                # grace for SDL2 to spool the decoder.
                verified = False
                for _ in range(5):
                    time.sleep(0.04)
                    if self._pygame.mixer.music.get_busy() and self._pygame.mixer.music.get_pos() >= 0:
                        verified = True
                        break

                if verified:
                    logger.info(
                        f"Background music segment start: offset={attempt_offset:.1f}s"
                    )
                    return True

                logger.warning(
                    f"Background music seek at offset={attempt_offset:.1f}s produced "
                    "no audio; retrying with shallower offset."
                )
            return False

        def _play_at_random_offset(cross_fade: bool = False):
            _play_verified(self._pick_random_offset(), cross_fade=cross_fade)

        try:
            self._pygame.mixer.music.load(track)
            self._pygame.mixer.music.set_volume(self._target_volume())
            _play_at_random_offset(cross_fade=False)
            logger.info(f"Background music started: {track}")
        except Exception as e:
            logger.warning(f"Failed starting background music: {e}")
            with self._lock:
                self._running = False
            return

        while not self._stop_event.wait(0.2):
            try:
                # Explicit segment-change request from a caller
                # (e.g., new intermission MUSIC block wants fresh audio).
                if self._segment_change_requested.is_set():
                    self._segment_change_requested.clear()
                    _play_at_random_offset(cross_fade=True)

                # Natural loop restart: pygame reports not-busy when the
                # track reaches its end.  Instead of looping from 0 (the
                # default loops=-1 behaviour), restart at a new random
                # offset so long tracks stay fresh turn after turn.
                if not self._pygame.mixer.music.get_busy():
                    _play_at_random_offset(cross_fade=False)

                self._pygame.mixer.music.set_volume(self._target_volume())
            except Exception as e:
                logger.warning(f"Background music loop stopped unexpectedly: {e}")
                break

        # Soft shutdown: short fadeout() before the hard stop so there
        # is never a click on session end / track swap.
        try:
            if self._pygame.mixer.music.get_busy():
                self._pygame.mixer.music.fadeout(max(self._segment_fade_ms, 120))
                time.sleep(0.15)
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
            # A hard volume set cancels any in-flight fade.
            self._fade_active = False

    def jump_to_random_segment(self):
        """Request the worker thread to restart playback at a new random offset.

        No-op until the worker is running; the actual seek happens on
        the worker's next ~200 ms poll so call sites don't block on
        audio-device I/O.
        """
        self._segment_change_requested.set()

    def fade_to(self, target_volume: float, duration: float = 1.5):
        """Smoothly ramp the music bed to ``target_volume`` over ``duration`` seconds.

        The ramp is evaluated by the running worker thread on each poll
        (~200 ms), so short fades are quantised to that interval.  Callers
        that need the fade to have visibly completed should sleep for
        ``duration + 0.25`` before proceeding.
        """
        target = max(0.0, min(1.0, float(target_volume)))
        dur = max(0.0, float(duration))
        with self._lock:
            # Starting volume for the ramp is whatever the user would be
            # hearing right now — if a prior fade is still in flight we
            # snap the ramp's origin to its current interpolated position
            # so we do not rubber-band.
            if self._fade_active:
                elapsed = time.monotonic() - self._fade_started_at
                if self._fade_duration > 0 and elapsed < self._fade_duration:
                    progress = elapsed / self._fade_duration
                    self._fade_start_volume = (
                        self._fade_start_volume
                        + (self._fade_target_volume - self._fade_start_volume) * progress
                    )
                else:
                    self._fade_start_volume = self._fade_target_volume
            else:
                self._fade_start_volume = self.base_volume
            self._fade_target_volume = target
            self._fade_duration = dur
            self._fade_started_at = time.monotonic()
            self._fade_active = dur > 0
            if not self._fade_active:
                # Zero-duration fade == hard set, matches set_base_volume.
                self.base_volume = target


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
                    set_user_speaking(True)  # duck music while user talks
                    logger.info("Speech detected, recording started.")
                    break
            except IOError as e:
                logger.warning(f"Audio read error during wait: {e}")
                time.sleep(0.01)

        if not has_speech:
            logger.info("No speech detected, stopping recording window.")
            self.stop_stream()
            self._set_vad_state(False)
            set_user_speaking(False)
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
        set_user_speaking(False)  # restore music volume

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
