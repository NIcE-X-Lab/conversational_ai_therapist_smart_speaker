"""Background therapeutic music loop with ducking support."""

from __future__ import annotations

import os
import threading

from src.utils.log_util import get_logger

logger = get_logger("MusicPlayer")

_PRIMARY_TRACK = "assets/audio/ambient_therapy.mp3"
_FALLBACK_TRACK = "assets/audio/waiting_music.wav"


class BackgroundMusicService:
    """Keeps ambient music running in a non-blocking loop thread."""

    SIGNAL_DUCK = "DUCK"
    SIGNAL_RESTORE = "RESTORE"

    def __init__(self, base_volume: float = 0.25, duck_volume: float = 0.05):
        self.base_volume = float(base_volume)
        self.duck_volume = float(duck_volume)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = None
        self._mixer_ready = False
        self._ducked = False
        self._running = False
        self._track_path = _PRIMARY_TRACK
        self._pygame = None

    def _resolve_track(self) -> str | None:
        if self._track_path and os.path.isfile(self._track_path):
            return self._track_path
        if os.path.isfile(_FALLBACK_TRACK):
            return _FALLBACK_TRACK
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
            logger.warning(f"Background music disabled: pygame mixer init failed: {e}")
            self._mixer_ready = False
            return False

    def _active_volume(self) -> float:
        return self.duck_volume if self._ducked else self.base_volume

    def _run_loop(self):
        if not self._ensure_mixer():
            with self._lock:
                self._running = False
            return

        track = self._resolve_track()
        if not track:
            logger.warning("Background music track not found; music loop disabled.")
            with self._lock:
                self._running = False
            return

        try:
            self._pygame.mixer.music.load(track)
            self._pygame.mixer.music.set_volume(self._active_volume())
            self._pygame.mixer.music.play(loops=-1)
            logger.info(f"Background music loop started: {track}")
        except Exception as e:
            logger.warning(f"Failed to start background loop: {e}")
            with self._lock:
                self._running = False
            return

        while not self._stop_event.wait(0.5):
            try:
                if not self._pygame.mixer.music.get_busy():
                    self._pygame.mixer.music.play(loops=-1)
                self._pygame.mixer.music.set_volume(self._active_volume())
            except Exception as e:
                logger.warning(f"Background loop worker stopped: {e}")
                break

        try:
            self._pygame.mixer.music.stop()
        except Exception:
            pass

        with self._lock:
            self._running = False
        logger.info("Background music loop stopped.")

    def start_loop(self, track_path: str | None = None):
        with self._lock:
            if track_path:
                self._track_path = track_path
            if self._running:
                return
            self._stop_event.clear()
            self._running = True
            self._worker = threading.Thread(target=self._run_loop, daemon=True)
            self._worker.start()

    def stop(self):
        with self._lock:
            self._stop_event.set()
            worker = self._worker

        if worker:
            worker.join(timeout=1.0)

    def duck(self):
        with self._lock:
            self._ducked = True
        if self._mixer_ready and self._pygame is not None:
            try:
                self._pygame.mixer.music.set_volume(self.duck_volume)
            except Exception:
                pass

    def restore(self):
        with self._lock:
            self._ducked = False
        if self._mixer_ready and self._pygame is not None:
            try:
                self._pygame.mixer.music.set_volume(self.base_volume)
            except Exception:
                pass

    def handle_signal(self, signal: str):
        if signal == self.SIGNAL_DUCK:
            self.duck()
        elif signal == self.SIGNAL_RESTORE:
            self.restore()
