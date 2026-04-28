#!/usr/bin/env python3
"""Smoke-test the intermission early-exit watcher in isolation.

Simulates the race we care about: a breathing block is running when the
handler enqueues its LLM response.  Asserts:
  1. The watcher trips `llm_done` within ~1 s of the queue getting a
     response (after the engagement floor).
  2. The watcher sets `stop_playback_event` so an in-flight TTS would
     be cut short.
  3. The engagement floor actually prevents a premature cut if the LLM
     is ready before the floor expires.

Does NOT load Whisper / Gemma / Piper — only drives the speech_service
scheduler logic.  Runs anywhere (laptop or Jetson).
"""
from __future__ import annotations
import os, sys, threading, time, queue
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _FakeService:
    """Minimal stand-in that exposes just what
    _start_output_ready_watcher touches: output_queue and
    stop_playback_event.
    """
    def __init__(self):
        self.output_queue = queue.Queue()
        self.stop_playback_event = threading.Event()


def _run_watcher(service, llm_done):
    # Import here so we pick up the patched module.
    from src.services.speech_service import SpeechInteractionService
    return SpeechInteractionService._start_output_ready_watcher(service, llm_done)


def _case_immediate_ready():
    """LLM answer already present → watcher should trip after engagement floor."""
    svc = _FakeService()
    svc.output_queue.put("already-ready response")
    llm_done = threading.Event()
    t0 = time.monotonic()
    stop = _run_watcher(svc, llm_done)
    fired = llm_done.wait(timeout=10.0)
    dt = time.monotonic() - t0
    stop.set()
    assert fired, "llm_done did not fire"
    assert svc.stop_playback_event.is_set(), "stop_playback_event not set"
    assert dt >= 5.5, f"fired too early: {dt:.2f}s (min engagement ~6s)"
    assert dt <= 7.5, f"fired too late: {dt:.2f}s"
    print(f"  [PASS] immediate-ready: llm_done fired at {dt:.2f}s (engagement floor respected)")


def _case_late_ready():
    """LLM answer arrives mid-activity → watcher trips ~immediately after."""
    svc = _FakeService()
    llm_done = threading.Event()
    stop = _run_watcher(svc, llm_done)
    time.sleep(9.0)  # past engagement floor
    t0 = time.monotonic()
    svc.output_queue.put("late response")
    fired = llm_done.wait(timeout=2.0)
    dt = time.monotonic() - t0
    stop.set()
    assert fired, "llm_done did not fire"
    assert svc.stop_playback_event.is_set(), "stop_playback_event not set"
    assert dt <= 0.5, f"watcher lagged: {dt:.2f}s after queue write (expected ~0.1s)"
    print(f"  [PASS] late-ready: llm_done fired {dt:.2f}s after queue write")


def _case_stop_before_ready():
    """Stop called before output → watcher exits cleanly, llm_done not set."""
    svc = _FakeService()
    llm_done = threading.Event()
    stop = _run_watcher(svc, llm_done)
    time.sleep(0.5)
    stop.set()
    time.sleep(0.3)
    assert not llm_done.is_set(), "llm_done fired after stop_event"
    assert not svc.stop_playback_event.is_set(), "stop_playback_event wrongly set"
    print(f"  [PASS] stop-before-ready: watcher exited cleanly, no spurious fires")


if __name__ == "__main__":
    print("[TEST] Intermission early-exit watcher")
    _case_stop_before_ready()
    _case_late_ready()
    _case_immediate_ready()
    print("[TEST] All cases passed.")
