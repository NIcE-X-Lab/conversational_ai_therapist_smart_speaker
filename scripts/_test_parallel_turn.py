#!/usr/bin/env python3
"""Smoke-test the parallel STT + intermission flow in isolation.

No real models are loaded.  The test stubs:
  - Recorder (returns a fixed frame count / WAV path).
  - STT (configurable delay + configurable transcript).
  - LLM output_queue (driven by the test with sleeps to simulate
    handler running in parallel).
  - Intermission manager + blocks (we just time how long the main
    thread spends in `_run_one_intermission_activity`).

Covers these cases:
  1. Happy path: long utterance -> parallel STT + intermission ->
     transcript queued -> delivered.
  2. Silence: mic returned no frames -> return "silence" without
     starting the intermission.
  3. END command: STT worker sets __CMD_END__ -> return "session_end"
     and stop_playback_event fires.
  4. Short utterance (< _PARALLEL_MIC_WINDOW_MIN_SEC): serial fallback
     path runs, intermission does NOT run concurrently with STT.
"""
from __future__ import annotations
import os, sys, threading, time, queue, tempfile, wave, struct
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import BEFORE we set up mocks so module-level code doesn't try to
# load Whisper/Piper/Gemma when we instantiate fakes.
import src.services.speech_service as ss


def _make_silent_wav(path: str, seconds: float = 3.0, rate: int = 16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        n = int(seconds * rate)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


class _FakeRecorder:
    """Stands in for AudioRecorder.  `frames` controls the fake payload."""
    def __init__(self, frames_count: int = 200, chunk: int = 480, rate: int = 16000):
        self.chunk = chunk
        self.rate = rate
        self._frames_count = frames_count

    def record_until_silence(self, max_duration=15.0, **kw):
        # Return a list of bytes objects — the real method returns audio
        # chunks; _record_utterance_to_wav only looks at len(frames) and
        # calls save_wav on them.  The bytes content doesn't matter.
        return [b"\x00" * self.chunk] * self._frames_count

    def compute_rms(self, frames):
        # Above the 0.005 threshold so we don't short-circuit to silence.
        return 0.1

    def save_wav(self, frames, path):
        # Write a minimal valid WAV so the STT mock can "transcribe" it.
        _make_silent_wav(path, seconds=0.1)


class _FakeSTT:
    """Stands in for STTGenerator.  `transcript` controls the output.
    `delay_sec` simulates Whisper decode time."""
    def __init__(self, transcript: str = "I have been feeling a bit down lately", delay_sec: float = 1.0):
        self._transcript = transcript
        self._delay = delay_sec
        self.suspend_called = threading.Event()
        self.resume_called = 0

    def transcribe(self, path):
        time.sleep(self._delay)
        import json as _json
        return _json.dumps({"transcript": self._transcript, "detected_emotion": "neu"})

    def suspend_all(self):
        self.suspend_called.set()

    def resume_all(self):
        self.resume_called += 1


def _mk_service(stt_transcript="I have been feeling a bit down lately",
                stt_delay=1.0,
                frames_count=200):
    """Build a SpeechInteractionService with the real parallel-turn
    machinery but all I/O-heavy deps stubbed.  Enough to exercise
    _run_parallel_turn end-to-end."""
    svc = ss.SpeechInteractionService.__new__(ss.SpeechInteractionService)
    svc.input_queue = queue.Queue()
    svc.output_queue = queue.Queue()
    svc.stop_playback_event = threading.Event()
    svc.recorder = _FakeRecorder(frames_count=frames_count)
    svc.stt = _FakeSTT(transcript=stt_transcript, delay_sec=stt_delay)
    svc.state = "idle"
    svc.running = True
    svc.is_hands_free = True
    svc._consecutive_silence_count = 0
    # Stub LED / chat / intermission helpers so only the parallel-turn
    # flow logic is under test.
    svc._led_on = lambda: None
    svc._led_off = lambda: None
    svc._sync_intermission_state_from_db = lambda: None
    svc.intermission_ladder = MagicMock()
    svc.intermission_ladder.screening_available.return_value = False
    class _Stage:
        value = "breathing"
    svc.intermission_ladder.next_activity.return_value = _Stage()
    svc.intermission_ladder.mark_activity = MagicMock()
    svc.intermission_ladder.next_breathing_exercise.return_value = "breathe in, breathe out"
    # Track intermission invocations
    svc._intermission_calls = []
    def _fake_intermission():
        svc._intermission_calls.append(time.monotonic())
        # Simulate the proactive activity taking ~2 s (well under the
        # real breathing 30 s budget but enough to overlap with the STT
        # worker in the happy-path test).
        time.sleep(2.0)
    svc._run_one_intermission_activity = _fake_intermission
    # Stub delivery phase — just record it was called.
    svc._delivery_calls = []
    def _fake_delivery(is_session_start=False):
        svc._delivery_calls.append(time.monotonic())
    svc._wait_for_output_with_intermission = _fake_delivery
    # Stub TTS helpers
    svc.say = lambda *a, **k: None
    svc._apply_global_command_priority = lambda s: None  # no command
    # Minimum-parallel threshold: default value is fine but lower it for
    # the "short utterance" test via a second instance.
    return svc


def _case_happy_path():
    print("[CASE] happy path (parallel intermission + STT, normal transcript)")
    svc = _mk_service(stt_transcript="I have been feeling a bit down lately",
                      stt_delay=1.0, frames_count=200)  # 200*480/16000 = 6s audio

    # Instrument the fake intermission to record the input_queue state
    # at the moment it starts, so we can verify the transcript is NOT
    # yet queued (STT hasn't finished) but gets queued during the
    # intermission body.
    stt_queue_at_start = []
    def _fake_intermission():
        svc._intermission_calls.append(time.monotonic())
        stt_queue_at_start.append(svc.input_queue.qsize())
        time.sleep(2.0)
        # By now STT (1s delay) should have finished and queued.
        svc._stt_queue_at_end = svc.input_queue.qsize()
    svc._run_one_intermission_activity = _fake_intermission

    t0 = time.monotonic()
    outcome = svc._run_parallel_turn()
    dt = time.monotonic() - t0
    print(f"  outcome: {outcome}  (took {dt:.2f}s)")
    assert outcome == "delivered", f"expected delivered, got {outcome!r}"
    # Intermission should have started BEFORE STT finished; check order.
    assert svc._intermission_calls, "intermission did not run"
    intermission_started = svc._intermission_calls[0]
    # STT delay was 1 s, intermission was called immediately after mic
    # closes. Gap from _run_parallel_turn start to intermission call
    # must be < 0.3 s.
    assert intermission_started - t0 < 0.3, \
        f"intermission was too slow to start: {intermission_started - t0:.2f}s"
    # CRITICAL: input_queue must have been EMPTY when intermission
    # started (STT still running) but POPULATED by the time intermission
    # finished. This proves the handler could start its LLM call in
    # parallel with the intermission rather than being blocked until it
    # returns.
    assert stt_queue_at_start[0] == 0, \
        f"transcript was queued BEFORE intermission started (qsize={stt_queue_at_start[0]}) — " \
        "that defeats parallelism"
    assert svc._stt_queue_at_end == 1, \
        f"transcript was NOT queued during intermission (qsize={svc._stt_queue_at_end}) — " \
        "handler would be starved until intermission finishes"
    # Input queue must have received the transcript (worker, not main).
    assert svc.input_queue.qsize() == 1, "transcript not queued"
    queued = svc.input_queue.get_nowait()
    assert queued == "I have been feeling a bit down lately", f"wrong queued text: {queued!r}"
    # Whisper must be suspended (by the worker).
    assert svc.stt.suspend_called.is_set(), "STT never suspended"
    # STT must be resumed (by parallel-turn entry).
    assert svc.stt.resume_called >= 1, "STT never resumed"
    print("  [PASS] transcript queued mid-intermission (handler runs in parallel)")


def _case_silence_no_frames():
    print("[CASE] silence (recorder returned no frames)")
    svc = _mk_service()
    svc.recorder = _FakeRecorder(frames_count=0)
    outcome = svc._run_parallel_turn()
    assert outcome == "silence", f"expected silence, got {outcome!r}"
    # No intermission should have started.
    assert not svc._intermission_calls, "intermission wrongly ran on silent input"
    # Nothing should be queued to handler.
    assert svc.input_queue.empty(), "queued unexpectedly on silence"
    print("  [PASS]")


def _case_end_command():
    print("[CASE] END command via STT priority gate")
    svc = _mk_service()
    # Stub transcribe() to return __CMD_END__ directly (as it would after
    # the priority gate flag fires).
    def _transcribe_end(path, apply_priority_gate=False):
        time.sleep(0.5)
        return "__CMD_END__"
    svc.transcribe = _transcribe_end
    outcome = svc._run_parallel_turn()
    assert outcome == "session_end", f"expected session_end, got {outcome!r}"
    # stop_playback_event must have fired so any in-flight intermission
    # is cut.
    assert svc.stop_playback_event.is_set(), "stop_playback_event not set on END"
    # CRITICAL: the worker must NOT have queued __CMD_END__ on
    # input_queue — the handler would treat it as a clinical answer
    # and try to classify it, which would corrupt the session.
    assert svc.input_queue.empty(), \
        f"sentinel wrongly queued on END (qsize={svc.input_queue.qsize()})"
    print("  [PASS]")


def _case_short_utterance_serial():
    print("[CASE] short utterance falls back to serial path (no parallel intermission during STT)")
    # 50 frames * 480 / 16000 = 1.5 s — well below _PARALLEL_MIC_WINDOW_MIN_SEC=3.0
    svc = _mk_service(stt_transcript="yeah", stt_delay=1.0, frames_count=50)
    t0 = time.monotonic()
    outcome = svc._run_parallel_turn()
    dt = time.monotonic() - t0
    print(f"  outcome: {outcome}  (took {dt:.2f}s)")
    assert outcome == "delivered", f"expected delivered, got {outcome!r}"
    # In the serial path, intermission runs AFTER transcription. So the
    # intermission call timestamp should be at least stt_delay after t0.
    intermission_started = svc._intermission_calls[0]
    assert intermission_started - t0 >= 1.0, \
        f"serial path ran intermission too early ({intermission_started - t0:.2f}s)"
    print("  [PASS]")


if __name__ == "__main__":
    print("[TEST] Parallel STT + intermission flow")
    _case_silence_no_frames()
    _case_happy_path()
    _case_end_command()
    _case_short_utterance_serial()
    print("[TEST] All cases passed.")
