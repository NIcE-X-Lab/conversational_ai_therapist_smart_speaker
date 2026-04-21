#!/usr/bin/env python3
"""Latency benchmark harness for CaiTI.

This script can measure:
- STT latency from a WAV file or a live microphone recording
- LLM latency using llm_complete()
- TTS synthesis latency using Piper / espeak-ng fallback
- A mic-driven end-to-end case test that exercises STT -> LLM -> TTS -> playback

Examples:
    python scripts/latency_benchmark.py --audio-path sample.wav
    python scripts/latency_benchmark.py --record-mic --case
    python scripts/latency_benchmark.py --samples 8 --playback
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

LITERT_BACKEND = os.environ.get("LITERT_BACKEND", "cpu").strip().lower()
LITERT_CONTEXT_LENGTH = int(os.environ.get("LITERT_CONTEXT_LENGTH", "512"))
LITERT_MAX_TOKENS = int(os.environ.get("LITERT_MAX_TOKENS", "80"))


@dataclass
class SampleResult:
    component: str
    seconds: float
    note: str = ""


@dataclass
class CaseResult:
    mic_capture_seconds: float
    stt_seconds: float
    llm_seconds: float
    tts_seconds: float
    playback_signal_seconds: float | None
    playback_block_seconds: float
    full_turn_seconds: float
    transcript: str
    response: str


def _parse_transcript(payload: str) -> str:
    try:
        data = json.loads(payload)
        return str(data.get("transcript", "")).strip()
    except Exception:
        return str(payload).strip()


def _quantile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile / 100.0) * (len(ordered) - 1)))
    index = max(0, min(len(ordered) - 1, index))
    return ordered[index]


def _print_summary(title: str, values: list[float]) -> None:
    if not values:
        print(f"{title}: no samples")
        return
    print(f"{title}: samples={len(values)} mean={statistics.mean(values):.3f}s p50={_quantile(values, 50):.3f}s p95={_quantile(values, 95):.3f}s max={max(values):.3f}s")


def _record_audio(timeout: float):
    from src.drivers.audio import AudioRecorder

    recorder = AudioRecorder()
    start = time.monotonic()
    try:
        frames = recorder.record_until_silence(max_duration=timeout)
    finally:
        capture_seconds = time.monotonic() - start
    return frames, recorder, capture_seconds


def _load_audio_source(audio_path: str | None, record_mic: bool, record_timeout: float) -> tuple[str | None, float, AudioRecorder | None]:
    if audio_path:
        return audio_path, 0.0, None

    if not record_mic:
        return None, 0.0, None

    print("Speak now. Recording from microphone...")
    frames, recorder, capture_seconds = _record_audio(record_timeout)
    if not frames:
        recorder.terminate()
        raise RuntimeError("No speech detected during microphone capture.")

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    recorder.save_wav(frames, tmp_wav.name)
    recorder.terminate()
    return tmp_wav.name, capture_seconds, recorder


def _run_stt(stt: STTGenerator, wav_path: str) -> tuple[float, str]:
    from src.models.stt import STTGenerator

    started = time.monotonic()
    payload = stt.transcribe(wav_path)
    elapsed = time.monotonic() - started
    return elapsed, _parse_transcript(payload)


def _run_llm(prompt: str, samples: int, warmup: bool = True) -> tuple[list[float], list[str]]:
    from src.models.llm_client import llm_complete

    if warmup:
        try:
            _ = llm_complete(
                "You are concise.",
                "Warmup run only.",
                inject_context=False,
            )
        except Exception:
            pass

    durations: list[float] = []
    outputs: list[str] = []
    for idx in range(samples):
        started = time.monotonic()
        output = llm_complete(
            "You are concise and supportive.",
            prompt,
            inject_context=False,
        )
        elapsed = time.monotonic() - started
        durations.append(elapsed)
        outputs.append(output)
        preview = str(output).strip().replace("\n", " ")[:80]
        print(f"LLM sample {idx + 1}: {elapsed:.3f}s | {preview!r}")
    return durations, outputs


def _run_tts(tts: TTSGenerator, text: str, playback: bool = False) -> tuple[float, float | None, float, Path | None]:
    from src.drivers.player import AudioPlayer

    tmp_wav = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    playback_signal_time: float | None = None
    playback_signal_event = threading.Event()

    def _playback_signal_handler(signal: str):
        nonlocal playback_signal_time
        if signal == "DUCK" and playback_signal_time is None:
            playback_signal_time = time.monotonic()
            playback_signal_event.set()

    player = AudioPlayer(playback_signal_handler=_playback_signal_handler) if playback else None

    started = time.monotonic()
    out = tts.generate(text, str(tmp_wav))
    tts_seconds = time.monotonic() - started

    if not out:
        if tmp_wav.exists():
            tmp_wav.unlink(missing_ok=True)
        return tts_seconds, None, 0.0, None

    playback_seconds = 0.0
    if playback and player is not None:
        started = time.monotonic()
        player.play(str(tmp_wav), duck=True)
        playback_seconds = time.monotonic() - started
        if not playback_signal_event.is_set():
            playback_signal_time = None

    return tts_seconds, playback_signal_time, playback_seconds, tmp_wav


def _benchmark_components(audio_path: str | None, record_mic: bool, record_timeout: float, samples: int, playback: bool, save_json: str | None) -> int:
    from src.models.stt import STTGenerator
    from src.models.tts import TTSGenerator

    stt = STTGenerator()
    tts = TTSGenerator()

    if audio_path is None and not record_mic:
        print("STT benchmark skipped: provide --audio-path or --record-mic")
        transcript_for_llm = "Tell me one practical grounding tip."
        stt_results: list[float] = []
    else:
        source_path, capture_seconds, _ = _load_audio_source(audio_path, record_mic, record_timeout)
        if source_path is None:
            raise RuntimeError("No audio source available for STT benchmark.")
        stt_seconds, transcript_for_llm = _run_stt(stt, source_path)
        stt_results = [stt_seconds]
        print(f"STT: {stt_seconds:.3f}s | transcript={transcript_for_llm!r}")
        if record_mic and audio_path is None:
            print(f"Mic capture: {capture_seconds:.3f}s")
            Path(source_path).unlink(missing_ok=True)

    prompts = [
        "What is CBT in one sentence?",
        "Give one grounding exercise.",
        "How can I slow racing thoughts?",
        "Explain box breathing briefly.",
        "Give one actionable coping step.",
        "Name one journaling prompt for anxiety.",
        "How can I reframe negative thoughts?",
        "Give one self-compassion sentence.",
    ]
    llm_prompt = transcript_for_llm if transcript_for_llm else prompts[0]
    llm_values, llm_outputs = _run_llm(llm_prompt, samples=samples, warmup=True)

    tts_texts = [
        llm_outputs[0] if llm_outputs else "Take one slow breath in, then out.",
        "Take one slow breath in, then out.",
        "Name three things you can see right now.",
    ]
    tts_values: list[float] = []
    playback_values: list[float] = []
    for idx, text in enumerate(tts_texts, start=1):
        tts_seconds, playback_signal_time, playback_seconds, tmp_wav = _run_tts(tts, text, playback=playback)
        tts_values.append(tts_seconds)
        if playback_seconds:
            playback_values.append(playback_seconds)
        print(f"TTS sample {idx}: synth={tts_seconds:.3f}s" + (f" playback_block={playback_seconds:.3f}s" if playback_seconds else ""))
        if tmp_wav is not None:
            tmp_wav.unlink(missing_ok=True)

    print("\n=== Component Summary ===")
    _print_summary("STT", stt_results)
    _print_summary("LLM", llm_values)
    _print_summary("TTS", tts_values)
    if playback_values:
        _print_summary("Playback block", playback_values)

    if save_json:
        payload = {
            "stt": stt_results,
            "llm": llm_values,
            "tts": tts_values,
            "playback_block": playback_values,
        }
        Path(save_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON results to {save_json}")

    return 0


def _run_case(record_mic: bool, audio_path: str | None, record_timeout: float, playback: bool, save_json: str | None) -> int:
    from src.models.llm_client import llm_complete
    from src.models.stt import STTGenerator
    from src.models.tts import TTSGenerator

    stt = STTGenerator()
    tts = TTSGenerator()

    case_started = time.monotonic()
    source_path, capture_seconds, _ = _load_audio_source(audio_path, record_mic, record_timeout)
    if source_path is None:
        raise RuntimeError("Case test requires --record-mic or --audio-path.")

    stt_started = time.monotonic()
    stt_seconds, transcript = _run_stt(stt, source_path)
    stt_total = time.monotonic() - stt_started

    prompt = transcript or "I feel overwhelmed today."
    llm_started = time.monotonic()
    response = llm_complete(
        "You are warm, brief, and practical. Reply in 2-4 sentences.",
        prompt,
        inject_context=False,
    )
    llm_seconds = time.monotonic() - llm_started

    tts_seconds, playback_signal_time, playback_seconds, tmp_wav = _run_tts(tts, response, playback=playback)

    playback_signal_seconds = None if playback_signal_time is None else playback_signal_time - case_started
    if playback_signal_seconds is None:
        full_turn_seconds = capture_seconds + stt_seconds + llm_seconds + tts_seconds
    else:
        full_turn_seconds = playback_signal_seconds

    result = CaseResult(
        mic_capture_seconds=capture_seconds,
        stt_seconds=stt_seconds,
        llm_seconds=llm_seconds,
        tts_seconds=tts_seconds,
        playback_signal_seconds=playback_signal_seconds,
        playback_block_seconds=playback_seconds,
        full_turn_seconds=full_turn_seconds,
        transcript=transcript,
        response=response,
    )

    print("\n=== Interactive Case Test ===")
    print(f"Transcript: {result.transcript!r}")
    print(f"Response: {result.response!r}")
    print(f"Mic capture: {result.mic_capture_seconds:.3f}s")
    print(f"STT: {result.stt_seconds:.3f}s")
    print(f"LLM: {result.llm_seconds:.3f}s")
    print(f"TTS synth: {result.tts_seconds:.3f}s")
    if playback_seconds:
        print(f"Playback block: {result.playback_block_seconds:.3f}s")
    print(f"Full turn (capture -> prepared playback): {result.full_turn_seconds:.3f}s")
    print(f"Backend: {LITERT_BACKEND} | context={LITERT_CONTEXT_LENGTH} | max_tokens={LITERT_MAX_TOKENS}")

    if save_json:
        Path(save_json).write_text(json.dumps(result.__dict__, indent=2, default=str), encoding="utf-8")
        print(f"Saved JSON results to {save_json}")

    if tmp_wav is not None:
        tmp_wav.unlink(missing_ok=True)
    if audio_path is None and source_path:
        Path(source_path).unlink(missing_ok=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark CaiTI latency components and interactive turn latency.")
    parser.add_argument("--audio-path", type=str, default=None, help="WAV file to use for STT benchmark instead of microphone capture.")
    parser.add_argument("--record-mic", action="store_true", help="Record a fresh utterance from the microphone for STT / case tests.")
    parser.add_argument("--record-timeout", type=float, default=15.0, help="Maximum seconds to wait for microphone recording.")
    parser.add_argument("--samples", type=int, default=5, help="LLM benchmark sample count.")
    parser.add_argument("--playback", action="store_true", help="Actually play synthesized TTS audio using the audio output device.")
    parser.add_argument("--case", action="store_true", help="Run a mic-driven end-to-end case test (STT -> LLM -> TTS -> playback).")
    parser.add_argument("--save-json", type=str, default=None, help="Optional path for JSON results.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.case:
        return _run_case(
            record_mic=args.record_mic,
            audio_path=args.audio_path,
            record_timeout=args.record_timeout,
            playback=True,
            save_json=args.save_json,
        )

    return _benchmark_components(
        audio_path=args.audio_path,
        record_mic=args.record_mic,
        record_timeout=args.record_timeout,
        samples=args.samples,
        playback=args.playback,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
