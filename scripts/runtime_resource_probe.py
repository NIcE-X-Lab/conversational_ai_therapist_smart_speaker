#!/usr/bin/env python3
"""Runs a minimal STT/SER/LLM probe to populate resource audit measurements."""

from __future__ import annotations

import json
import os
import tempfile
import wave

import numpy as np

from src.utils.resource_audit import get_resource_audit


def _write_silence_wav(path: str, sample_rate: int = 16000, duration_sec: float = 1.0) -> None:
    frames = int(sample_rate * duration_sec)
    audio = np.zeros(frames, dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def main() -> int:
    audit = get_resource_audit()
    audit.capture_point("runtime_probe_start")

    # STT/SER probe
    stt_payload = ""
    try:
        from src.models.stt import STTGenerator

        stt = STTGenerator()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _write_silence_wav(tmp_path)
            stt_payload = stt.transcribe(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        stt_payload = json.dumps({"error": f"stt_probe_failed: {e}"})

    # LLM probe
    llm_output = ""
    try:
        from src.models.llm_client import llm_complete

        llm_output = llm_complete(
            "You are a terse assistant. Return exactly: PROBE_OK",
            "Return exactly: PROBE_OK",
        )
    except Exception as e:
        llm_output = f"llm_probe_failed: {e}"

    audit.capture_point("runtime_probe_end")
    resource_map = audit.emit_resource_map()

    print("=== Runtime Resource Probe ===")
    print(f"STT payload: {stt_payload}")
    print(f"LLM output: {llm_output}")
    print("Resource Map:")
    print(json.dumps(resource_map, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
