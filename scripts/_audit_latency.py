#!/usr/bin/env python3
"""Audit per-module latency AND the real per-turn LLM call chain.

Designed so a single run tells us:
  1. STT, TTS, LLM (short + long prompt) single-call cost
  2. Cost of each REAL role prompt used in production (REPHRASER, ANALYZER,
     RV_REASONER, RV_VALIDATOR) — the actual system prompts are several KB,
     which matters a lot on Gemma-4-E2B CPU.
  3. Sum-of-parts vs end-to-end to detect integration overhead.
  4. Happy-path turn simulation: score-0/1 (no follow-up) and score-2 (RV).

Not a replacement for Jetson testing, but enough to prove which calls
dominate and whether the total matches the field-observed 1-2 min/turn.
"""
from __future__ import annotations
import json, os, struct, sys, tempfile, time, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.stt import STTGenerator
from src.models.tts import TTSGenerator
from src.models.llm_client import llm_complete, LLMRole
from src.core.response_analyzer import (
    INIT_ASKER_SYSTEM_PROMPT_V2,
    MULTI_DIM_SYSTEM_PROMPT,
    REPHRASER_PROMPT,
)
from src.core.reflection_validation import (
    RV_REASONER_SYSTEM_PROMPT,
    RV_VALIDATOR_OARS_SYSTEM_PROMPT,
    RV_GUIDE_SYSTEM_PROMPT,
)


def _mk_wav(path: str, seconds: float = 3.0, sample_rate: int = 16000):
    n = int(seconds * sample_rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


def _time(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:42s} {dt:7.2f}s")
    return dt, out


def main():
    print("="*72)
    print(f"HOST: {os.uname().nodename}  pid={os.getpid()}")
    print("="*72)

    # ── Initialization costs ────────────────────────────────────────────────
    print("\n[INIT] loading models (one-time cost)")
    t0 = time.perf_counter(); stt = STTGenerator(); print(f"  STT init: {time.perf_counter()-t0:.2f}s")
    t0 = time.perf_counter(); tts = TTSGenerator(); print(f"  TTS init: {time.perf_counter()-t0:.2f}s")

    tmpdir = tempfile.gettempdir()
    wav_in  = os.path.join(tmpdir, "audit_silence_3s.wav"); _mk_wav(wav_in, 3.0)
    wav_out = os.path.join(tmpdir, "audit_tts_out.wav")

    # ── Warmup ──────────────────────────────────────────────────────────────
    print("\n[WARMUP] (cold-start cost NOT counted)")
    stt.transcribe(wav_in)
    # One short LLM call warms the LiteRT engine singleton.
    t0 = time.perf_counter()
    llm_complete("You are helpful.", "Say hi.", role=LLMRole.GENERAL)
    print(f"  LLM cold+warm call: {time.perf_counter()-t0:.2f}s")
    tts.generate("Hello world.", wav_out)

    # ── Individual module (warm) ────────────────────────────────────────────
    print("\n[INDIVIDUAL] warm calls, 3s silence / short prompt / short text")
    stt_dt, stt_res = _time("STT transcribe 3s silence",
                             lambda: stt.transcribe(wav_in))
    short_llm_dt, short_llm_res = _time("LLM short prompt (~50 tok sys)",
                             lambda: llm_complete(
                                 "You are a therapist. Answer in one short sentence.",
                                 "I'm feeling stressed today.",
                                 role=LLMRole.GENERAL))
    tts_dt, _ = _time("TTS generate 'Hello world.'",
                      lambda: tts.generate("Hello world.", wav_out))

    # ── Real prompt sizes used in production ────────────────────────────────
    print("\n[REAL PROMPTS] measuring the actual system prompts used per turn")
    # Report prompt sizes so the reader can see the inference floor
    sizes = {
        "REPHRASER_PROMPT":                len(REPHRASER_PROMPT),
        "INIT_ASKER_SYSTEM_PROMPT_V2":     len(INIT_ASKER_SYSTEM_PROMPT_V2),
        "MULTI_DIM_SYSTEM_PROMPT":         len(MULTI_DIM_SYSTEM_PROMPT),
        "RV_REASONER_SYSTEM_PROMPT":       len(RV_REASONER_SYSTEM_PROMPT),
        "RV_VALIDATOR_OARS_SYSTEM_PROMPT": len(RV_VALIDATOR_OARS_SYSTEM_PROMPT),
        "RV_GUIDE_SYSTEM_PROMPT":          len(RV_GUIDE_SYSTEM_PROMPT),
    }
    print("  Prompt-size table (bytes, includes examples/schema):")
    for k, v in sizes.items():
        print(f"    {k:38s} {v:6d} B  (~{v//4:4d} tokens)")

    sample_q  = "How has your mood been recently?"
    sample_a  = "It's been rough. I've been really down and sometimes I cry for no reason."
    sample_fu = "I guess it started after my partner moved out. I miss them a lot."

    # Call each role with its real prompt and a realistic payload
    dt_reph, _ = _time("REPHRASER call (real prompt)", lambda: llm_complete(
        REPHRASER_PROMPT,
        f'{{"Original Question": "{sample_q}"}}',
        role=LLMRole.REPHRASER))
    dt_analyzer, _ = _time("ANALYZER call (real prompt)", lambda: llm_complete(
        INIT_ASKER_SYSTEM_PROMPT_V2,
        f"Question: {sample_q}\nAnswer: {sample_a}",
        role=LLMRole.ANALYZER))
    dt_multidim, _ = _time("MULTI-DIM ANALYZER call (real prompt)", lambda: llm_complete(
        MULTI_DIM_SYSTEM_PROMPT,
        f"Question: {sample_q}\nAnswer: {sample_a}",
        role=LLMRole.ANALYZER))
    dt_rv_reas, _ = _time("RV_REASONER call (real prompt)", lambda: llm_complete(
        RV_REASONER_SYSTEM_PROMPT,
        f'{{"Topic": "mood", "Original Question": "{sample_q}", "Original Response": "{sample_a}", "Follow-up Response": "{sample_fu}"}}',
        role=LLMRole.RV_REASONER))
    dt_rv_val, _ = _time("RV_VALIDATOR call (real prompt)", lambda: llm_complete(
        RV_VALIDATOR_OARS_SYSTEM_PROMPT,
        f'{{"Topic": "mood", "Original Question": "{sample_q}", "Original Response": "{sample_a}", "Follow-up Response": "{sample_fu}"}}',
        role=LLMRole.RV_VALIDATOR))
    dt_rv_guide, _ = _time("RV_GUIDE call (real prompt)", lambda: llm_complete(
        RV_GUIDE_SYSTEM_PROMPT,
        f'{{"Topic": "mood", "Original Question": "{sample_q}", "Original Response": "{sample_a}", "Follow-up Response": "{sample_fu}"}}',
        role=LLMRole.RV_GUIDE))

    # ── Simulated happy-path turn (score 0/1, no RV) ────────────────────────
    print("\n[TURN SIM] score-0/1 turn: STT + REPHRASER + [speak Q] + STT + ANALYZER + MULTI-DIM + [speak next Q]")
    # We don't simulate the user/TTS here; just the *serialized LLM+STT work*
    # the handler does between "user done speaking" and "next question ready"
    t0 = time.perf_counter()
    _ = stt.transcribe(wav_in)                                      # STT for prior user turn
    _ = llm_complete(REPHRASER_PROMPT,
                     f'{{"Original Question": "{sample_q}"}}',
                     role=LLMRole.REPHRASER)                        # REPHRASER for next Q
    # After the user answers THIS question (another STT -> ANALYZER -> multidim):
    _ = stt.transcribe(wav_in)
    _ = llm_complete(INIT_ASKER_SYSTEM_PROMPT_V2,
                     f"Question: {sample_q}\nAnswer: {sample_a}",
                     role=LLMRole.ANALYZER)
    _ = llm_complete(MULTI_DIM_SYSTEM_PROMPT,
                     f"Question: {sample_q}\nAnswer: {sample_a}",
                     role=LLMRole.ANALYZER)
    _ = tts.generate("Thank you for sharing. Next, let me ask...", wav_out)
    turn_score01_dt = time.perf_counter() - t0
    print(f"  score-0/1 turn serialized work: {turn_score01_dt:.2f}s")

    # ── Simulated RV-2 turn (score 2: on-topic, single follow-up) ───────────
    print("\n[TURN SIM] score-2 turn (ON-TOPIC): ANALYZER + speak follow-up + STT + RV_REASONER + RV_VALIDATOR")
    t0 = time.perf_counter()
    _ = llm_complete(INIT_ASKER_SYSTEM_PROMPT_V2,
                     f"Question: {sample_q}\nAnswer: {sample_a}",
                     role=LLMRole.ANALYZER)
    _ = tts.generate(f"You mentioned that {sample_a[:40]}. Can you tell me more?", wav_out)
    _ = stt.transcribe(wav_in)
    _ = llm_complete(RV_REASONER_SYSTEM_PROMPT,
                     f'{{"Topic": "mood", "Original Question": "{sample_q}", "Original Response": "{sample_a}", "Follow-up Response": "{sample_fu}"}}',
                     role=LLMRole.RV_REASONER)
    _ = llm_complete(RV_VALIDATOR_OARS_SYSTEM_PROMPT,
                     f'{{"Topic": "mood", "Original Question": "{sample_q}", "Original Response": "{sample_a}", "Follow-up Response": "{sample_fu}"}}',
                     role=LLMRole.RV_VALIDATOR)
    turn_score2_dt = time.perf_counter() - t0
    print(f"  score-2 on-topic turn serialized work: {turn_score2_dt:.2f}s")

    # ── Report ──────────────────────────────────────────────────────────────
    sum_of_parts = stt_dt + short_llm_dt + tts_dt
    print("\n" + "="*72)
    print("SUMMARY (all warm)")
    print("="*72)
    print(f"  STT (3s silence)                   : {stt_dt:7.2f}s")
    print(f"  LLM (short prompt)                 : {short_llm_dt:7.2f}s")
    print(f"  TTS (3-word text)                  : {tts_dt:7.2f}s")
    print(f"  sum-of-trivial-parts               : {sum_of_parts:7.2f}s")
    print()
    print(f"  LLM REPHRASER  (real prompt)       : {dt_reph:7.2f}s")
    print(f"  LLM ANALYZER   (real prompt)       : {dt_analyzer:7.2f}s")
    print(f"  LLM MULTI-DIM  (real prompt)       : {dt_multidim:7.2f}s")
    print(f"  LLM RV_REASONER (real prompt)      : {dt_rv_reas:7.2f}s")
    print(f"  LLM RV_VALIDATOR (real prompt)     : {dt_rv_val:7.2f}s")
    print(f"  LLM RV_GUIDE (real prompt)         : {dt_rv_guide:7.2f}s")
    print()
    print(f"  Simulated score-0/1 turn work      : {turn_score01_dt:7.2f}s")
    print(f"  Simulated score-2  on-topic work   : {turn_score2_dt:7.2f}s")
    print()
    # Overhead check: the combined-pipeline test should be close to sum.
    # We reconstruct the score-0/1 sum directly so the user can see it.
    parts_01 = stt_dt + dt_reph + stt_dt + dt_analyzer + dt_multidim + tts_dt
    parts_2  = dt_analyzer + tts_dt + stt_dt + dt_rv_reas + dt_rv_val
    print(f"  Expected score-0/1 parts-sum       : {parts_01:7.2f}s  (overhead={turn_score01_dt - parts_01:+.2f}s)")
    print(f"  Expected score-2 parts-sum         : {parts_2:7.2f}s  (overhead={turn_score2_dt - parts_2:+.2f}s)")
    print()
    print("NOTE: running on WSL2 / Intel i7-1355U, NOT Jetson Orin Nano.")
    print("      Jetson CPU is ARM Cortex-A78AE @ 1.5 GHz x 6 — expect ~2-3x slower for LLM.")


if __name__ == "__main__":
    main()
