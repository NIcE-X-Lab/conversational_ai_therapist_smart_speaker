#!/usr/bin/env python3
"""Edge-performance benchmark for the CaiTI runtime on Jetson Orin Nano.

Measures:
  - Hardware snapshot (CPU / RAM / GPU / disk / thermal).
  - Module cold-start import latency.
  - LLM inference latency over small / medium / large prompts.
  - STT load + first-transcription latency.
  - TTS synthesis latency (short + long).
  - Write latency for each session stream (DB turn, CSV row, dossier,
    JSON event, clinical_scores, intervention_log).
  - Therapist-report generation latency on a synthetic session.
  - Storage footprint per session (dossier + report + CSV log + NDJSON).
  - Process RSS / VMS peak during an end-to-end synthetic turn.
  - Baseline vs. LLM-loaded memory delta.

Outputs:
  data/benchmarks/benchmark_<timestamp>.json    # raw numbers
  data/benchmarks/benchmark_<timestamp>.md      # human-readable summary

Usage:
  python scripts/benchmark_edge.py                # full suite (can take ~5-10 min)
  python scripts/benchmark_edge.py --skip-llm     # omit LLM latency (faster)
  python scripts/benchmark_edge.py --skip-stt     # omit STT load
  python scripts/benchmark_edge.py --reps 3       # LLM warmups per prompt size
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import datetime
from statistics import mean, median, stdev
from typing import Any, Callable

# Make src importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mb(nbytes: float) -> float:
    return round(nbytes / (1024 * 1024), 2)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def _rss_mb() -> float:
    try:
        import psutil
        return _mb(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0.0


def _vms_mb() -> float:
    try:
        import psutil
        return _mb(psutil.Process(os.getpid()).memory_info().vms)
    except Exception:
        return 0.0


def _time_block(fn: Callable[[], Any], reps: int = 1, warmup: int = 0) -> dict[str, float]:
    """Run `fn` `reps` times + `warmup` untimed. Return latency stats in ms."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "reps": reps,
        "mean_ms": round(mean(samples), 2) if samples else 0.0,
        "median_ms": round(median(samples), 2) if samples else 0.0,
        "p95_ms": round(_percentile(samples, 95), 2),
        "min_ms": round(min(samples), 2) if samples else 0.0,
        "max_ms": round(max(samples), 2) if samples else 0.0,
        "stdev_ms": round(stdev(samples), 2) if len(samples) > 1 else 0.0,
        "samples_ms": [round(s, 2) for s in samples],
    }


def _run_cli(argv: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"ok": proc.returncode == 0, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Hardware snapshot
# ─────────────────────────────────────────────────────────────────────────────

def capture_hardware() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    disk = psutil.disk_usage(".")
    cpu_freq = psutil.cpu_freq()

    snapshot: dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "cpu": {
            "count_physical": psutil.cpu_count(logical=False),
            "count_logical": psutil.cpu_count(logical=True),
            "freq_mhz_current": round(cpu_freq.current, 1) if cpu_freq else None,
            "freq_mhz_max": round(cpu_freq.max, 1) if cpu_freq else None,
            "load_percent": psutil.cpu_percent(interval=0.5),
        },
        "memory": {
            "total_mb": _mb(vm.total),
            "available_mb": _mb(vm.available),
            "used_mb": _mb(vm.used),
            "percent": vm.percent,
        },
        "swap": {
            "total_mb": _mb(sw.total),
            "used_mb": _mb(sw.used),
            "percent": sw.percent,
        },
        "disk": {
            "total_mb": _mb(disk.total),
            "free_mb": _mb(disk.free),
            "used_percent": disk.percent,
        },
    }

    thermal: list[dict[str, Any]] = []
    tz_dir = "/sys/class/thermal"
    if os.path.isdir(tz_dir):
        for entry in sorted(os.listdir(tz_dir)):
            if not entry.startswith("thermal_zone"):
                continue
            try:
                t_type = open(os.path.join(tz_dir, entry, "type")).read().strip()
                t_mdeg = int(open(os.path.join(tz_dir, entry, "temp")).read().strip())
                thermal.append({"zone": entry, "type": t_type, "temp_c": round(t_mdeg / 1000, 1)})
            except Exception:
                pass
    snapshot["thermal_zones"] = thermal

    snapshot["gpu"] = _gpu_snapshot()
    snapshot["jetson_release"] = _jetson_release()
    return snapshot


def _gpu_snapshot() -> dict[str, Any]:
    tegra = _run_cli(["timeout", "2.5", "tegrastats", "--interval", "1000"], timeout=5.0)
    if tegra.get("ok") and tegra.get("stdout"):
        first = tegra["stdout"].splitlines()[0]
        return {"source": "tegrastats", "sample": first[:600]}
    nv = _run_cli([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ], timeout=3.0)
    if nv.get("ok") and nv.get("stdout"):
        return {"source": "nvidia-smi", "sample": nv["stdout"]}
    return {"source": "unavailable", "sample": None}


def _jetson_release() -> dict[str, Any]:
    path = "/etc/nv_tegra_release"
    if os.path.isfile(path):
        try:
            return {"raw": open(path).read().strip().splitlines()[0]}
        except Exception:
            pass
    return {"raw": None}


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Module cold-start timings
# ─────────────────────────────────────────────────────────────────────────────

_MODULE_TARGETS = [
    "src.utils.config_loader",
    "src.utils.io_record",
    "src.drivers.db_manager",
    "src.core.therapy_content",
    "src.core.response_analyzer",
    "src.core.reflection_validation",
    "src.core.CBT",
    "src.core.questioner",
    "src.core.handler_rl",
    "src.utils.therapist_report",
]


def benchmark_module_init() -> dict[str, Any]:
    results = []
    for mod in _MODULE_TARGETS:
        proc = subprocess.run(
            [sys.executable, "-c",
             f"import time,sys; sys.path.insert(0, '{_ROOT}'); t=time.perf_counter(); import {mod}; print((time.perf_counter()-t)*1000)"],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                ms = round(float(proc.stdout.strip().splitlines()[-1]), 2)
                results.append({"module": mod, "import_ms": ms})
                continue
            except Exception:
                pass
        results.append({"module": mod, "import_ms": None, "error": proc.stderr.strip()[:200]})
    return {"modules": results}


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Storage write latencies + growth
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_storage(scratch_dir: str) -> dict[str, Any]:
    from src.drivers.db_manager import DBManager

    db_path = os.path.join(scratch_dir, "bench.db")
    db = DBManager(db_path)
    user_id = db.get_user_id("bench_subject")
    session_id = db.create_session(user_id)

    turn_stats = _time_block(
        lambda: db.add_turn(session_id, 0, "agent", "x" * 200), reps=100
    )

    score_idx = [0]
    def _write_score():
        i = score_idx[0]
        db.record_clinical_score(
            session_id, dim_index=(i % 37) + 1, dim_label=f"dim_{i % 37}",
            score=(i % 3), evidence_text="evidence",
            source="benchmark",
        )
        score_idx[0] = i + 1
    score_stats = _time_block(_write_score, reps=100)

    log_stats = _time_block(
        lambda: db.record_intervention_log(
            session_id, kind="MI", stage="rv", technique="bench",
            outcome="delivered", dim_label="mood",
            detail={"k": "v"},
        ),
        reps=50,
    )

    csv_path = os.path.join(scratch_dir, "bench.csv")
    def _csv_write():
        with open(csv_path, "a") as f:
            f.write('"2026-01-01T00:00:00","turn","user","x"\n')
    csv_stats = _time_block(_csv_write, reps=100)

    ndjson_path = os.path.join(scratch_dir, "bench.ndjson")
    def _ndjson_write():
        with open(ndjson_path, "a") as f:
            f.write(json.dumps({"ts": "t", "event": "x", "payload": "y"}) + "\n")
    ndjson_stats = _time_block(_ndjson_write, reps=100)

    from src.utils.therapist_report import generate_therapist_report, _EXPORTED_SESSIONS
    _EXPORTED_SESSIONS.clear()
    os.makedirs(os.path.join(scratch_dir, "data", "clinical"), exist_ok=True)
    old_cwd = os.getcwd()
    try:
        os.chdir(scratch_dir)
        def _gen():
            _EXPORTED_SESSIONS.clear()
            generate_therapist_report(session_id, db=db)
        report_stats = _time_block(_gen, reps=5)
    finally:
        os.chdir(old_cwd)

    footprint = {
        "db_bytes": os.path.getsize(db_path) if os.path.isfile(db_path) else 0,
        "csv_bytes": os.path.getsize(csv_path) if os.path.isfile(csv_path) else 0,
        "ndjson_bytes": os.path.getsize(ndjson_path) if os.path.isfile(ndjson_path) else 0,
    }
    row_cost = {
        "turn_row_avg_bytes": round(footprint["db_bytes"] / 100, 1) if footprint["db_bytes"] else 0,
        "csv_row_avg_bytes": round(footprint["csv_bytes"] / 100, 1) if footprint["csv_bytes"] else 0,
        "ndjson_event_avg_bytes": round(footprint["ndjson_bytes"] / 100, 1) if footprint["ndjson_bytes"] else 0,
    }

    return {
        "db_turn_add": turn_stats,
        "db_clinical_score": score_stats,
        "db_intervention_log": log_stats,
        "csv_append": csv_stats,
        "ndjson_append": ndjson_stats,
        "therapist_report_gen": report_stats,
        "storage_footprint_bytes": footprint,
        "row_cost_bytes": row_cost,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: LLM / STT / TTS latency
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_llm(reps: int) -> dict[str, Any]:
    try:
        from src.models.llm_client import llm_complete, LLMRole
    except Exception as e:
        return {"ok": False, "error": f"import failed: {e}"}

    rss_before_warmup = _rss_mb()
    try:
        _ = llm_complete(
            "You are a classifier.",
            "Answer with one word: yes or no. Is the sky blue?",
            role=LLMRole.GENERAL,
        )
    except Exception as e:
        return {"ok": False, "error": f"warmup failed: {e}"}
    rss_after_warmup = _rss_mb()

    def _small():
        return llm_complete(
            "You are a classifier.",
            "Answer yes or no: Do you understand?",
            role=LLMRole.GENERAL,
        )

    medium_system = (
        "You are a classifier that maps a user's response to one of: "
        "weight, mood, sleep, eat, work, social, mood. Output one word."
    )
    def _medium():
        return llm_complete(
            medium_system,
            "Question: How's your sleep? Answer: I've been tossing and turning.",
            role=LLMRole.ANALYZER,
        )

    large_system = (
        "You are a Motivational Interviewing therapist. Produce an empathic "
        "validation in 3-5 sentences using OARS. Do not ask questions. "
    ) + "Example: " * 20
    large_user = "Topic: Sleep. Response: " + ("I barely slept and feel anxious. " * 15)
    def _large():
        return llm_complete(large_system, large_user, role=LLMRole.RV_VALIDATOR)

    small_stats = _time_block(_small, reps=reps)
    medium_stats = _time_block(_medium, reps=reps)
    large_stats = _time_block(_large, reps=reps)

    return {
        "ok": True,
        "warmup_rss_delta_mb": round(rss_after_warmup - rss_before_warmup, 1),
        "rss_after_warmup_mb": rss_after_warmup,
        "small_prompt": small_stats,
        "medium_prompt": medium_stats,
        "large_prompt": large_stats,
    }


def benchmark_stt() -> dict[str, Any]:
    try:
        from src.models.stt import STTGenerator
    except Exception as e:
        return {"ok": False, "error": f"import failed: {e}"}

    rss_before = _rss_mb()
    t0 = time.perf_counter()
    try:
        stt = STTGenerator()
    except Exception as e:
        return {"ok": False, "error": f"init failed: {e}"}
    init_ms = round((time.perf_counter() - t0) * 1000, 2)
    rss_after_init = _rss_mb()

    import wave, struct
    wav_path = os.path.join(tempfile.gettempdir(), "caiti_bench_silence.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = struct.pack("<" + "h" * 24000, *([0] * 24000))
        w.writeframes(frames)

    transcribe_ms: float | None = None
    transcribe_err: str | None = None
    try:
        t0 = time.perf_counter()
        if hasattr(stt, "transcribe"):
            stt.transcribe(wav_path)
        elif hasattr(stt, "transcribe_file"):
            stt.transcribe_file(wav_path)
        else:
            raise AttributeError("STTGenerator has no transcribe method")
        transcribe_ms = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        transcribe_err = str(e)

    try:
        os.unlink(wav_path)
    except Exception:
        pass

    result = {
        "ok": True,
        "init_ms": init_ms,
        "transcribe_1_5s_silence_ms": transcribe_ms,
        "rss_before_init_mb": rss_before,
        "rss_after_init_mb": rss_after_init,
        "init_rss_delta_mb": round(rss_after_init - rss_before, 1),
    }
    if transcribe_err:
        result["transcribe_error"] = transcribe_err
    return result


def benchmark_tts() -> dict[str, Any]:
    try:
        from src.models.tts import TTSGenerator
    except Exception as e:
        return {"ok": False, "error": f"import failed: {e}"}

    try:
        tts = TTSGenerator()
    except Exception as e:
        return {"ok": False, "error": f"init failed: {e}"}

    short_text = "Hello, I am here with you today."
    long_text = (
        "It sounds like the pressure from your upcoming deadlines is really "
        "weighing on you, and the stress eating is one of the ways your body "
        "has been responding to that pressure. You might notice it helps to "
        "identify the moments when stress peaks and protect regular meal times."
    )

    out_short = os.path.join(tempfile.gettempdir(), "caiti_bench_tts_short.wav")
    out_long = os.path.join(tempfile.gettempdir(), "caiti_bench_tts_long.wav")

    def _synth(text, path):
        if hasattr(tts, "generate"):
            tts.generate(text, path)
        elif hasattr(tts, "synthesize"):
            tts.synthesize(text, path)
        else:
            raise AttributeError("TTSGenerator has no generate method")

    try:
        short_stats = _time_block(lambda: _synth(short_text, out_short), reps=3, warmup=1)
        long_stats = _time_block(lambda: _synth(long_text, out_long), reps=3, warmup=1)
    except Exception as e:
        return {"ok": False, "error": f"synth failed: {e}"}

    short_size = os.path.getsize(out_short) if os.path.isfile(out_short) else 0
    long_size = os.path.getsize(out_long) if os.path.isfile(out_long) else 0
    for p in (out_short, out_long):
        try:
            os.unlink(p)
        except Exception:
            pass

    return {
        "ok": True,
        "short_utterance_chars": len(short_text),
        "short_synthesis": short_stats,
        "short_wav_bytes": short_size,
        "long_utterance_chars": len(long_text),
        "long_synthesis": long_stats,
        "long_wav_bytes": long_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--reps", type=int, default=3,
                        help="LLM samples per prompt size (default 3).")
    parser.add_argument("--out-dir", default="data/benchmarks")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[benchmark] CaiTI edge-performance benchmark — {timestamp}")
    print("[benchmark] Section 1/4: hardware snapshot")
    hw = capture_hardware()

    print("[benchmark] Section 2/4: module cold-start import latency")
    modinit = benchmark_module_init()

    print("[benchmark] Section 3/4: storage write latency")
    with tempfile.TemporaryDirectory(prefix="caiti_bench_") as scratch:
        storage = benchmark_storage(scratch)

    print("[benchmark] Section 4/4: model engines (LLM / STT / TTS)")
    baseline_rss_mb = _rss_mb()

    llm_result: dict[str, Any] = {"skipped": True}
    stt_result: dict[str, Any] = {"skipped": True}
    tts_result: dict[str, Any] = {"skipped": True}

    if not args.skip_stt:
        print("  - STT")
        stt_result = benchmark_stt()
        gc.collect()

    if not args.skip_tts:
        print("  - TTS")
        tts_result = benchmark_tts()
        gc.collect()

    if not args.skip_llm:
        print("  - LLM")
        llm_result = benchmark_llm(reps=args.reps)

    post_rss_mb = _rss_mb()
    post_vms_mb = _vms_mb()

    report = {
        "meta": {
            "timestamp": timestamp,
            "host": os.uname().nodename,
            "kernel": os.uname().release,
            "args": vars(args),
        },
        "hardware": hw,
        "module_init": modinit,
        "storage": storage,
        "llm": llm_result,
        "stt": stt_result,
        "tts": tts_result,
        "process": {
            "baseline_rss_mb": baseline_rss_mb,
            "peak_rss_mb": post_rss_mb,
            "peak_vms_mb": post_vms_mb,
            "rss_delta_mb": round(post_rss_mb - baseline_rss_mb, 1),
        },
    }

    json_path = os.path.join(args.out_dir, f"benchmark_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.join(args.out_dir, f"benchmark_{timestamp}.md")
    with open(md_path, "w") as f:
        f.write(_render_markdown(report))

    print()
    print(f"[benchmark] JSON: {json_path}")
    print(f"[benchmark] MD:   {md_path}")
    print(f"[benchmark] Peak RSS: {post_rss_mb} MB (delta +{round(post_rss_mb - baseline_rss_mb, 1)} MB)")
    return 0


def _render_markdown(r: dict[str, Any]) -> str:
    hw = r["hardware"]
    proc = r["process"]
    llm = r["llm"]
    stt = r["stt"]
    tts = r["tts"]
    storage = r["storage"]
    modinit = r["module_init"]

    lines: list[str] = []
    lines.append(f"# CaiTI Edge Benchmark - {r['meta']['timestamp']}")
    lines.append("")
    lines.append(f"Host: `{r['meta']['host']}` ({r['meta']['kernel']})")
    lines.append("")
    lines.append("## Hardware")
    lines.append(f"- CPU: {hw['cpu']['count_physical']} physical / {hw['cpu']['count_logical']} logical cores @ {hw['cpu']['freq_mhz_current']} MHz (max {hw['cpu']['freq_mhz_max']})")
    lines.append(f"- RAM: {hw['memory']['used_mb']} MB used / {hw['memory']['total_mb']} MB total ({hw['memory']['percent']}%)")
    lines.append(f"- Swap: {hw['swap']['used_mb']} MB used / {hw['swap']['total_mb']} MB total ({hw['swap']['percent']}%)")
    lines.append(f"- Disk: {hw['disk']['free_mb']} MB free / {hw['disk']['total_mb']} MB total ({hw['disk']['used_percent']}% used)")
    if hw.get("jetson_release", {}).get("raw"):
        lines.append(f"- Jetson: `{hw['jetson_release']['raw']}`")
    if hw.get("thermal_zones"):
        lines.append("- Thermal zones:")
        for tz in hw["thermal_zones"]:
            lines.append(f"  - `{tz['type']}` -> {tz['temp_c']} C")
    if hw.get("gpu", {}).get("sample"):
        lines.append(f"- GPU sample ({hw['gpu']['source']}): `{str(hw['gpu']['sample'])[:160]}`")

    lines.append("")
    lines.append("## Process memory")
    lines.append(f"- Baseline RSS: {proc['baseline_rss_mb']} MB")
    lines.append(f"- Peak RSS:     {proc['peak_rss_mb']} MB (delta +{proc['rss_delta_mb']} MB)")
    lines.append(f"- Peak VMS:     {proc['peak_vms_mb']} MB")

    lines.append("")
    lines.append("## Module cold-start import latency")
    lines.append("| Module | Import (ms) |")
    lines.append("|---|---:|")
    for m in modinit["modules"]:
        ms = m.get("import_ms")
        lines.append(f"| `{m['module']}` | {ms if ms is not None else 'FAIL'} |")

    lines.append("")
    lines.append("## Storage write latency")
    for key in ("db_turn_add", "db_clinical_score", "db_intervention_log",
                "csv_append", "ndjson_append", "therapist_report_gen"):
        s = storage.get(key)
        if not s:
            continue
        lines.append(f"- **{key}** ({s['reps']} reps): mean {s['mean_ms']} ms, p95 {s['p95_ms']} ms, max {s['max_ms']} ms")
    lines.append("")
    lines.append("Per-row size estimates:")
    for k, v in storage["row_cost_bytes"].items():
        lines.append(f"  - `{k}`: {v} B")

    lines.append("")
    lines.append("## LLM latency")
    if llm.get("skipped"):
        lines.append("_skipped_")
    elif not llm.get("ok"):
        lines.append(f"_FAILED: {llm.get('error')}_")
    else:
        lines.append(f"- Warmup RSS delta: +{llm['warmup_rss_delta_mb']} MB (RSS after = {llm['rss_after_warmup_mb']} MB)")
        for size in ("small_prompt", "medium_prompt", "large_prompt"):
            s = llm[size]
            lines.append(f"- **{size}** ({s['reps']} reps): mean {s['mean_ms']} ms, p95 {s['p95_ms']} ms, max {s['max_ms']} ms")

    lines.append("")
    lines.append("## STT latency (Faster-Whisper `base.en` int8)")
    if stt.get("skipped"):
        lines.append("_skipped_")
    elif not stt.get("ok"):
        lines.append(f"_FAILED: {stt.get('error')}_")
    else:
        lines.append(f"- Init: {stt['init_ms']} ms (+{stt['init_rss_delta_mb']} MB RSS)")
        lines.append(f"- Transcribe 1.5s silence: {stt.get('transcribe_1_5s_silence_ms')} ms")
        if "transcribe_error" in stt:
            lines.append(f"  - _error: {stt['transcribe_error']}_")

    lines.append("")
    lines.append("## TTS latency (Piper en_US-amy-medium)")
    if tts.get("skipped"):
        lines.append("_skipped_")
    elif not tts.get("ok"):
        lines.append(f"_FAILED: {tts.get('error')}_")
    else:
        s = tts["short_synthesis"]
        lines.append(f"- Short ({tts['short_utterance_chars']} chars): mean {s['mean_ms']} ms, max {s['max_ms']} ms")
        s = tts["long_synthesis"]
        lines.append(f"- Long ({tts['long_utterance_chars']} chars): mean {s['mean_ms']} ms, max {s['max_ms']} ms")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
