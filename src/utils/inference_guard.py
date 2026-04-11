"""Utilities to serialize heavy inference and clear cache between phases."""

from __future__ import annotations

import gc
import subprocess
import threading
from contextlib import contextmanager

from src.utils.log_util import get_logger

logger = get_logger("InferenceGuard")
_HEAVY_STAGE_LOCK = threading.Lock()


@contextmanager
def heavy_stage(stage_name: str):
    """Ensure only one heavy inference stage runs at a time in-process."""
    with _HEAVY_STAGE_LOCK:
        logger.debug(f"Entering heavy stage: {stage_name}")
        try:
            yield
        finally:
            logger.debug(f"Exiting heavy stage: {stage_name}")


def clear_inference_cache(reason: str = "") -> None:
    """Best-effort cache cleanup between STT/SER/LLM phases."""
    if reason:
        logger.info(f"Clear Cache command: {reason}")
    else:
        logger.info("Clear Cache command")

    gc.collect()

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        # torch is optional in this project; cache cleanup remains best-effort.
        pass


def _read_sysfs_gpu() -> str | None:
    """
    Read GPU frequency and utilization directly from sysfs on Jetson.
    These files are zero-cost reads (no subprocess), making them ideal
    for high-frequency telemetry on embedded devices.
    """
    gpu_info_parts = []

    # GPU clock frequency (MHz)
    freq_paths = [
        "/sys/class/devfreq/fb000000.gpu/device/gr3d_freq",  # Orin Nano
        "/sys/class/devfreq/17000000.ga10b/cur_freq",        # Orin NX
        "/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq",  # Xavier NX
    ]
    for fpath in freq_paths:
        try:
            with open(fpath) as f:
                freq_hz = int(f.read().strip())
                freq_mhz = freq_hz // 1_000_000 if freq_hz > 1_000_000 else freq_hz
                gpu_info_parts.append(f"GPU Freq: {freq_mhz}MHz")
                break
        except (FileNotFoundError, ValueError, PermissionError):
            continue

    # GPU utilisation (percentage)
    load_paths = [
        "/sys/devices/gpu.0/load",
        "/sys/devices/platform/gpu.0/load",
        "/sys/class/devfreq/fb000000.gpu/device/load",
    ]
    for lpath in load_paths:
        try:
            with open(lpath) as f:
                raw = int(f.read().strip())
                # Jetson reports load as 0-1000 (permille), convert to percent
                pct = raw / 10.0 if raw > 100 else float(raw)
                gpu_info_parts.append(f"GPU Load: {pct:.1f}%")
                break
        except (FileNotFoundError, ValueError, PermissionError):
            continue

    # Available shared memory (Jetson unified memory)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                    gpu_info_parts.append(f"MemAvail: {avail_kb // 1024}MB")
                    break
    except Exception:
        pass

    return " | ".join(gpu_info_parts) if gpu_info_parts else None


def get_system_memory_snapshot() -> str:
    """
    Clinical-grade memory transparency for Jetson debugging.
    Returns a multi-line string with:
      - System RAM from 'free -h'
      - GPU/VRAM from sysfs (zero-cost), nvidia-smi, or tegrastats
    All sources are best-effort — missing tools are silently skipped.
    """
    parts = []

    # ── System RAM ──────────────────────────────────────────────────────
    try:
        proc = subprocess.run(
            ["free", "-h"],
            check=False, capture_output=True, text=True, timeout=4,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            for line in proc.stdout.splitlines():
                if line.strip().startswith("Mem:"):
                    parts.append(f"RAM: {line.strip()}")
                    break
            else:
                parts.append(f"RAM: {proc.stdout.strip().splitlines()[-1]}")
    except Exception:
        parts.append("RAM: unavailable")

    # ── GPU/VRAM — priority: sysfs → nvidia-smi → tegrastats ──────────
    gpu_info = _read_sysfs_gpu()

    if gpu_info is None:
        try:
            proc = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                check=False, capture_output=True, text=True, timeout=4,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                fields = [f.strip() for f in proc.stdout.strip().split(",")]
                if len(fields) >= 4:
                    gpu_info = (
                        f"GPU: {fields[0]} | VRAM: {fields[1]}MB/{fields[2]}MB "
                        f"| Util: {fields[3]}%"
                    )
                else:
                    gpu_info = f"GPU: {proc.stdout.strip()}"
        except FileNotFoundError:
            pass
        except Exception:
            pass

    if gpu_info is None:
        try:
            proc = subprocess.run(
                ["tegrastats", "--interval", "1", "--count", "1"],
                check=False, capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                line = proc.stdout.strip().splitlines()[0]
                gpu_info = f"Tegrastats: {line}"
        except FileNotFoundError:
            pass
        except Exception:
            pass

    parts.append(gpu_info if gpu_info else "GPU: no sysfs, nvidia-smi, or tegrastats")

    return " | ".join(parts)
