"""Low-overhead runtime resource forensics for memory-constrained deployments."""

from __future__ import annotations

import atexit
import json
import math
import os
import threading
import time
import tracemalloc
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.utils.log_util import get_logger

logger = get_logger("ResourceAudit")

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    psutil = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(raw: Optional[str], default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _safe_int(raw: Optional[str], default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _read_proc_rss_mb_fallback() -> float:
    """Best-effort Linux fallback when psutil is unavailable."""
    status_path = "/proc/self/status"
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Format: VmRSS:\t  123456 kB
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        pass
    return 0.0


class ResourceAudit:
    """Tracks memory/threads/process pressure and emits a terminal resource map."""

    def __init__(self) -> None:
        self.enabled = _env_flag("RESOURCE_AUDIT_ENABLED", True)
        self.sample_interval_sec = max(
            0.02,
            _safe_float(os.environ.get("RESOURCE_AUDIT_SAMPLE_INTERVAL_SEC"), 0.05),
        )
        self.context_warn_ratio = min(
            1.0,
            max(0.1, _safe_float(os.environ.get("RESOURCE_AUDIT_CONTEXT_WARN_RATIO"), 0.7)),
        )
        self.chars_per_token_est = max(
            1,
            _safe_int(os.environ.get("RESOURCE_AUDIT_CHARS_PER_TOKEN"), 4),
        )
        self.report_path = os.environ.get(
            "RESOURCE_AUDIT_REPORT_PATH",
            "data/logs/resource_map_latest.json",
        )

        self._lock = threading.RLock()
        self._module_records: List[Dict[str, Any]] = []
        self._zone_records: List[Dict[str, Any]] = []
        self._prompt_records: List[Dict[str, Any]] = []
        self._process_records: List[Dict[str, Any]] = []
        self._snapshots: List[Dict[str, Any]] = []
        self._resource_map_cache: Dict[str, Any] = {}

        self._process = None
        if psutil is not None:
            try:
                self._process = psutil.Process(os.getpid())
            except Exception:
                self._process = None

        if self.enabled and not tracemalloc.is_tracing():
            tracemalloc.start()

        self._baseline_snapshot = self._snapshot()
        self._baseline_snapshot["label"] = "process_baseline"
        self._snapshots.append(dict(self._baseline_snapshot))

        if self.enabled:
            logger.info(
                "[RESOURCE] baseline_rss=%.1fMB child_rss=%.1fMB threads=%s",
                self._baseline_snapshot.get("rss_mb", 0.0),
                self._baseline_snapshot.get("child_rss_mb", 0.0),
                self._baseline_snapshot.get("thread_count", 0),
            )

        atexit.register(self._emit_on_exit)

    def _snapshot(self) -> Dict[str, Any]:
        now = time.time()
        snap: Dict[str, Any] = {
            "ts": now,
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "uss_mb": 0.0,
            "child_rss_mb": 0.0,
            "thread_count": 0,
            "child_count": 0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
            "tracemalloc_current_mb": 0.0,
            "tracemalloc_peak_mb": 0.0,
        }

        if self._process is not None:
            try:
                mem = self._process.memory_info()
                snap["rss_mb"] = float(mem.rss) / (1024.0 * 1024.0)
                snap["vms_mb"] = float(mem.vms) / (1024.0 * 1024.0)
            except Exception:
                pass

            try:
                full_mem = self._process.memory_full_info()
                uss = getattr(full_mem, "uss", 0)
                snap["uss_mb"] = float(uss) / (1024.0 * 1024.0)
            except Exception:
                pass

            try:
                children = self._process.children(recursive=True)
                child_rss = 0
                for child in children:
                    try:
                        child_rss += child.memory_info().rss
                    except Exception:
                        continue
                snap["child_rss_mb"] = float(child_rss) / (1024.0 * 1024.0)
                snap["child_count"] = len(children)
            except Exception:
                pass

            try:
                snap["thread_count"] = int(self._process.num_threads())
            except Exception:
                pass

        if snap["rss_mb"] <= 0.0:
            snap["rss_mb"] = _read_proc_rss_mb_fallback()

        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                snap["ram_used_gb"] = float(vm.used) / (1024.0**3)
                snap["ram_total_gb"] = float(vm.total) / (1024.0**3)
            except Exception:
                pass

        if tracemalloc.is_tracing():
            try:
                current, peak = tracemalloc.get_traced_memory()
                snap["tracemalloc_current_mb"] = float(current) / (1024.0 * 1024.0)
                snap["tracemalloc_peak_mb"] = float(peak) / (1024.0 * 1024.0)
            except Exception:
                pass

        return snap

    def capture_point(
        self,
        label: str,
        extra: Optional[Dict[str, Any]] = None,
        *,
        log: bool = True,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        snap = self._snapshot()
        snap["label"] = label
        if extra:
            snap.update(extra)
        with self._lock:
            self._snapshots.append(dict(snap))
        if log:
            logger.info(
                "[RESOURCE] %s rss=%.1fMB child=%.1fMB threads=%s trace_cur=%.1fMB trace_peak=%.1fMB",
                label,
                snap.get("rss_mb", 0.0),
                snap.get("child_rss_mb", 0.0),
                snap.get("thread_count", 0),
                snap.get("tracemalloc_current_mb", 0.0),
                snap.get("tracemalloc_peak_mb", 0.0),
            )
        return snap

    @contextmanager
    def track_module_init(self, module_name: str) -> Generator[None, None, None]:
        if not self.enabled:
            yield
            return

        before = self._snapshot()
        start = time.monotonic()
        try:
            yield
        finally:
            after = self._snapshot()
            elapsed = time.monotonic() - start
            delta = after.get("rss_mb", 0.0) - before.get("rss_mb", 0.0)
            record = {
                "module": module_name,
                "elapsed_sec": elapsed,
                "rss_before_mb": before.get("rss_mb", 0.0),
                "rss_after_mb": after.get("rss_mb", 0.0),
                "rss_delta_mb": delta,
            }
            with self._lock:
                self._module_records.append(record)

            logger.info(
                "[RESOURCE_INIT] module=%s rss_delta=%+.1fMB rss_after=%.1fMB elapsed=%.2fs",
                module_name,
                delta,
                after.get("rss_mb", 0.0),
                elapsed,
            )

    @contextmanager
    def track_peak(
        self,
        zone_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[None, None, None]:
        if not self.enabled:
            yield
            return

        start_snapshot = self._snapshot()
        start_time = time.monotonic()
        peak_rss = float(start_snapshot.get("rss_mb", 0.0))
        peak_child_rss = float(start_snapshot.get("child_rss_mb", 0.0))
        sample_count = 1
        stop_event = threading.Event()

        def _sampler() -> None:
            nonlocal peak_rss, peak_child_rss, sample_count
            while not stop_event.wait(self.sample_interval_sec):
                snap = self._snapshot()
                peak_rss = max(peak_rss, float(snap.get("rss_mb", 0.0)))
                peak_child_rss = max(peak_child_rss, float(snap.get("child_rss_mb", 0.0)))
                sample_count += 1

        sampler_thread = threading.Thread(target=_sampler, daemon=True)
        sampler_thread.start()

        try:
            yield
        finally:
            stop_event.set()
            sampler_thread.join(timeout=1.0)

            end_snapshot = self._snapshot()
            peak_rss = max(peak_rss, float(end_snapshot.get("rss_mb", 0.0)))
            peak_child_rss = max(peak_child_rss, float(end_snapshot.get("child_rss_mb", 0.0)))
            elapsed = time.monotonic() - start_time
            spike = max(0.0, peak_rss - float(start_snapshot.get("rss_mb", 0.0)))
            total_peak = peak_rss + peak_child_rss

            record = {
                "zone": zone_name,
                "elapsed_sec": elapsed,
                "sample_count": sample_count,
                "rss_start_mb": float(start_snapshot.get("rss_mb", 0.0)),
                "rss_end_mb": float(end_snapshot.get("rss_mb", 0.0)),
                "rss_peak_mb": peak_rss,
                "child_rss_peak_mb": peak_child_rss,
                "total_peak_mb": total_peak,
                "spike_mb": spike,
            }
            if metadata:
                record["metadata"] = metadata

            with self._lock:
                self._zone_records.append(record)

            logger.info(
                "[RESOURCE_PEAK] zone=%s spike=%+.1fMB rss_peak=%.1fMB total_peak=%.1fMB elapsed=%.2fs samples=%s",
                zone_name,
                spike,
                peak_rss,
                total_peak,
                elapsed,
                sample_count,
            )

    def capture_process_inventory(self, label: str = "process_inventory") -> Dict[str, Any]:
        if not self.enabled:
            return {}

        record: Dict[str, Any] = {
            "label": label,
            "ts": time.time(),
            "process_pid": os.getpid(),
            "python_threads": threading.active_count(),
            "child_processes": 0,
            "keyword_counts": {"ollama": 0, "python": 0, "piper": 0, "ffmpeg": 0},
        }

        if self._process is not None:
            try:
                record["child_processes"] = len(self._process.children(recursive=True))
            except Exception:
                pass

        if psutil is not None:
            keywords = record["keyword_counts"]
            try:
                for proc in psutil.process_iter(attrs=["name", "cmdline"]):
                    info = proc.info or {}
                    name = str(info.get("name") or "")
                    cmdline = info.get("cmdline") or []
                    cmd0 = str(cmdline[0]).lower() if cmdline else ""
                    merged = f"{name.lower()} {cmd0}"
                    for key in keywords:
                        if key in merged:
                            keywords[key] += 1
            except Exception:
                pass

        with self._lock:
            self._process_records.append(dict(record))

        logger.info(
            "[RESOURCE_PROC] %s child_processes=%s python_threads=%s keywords=%s",
            label,
            record.get("child_processes", 0),
            record.get("python_threads", 0),
            record.get("keyword_counts", {}),
        )
        return record

    def _model_kv_shape(self, model_name: str) -> Tuple[int, int, int]:
        """
        Returns a coarse (layers, kv_heads, head_dim) tuple for KV pressure estimates.
        This is intentionally approximate for early-warning diagnostics.
        """
        m = (model_name or "").lower()
        if "llama3.2" in m and "1b" in m:
            return (16, 8, 128)
        if "llama3.2" in m and "3b" in m:
            return (28, 8, 128)
        if "gemma:2b" in m or "gemma2:2b" in m:
            return (26, 8, 128)
        if "gemma4" in m or "5.1b" in m:
            return (36, 16, 128)
        return (28, 8, 128)

    def estimate_kv_cache_mb(self, model_name: str, num_ctx: int, f16_kv: bool) -> float:
        layers, kv_heads, head_dim = self._model_kv_shape(model_name)
        bytes_per_scalar = 2.0 if f16_kv else 1.0
        kv_bytes = float(num_ctx) * 2.0 * layers * kv_heads * head_dim * bytes_per_scalar
        return kv_bytes / (1024.0 * 1024.0)

    def record_prompt_budget(
        self,
        model_name: str,
        system_content: str,
        user_content: str,
        num_ctx: int,
        num_predict: int,
        *,
        f16_kv: bool,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        merged = f"{system_content or ''}\n{user_content or ''}"
        prompt_chars = len(merged)
        prompt_tokens_est = max(1, int(math.ceil(prompt_chars / float(self.chars_per_token_est))))
        utilization = float(prompt_tokens_est) / float(max(1, num_ctx))
        kv_cache_est_mb = self.estimate_kv_cache_mb(model_name, num_ctx, f16_kv)

        growth_tokens = 0
        with self._lock:
            if self._prompt_records:
                growth_tokens = prompt_tokens_est - int(self._prompt_records[-1].get("prompt_tokens_est", 0))

        record = {
            "model": model_name,
            "prompt_chars": prompt_chars,
            "prompt_tokens_est": prompt_tokens_est,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "ctx_utilization": utilization,
            "kv_cache_est_mb": kv_cache_est_mb,
            "growth_tokens": growth_tokens,
            "f16_kv": bool(f16_kv),
            "ts": time.time(),
        }
        with self._lock:
            self._prompt_records.append(record)

        logger.info(
            "[PROMPT_BUDGET] model=%s prompt_tokens_est=%s num_ctx=%s utilization=%.1f%% kv_est=%.1fMB growth_tokens=%+d",
            model_name,
            prompt_tokens_est,
            num_ctx,
            utilization * 100.0,
            kv_cache_est_mb,
            growth_tokens,
        )

        if utilization >= self.context_warn_ratio:
            logger.warning(
                "[CONTEXT_BLOAT] Prompt utilization %.1f%% exceeds warning threshold %.1f%%. "
                "Consider reducing context pack size or lowering OLLAMA_NUM_CTX.",
                utilization * 100.0,
                self.context_warn_ratio * 100.0,
            )
        return record

    def _top_static_record(self) -> Dict[str, Any]:
        if not self._module_records:
            return {}
        return max(self._module_records, key=lambda x: float(x.get("rss_delta_mb", 0.0)))

    def _top_dynamic_record(self) -> Dict[str, Any]:
        if not self._zone_records:
            return {}
        return max(self._zone_records, key=lambda x: float(x.get("spike_mb", 0.0)))

    def build_resource_map(self) -> Dict[str, Any]:
        baseline_mb = float(self._baseline_snapshot.get("rss_mb", 0.0))
        static_mb = sum(max(0.0, float(r.get("rss_delta_mb", 0.0))) for r in self._module_records)
        top_static = self._top_static_record()

        top_dynamic = self._top_dynamic_record()
        dynamic_mb = max(0.0, float(top_dynamic.get("spike_mb", 0.0)))

        top_prompt = {}
        if self._prompt_records:
            top_prompt = max(self._prompt_records, key=lambda x: float(x.get("ctx_utilization", 0.0)))

        culprit = "Mixed memory pressure from model initialization and runtime inference."
        recommendation = "Capture additional traces with RESOURCE_AUDIT_ENABLED=1 during a full turn loop."

        top_dynamic_zone = str(top_dynamic.get("zone", "n/a"))
        top_static_module = str(top_static.get("module", "n/a"))
        prompt_util = float(top_prompt.get("ctx_utilization", 0.0))

        if "llm" in top_dynamic_zone.lower() and prompt_util >= self.context_warn_ratio:
            culprit = "Context-window growth and KV cache pressure in the LLM handshake layer."
            recommendation = (
                "Cap OLLAMA_NUM_CTX at 512, prune context-pack fields per turn, "
                "and keep f16_kv disabled on constrained devices."
            )
        elif "stt" in top_static_module.lower() or "whisper" in top_static_module.lower():
            culprit = "Static STT model load dominates resident memory before first user turn."
            recommendation = (
                "Keep Faster-Whisper at int8/base.en or smaller and avoid loading extra SER backends in parallel."
            )
        elif self._process_records:
            last_proc = self._process_records[-1]
            counts = last_proc.get("keyword_counts", {})
            ollama_count = int(counts.get("ollama", 0)) if isinstance(counts, dict) else 0
            if ollama_count > 1:
                culprit = "Potential duplicate Ollama runners/processes increasing memory pressure."
                recommendation = "Ensure a single ollama serve instance and enforce keep_alive=0 between turns."

        resource_map = {
            "baseline_ram_gb": round(baseline_mb / 1024.0, 3),
            "static_load_gb": round(static_mb / 1024.0, 3),
            "dynamic_spike_gb": round(dynamic_mb / 1024.0, 3),
            "top_static_module": top_static_module,
            "top_dynamic_zone": top_dynamic_zone,
            "max_prompt_utilization": round(prompt_util, 3),
            "culprit": culprit,
            "recommendation": recommendation,
            "samples": {
                "module_records": len(self._module_records),
                "zone_records": len(self._zone_records),
                "prompt_records": len(self._prompt_records),
                "process_records": len(self._process_records),
                "snapshots": len(self._snapshots),
            },
        }
        self._resource_map_cache = resource_map
        return resource_map

    def write_report(self, report_path: Optional[str] = None) -> str:
        path = report_path or self.report_path
        report = self.build_resource_map()
        payload = {
            "resource_map": report,
            "baseline": self._baseline_snapshot,
            "module_init": self._module_records,
            "zone_peaks": self._zone_records,
            "prompt_budget": self._prompt_records,
            "process_inventory": self._process_records,
        }
        try:
            folder = os.path.dirname(path)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return path
        except Exception as e:
            logger.error("Failed to write resource audit report to %s: %s", path, e)
            return ""

    def emit_resource_map(self) -> Dict[str, Any]:
        report = self.build_resource_map()
        baseline = report.get("baseline_ram_gb", 0.0)
        static = report.get("static_load_gb", 0.0)
        dynamic = report.get("dynamic_spike_gb", 0.0)
        top_static = report.get("top_static_module", "n/a")
        top_dynamic = report.get("top_dynamic_zone", "n/a")

        logger.info("[Baseline RAM]: %.2f GB used at process baseline", baseline)
        logger.info("[Static Load]: +%.2f GB consumed on module init (%s)", static, top_static)
        logger.info("[Dynamic Spike]: +%.2f GB during peak zone (%s)", dynamic, top_dynamic)
        logger.info("[The Culprit]: %s", report.get("culprit", "Unknown"))
        logger.info("[Refactor/CAP Recommendation]: %s", report.get("recommendation", "n/a"))

        output_path = self.write_report()
        if output_path:
            logger.info("[RESOURCE_REPORT] %s", output_path)
        return report

    def _emit_on_exit(self) -> None:
        if not self.enabled:
            return
        try:
            self.emit_resource_map()
        except Exception as e:
            logger.error("Failed to emit resource map on exit: %s", e)


_RESOURCE_AUDIT = ResourceAudit()


def get_resource_audit() -> ResourceAudit:
    return _RESOURCE_AUDIT
