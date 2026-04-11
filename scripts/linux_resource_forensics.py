#!/usr/bin/env python3
"""Collect Linux forensic evidence for memory crashes and process duplication."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from typing import Dict, List, Tuple


def _run(cmd: List[str], timeout: int = 8) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 127, "", str(e)


def _collect_command_block() -> Dict[str, Dict[str, object]]:
    commands = {
        "free_h": ["free", "-h"],
        "top_rss_processes": ["sh", "-lc", "ps -eo pid,ppid,pmem,rss,comm,args --sort=-rss | head -n 30"],
        "tracked_processes": ["sh", "-lc", "pgrep -af 'python|piper|uvicorn|main.py' || true"],
        "journal_kernel": ["journalctl", "-k", "--no-pager", "-n", "500"],
        "dmesg_tail": ["dmesg", "-T"],
    }
    out: Dict[str, Dict[str, object]] = {}
    for name, cmd in commands.items():
        code, stdout, stderr = _run(cmd)
        if name == "dmesg_tail" and stdout:
            lines = stdout.splitlines()
            stdout = "\n".join(lines[-500:])
        out[name] = {
            "returncode": code,
            "stdout": stdout,
            "stderr": stderr,
            "cmd": cmd,
        }
    return out


def _oom_evidence(text: str) -> List[str]:
    if not text:
        return []
    patterns = [
        r"out of memory",
        r"oom-killer",
        r"killed process",
        r"cudaMalloc failed",
        r"runner process has terminated",
    ]
    hits: List[str] = []
    for line in text.splitlines():
        low = line.lower()
        if any(re.search(p, low) for p in patterns):
            hits.append(line.strip())
    return hits


def _count_keyword_processes(process_block: str) -> Dict[str, int]:
    counts = {"python": 0, "piper": 0, "uvicorn": 0}
    if not process_block:
        return counts
    for line in process_block.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        executable = os.path.basename(parts[1]).lower()
        for key in list(counts.keys()):
            if key in executable:
                counts[key] += 1
    return counts


def _parse_baseline_ram(free_h_stdout: str) -> str:
    if not free_h_stdout:
        return "unknown"
    for line in free_h_stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Mem:"):
            return stripped
    return "unknown"


def _load_resource_report(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _compose_resource_map(
    baseline_ram_line: str,
    dependency_report: Dict[str, object],
    resource_report: Dict[str, object],
    process_counts: Dict[str, int],
    kernel_oom_hits: List[str],
) -> Dict[str, str]:
    resource_map_data = resource_report.get("resource_map", {}) if isinstance(resource_report, dict) else {}
    if not isinstance(resource_map_data, dict):
        resource_map_data = {}

    baseline = baseline_ram_line
    static_gb = float(resource_map_data.get("static_load_gb", 0.0) or 0.0)
    dynamic_gb = float(resource_map_data.get("dynamic_spike_gb", 0.0) or 0.0)
    top_static = str(resource_map_data.get("top_static_module", "unknown"))
    top_dynamic = str(resource_map_data.get("top_dynamic_zone", "unknown"))

    dep_risks = dependency_report.get("memory_risk_added_vs_legacy", []) if isinstance(dependency_report, dict) else []
    dep_risk_count = len(dep_risks) if isinstance(dep_risks, list) else 0

    culprit = "Memory pressure appears mixed and requires staged isolation runs."
    if kernel_oom_hits:
        culprit = "Kernel OOM evidence indicates runtime memory exhaustion during model load."
    elif top_dynamic.lower().startswith("llm") or "llm" in top_dynamic.lower():
        culprit = "LLM inference/context expansion is the highest dynamic spike region."
    elif top_static != "unknown":
        culprit = f"Static model load concentration in {top_static}."

    recommendation = "Cap LITERT_CONTEXT_LENGTH to 512, keep f16_kv disabled, and monitor in-process LLM memory."
    if dep_risk_count > 10:
        recommendation += " Also trim unused high-memory dependencies from the runtime environment."

    return {
        "Baseline RAM": baseline,
        "Static Load": f"+{static_gb:.2f} GB (top static module: {top_static})",
        "Dynamic Spike": f"+{dynamic_gb:.2f} GB (top dynamic zone: {top_dynamic})",
        "The Culprit": f"{culprit} Recommendation: {recommendation}",
    }


def _print_resource_map(resource_map: Dict[str, str]) -> None:
    print("[Baseline RAM]: " + resource_map.get("Baseline RAM", "unknown"))
    print("[Static Load]: " + resource_map.get("Static Load", "unknown"))
    print("[Dynamic Spike]: " + resource_map.get("Dynamic Spike", "unknown"))
    print("[The Culprit]: " + resource_map.get("The Culprit", "unknown"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Linux forensic collector")
    parser.add_argument(
        "--resource-report",
        default="data/logs/resource_map_latest.json",
        help="Path to runtime resource audit JSON.",
    )
    parser.add_argument(
        "--dependency-report",
        default="data/logs/dependency_delta_report.json",
        help="Path to dependency delta JSON.",
    )
    parser.add_argument(
        "--output",
        default="data/logs/linux_forensics_report.json",
        help="Output report path.",
    )
    args = parser.parse_args()

    command_data = _collect_command_block()

    kernel_text = str(command_data.get("journal_kernel", {}).get("stdout", "")) + "\n" + str(
        command_data.get("dmesg_tail", {}).get("stdout", "")
    )

    kernel_hits = _oom_evidence(kernel_text)
    process_counts = _count_keyword_processes(str(command_data.get("tracked_processes", {}).get("stdout", "")))
    baseline_line = _parse_baseline_ram(str(command_data.get("free_h", {}).get("stdout", "")))

    resource_report = _load_resource_report(args.resource_report)
    dependency_report = _load_resource_report(args.dependency_report)

    resource_map = _compose_resource_map(
        baseline_line,
        dependency_report,
        resource_report,
        process_counts,
        kernel_hits,
    )

    final_report = {
        "resource_map": resource_map,
        "process_counts": process_counts,
        "kernel_oom_hits": kernel_hits[:100],
        "commands": command_data,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("=== Linux Resource Forensics ===")
    print(f"Report written: {args.output}")
    print(f"Kernel/OOM hits: {len(kernel_hits)}")
    print(f"Process counts: {process_counts}")
    _print_resource_map(resource_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
