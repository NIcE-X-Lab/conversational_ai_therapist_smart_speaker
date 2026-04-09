#!/usr/bin/env python3
"""Compare current installed packages against legacy/current environment manifests."""

from __future__ import annotations

import argparse
import json
import os
import re
from importlib import metadata
from typing import Dict, List, Set, Tuple

import yaml


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _extract_name(dep: str) -> str:
    # Handles forms like "numpy>=1.21", "flask=3.1.2", "python-dotenv"
    token = re.split(r"[<>=!~\s]", dep.strip(), maxsplit=1)[0]
    return _normalize_name(token)


def _load_env_packages(path: str) -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    deps = data.get("dependencies", [])
    out: Set[str] = set()
    for entry in deps:
        if isinstance(entry, str):
            out.add(_extract_name(entry))
        elif isinstance(entry, dict):
            pip_list = entry.get("pip", [])
            for pip_dep in pip_list:
                out.add(_extract_name(str(pip_dep)))
    return {x for x in out if x}


def _installed_packages() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary") or dist.name
        if not name:
            continue
        norm = _normalize_name(str(name))
        out[norm] = dist.version or "unknown"
    return out


def _is_memory_risk(pkg: str) -> bool:
    risk_tokens = (
        "torch",
        "tensorflow",
        "transformers",
        "ctranslate2",
        "faster-whisper",
        "speechbrain",
        "onnx",
        "llama",
        "cuda",
        "nvidia",
        "triton",
        "mlc",
        "huggingface",
        "mlx",
        "bitsandbytes",
        "openai",
    )
    return any(tok in pkg for tok in risk_tokens)


def _build_report(
    installed: Dict[str, str],
    legacy_set: Set[str],
    current_manifest_set: Set[str],
) -> Dict[str, object]:
    installed_set = set(installed.keys())

    added_vs_legacy = sorted(installed_set - legacy_set)
    removed_vs_legacy = sorted(legacy_set - installed_set)
    missing_from_installed_vs_current_manifest = sorted(current_manifest_set - installed_set)
    extra_vs_current_manifest = sorted(installed_set - current_manifest_set)

    memory_risk_added = [p for p in added_vs_legacy if _is_memory_risk(p)]
    memory_risk_removed = [p for p in removed_vs_legacy if _is_memory_risk(p)]
    memory_risk_extra = [p for p in extra_vs_current_manifest if _is_memory_risk(p)]

    return {
        "installed_count": len(installed_set),
        "legacy_manifest_count": len(legacy_set),
        "current_manifest_count": len(current_manifest_set),
        "added_vs_legacy": [{"name": p, "version": installed.get(p, "unknown")} for p in added_vs_legacy],
        "removed_vs_legacy": removed_vs_legacy,
        "missing_from_installed_vs_current_manifest": missing_from_installed_vs_current_manifest,
        "extra_vs_current_manifest": [
            {"name": p, "version": installed.get(p, "unknown")} for p in extra_vs_current_manifest
        ],
        "memory_risk_added_vs_legacy": [
            {"name": p, "version": installed.get(p, "unknown")} for p in memory_risk_added
        ],
        "memory_risk_removed_vs_legacy": memory_risk_removed,
        "memory_risk_extra_vs_current_manifest": [
            {"name": p, "version": installed.get(p, "unknown")} for p in memory_risk_extra
        ],
    }


def _print_human_summary(report: Dict[str, object]) -> None:
    print("=== Dependency Delta Audit ===")
    print(f"Installed packages: {report['installed_count']}")
    print(f"Legacy manifest packages: {report['legacy_manifest_count']}")
    print(f"Current manifest packages: {report['current_manifest_count']}")

    added = report.get("added_vs_legacy", [])
    removed = report.get("removed_vs_legacy", [])
    missing = report.get("missing_from_installed_vs_current_manifest", [])
    risks = report.get("memory_risk_added_vs_legacy", [])

    print(f"Added vs legacy: {len(added)}")
    print(f"Removed vs legacy: {len(removed)}")
    print(f"Missing from installed vs current manifest: {len(missing)}")
    print(f"Memory-risk additions vs legacy: {len(risks)}")

    if risks:
        print("Top memory-risk additions:")
        for row in risks[:15]:
            print(f"- {row['name']} ({row['version']})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dependency delta audit")
    parser.add_argument(
        "--legacy-env",
        default="environment_upgradable.yml",
        help="Path to baseline legacy environment file.",
    )
    parser.add_argument(
        "--current-env",
        default="environment.yml",
        help="Path to current environment manifest.",
    )
    parser.add_argument(
        "--output",
        default="data/logs/dependency_delta_report.json",
        help="Path to write JSON report.",
    )
    args = parser.parse_args()

    installed = _installed_packages()
    legacy_set = _load_env_packages(args.legacy_env)
    current_manifest_set = _load_env_packages(args.current_env)
    report = _build_report(installed, legacy_set, current_manifest_set)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_human_summary(report)
    print(f"Report written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
