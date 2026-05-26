"""Debug single-persona run with verbose tracing."""
from __future__ import annotations

import os
import sys
import time
import threading
import queue
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dev.clinical_benchmark.personas import get_persona  # noqa: E402
from dev.clinical_benchmark.harness import run_persona  # noqa: E402

if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "P02"
    persona = get_persona(pid)
    run_dir = ROOT / "dev" / "clinical_benchmark" / "_debug"
    run_dir.mkdir(parents=True, exist_ok=True)

    def watchdog():
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            time.sleep(5)
        print("\n!!! WATCHDOG TIMEOUT — dumping threads !!!\n")
        import faulthandler
        faulthandler.dump_traceback()
        os._exit(2)

    threading.Thread(target=watchdog, daemon=True).start()

    print(f"Running persona {persona.pid} {persona.name}")
    t = run_persona(persona, run_dir=run_dir)
    print(f"Agent turns: {len(t.agent_turns)}")
    print(f"User turns:  {len(t.user_turns)}")
    print(f"Last 6 agent turns:")
    for i, a in enumerate(t.agent_turns[-6:]):
        print(f"  [-{6-i}] {a[:200]!r}")
    print(f"Last 6 user replies:")
    for i, u in enumerate(t.user_turns[-6:]):
        print(f"  [-{6-i}] {u[:200]!r}")
    print(f"Recorded scores per dim (sorted):")
    for label in sorted(t.recorded_scores.keys()):
        print(f"  {label}: {t.recorded_scores[label]}")
    print(f"Exceptions: {t.exceptions}")
