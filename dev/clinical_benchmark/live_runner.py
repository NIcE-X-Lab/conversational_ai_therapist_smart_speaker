"""Live-LLM persona runner — real Jetson Gemma engine, Claude-as-user.

Architecture
------------
This is a long-running process that:

  1. Starts HandlerRL.run() in a background thread.
  2. Watches OUTPUT_QUEUE → appends every agent turn to live_session.log.
  3. Blocks on INPUT_QUEUE; replies arrive via a "reply file" the user
     (Claude) writes between turns.
  4. After the handler exits, dumps a JSON telemetry blob.

The reply protocol
------------------
Claude drives this from the laptop side via SSH "one-turn" RPC commands.
The runner exposes a tiny file-based protocol in `data/bench_runs/<run_id>/`:

  • live_session.log        — newline-delimited agent + user turns,
                               prefixed with [AGENT] / [USER] / [SYS].
  • pending_reply.txt       — Claude writes a reply here. The runner
                               drains it and forwards to INPUT_QUEUE.
  • runner_state.json       — runner heartbeat (phase, n_agent_turns,
                               n_user_turns, last_seen_agent_idx).
  • telemetry.json          — final dump on exit.

Audio is fully stubbed:
  • src.services.speech_service is NOT instantiated.
  • TTS/STT/GPIO are not loaded.

LLM is REAL:
  • src.models.llm_client.llm_complete is NOT patched.  Calls hit the
    on-device Gemma engine via litert_lm.  Each call's role + wall-clock
    is recorded into telemetry.

DB is REAL:
  • init_record creates a real session row keyed by a bench subject id
    (e.g. bench_PA_<timestamp>) so production DB rows are clearly tagged.
"""
from __future__ import annotations

import datetime
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

# Repo root must already be the cwd; the runner is launched from there.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ── Config ──────────────────────────────────────────────────────────────
DEFAULT_RUN_DIR = ROOT / "data" / "bench_runs"
TURN_TIMEOUT_S = 600.0          # max wait for the user to write a reply
HARDER_TIMEOUT_S = 1800.0       # absolute upper bound for the whole session


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _utcstamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Telemetry collector ────────────────────────────────────────────────
class LiveTelemetry:
    def __init__(self, persona_id: str, persona_name: str, profile: str, run_dir: Path):
        self.pid = persona_id
        self.name = persona_name
        self.profile = profile
        self.run_dir = run_dir
        self.started_at = time.monotonic()
        self.ended_at = 0.0
        self.agent_turns: list[str] = []
        self.user_turns: list[str] = []
        self.llm_calls: list[dict] = []   # {role, latency_s, prompt_len, response_len, error?}
        self.exceptions: list[str] = []
        self.recorded_scores: dict[str, list[int]] = {}
        self.rv_decisions: list[str] = []
        self.cbt_dim: str | None = None
        self.cbt_outcomes: list[str] = []
        self.crisis_triggered = False
        self.crisis_dim: str | None = None
        self.completed = False

    def llm_call_summary(self) -> dict:
        by_role: dict[str, dict] = {}
        for c in self.llm_calls:
            r = by_role.setdefault(c["role"], {"n": 0, "total_s": 0.0, "errors": 0})
            r["n"] += 1
            r["total_s"] += c["latency_s"]
            if c.get("error"):
                r["errors"] += 1
        for r in by_role.values():
            r["avg_s"] = round(r["total_s"] / r["n"], 2) if r["n"] else 0.0
            r["total_s"] = round(r["total_s"], 2)
        return by_role

    def as_dict(self) -> dict:
        return {
            "pid": self.pid,
            "name": self.name,
            "profile": self.profile,
            "started_at": _now_iso(),
            "duration_s": round(self.ended_at - self.started_at, 2),
            "agent_turns_count": len(self.agent_turns),
            "user_turns_count": len(self.user_turns),
            "agent_turns": self.agent_turns,
            "user_turns": self.user_turns,
            "llm_call_summary": self.llm_call_summary(),
            "llm_calls": self.llm_calls,
            "rv_decisions": self.rv_decisions,
            "cbt_dim": self.cbt_dim,
            "cbt_outcomes": self.cbt_outcomes,
            "crisis_triggered": self.crisis_triggered,
            "crisis_dim": self.crisis_dim,
            "recorded_scores": {k: v for k, v in self.recorded_scores.items()},
            "exceptions": self.exceptions,
            "completed": self.completed,
        }


# ── File protocol primitives ────────────────────────────────────────────
def _append_session_log(run_dir: Path, line: str):
    log = run_dir / "live_session.log"
    with log.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _write_state(run_dir: Path, payload: dict):
    (run_dir / "runner_state.json").write_text(json.dumps(payload, indent=2))


def _read_pending_reply(run_dir: Path) -> str | None:
    p = run_dir / "pending_reply.txt"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8").rstrip("\n")
    p.unlink()
    return txt


# ── LLM telemetry shim ─────────────────────────────────────────────────
def _install_llm_telemetry(telemetry: LiveTelemetry):
    """Wrap llm_complete so every real call gets timed + recorded.

    Does NOT replace the call — the wrapped function still reaches the
    real LiteRT engine on the Jetson.
    """
    import src.models.llm_client as llm_mod

    real_call = llm_mod.llm_complete

    def wrapped(system_content: str, user_content: str, role=None, **kwargs):
        role_value = role.value if hasattr(role, "value") else str(role or "general")
        started = time.monotonic()
        err = None
        result = ""
        try:
            result = real_call(system_content, user_content, role=role, **kwargs)
            return result
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            elapsed = time.monotonic() - started
            telemetry.llm_calls.append({
                "role": role_value,
                "latency_s": round(elapsed, 2),
                "prompt_chars": len(str(system_content) or "") + len(str(user_content) or ""),
                "response_chars": len(str(result) or ""),
                "error": err,
                "ts": _now_iso(),
            })

    # Monkey-patch every module that imported llm_complete by name.
    # (Same set the previous benchmark used.)
    import importlib
    for mod_path in (
        "src.models.llm_client",
        "src.core.CBT",
        "src.core.reflection_validation",
        "src.core.response_analyzer",
        "src.core.questioner",
        "src.core.handler_rl",
        "src.utils.text_generators",
        "src.services.response_bridge",
    ):
        try:
            m = importlib.import_module(mod_path)
            if hasattr(m, "llm_complete"):
                setattr(m, "llm_complete", wrapped)
        except Exception:
            pass


def _install_score_telemetry(telemetry: LiveTelemetry):
    """Hook the questioner._record_clinical_score so we get per-dim scores."""
    import src.core.questioner as q

    real_fn = q._record_clinical_score

    def wrapped(question_lib, i_key, score, *, evidence_text=None, source="response_analyzer"):
        try:
            entry = question_lib.get(i_key, {}).get("1", {})
            label = str(entry.get("label", ""))
            telemetry.recorded_scores.setdefault(label, []).append(int(score))
        except Exception:
            pass
        return real_fn(question_lib, i_key, score, evidence_text=evidence_text, source=source)

    q._record_clinical_score = wrapped


def _install_rv_telemetry(telemetry: LiveTelemetry):
    """Capture rv_consolidated decision tokens."""
    import src.core.reflection_validation as rv

    real = rv.rv_consolidated

    def wrapped(topic, original_question, original_response, follow_up_response):
        out = real(topic, original_question, original_response, follow_up_response)
        try:
            telemetry.rv_decisions.append(str(out[0]))
        except Exception:
            pass
        return out

    rv.rv_consolidated = wrapped


# ── Live user-thread (file-based) ──────────────────────────────────────
def _live_user_thread(
    run_dir: Path,
    telemetry: LiveTelemetry,
    output_queue,
    input_queue,
    end_session_event,
    stop_event: threading.Event,
):
    """Drain OUTPUT_QUEUE; for each agent turn, wait for the user's reply.

    Blocks on `pending_reply.txt`. Writes [AGENT]/[USER]/[SYS] lines to
    the session log so the user (Claude) can reconstruct context by
    reading the log file.
    """
    pending_reply_path = run_dir / "pending_reply.txt"

    while not stop_event.is_set():
        try:
            agent_text = output_queue.get(timeout=0.3)
        except queue.Empty:
            if end_session_event.is_set():
                _append_session_log(run_dir, "[SYS] END_SESSION_EVENT set; user thread exiting.")
                return
            continue

        agent_text_str = str(agent_text).strip()
        if not agent_text_str:
            continue
        telemetry.agent_turns.append(agent_text_str)
        idx = len(telemetry.agent_turns)
        _append_session_log(run_dir, f"[AGENT #{idx:02d}] {agent_text_str}")
        _write_state(run_dir, {
            "phase": "awaiting_user_reply",
            "n_agent_turns": len(telemetry.agent_turns),
            "n_user_turns": len(telemetry.user_turns),
            "last_agent_idx": idx,
            "last_agent_preview": agent_text_str[:200],
            "ts": _now_iso(),
        })

        # Wait for the user (Claude) to write a reply file.
        deadline = time.monotonic() + TURN_TIMEOUT_S
        reply = None
        while time.monotonic() < deadline:
            if stop_event.is_set() or end_session_event.is_set():
                return
            reply = _read_pending_reply(run_dir)
            if reply is not None:
                break
            time.sleep(0.5)

        if reply is None:
            _append_session_log(run_dir, f"[SYS] User reply timeout ({TURN_TIMEOUT_S:.0f}s) — sending SESSION_END.")
            telemetry.exceptions.append(f"User reply timeout at agent turn #{idx}")
            input_queue.put("SESSION_END")
            end_session_event.set()
            return

        telemetry.user_turns.append(reply)
        _append_session_log(run_dir, f"[USER #{len(telemetry.user_turns):02d}] {reply}")
        _write_state(run_dir, {
            "phase": "user_reply_sent",
            "n_agent_turns": len(telemetry.agent_turns),
            "n_user_turns": len(telemetry.user_turns),
            "ts": _now_iso(),
        })
        input_queue.put(reply)


# ── DB-side stub for crisis flag observability ─────────────────────────
def _install_crisis_observer(telemetry: LiveTelemetry):
    """Wrap DB.log_clinical_flag so we know if a critical-dim flag fired."""
    import src.utils.io_record as io_rec

    # We hook AFTER init_record creates DB; defer via lazy wrap.
    # Done by patching DBManager.log_clinical_flag at the class level.
    try:
        from src.drivers.db_manager import DBManager
    except Exception:
        return

    real_fn = DBManager.log_clinical_flag

    def wrapped(self, session_id, flag_type, details=None):
        try:
            telemetry.crisis_triggered = True
            if isinstance(details, dict):
                telemetry.crisis_dim = details.get("critical_dim", telemetry.crisis_dim)
        except Exception:
            pass
        return real_fn(self, session_id, flag_type, details)

    DBManager.log_clinical_flag = wrapped


# ── Main ───────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live persona runner.")
    parser.add_argument("--persona-id", required=True,
                        help="Short id like P-A, P-B, P-C — used for the run dir.")
    parser.add_argument("--persona-name", required=True,
                        help="Persona name (becomes part of the bench subject id).")
    parser.add_argument("--profile", required=True,
                        help="One-line description of the persona for the report.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR),
                        help="Parent directory for run artefacts.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) / f"{_utcstamp()}_{args.persona_id}_{args.persona_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[live_runner] run_dir: {run_dir}")
    _append_session_log(run_dir, f"[SYS] Starting live persona run for {args.persona_id} ({args.persona_name}) at {_now_iso()}")
    _append_session_log(run_dir, f"[SYS] Profile: {args.profile}")

    telemetry = LiveTelemetry(args.persona_id, args.persona_name, args.profile, run_dir)
    _install_llm_telemetry(telemetry)

    # ── Setup io_record state ──
    import src.utils.io_record as io_rec
    bench_subject = f"bench_{args.persona_name.lower()}"
    io_rec.SUBJECT_BASE_ID = bench_subject
    # Force a fresh session row.
    io_rec._INIT_DONE = False
    io_rec.END_SESSION_EVENT.clear()
    io_rec.START_SESSION_EVENT.set()
    # Initialise real DB + dossier.
    io_rec.init_record(user_id_override=bench_subject, force=True)
    _append_session_log(run_dir, f"[SYS] DB session id = {io_rec.SESSION_ID}")

    # Install per-dim score / RV / crisis observers AFTER io_rec is up.
    _install_score_telemetry(telemetry)
    _install_rv_telemetry(telemetry)
    _install_crisis_observer(telemetry)

    # ── Spawn user thread ──
    stop_event = threading.Event()
    user_t = threading.Thread(
        target=_live_user_thread,
        args=(run_dir, telemetry, io_rec.OUTPUT_QUEUE, io_rec.INPUT_QUEUE,
              io_rec.END_SESSION_EVENT, stop_event),
        daemon=True,
        name=f"LiveUser-{args.persona_id}",
    )
    user_t.start()

    # ── Spawn handler thread ──
    from src.core.handler_rl import HandlerRL
    handler = HandlerRL()
    handler_done = threading.Event()
    handler_exc: list[BaseException] = []

    def _handler_runner():
        try:
            handler.run()
        except BaseException as e:
            handler_exc.append(e)
        finally:
            handler_done.set()

    handler_t = threading.Thread(target=_handler_runner, daemon=True, name="Handler")
    handler_t.start()
    _write_state(run_dir, {"phase": "handler_started", "ts": _now_iso()})
    _append_session_log(run_dir, "[SYS] Handler thread launched.")

    # Watch loop — handler may take many minutes (Gemma is slow).
    deadline = time.monotonic() + HARDER_TIMEOUT_S
    while time.monotonic() < deadline:
        if handler_done.wait(timeout=2.0):
            break

    if not handler_done.is_set():
        msg = f"WATCHDOG: handler did not finish within {HARDER_TIMEOUT_S:.0f}s — forcing END_SESSION_EVENT."
        telemetry.exceptions.append(msg)
        _append_session_log(run_dir, f"[SYS] {msg}")
        io_rec.END_SESSION_EVENT.set()
        try:
            io_rec.INPUT_QUEUE.put_nowait("SESSION_END")
        except Exception:
            pass
        handler_done.wait(timeout=30.0)

    if handler_exc:
        e = handler_exc[0]
        telemetry.exceptions.append(f"{type(e).__name__}: {e}")
        _append_session_log(run_dir, f"[SYS] Handler raised: {e}")
    else:
        telemetry.completed = True

    stop_event.set()
    user_t.join(timeout=3.0)

    # Pull per-dim scores from the handler's question_lib (already mirrored
    # via _record_clinical_score, but include the final state for safety).
    try:
        for i_key in handler.question_lib.keys():
            if i_key == "0":
                continue
            entry = handler.question_lib[i_key].get("1", {})
            label = str(entry.get("label", ""))
            scores = [s for s in entry.get("score", []) if isinstance(s, int) and 0 <= s <= 2]
            if scores:
                # de-dupe with the live mirror
                existing = telemetry.recorded_scores.setdefault(label, [])
                for s in scores:
                    if s not in existing:
                        existing.append(s)
    except Exception:
        pass

    telemetry.ended_at = time.monotonic()
    (run_dir / "telemetry.json").write_text(json.dumps(telemetry.as_dict(), indent=2))
    _write_state(run_dir, {
        "phase": "complete",
        "n_agent_turns": len(telemetry.agent_turns),
        "n_user_turns": len(telemetry.user_turns),
        "completed": telemetry.completed,
        "exceptions": telemetry.exceptions,
        "ts": _now_iso(),
    })
    _append_session_log(run_dir, f"[SYS] Run complete at {_now_iso()}.")
    print(f"[live_runner] telemetry → {run_dir / 'telemetry.json'}")


if __name__ == "__main__":
    main()
