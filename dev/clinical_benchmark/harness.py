"""End-to-end persona-driven harness for clinical-trial benchmarking.

Drives a HandlerRL session through a full screening + CBT loop with:

  • A stub LLM that returns role-aware deterministic responses, so the
    session never hits the real Gemma engine — but every paper-aligned
    LLMRole call site is still exercised (counts are recorded per role).

  • A stub Response Analyzer that maps the persona's reply phrasing to
    the legacy (dim_label, Yes/No/Maybe/Question/Stop) tokens or scored
    (dim, 0-2) tuples — the same contract the real
    `services.response_bridge.get_openai_resp` returns.

  • A "user" thread that drains OUTPUT_QUEUE, identifies the dimension
    being asked from the agent question text (reverse-mapped via the
    question library's `name`/`label`), looks up the persona's intent
    for that dim, and pushes the corresponding phrasing on INPUT_QUEUE.

  • Full telemetry per run: every agent + user turn, LLM-role counts,
    retry_guide / brainstorm activations, RV decisions, recorded scores
    per dim, CBT stage outcomes, exceptions, wall-clock duration.
"""
from __future__ import annotations

import importlib
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dev.clinical_benchmark.personas import (  # noqa: E402
    Persona,
    s0, s1, s2,
    STUCK, STUCK_ONCE, UNSURE, OFF_TOPIC, QUESTION, STOP,
    EMPTY, UNICODE, LONG, REPEAT_STOP,
    _SCORE_0_PHRASES, _SCORE_1_PHRASES, _SCORE_2_PHRASES,
)


# ── Telemetry collector ────────────────────────────────────────────────
@dataclass
class RunTelemetry:
    pid: str
    name: str
    profile: str
    started_at: float = 0.0
    ended_at: float = 0.0
    agent_turns: List[str] = field(default_factory=list)
    user_turns: List[str] = field(default_factory=list)
    llm_calls_by_role: Dict[str, int] = field(default_factory=dict)
    analyzer_results: List[Tuple[str, Any]] = field(default_factory=list)
    retry_guide_triggers: int = 0
    brainstorm_triggers: int = 0
    rv_decisions: List[str] = field(default_factory=list)  # "0"/"1"
    cbt_dimension_picked: Optional[str] = None
    cbt_stage_outcomes: List[str] = field(default_factory=list)
    crisis_triggered: bool = False
    crisis_dim: Optional[str] = None
    exceptions: List[str] = field(default_factory=list)
    recorded_scores: Dict[str, List[int]] = field(default_factory=dict)
    intended_intents: Dict[str, Any] = field(default_factory=dict)
    stop_confirm_invoked: bool = False
    completed_screening: bool = False
    completed_cbt: bool = False
    cbt_success: bool = False

    def as_dict(self) -> dict:
        return {
            "pid": self.pid,
            "name": self.name,
            "profile": self.profile,
            "duration_sec": round(self.ended_at - self.started_at, 2),
            "agent_turn_count": len(self.agent_turns),
            "user_turn_count": len(self.user_turns),
            "llm_calls_by_role": dict(self.llm_calls_by_role),
            "retry_guide_triggers": self.retry_guide_triggers,
            "brainstorm_triggers": self.brainstorm_triggers,
            "rv_decision_breakdown": {
                "on_topic_0": sum(1 for d in self.rv_decisions if d == "0"),
                "off_topic_1": sum(1 for d in self.rv_decisions if d == "1"),
            },
            "cbt_dimension_picked": self.cbt_dimension_picked,
            "cbt_stage_outcomes": list(self.cbt_stage_outcomes),
            "cbt_success": self.cbt_success,
            "crisis_triggered": self.crisis_triggered,
            "crisis_dim": self.crisis_dim,
            "exceptions": list(self.exceptions),
            "recorded_scores": {k: list(v) for k, v in self.recorded_scores.items()},
            "intended_intents": {k: v for k, v in self.intended_intents.items()},
            "stop_confirm_invoked": self.stop_confirm_invoked,
            "completed_screening": self.completed_screening,
            "completed_cbt": self.completed_cbt,
        }


# ── LLM stub ─────────────────────────────────────────────────────────────
def _make_llm_stub(telemetry: RunTelemetry) -> Callable:
    """Return a fake `llm_complete` that records every call by role."""
    cbt_stage1_calls = {"reasoner": 0}

    def fake_llm(system, user, role=None, **kwargs):
        role_value = role.value if hasattr(role, "value") else str(role or "general")
        telemetry.llm_calls_by_role[role_value] = telemetry.llm_calls_by_role.get(role_value, 0) + 1

        sys_text = str(system or "")
        user_text = str(user or "")
        sys_low = sys_text.lower()

        if role_value == "rv_reasoner":
            # On-topic by default — most personas elaborate substantively
            # on the asked dimension. The harness can override via the
            # persona's "off_topic" intent which feeds an off-topic
            # follow-up text; we detect that and emit DECISION: 1.
            if " banana " in user_text.lower() or "weather" in user_text.lower():
                telemetry.rv_decisions.append("1")
                return "DECISION: 1"
            telemetry.rv_decisions.append("0")
            return "DECISION: 0"

        if role_value == "rv_validator":
            return ("VALIDATION: It makes sense that this has been hard. "
                    "What you're going through is real. A few small steps can "
                    "help: name what you need, lean on one trusted person, "
                    "and take it one day at a time. You're not alone in this.")

        if role_value == "rv_guide":
            # The RV_GUIDE role is reused by `questioner.retry_guide` for
            # clarify/different-angle/restate guidance on unscored
            # screening replies. Disambiguate by the payload's "Original
            # Answer" key (retry_guide) vs "Original Response"/"Follow-up
            # Response" (genuine RV Guide). Returning a retry-style
            # phrasing the user-thread recognises avoids the harness
            # hanging on Maybe/Question Maybe-only personas (P06).
            if "Original Answer" in user_text and "Follow-up Response" not in user_text:
                # retry_guide path
                return ("GUIDE: Let us try from a different perspective — "
                        "could you describe how that has been going for you "
                        "in the past week?")
            return "GUIDE: Coming back to what we were exploring, could you tell me a little more about that?"

        if role_value == "cbt_reasoner":
            # Stage 1 fails once on the first reasoner call, then accepts;
            # this exercises the retry loop. Stage 2/3 always accept.
            if "UNHELPFUL_THOUGHTS" in user_text and "CHALLENGE" not in user_text:
                cbt_stage1_calls["reasoner"] += 1
                if cbt_stage1_calls["reasoner"] == 1:
                    return "DECISION: 1"
                return "DECISION: 0"
            return "DECISION: 0"

        if role_value == "cbt_guide":
            if "STATEMENT:" in user_text and "UNHELPFUL_THOUGHTS:" not in user_text:
                return ("UNHELPFUL_THOUGHTS: you think you always fail at this; "
                        "you fear you can't change; you worry it's too late; "
                        "you see one slip as everything; you assume effort doesn't matter.")
            if "CHALLENGE:" not in user_text and "UNHELPFUL_THOUGHTS:" in user_text:
                return ("CHALLENGE: I have managed this before — even briefly. "
                        "What evidence do I actually have for the worst-case story?")
            return ("REFRAME: I can hold both — that this is hard AND that I have "
                    "tools and history that show progress is possible.")

        if role_value == "analyzer":
            # Real analyzer is bypassed by the response_bridge stub below.
            # If something does call this directly (e.g. multi-dim
            # back-fill), return an empty array.
            return "[]"

        if role_value == "rephraser":
            return "REPHRASER: " + user_text[-160:]

        if role_value == "reflective_summarizer":
            return ""

        # GENERAL: greeting, closing, session analysis, intent classifier.
        if "farewell" in sys_low or "wrapping up" in sys_low or "clinician" in sys_low:
            return ("SUMMARY: Thank you for sharing what you did today. "
                    "Take care, and I'll be here whenever you'd like to come back.\n"
                    "PREFERENCES:\n- session_engagement: completed\n"
                    "SAFETY_FLAGS:\nNONE\n")
        if "soap" in sys_low or "clinical documentation" in sys_low or "subjective" in sys_low:
            return ("SUBJECTIVE:\n- Client engaged with screening.\n"
                    "OBJECTIVE:\n- Scores recorded across dimensions.\n"
                    "ASSESSMENT:\n- Routine follow-up.\n"
                    "INTERVENTION:\n- Continued CBT focus next session.\n")
        if "greeting" in sys_low or "opening" in sys_low or "transition" in sys_low:
            return ("Let's begin with a few brief check-ins about your recent "
                    "day-to-day.")
        # Default — short acknowledgement
        return "Okay."

    return fake_llm


# ── Response Analyzer stub ───────────────────────────────────────────────
# Reverse-maps a persona's phrasing back to the legacy taxonomy.  Built
# from the same _SCORE_*_PHRASES tables the persona uses, so a Score-2
# phrasing for "mood" coming back from the user always classifies as
# (mood, 2).
_SUB_TO_TUPLE: List[Tuple[str, str, Any]] = []  # (lowercased substring, dim_label, score|tag)


def _build_substring_index() -> None:
    """Index every persona phrasing by a unique-ish substring for matching."""
    _SUB_TO_TUPLE.clear()
    for label, phrase in _SCORE_0_PHRASES.items():
        _SUB_TO_TUPLE.append((phrase.lower().strip(" ."), label, 0))
    for label, phrase in _SCORE_1_PHRASES.items():
        _SUB_TO_TUPLE.append((phrase.lower().strip(" ."), label, 1))
    for label, phrase in _SCORE_2_PHRASES.items():
        _SUB_TO_TUPLE.append((phrase.lower().strip(" ."), label, 2))


_build_substring_index()


def _make_analyzer_stub(telemetry: RunTelemetry) -> Callable:
    """Return a fake `get_openai_resp` that maps user replies to (dim, value)."""

    def fake_get_openai_resp(user_input, original_question, dimension_label):
        text = str(user_input or "").strip().lower()

        # Hard tokens first — these short-circuit before any score lookup.
        if not text:
            telemetry.analyzer_results.append(("NA", 99))
            return "NA", 99
        if text in ("stop", "stop.", "stop!"):
            telemetry.analyzer_results.append((dimension_label, "Stop"))
            return dimension_label, "Stop"
        if "i don't know" in text or "i dont know" in text or "no idea" in text:
            telemetry.analyzer_results.append((dimension_label, "Maybe"))
            return dimension_label, "Maybe"
        if "i'm not sure" in text or "im not sure" in text or "not sure" in text:
            telemetry.analyzer_results.append((dimension_label, "Maybe"))
            return dimension_label, "Maybe"
        if "what do you mean" in text or "i don't get it" in text or "dont get it" in text:
            telemetry.analyzer_results.append((dimension_label, "Question"))
            return dimension_label, "Question"

        # Score-token short-circuits for trivial 1-3 word affirm/deny.
        tokens = text.split()
        if len(tokens) <= 3:
            if "yes" in tokens:
                telemetry.analyzer_results.append((dimension_label, "Yes"))
                return dimension_label, "Yes"
            if "no" in tokens:
                telemetry.analyzer_results.append((dimension_label, "No"))
                return dimension_label, "No"

        # Persona-phrasing match — find the best-matching indexed phrase.
        # We iterate the index and return the FIRST substring hit so the
        # ordering of _build_substring_index defines tie-breaking
        # (Score-0 first, then 1, then 2). To bias to the asked dim, we
        # do a two-pass scan: prefer matches where the matched dim equals
        # the asked dim_label.
        best = None
        for sub, lbl, sc in _SUB_TO_TUPLE:
            if sub and sub in text:
                if lbl == dimension_label:
                    best = (lbl, sc)
                    break
                if best is None:
                    best = (lbl, sc)
        if best is not None:
            telemetry.analyzer_results.append(best)
            return best

        # Fallback — unknown reply, classify as Other.
        telemetry.analyzer_results.append(("NA", 99))
        return "NA", 99

    return fake_get_openai_resp


# ── User thread: drains OUTPUT_QUEUE, replies on INPUT_QUEUE ────────────
def _identify_dim_in_question(question_text: str, question_lib: dict) -> Optional[str]:
    """Reverse-look the dimension label being asked, given the agent's text.

    Strategy: each dim has a `name` (e.g. "Maintaining Stable Weight") and
    a `question` list (the canonical phrasings the questioner picks from).
    A question text matches a dim if the dim's name OR any of its canonical
    questions OR any of its `question_synthetic` rephrasings shows up
    substring-wise in the agent text.
    """
    text = str(question_text or "").lower()
    if not text:
        return None

    # Canonical-question substring match — most reliable.
    best = None
    best_len = 0
    for i_key, slot in question_lib.items():
        if i_key == "0":
            continue
        entry = slot.get("1", {})
        label = str(entry.get("label", "")).lower()
        if not label:
            continue
        for src in ("question", "question_synthetic"):
            for q in entry.get(src, []):
                q_low = str(q or "").lower().strip()
                if not q_low:
                    continue
                if q_low in text and len(q_low) > best_len:
                    best = label
                    best_len = len(q_low)
        # Name fallback — handles CBT Stage 0 listing where `name` shows.
        name = str(entry.get("name", "")).lower()
        if name and name in text and len(name) > best_len:
            best = label
            best_len = len(name)
    return best


# Phrases that mean "the agent is mid-CBT, not asking a screening dim".
_CBT_STAGE_MARKERS = (
    "identify any unhelpful thoughts",
    "challenge those unhelpful thoughts",
    "reframe the unhelpful thought",
    "you have issue in",
)


def _is_cbt_stage_prompt(text: str) -> str:
    """Identify which CBT stage the agent is currently asking about.

    Handles both the initial Stage prompts and the Stage retry prompts —
    Stage 1/2/3 retries fold the Guide example + re-ask into one
    OUTPUT_QUEUE entry whose phrasing is "Please share those unhelpful
    thoughts again" / "Please try to challenge the unhelpful thoughts
    again" / "Please try to reframe that again". Without this, the user
    thread would hang on Stage retries.
    """
    low = text.lower()
    if "you have issue in" in low and "tell me the dimension number" in low:
        return "cbt_pick"
    # CBT dim-pick retry on a parse failure: "Please reply with a single
    # number between 1 and N. Example: 1. Options: ..."
    if ("please reply with a single number between" in low
            and "options:" in low):
        return "cbt_pick"
    # Stage 1 — initial + retry phrasings
    if ("identify any unhelpful thoughts" in low
            or "share those unhelpful thoughts again" in low):
        return "cbt_unhelpful"
    # Stage 2 — initial + retry phrasings
    if ("challenge those unhelpful thoughts" in low
            or "challenge the unhelpful thoughts again" in low):
        return "cbt_challenge"
    # Stage 3 — initial + retry phrasings
    if ("reframe the unhelpful thought" in low
            or "reframe that again" in low):
        return "cbt_reframe"
    if "great work today" in low and "cbt" in low:
        return "cbt_success"
    if "let's pause cbt" in low or "pause cbt and revisit" in low:
        return "cbt_failed"
    if "could not determine your choice" in low:
        return "cbt_failed"
    return ""


_FOLLOWUP_MARKERS = (
    "tell me more about it",
    "tell me more",
    "can you tell me more",
)


def _is_followup_prompt(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _FOLLOWUP_MARKERS)


_RETRY_GUIDE_MARKERS = (
    "could you describe",
    "let us try from a different perspective",
    "let us focus on",
    "even briefly, is useful",  # brainstorm scaffold tail
    "we can think about it together",
    "we can take it slow",
)


def _is_retry_or_brainstorm(text: str) -> tuple[bool, bool]:
    """Return (is_retry_prompt, is_brainstorm_prompt)."""
    low = text.lower()
    is_brainstorm = (
        "we can think about it together" in low
        or "we can take it slow" in low
        or "even briefly, is useful" in low
        or "anything that comes to mind" in low
    )
    is_retry = is_brainstorm or any(m in low for m in _RETRY_GUIDE_MARKERS)
    return is_retry, is_brainstorm


def _persona_reply_for_dim(persona: Persona, label: str, retry: bool) -> str:
    """Resolve persona intent for `label` into a concrete reply string."""
    intent = persona.intents.get(label, 0)

    if intent == 0:
        return s0(label)
    if intent == 1:
        return s1(label)
    if intent == 2:
        return s2(label)
    if intent == STOP:
        return "stop"
    if intent == STUCK:
        return "I don't know."
    if intent == STUCK_ONCE:
        # First-pass: stuck. Retry-pass: Score-1 reply (the brainstorm
        # scaffold gave us angles to react to).
        return s1(label) if retry else "I don't know."
    if intent == UNSURE:
        return s1(label) if retry else "I'm not sure."
    if intent == QUESTION:
        return s0(label) if retry else "What do you mean?"
    if intent == OFF_TOPIC:
        return ("I was thinking about the weather earlier, it's been raining a lot."
                if not retry else s0(label))
    if intent == EMPTY:
        return s0(label) if retry else ""  # blank → analyzer NA → retry_guide
    if intent == UNICODE:
        # Score-2 dim reply with non-ASCII — must still classify by
        # phrasing substring match.
        return s2(label) + " (it's been brutal — really hard 😔)"
    if intent == LONG:
        # Multi-segment — first segment is the Score-2 phrase, then
        # extra unrelated content. The legacy segmenter splits on ".",
        # ", and", " but ".
        return (s2(label) + ". And by the way, I haven't been to the gym, "
                "but I do walk to work most days, and I cooked dinner last night "
                "for the first time in a while.")
    if intent == REPEAT_STOP:
        return "stop"
    return s0(label)


def _user_thread(
    persona: Persona,
    question_lib: dict,
    telemetry: RunTelemetry,
    output_queue: "queue.Queue",
    input_queue: "queue.Queue",
    end_session_event,
    cbt_started_event,
    stop_event: threading.Event,
):
    """User-simulation thread.

    Watches the OUTPUT_QUEUE, identifies what's being asked, looks up
    the persona's intent for the relevant dim, and pushes an
    appropriate reply on INPUT_QUEUE.  The handler thread is unaware
    that the "user" is a stub — it just sees turns arrive.
    """
    last_dim: Optional[str] = None  # remembers the most recent screening dim
                                    # (so a follow-up "tell me more" knows
                                    # which dim to elaborate on)
    last_user_text: str = ""        # previous reply, used for stuck-on-retry
    asked_dims: List[str] = []
    repeat_stop_count = 0
    agent_turn_count = 0
    cbt_dim_picked = False

    while not stop_event.is_set():
        try:
            agent_text = output_queue.get(timeout=0.2)
        except queue.Empty:
            if end_session_event.is_set():
                return
            continue

        agent_text_str = str(agent_text)
        telemetry.agent_turns.append(agent_text_str)
        agent_turn_count += 1

        # Reset per-turn flags.
        cbt_marker = _is_cbt_stage_prompt(agent_text_str)
        is_followup = _is_followup_prompt(agent_text_str)
        is_retry, is_brainstorm = _is_retry_or_brainstorm(agent_text_str)
        if is_retry:
            telemetry.retry_guide_triggers += 1
        if is_brainstorm:
            telemetry.brainstorm_triggers += 1

        # Stop double-confirm prompt — coming from speech_service. The
        # handler-only harness doesn't run speech_service, so this won't
        # appear in normal flow; included defensively.
        if "should we go ahead and end" in agent_text_str.lower():
            telemetry.stop_confirm_invoked = True
            input_queue.put(persona.stop_confirm)
            telemetry.user_turns.append(persona.stop_confirm)
            continue

        # ── CBT Stage routing ──
        if cbt_marker == "cbt_pick":
            cbt_dim_picked = True
            reply = str(persona.cbt_pick)
            input_queue.put(reply)
            telemetry.user_turns.append(reply)
            # Try to record which dim that index resolved to (best-effort).
            telemetry.cbt_dimension_picked = telemetry.cbt_dimension_picked or "(picked by index)"
            continue
        if cbt_marker == "cbt_unhelpful":
            telemetry.cbt_stage_outcomes.append("stage1_entered")
            reply = persona.cbt_unhelpful
            input_queue.put(reply)
            telemetry.user_turns.append(reply)
            continue
        if cbt_marker == "cbt_challenge":
            telemetry.cbt_stage_outcomes.append("stage2_entered")
            reply = persona.cbt_challenge
            input_queue.put(reply)
            telemetry.user_turns.append(reply)
            continue
        if cbt_marker == "cbt_reframe":
            telemetry.cbt_stage_outcomes.append("stage3_entered")
            reply = persona.cbt_reframe
            input_queue.put(reply)
            telemetry.user_turns.append(reply)
            continue
        if cbt_marker == "cbt_success":
            telemetry.cbt_stage_outcomes.append("success")
            telemetry.cbt_success = True
            telemetry.completed_cbt = True
            # No reply needed — handler exits the run() loop next.
            continue
        if cbt_marker == "cbt_failed":
            telemetry.cbt_stage_outcomes.append("failed")
            continue

        # ── Follow-up "tell me more" — elaborate on the same dim ──
        if is_followup and last_dim:
            # Substantive elaboration so RV Reasoner returns DECISION: 0
            elab = f"It really does affect me. {s2(last_dim)}"
            # If the persona is OFF_TOPIC for this dim, pivot to weather
            if persona.intents.get(last_dim) == OFF_TOPIC:
                elab = "I started thinking about the weather instead — really sunny outside today."
            input_queue.put(elab)
            telemetry.user_turns.append(elab)
            continue

        # ── Retry / brainstorm — answer using last_dim with retry=True ──
        if is_retry and last_dim:
            reply = _persona_reply_for_dim(persona, last_dim, retry=True)
            input_queue.put(reply)
            telemetry.user_turns.append(reply)
            last_user_text = reply
            continue

        # ── Screening question — identify the dim and reply ──
        dim = _identify_dim_in_question(agent_text_str, question_lib)
        if dim is not None:
            last_dim = dim
            asked_dims.append(dim)
            reply = _persona_reply_for_dim(persona, dim, retry=False)
            # REPEAT_STOP intent: spam stop a few times for adversarial
            if persona.intents.get(dim) == REPEAT_STOP:
                repeat_stop_count += 1
            input_queue.put(reply)
            telemetry.user_turns.append(reply)
            last_user_text = reply
            continue

        # ── Unknown agent text — push a polite filler reply ──
        # This handles validator/guide-style prefixes, RV validations
        # prepended to the next question, etc., that we didn't already
        # interpret. We let the handler proceed without artificial
        # interference.
        # If we *just* answered something and the queue is empty, sit tight.
        # No put — the handler must still emit the next question for us.
        continue


# ── Run a single persona ────────────────────────────────────────────────
def run_persona(persona: Persona, *, run_dir: Path) -> RunTelemetry:
    """Drive one persona through a full HandlerRL session and collect telemetry."""
    telemetry = RunTelemetry(
        pid=persona.pid,
        name=persona.name,
        profile=persona.profile,
        intended_intents=dict(persona.intents),
    )
    telemetry.started_at = time.monotonic()

    # Per-run isolation: fresh queues, fresh DB stub, no Q-table file.
    import src.utils.io_record as io_rec

    # Wipe any previous Q-table for the test subject id so the warm-start
    # path doesn't carry state across personas.
    qfile = Path(f"data/q_tables/item_qtable_{persona.name.lower()}.csv")
    if qfile.exists():
        qfile.unlink()
    qfile_default = Path("data/q_tables/item_qtable_8080.csv")
    if qfile_default.exists():
        qfile_default.unlink()

    # ── Patch LLM and Analyzer ──
    fake_llm = _make_llm_stub(telemetry)
    fake_analyzer = _make_analyzer_stub(telemetry)

    patched_modules: List[Tuple[Any, str, Any]] = []

    def patch(mod_path: str, attr: str, value: Any):
        m = importlib.import_module(mod_path)
        if hasattr(m, attr):
            patched_modules.append((m, attr, getattr(m, attr)))
            setattr(m, attr, value)

    for mp in (
        "src.models.llm_client",
        "src.core.CBT",
        "src.core.reflection_validation",
        "src.core.response_analyzer",
        "src.core.questioner",
        "src.core.handler_rl",
        "src.utils.text_generators",
        "src.services.response_bridge",
    ):
        patch(mp, "llm_complete", fake_llm)
    for mp in ("src.services.response_bridge", "src.core.questioner"):
        patch(mp, "get_openai_resp", fake_analyzer)

    # ── Stub DB ──
    class StubDB:
        def __init__(self):
            self.history: List[Dict[str, Any]] = []
            self.summaries: List[str] = []
            self.preferences: Dict[str, Any] = {}
            self.safety_flags: List[Tuple] = []
            self.clinical_flags: List[Tuple] = []

        def get_session_history(self, sid):
            return list(self.history)

        def get_screening_scores(self, sid):
            return {}

        def add_summary(self, sid, summary):
            self.summaries.append(summary)

        def get_user_id(self, subj):
            return 1

        def set_preference(self, uid, k, v):
            self.preferences[k] = v

        def log_safety_flag(self, *a, **kw):
            self.safety_flags.append((a, kw))

        def log_clinical_flag(self, *a, **kw):
            self.clinical_flags.append((a, kw))
            telemetry.crisis_triggered = True
            try:
                detail = kw.get("details") or (a[2] if len(a) > 2 else {})
                if isinstance(detail, dict):
                    telemetry.crisis_dim = detail.get("critical_dim", telemetry.crisis_dim)
            except Exception:
                pass

        def log_safety_delivery(self, *a, **kw):
            pass

        def record_intervention_log(self, *a, **kw):
            pass

        def record_clinical_score(self, *a, **kw):
            # Mirror score into telemetry so we have a per-dim ground truth.
            try:
                kw_or_args = kw or {}
                dim_label = kw_or_args.get("dim_label")
                score = kw_or_args.get("score")
                if dim_label is not None and isinstance(score, int):
                    telemetry.recorded_scores.setdefault(dim_label, []).append(score)
            except Exception:
                pass

        def add_turn(self, sid, idx, speaker, text, meta_data=None):
            self.history.append({"speaker": speaker, "text": str(text)})

        def get_recent_screening_scores(self, *a, **kw):
            return []

        def get_all_preferences(self, *a, **kw):
            return {}

        def load_rl_state(self, *a, **kw):
            return None

        def save_rl_state(self, *a, **kw):
            pass

        def get_clinical_scores(self, *a, **kw):
            return []

        def upsert_intermission_screening_status(self, *a, **kw):
            pass

        def get_intermission_screening_statuses(self, *a, **kw):
            return {}

        def close_open_sessions_for_user(self, *a, **kw):
            return []

        def create_session(self, *a, **kw):
            return f"session_{persona.pid}"

        def close_session(self, *a, **kw):
            pass

        def get_user_context_string(self, *a, **kw):
            return ""

    # ── Reset io_rec state ──
    io_rec.DB = StubDB()
    io_rec.SESSION_ID = f"bench_{persona.pid}"
    io_rec.SUBJECT_ID = persona.name.lower()
    io_rec.SUBJECT_BASE_ID = persona.name.lower()
    io_rec._INIT_DONE = True
    io_rec.CBT_STARTED_EVENT.clear()
    io_rec.END_SESSION_EVENT.clear()
    io_rec.START_SESSION_EVENT.set()
    io_rec.INPUT_QUEUE = queue.Queue()
    io_rec.OUTPUT_QUEUE = queue.Queue()
    # init_record is a no-op so we don't reopen the real DB
    io_rec.init_record = lambda *a, **kw: None
    io_rec._LAST_AGENT_LOGGED = ""

    # ── Load question lib (used by the user thread for dim ID) ──
    from src.utils.io_question_lib import load_question_lib
    from src.utils.config_loader import QUESTION_LIB_FILENAME
    question_lib = load_question_lib(QUESTION_LIB_FILENAME)

    # ── Spawn user thread ──
    stop_event = threading.Event()
    user_t = threading.Thread(
        target=_user_thread,
        args=(persona, question_lib, telemetry,
              io_rec.OUTPUT_QUEUE, io_rec.INPUT_QUEUE,
              io_rec.END_SESSION_EVENT, io_rec.CBT_STARTED_EVENT,
              stop_event),
        daemon=True,
        name=f"User-{persona.pid}",
    )
    user_t.start()

    # ── Run handler in a background thread with a hard watchdog ──
    # Without a watchdog, a single misclassified agent prompt causes the
    # user thread to never push a reply; HandlerRL then blocks forever
    # on `INPUT_QUEUE.get()`. The watchdog flips END_SESSION_EVENT after
    # a fixed wall-clock deadline so the handler can drain cleanly.
    from src.core.handler_rl import HandlerRL
    handler = HandlerRL()
    handler_done = threading.Event()
    handler_exc: List[BaseException] = []

    def _handler_runner():
        try:
            handler.run()
        except SystemExit:
            pass
        except BaseException as e:
            handler_exc.append(e)
        finally:
            handler_done.set()

    handler_t = threading.Thread(
        target=_handler_runner, daemon=True, name=f"Handler-{persona.pid}",
    )
    handler_t.start()

    # Per-persona deadline: 60s is generous for ~40 turns even on slow
    # CI; the LLM is stubbed so most turns are sub-millisecond.
    deadline = 60.0
    if not handler_done.wait(timeout=deadline):
        telemetry.exceptions.append(
            f"WATCHDOG: handler did not finish within {deadline:.0f}s — "
            f"setting END_SESSION_EVENT and force-draining."
        )
        io_rec.END_SESSION_EVENT.set()
        # Drop a sentinel so any blocking get_resp_log / get_answer
        # unblocks instantly.
        try:
            io_rec.INPUT_QUEUE.put_nowait("SESSION_END")
        except Exception:
            pass
        handler_done.wait(timeout=10.0)

    if handler_exc:
        e = handler_exc[0]
        telemetry.exceptions.append(f"{type(e).__name__}: {e}")
        try:
            traceback.print_exception(type(e), e, e.__traceback__)
        except Exception:
            pass
    else:
        telemetry.completed_screening = True

    # Stop user thread
    stop_event.set()
    user_t.join(timeout=2.0)

    telemetry.ended_at = time.monotonic()

    # ── Pull any final score state from the question_lib in handler ──
    try:
        for i_key in handler.question_lib.keys():
            if i_key == "0":
                continue
            entry = handler.question_lib[i_key].get("1", {})
            label = entry.get("label", "")
            scores = [s for s in entry.get("score", []) if isinstance(s, int) and 0 <= s <= 2]
            if scores:
                telemetry.recorded_scores.setdefault(label, []).extend(scores)
    except Exception:
        pass

    # ── Restore patched modules ──
    for mod, attr, original in patched_modules:
        setattr(mod, attr, original)

    # ── Persist telemetry ──
    out_path = run_dir / f"{persona.pid}_{persona.name}.json"
    out_path.write_text(json.dumps(telemetry.as_dict(), indent=2))

    return telemetry


__all__ = ["run_persona", "RunTelemetry"]
