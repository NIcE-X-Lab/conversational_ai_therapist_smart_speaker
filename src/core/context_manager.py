"""Rolling clinical takeaway engine — extracts a concise clinical snapshot
every N turns and keeps it prepended to the LLM system prompt so the
"Brain" always knows the user's status regardless of context trimming."""

from __future__ import annotations

import threading
import time
from typing import List

from src.utils.log_util import get_logger

logger = get_logger("ContextManager")

_EXTRACTION_INTERVAL = 4  # run extraction every N turns

_EXTRACTION_PROMPT = (
    "Extract the following from the conversation turns below:\n"
    "1. Numerical screening scores (GAD-2, PHQ-2, PHQ-4) if mentioned.\n"
    "2. Current RL focus area / therapy dimension being explored.\n"
    "3. Key behavioral facts the user reported (sleep, eating, mood, etc.).\n"
    "4. Overall emotional trend (improving, declining, stable, mixed).\n\n"
    "Format as a concise bulleted list.  No preamble.  Start with bullets.\n\n"
    "Turns:\n"
)


class ClinicalContextManager:
    """Maintains a rolling clinical summary that survives context trimming.

    Usage:
        mgr = ClinicalContextManager()
        mgr.record_turn("user", "I haven't been sleeping well.")
        mgr.record_turn("agent", "That sounds difficult ...")
        # ... after 4 turns, a background extraction updates the summary.
        system_prompt = mgr.inject_into_prompt(base_system_prompt)
    """

    def __init__(self, extraction_interval: int = _EXTRACTION_INTERVAL):
        self._turns: List[dict] = []
        self._summary: str = ""
        self._lock = threading.Lock()
        self._interval = max(1, extraction_interval)
        self._pending_turns: List[dict] = []
        self._extraction_running = False

    # ── Public API ────────────────────────────────────────────────────────

    def record_turn(self, speaker: str, text: str) -> None:
        """Record a conversation turn.  Triggers extraction every N turns."""
        with self._lock:
            entry = {"speaker": speaker, "text": text}
            self._turns.append(entry)
            self._pending_turns.append(entry)

            if len(self._pending_turns) >= self._interval and not self._extraction_running:
                batch = list(self._pending_turns)
                self._pending_turns.clear()
                self._extraction_running = True
                thread = threading.Thread(
                    target=self._run_extraction,
                    args=(batch,),
                    daemon=True,
                )
                thread.start()

    def inject_into_prompt(self, system_prompt: str) -> str:
        """Prepend the latest clinical summary to a system prompt."""
        with self._lock:
            if not self._summary:
                return system_prompt
            return (
                "[Rolling Clinical Takeaway — auto-updated every "
                f"{self._interval} turns]\n"
                f"{self._summary}\n"
                "[End Takeaway]\n\n"
                f"{system_prompt}"
            )

    @property
    def summary(self) -> str:
        with self._lock:
            return self._summary

    def generate_closing_reflection(self) -> str:
        """Generate a 2-3 sentence spoken reflection summarising the session.

        Uses the rolling clinical summary and the raw turn history to give
        the user a personalised closing message before the goodbye sequence.
        Returns empty string on failure so callers can fall back gracefully.
        """
        with self._lock:
            summary = self._summary
            turns = list(self._turns[-12:])  # last 12 turns for recency

        if not turns:
            return ""

        turns_text = "\n".join(f"  {t['speaker']}: {t['text']}" for t in turns)
        system_prompt = (
            "You are a warm, empathetic therapist-assistant wrapping up a session.\n"
            "Generate a 2-3 sentence spoken closing reflection for the user.\n"
            "Reference specific topics, scores, or feelings from the session.\n"
            "Tone: caring, concise, validating.  No headers or labels.\n"
        )
        user_payload = (
            f"Clinical Summary:\n{summary or '(not yet available)'}\n\n"
            f"Recent Turns:\n{turns_text}"
        )
        try:
            from src.models.llm_client import llm_complete
            result = llm_complete(system_prompt, user_payload, inject_context=False).strip()
            if result:
                logger.info(f"[TAKEAWAY] Closing reflection generated ({len(result)} chars).")
                return result
        except Exception as e:
            logger.warning(f"[TAKEAWAY] Closing reflection failed: {e}")
        return ""

    def reset(self) -> None:
        """Clear all state for a new session."""
        with self._lock:
            self._turns.clear()
            self._pending_turns.clear()
            self._summary = ""
            self._extraction_running = False

    # ── Internal ──────────────────────────────────────────────────────────

    def _run_extraction(self, batch: List[dict]) -> None:
        """Background LLM call to extract clinical takeaway."""
        try:
            from src.models.llm_client import llm_complete

            turns_text = "\n".join(
                f"  {t['speaker']}: {t['text']}" for t in batch
            )
            prompt = f"{_EXTRACTION_PROMPT}{turns_text}"

            result = llm_complete(
                "You are a clinical note-taking assistant. Be concise.",
                prompt,
                inject_context=False,
            ).strip()

            if result:
                with self._lock:
                    self._summary = result
                logger.info(
                    f"[TAKEAWAY] Updated rolling clinical summary "
                    f"({len(batch)} turns)."
                )
        except Exception as e:
            logger.warning(f"[TAKEAWAY] Extraction failed: {e}")
        finally:
            with self._lock:
                self._extraction_running = False


# ── Module-level singleton ────────────────────────────────────────────────
_INSTANCE: ClinicalContextManager | None = None
_INSTANCE_LOCK = threading.Lock()


def get_context_manager() -> ClinicalContextManager:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = ClinicalContextManager()
    return _INSTANCE
