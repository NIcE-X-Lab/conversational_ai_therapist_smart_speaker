"""Session closing-reflection helper.

Legacy/paper pattern: the LLM never sees rolling conversation state — each
call is self-contained.  The only legitimate "context manager" responsibility
that survived the cut is generating a spoken closing reflection at session
end, which explicitly pulls the last N turns from the DB and passes them as
the payload of a single self-contained LLM call.
"""

from __future__ import annotations

import threading

from src.utils.log_util import get_logger

logger = get_logger("ContextManager")


class ClinicalContextManager:
    """Thin facade for session-end closing reflection.

    Usage:
        mgr = get_context_manager()
        reflection = mgr.generate_closing_reflection()
    """

    def __init__(self):
        pass

    def generate_closing_reflection(self) -> str:
        """Generate a short spoken closing reflection for session end.

        Pulls the last 12 turns from the session DB and passes them
        explicitly to a single self-contained LLM call — no rolling
        history, no cross-call state.  Returns empty string on failure.
        """
        try:
            from src.utils import io_record as _io
            if not _io.DB or not _io.SESSION_ID:
                return ""
            history = _io.DB.get_session_history(_io.SESSION_ID) or []
        except Exception as e:
            logger.warning(f"[TAKEAWAY] Failed to load session history: {e}")
            return ""

        if not history:
            return ""

        recent = history[-12:]
        turns_text = "\n".join(
            f"  {str(t.get('speaker', '')).lower()}: {t.get('text', '')}" for t in recent
        )

        system_prompt = (
            "You are a warm, empathetic therapist-assistant wrapping up a session.\n"
            "Generate a 2-3 sentence spoken closing reflection for the user.\n"
            "Reference specific topics or feelings from the session.\n"
            "Tone: caring, concise, validating.  No headers or labels.\n"
        )
        user_payload = f"Recent Turns:\n{turns_text}"

        try:
            from src.models.llm_client import llm_complete, LLMRole
            # Paper role: GENERAL (session-end reflection, not microbenchmarked).
            result = llm_complete(system_prompt, user_payload, role=LLMRole.GENERAL).strip()
            if result:
                logger.info(f"[TAKEAWAY] Closing reflection generated ({len(result)} chars).")
                return result
        except Exception as e:
            logger.warning(f"[TAKEAWAY] Closing reflection failed: {e}")
        return ""


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
