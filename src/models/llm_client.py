"""AI model wrapper abstracting communications with the primary LLM engine.

Uses Google's LiteRT-LM framework (litert-lm-api) for in-process Gemma 4 E2B
inference on Jetson Orin Nano.  No external server required — the model runs
directly inside this Python process via litert_lm.Engine.

Task-specific LLM roles (paper §5, Fig. 10)
-------------------------------------------
The CaiTI paper assigns different LLMs to different subtasks so each task
uses the model that microbenchmarks showed was best at it:

    Role                Paper mapping                       Why
    ──────────────────────────────────────────────────────────────────────────
    ANALYZER            fine-tuned GPT-3.5-Turbo            best (Dim, Score)
    REPHRASER           GPT-4                               structural rewrite
    REFLECTIVE_SUMMARIZER  GPT-4                            1st→3rd-person
    RV_REASONER         GPT-4                               best validity judge
    RV_GUIDE            GPT-3.5-Turbo                       fewer "read-mind" drifts
    RV_VALIDATOR        GPT-3.5-Turbo                       less "reads into feelings"
    CBT_REASONER        GPT-4                               hardest reasoning task
    CBT_GUIDE           GPT-3.5-Turbo                       empathic, non-assumptive
    GENERAL             any capable LLM                     greetings / closings / intent

On this deployment (Jetson Orin Nano, 8 GB budget), every role maps to the
same local Gemma 4 E2B model. The role plumbing is in place so that future
deployments can swap per-role models without touching call sites — just
update `ROLE_MODEL_MAP`.

Every call site must pass a `role`; call sites record which task they serve
in the log, which makes it straightforward to later split roles across
model backends (e.g., a larger Gemma for reasoning, a smaller one for
validation).
"""
import os
import time
import threading
from enum import Enum
from src.utils.config_loader import (
    LLM_MODEL,
    LITERT_MODEL_PATH,
    LITERT_BACKEND,
)
from src.utils.inference_guard import heavy_stage
from src.utils.log_util import get_logger

logger = get_logger("LLMClient")


class LLMRole(str, Enum):
    """Paper-aligned roles for each LLM call site.

    Every call to `llm_complete` must pass one of these. Today they all
    resolve to the same Gemma model via ROLE_MODEL_MAP below, but once a
    multi-model deployment is viable, swapping the mapping is the only
    change required — call sites stay untouched.
    """

    # Paper §5.2 — Response Analyzer, fine-tuned GPT-3.5-Turbo in paper
    ANALYZER = "analyzer"

    # Paper §5.1 — Rephraser, GPT-4 in paper (structural rewrite)
    REPHRASER = "rephraser"

    # Paper §5.2 — ReflectiveSummarizer, GPT-4 in paper (1st→3rd person)
    REFLECTIVE_SUMMARIZER = "reflective_summarizer"

    # Paper §5.3 — R-V Reasoner, GPT-4 in paper (validity judgement)
    RV_REASONER = "rv_reasoner"

    # Paper §5.3 — R-V Guide, GPT-3.5-Turbo in paper (redirect off-topic)
    RV_GUIDE = "rv_guide"

    # Paper §5.3 — R-V Validator, GPT-3.5-Turbo in paper (MI empathic reflection)
    RV_VALIDATOR = "rv_validator"

    # Paper §5.4 — CBT Stage{1,2,3} Reasoner, GPT-4 in paper
    CBT_REASONER = "cbt_reasoner"

    # Paper §5.4 — CBT Stage{1,2,3} Guide, GPT-3.5-Turbo in paper
    CBT_GUIDE = "cbt_guide"

    # Generic tasks not explicitly microbenchmarked in the paper: greeting,
    # closing, SOAP summary, intent classifier, therapist chat.
    GENERAL = "general"


# Role → concrete model identifier. Today every entry points at the single
# on-device Gemma model; split this map to swap per-role models later.
# Keep this keyed by role.value (str) so config-file overrides stay simple.
ROLE_MODEL_MAP: dict[str, str] = {role.value: LLM_MODEL for role in LLMRole}

class LLMError(Exception):
    """C6: typed exception for hard LLM failures.

    Raised when the engine itself fails (crash, timeout, unrecoverable
    error) — distinct from an empty/short output which is returned as a
    string. Callers that care about clinical correctness (the Analyzer,
    R-V Reasoner, CBT Reasoner) can catch this and mark the turn SKIPPED
    instead of silently scoring the user as healthy.
    """


# ── LiteRT-LM engine singleton ────────────────────────────────────────────
_ENGINE = None
_ENGINE_LOCK = threading.Lock()

# H2: bounded re-init attempts per session. If the engine crashes 3 times
# in a row we stop trying — further calls raise LLMError immediately so
# the handler can close the session cleanly instead of hanging.
_ENGINE_FAILURE_COUNT = 0
_ENGINE_MAX_FAILURES = 3


def _invalidate_engine(reason: str):
    """H2: drop the singleton so the next call re-loads."""
    global _ENGINE, _ENGINE_FAILURE_COUNT
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            logger.warning(f"[LiteRT] Invalidating engine singleton ({reason}).")
            _ENGINE = None
        _ENGINE_FAILURE_COUNT += 1


def _reset_engine_failure_count():
    """Called after a successful call so transient hiccups don't exhaust the budget."""
    global _ENGINE_FAILURE_COUNT
    _ENGINE_FAILURE_COUNT = 0


def engine_is_healthy() -> bool:
    """Public check so handler_rl can decide whether to short-circuit to SKIPPED."""
    return _ENGINE_FAILURE_COUNT < _ENGINE_MAX_FAILURES


def _init_engine():
    """Lazy-initialise the LiteRT-LM inference engine (thread-safe singleton).

    Raises LLMError when repeated failures exhaust the budget so the caller
    can record a clean SKIPPED turn instead of spinning.
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    if _ENGINE_FAILURE_COUNT >= _ENGINE_MAX_FAILURES:
        raise LLMError(
            f"LiteRT engine has failed {_ENGINE_FAILURE_COUNT} times this "
            "session; refusing further attempts."
        )

    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE

        model_path = LITERT_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"LiteRT model not found at {model_path}. "
                "Run: python scripts/model_fetch.py"
            )

        logger.info(f"[LiteRT] Loading model from {model_path} (backend={LITERT_BACKEND})")
        rss_before = _rss_mb()

        try:
            import litert_lm

            backend = litert_lm.Backend.CPU
            if LITERT_BACKEND == "gpu":
                if hasattr(litert_lm.Backend, "GPU"):
                    backend = litert_lm.Backend.GPU
                    logger.info("[LiteRT] Using GPU backend (ML Drift).")
                else:
                    logger.warning(
                        "[LiteRT] GPU backend requested but not available in this "
                        "litert-lm-api version. Falling back to CPU."
                    )

            engine = litert_lm.Engine(
                model_path,
                backend=backend,
                cache_dir="/tmp/litert-lm-cache",
            )
        except Exception as e:
            logger.error(f"[LiteRT] Failed to load model: {e}")
            raise LLMError(f"LiteRT engine failed to initialize: {e}") from e

        rss_after = _rss_mb()
        logger.info(
            f"[LiteRT] Model loaded. RSS delta: +{rss_after - rss_before:.1f}MB "
            f"(now {rss_after:.1f}MB)"
        )
        _ENGINE = engine
        return _ENGINE


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _build_gemma_prompt(system_content: str, user_content: str) -> str:
    """Fold system + user content into a single user message.

    Gemma has no native system role.  The system prompt is prepended to
    the user message so the model sees instructions before the question.
    The litert_lm conversation API handles the chat template internally.
    """
    return f"{system_content}\n\n{user_content}"


def _resolve_model(role: LLMRole | str | None) -> tuple[str, str]:
    """Return (role_value, model_id) for a call.

    Accepts the enum, its string value, or None (→ GENERAL). Unknown role
    strings fall back to GENERAL with a warning so typos never crash a
    session.
    """
    if role is None:
        role_value = LLMRole.GENERAL.value
    elif isinstance(role, LLMRole):
        role_value = role.value
    else:
        role_value = str(role).strip().lower()
        if role_value not in ROLE_MODEL_MAP:
            logger.warning(
                f"[LLM_CLIENT] Unknown role '{role_value}' — routing to GENERAL."
            )
            role_value = LLMRole.GENERAL.value
    return role_value, ROLE_MODEL_MAP.get(role_value, LLM_MODEL)


def llm_complete(
    system_content: str,
    user_content: str,
    role: LLMRole | str | None = None,
) -> str:
    """Unified LLM caller — legacy/paper pattern: self-contained per call.

    Raises:
        LLMError — on engine crash, timeout, or exhausted retry budget.
                   Clinically-important callers (Analyzer, R-V Reasoner,
                   CBT Reasoner) MUST catch this and mark the turn SKIPPED
                   rather than silently accepting the legacy placeholder
                   string, which would corrupt downstream classification.

    Returns the model's response (stripped). On empty/whitespace output
    returns the legacy placeholder "I am currently unable to access my
    language model, but I am listening." — distinguishable from a real
    answer by callers that care.
    """
    role_value, model_id = _resolve_model(role)
    logger.info(
        f"[LLM_CLIENT] Requesting in-process LiteRT inference "
        f"(role={role_value}, model={model_id})"
    )
    started_at = time.monotonic()

    _heartbeat_stop = threading.Event()

    def _heartbeat():
        while not _heartbeat_stop.wait(10.0):
            elapsed = time.monotonic() - started_at
            logger.info(
                f"[Heartbeat] LLM inference in progress... "
                f"Time elapsed: {elapsed:.0f}s. Role: {role_value}. Model: {model_id}."
            )

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        engine = _init_engine()
        prompt = _build_gemma_prompt(system_content, user_content)

        try:
            with heavy_stage(f"LLM/{model_id}/{role_value}"):
                with engine.create_conversation() as conversation:
                    response = conversation.send_message(prompt)
        except LLMError:
            raise
        except Exception as e:
            # Engine-level crash — invalidate singleton so the next call
            # can attempt a fresh init. Budget-exhausted cases raise
            # LLMError from _init_engine on the next call.
            _invalidate_engine(reason=f"generate failure: {e}")
            elapsed = time.monotonic() - started_at
            logger.error(f"LLM engine crash after {elapsed:.2f}s: {e}")
            raise LLMError(f"LLM engine crash during generate: {e}") from e

        # Extract text from the conversation response
        if isinstance(response, dict):
            content_parts = response.get("content", [])
            if content_parts and isinstance(content_parts[0], dict):
                result = content_parts[0].get("text", "")
            else:
                result = str(response)
        elif isinstance(response, str):
            result = response
        else:
            result = str(response)

        if not result or not result.strip():
            logger.warning("[LLM_CLIENT] Empty response from LiteRT engine.")
            # Successful round-trip, empty output — reset failure counter.
            _reset_engine_failure_count()
            return "I am currently unable to access my language model, but I am listening."

        _reset_engine_failure_count()
        content = result.strip()
        elapsed = time.monotonic() - started_at
        logger.info(
            f"Received response from LLM in {elapsed:.2f}s "
            f"(role={role_value}, LiteRT in-process)"
        )
        return content

    finally:
        _heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)


__all__ = [
    "LLMRole",
    "LLMError",
    "ROLE_MODEL_MAP",
    "llm_complete",
    "engine_is_healthy",
]
