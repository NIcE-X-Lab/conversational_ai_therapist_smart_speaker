"""AI model wrapper abstracting communications with the primary LLM engine.

Uses Google's LiteRT-LM framework (litert-lm-api) for in-process Gemma 4 E2B
inference on Jetson Orin Nano.  No external server required — the model runs
directly inside this Python process via litert_lm.Engine.
"""
import atexit
import os
import time
import threading
from src.utils.config_loader import (
    LLM_MODEL,
    LITERT_MODEL_PATH,
    LITERT_BACKEND,
    LITERT_CONTEXT_LENGTH,
    LITERT_MAX_TOKENS,
    LLM_REQUEST_TIMEOUT_SECONDS,
    DISABLE_CONTEXT_HISTORY,
)
from src.utils.inference_guard import (
    clear_inference_cache,
    get_system_memory_snapshot,
    heavy_stage,
)
from src.utils.log_util import get_logger
from src.utils.resource_audit import get_resource_audit

logger = get_logger("LLMClient")
RESOURCE_AUDIT = get_resource_audit()

# ── LiteRT-LM engine singleton ────────────────────────────────────────────
_ENGINE = None
_ENGINE_LOCK = threading.Lock()

_INTERMISSION_MANAGER = None


def _get_intermission_manager():
    global _INTERMISSION_MANAGER
    if _INTERMISSION_MANAGER is None:
        from src.services.response_bridge import IntermissionManager
        _INTERMISSION_MANAGER = IntermissionManager()
    return _INTERMISSION_MANAGER


def _init_engine():
    """Lazy-initialise the LiteRT-LM inference engine (thread-safe singleton)."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

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

            # Select backend: try GPU if configured, fall back to CPU
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
            raise

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


def llm_complete(system_content: str, user_content: str, *, inject_context: bool = True) -> str:
    """
    Unified LLM caller used across the app.
    Inputs:
      - system_content: system prompt/instructions
      - user_content: user input/payload
      - inject_context: when True (default), appends user history and
        session context pack to the system prompt.  Set to False for
        utility calls (classification, rephrasing) where context bloats
        the prompt without benefit.
    Output:
      - plain text content returned by the model
    """
    if not inject_context:
        logger.debug("[LLM_CLIENT] Context injection skipped (inject_context=False).")
    elif DISABLE_CONTEXT_HISTORY:
        logger.info("[ZERO-HISTORY] Context injection disabled for diagnostic mode.")
    else:
        # Prepend rolling clinical takeaway so the Brain always has
        # up-to-date clinical status regardless of context trimming.
        try:
            from src.core.context_manager import get_context_manager
            system_content = get_context_manager().inject_into_prompt(system_content)
        except Exception:
            pass
        try:
            from src.utils.io_record import get_user_context, get_context_pack
            user_ctx = get_user_context()
            context_pack = get_context_pack()
            if user_ctx:
                system_content = f"{system_content}\n\n{user_ctx}"

            if context_pack:
                system_content = (
                    f"{system_content}\n\n"
                    "[Context Pack]\n"
                    f"User Transcript: {context_pack.get('user_transcript', '')}\n"
                    f"Emotion Tag: {context_pack.get('emotion_tag', 'Neutral')}\n"
                    f"RL State: {context_pack.get('rl_state', {})}\n"
                    f"Latest PHQ-4/GAD-2 Scores: {context_pack.get('screening_scores', {})}\n"
                )
        except ImportError:
            pass

    # ── Context Governance: enforce sliding window ──────────────────────
    _CHARS_PER_TOKEN = 4
    _CTX_BUDGET_RATIO = 0.75
    total_chars = len(system_content) + len(user_content)
    budget_chars = int(LITERT_CONTEXT_LENGTH * _CTX_BUDGET_RATIO * _CHARS_PER_TOKEN)
    if total_chars > budget_chars:
        overshoot = total_chars - budget_chars
        logger.warning(
            f"[CONTEXT GOV] Prompt est. {total_chars // _CHARS_PER_TOKEN} tokens "
            f"exceeds 75% budget ({LITERT_CONTEXT_LENGTH}). Trimming {overshoot} chars "
            "from system content tail (injected context)."
        )
        system_content = system_content[:len(system_content) - overshoot]

    logger.info(f"[LLM_CLIENT] Requesting in-process LiteRT inference ({LLM_MODEL})")
    started_at = time.monotonic()
    clear_inference_cache("Before LLM phase")
    logger.info(f"[Memory PRE-LLM] {get_system_memory_snapshot()}")
    RESOURCE_AUDIT.capture_process_inventory("llm_pre_call_inventory")

    # ── Heartbeat thread: logs every 10s so terminal never looks dead ──
    _heartbeat_stop = threading.Event()

    def _heartbeat():
        while not _heartbeat_stop.wait(10.0):
            elapsed = time.monotonic() - started_at
            logger.info(
                f"[Heartbeat] LLM inference in progress... "
                f"Time elapsed: {elapsed:.0f}s. Model: {LLM_MODEL}."
            )

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        engine = _init_engine()
        prompt = _build_gemma_prompt(system_content, user_content)

        RESOURCE_AUDIT.record_prompt_budget(
            model_name=LLM_MODEL,
            system_content=system_content,
            user_content=user_content,
            num_ctx=LITERT_CONTEXT_LENGTH,
            num_predict=LITERT_MAX_TOKENS,
            f16_kv=False,
        )

        with RESOURCE_AUDIT.track_peak(
            f"LLM/{LLM_MODEL}/generate",
            metadata={"context_length": LITERT_CONTEXT_LENGTH, "backend": LITERT_BACKEND},
        ):
            with heavy_stage(f"LLM/{LLM_MODEL}"):
                RESOURCE_AUDIT.capture_point(
                    f"LLM/{LLM_MODEL}/pre_dispatch",
                    extra={"phase": "generate", "context_length": LITERT_CONTEXT_LENGTH},
                )
                with engine.create_conversation() as conversation:
                    response = conversation.send_message(prompt)
                RESOURCE_AUDIT.capture_point(f"LLM/{LLM_MODEL}/post_dispatch")

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
            return "I am currently unable to access my language model, but I am listening."

        content = result.strip()
        elapsed = time.monotonic() - started_at
        logger.info(f"Received response from LLM in {elapsed:.2f}s (LiteRT in-process)")
        logger.info(f"[Memory POST-LLM] {get_system_memory_snapshot()}")
        RESOURCE_AUDIT.capture_process_inventory("llm_post_call_inventory")
        RESOURCE_AUDIT.write_report()
        return content

    except Exception as e:
        elapsed = time.monotonic() - started_at
        logger.info(f"[Memory POST-LLM FAIL] {get_system_memory_snapshot()}")
        RESOURCE_AUDIT.capture_process_inventory("llm_post_call_failure_inventory")
        RESOURCE_AUDIT.write_report()
        logger.error(f"LLM Call Failed after {elapsed:.2f}s: {e}")
        return "I am currently unable to access my language model, but I am listening."
    finally:
        _heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)


# Reuse a single thread-pool to avoid creating (and leaking) an executor per call.
_LLM_POOL = None
_LLM_POOL_LOCK = threading.Lock()


def _shutdown_llm_pool():
    global _LLM_POOL
    if _LLM_POOL is not None:
        _LLM_POOL.shutdown(wait=False)
        _LLM_POOL = None


def _get_llm_pool():
    global _LLM_POOL
    if _LLM_POOL is None:
        with _LLM_POOL_LOCK:
            if _LLM_POOL is None:
                import concurrent.futures
                _LLM_POOL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="llm_async"
                )
                atexit.register(_shutdown_llm_pool)
    return _LLM_POOL


def llm_complete_async(system_content: str, user_content: str):
    return _get_llm_pool().submit(llm_complete, system_content, user_content)


def _reset_screening():
    """Reset per-session interstitial state (screeners + meditation sequence)."""
    _get_intermission_manager().reset()


def llm_complete_with_interstitial(
    system_content: str,
    user_content: str,
    trigger_threshold: float = 2.0,
    led_controller=None,
) -> str:
    """
    Async LLM wrapper with a looping filler engine.
    This keeps the user engaged during slow LLM inference turns (e.g. on Jetson Orin Nano).

    Sequence Flow:
    1. Fast Path (< 2.0s): wait silently.
    2. Slow Path (>= 2.0s):
       a. Immediately trigger PHQ-4/GAD-2 clinical screening.
       b. If screening is done/refused, provide a guided Breathing Exercise.
       c. Stay in Exercise State until the LLM thread signals done.
       d. If exercises exhaust, fall back to neutral Waiting Music.

    LED Synchronization:
    Pin 18 (Green LED) is held LOW for the entire Thinking + Intermission phase
    to signal that the device is processing, not listening.  The caller passes
    in a `led_controller(bool)` callable (typically `gpio.set_led`).
    """
    future = llm_complete_async(system_content, user_content)
    _get_intermission_manager().engage_while_waiting(
        future=future,
        trigger_threshold=trigger_threshold,
        led_controller=led_controller,
    )

    try:
        return future.result(timeout=LLM_REQUEST_TIMEOUT_SECONDS + 5)
    except Exception as e:
        logger.error(f"LLM future timeout/failure in interstitial wrapper: {e}")
        return "I am currently unable to access my language model, but I am listening."


__all__ = ["llm_complete", "llm_complete_async", "llm_complete_with_interstitial", "_reset_screening"]
