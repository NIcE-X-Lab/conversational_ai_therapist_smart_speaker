"""AI model wrapper abstracting communications with the primary LLM engine."""
import os
import time
import threading
from openai import OpenAI
from src.utils.config_loader import (
    OPENAI_BASE_URL,
    LLM_MODEL,
    LLM_FALLBACK_MODELS,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
    LLM_REQUEST_TIMEOUT_SECONDS,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_NUM_GPU,
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

_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    logger.warning("OPENAI_API_KEY is not set. Using dummy key for local LLM.")
    _api_key = "dummy"
client = OpenAI(api_key=_api_key, base_url=OPENAI_BASE_URL)
_INTERMISSION_MANAGER = None


def _get_intermission_manager():
    global _INTERMISSION_MANAGER
    if _INTERMISSION_MANAGER is None:
        from src.services.response_bridge import IntermissionManager
        _INTERMISSION_MANAGER = IntermissionManager()
    return _INTERMISSION_MANAGER


def llm_complete(system_content: str, user_content: str) -> str:
    """
    Unified LLM caller used across the app.
    Inputs:
      - system_content: system prompt/instructions
      - user_content: user input/payload
    Output:
      - plain text content returned by the model
    """
    if DISABLE_CONTEXT_HISTORY:
        logger.info("[ZERO-HISTORY] Context injection disabled for diagnostic mode.")
    else:
        try:
            from src.utils.io_record import get_user_context, get_context_pack
            user_ctx = get_user_context()
            context_pack = get_context_pack()
            if user_ctx:
                system_content = f"{system_content}\n\n{user_ctx}"

            # Unified context pack for downstream CBT/RL/Reflection orchestration
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

    logger.info(f"[LLM_CLIENT] Requesting GPU-accelerated inference for model: {LLM_MODEL}")
    logger.debug({"model": LLM_MODEL, "user": user_content})
    started_at = time.monotonic()
    clear_inference_cache("Before LLM phase")
    logger.info(f"[Memory PRE-LLM] {get_system_memory_snapshot()}")
    RESOURCE_AUDIT.capture_process_inventory("llm_pre_call_inventory")

    # ── Strict VRAM Budget ─────────────────────────────────────────────────
    # The 3B model's cudaMalloc was requesting ~1918 MiB at num_ctx=512.
    # With num_ctx=256, f16_kv=false, num_batch=1, this should drop to
    # ~1200 MiB.  Auto-scaling decrements by 64 (finer steps at low ctx)
    # on OOM, with a floor of 128 tokens.
    _CTX_DECREMENT = 64
    _CTX_FLOOR = 128
    effective_num_ctx = OLLAMA_NUM_CTX
    logger.info(
        f"[VRAM BUDGET] Strict profile: num_ctx={effective_num_ctx}, "
        f"num_batch=1, f16_kv=false, num_gpu={OLLAMA_NUM_GPU}, "
        f"num_predict={min(OLLAMA_NUM_PREDICT, effective_num_ctx)}"
    )

    models_to_try = [LLM_MODEL] + [m for m in LLM_FALLBACK_MODELS if m != LLM_MODEL]
    last_error = None

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

    def _build_extra_body(num_ctx_val):
        """Build the Ollama extra_body payload with the given num_ctx."""
        return {
            "keep_alive": 0,
            "options": {
                "num_gpu": OLLAMA_NUM_GPU,
                "num_thread": 4,
                "num_ctx": num_ctx_val,
                "num_predict": min(OLLAMA_NUM_PREDICT, num_ctx_val),
                "num_batch": 1,
                "f16_kv": False,
                "temperature": OPENAI_TEMPERATURE,
            },
        }

    def _is_oom_error(exc: Exception) -> bool:
        """Detect Ollama / llama runner OOM crashes (HTTP 500, runner terminated)."""
        msg = str(exc).lower()
        return any(tok in msg for tok in ("500", "runner", "oom", "terminated", "killed"))

    def _dispatch_llm(model_name, extra_body, num_ctx_val):
        """Execute the LLM call.  Returns response text or raises."""
        logger.info(
            f"[LLM_CLIENT] GPU payload: num_gpu={OLLAMA_NUM_GPU}, num_thread=4, "
            f"num_ctx={num_ctx_val}, num_predict={min(OLLAMA_NUM_PREDICT, num_ctx_val)}, "
            f"num_batch=1, f16_kv=false"
        )
        RESOURCE_AUDIT.record_prompt_budget(
            model_name=model_name,
            system_content=system_content,
            user_content=user_content,
            num_ctx=num_ctx_val,
            num_predict=min(OLLAMA_NUM_PREDICT, num_ctx_val),
            f16_kv=False,
        )
        with RESOURCE_AUDIT.track_peak(
            f"LLM/{model_name}/handshake_and_generate",
            metadata={"num_ctx": num_ctx_val, "num_gpu": OLLAMA_NUM_GPU},
        ):
            with heavy_stage(f"LLM/{model_name}"):
                try:
                    RESOURCE_AUDIT.capture_point(
                        f"LLM/{model_name}/pre_dispatch",
                        extra={"phase": "handshake", "num_ctx": num_ctx_val},
                    )
                    resp = client.responses.create(
                        model=model_name,
                        reasoning={"effort": "low"},
                        instructions=system_content,
                        input=user_content,
                        max_output_tokens=OPENAI_MAX_TOKENS,
                        extra_body=extra_body,
                        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                    )
                    RESOURCE_AUDIT.capture_point(f"LLM/{model_name}/post_dispatch")
                    return resp.output_text
                except AttributeError:
                    RESOURCE_AUDIT.capture_point(
                        f"LLM/{model_name}/pre_dispatch_chat_completions",
                        extra={"phase": "handshake", "num_ctx": num_ctx_val},
                    )
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content},
                        ],
                        max_tokens=OPENAI_MAX_TOKENS,
                        temperature=OPENAI_TEMPERATURE,
                        extra_body=extra_body,
                        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                    )
                    RESOURCE_AUDIT.capture_point(f"LLM/{model_name}/post_dispatch_chat_completions")
                    return resp.choices[0].message.content

    try:
        for model_name in models_to_try:
            try:
                logger.info(f"LLM attempt using model: {model_name}")
                clear_inference_cache(f"Before LLM attempt ({model_name})")
                extra_body = _build_extra_body(effective_num_ctx)

                try:
                    result = _dispatch_llm(model_name, extra_body, effective_num_ctx)
                except Exception as first_err:
                    if _is_oom_error(first_err) and effective_num_ctx - _CTX_DECREMENT >= _CTX_FLOOR:
                        # ── Auto-Scale: decrement num_ctx and retry once ──
                        old_ctx = effective_num_ctx
                        effective_num_ctx -= _CTX_DECREMENT
                        logger.warning(
                            f"[AUTO-SCALE] OOM detected (num_ctx={old_ctx}). "
                            f"Decrementing to num_ctx={effective_num_ctx} and retrying."
                        )
                        clear_inference_cache(f"After OOM, before retry at num_ctx={effective_num_ctx}")
                        extra_body = _build_extra_body(effective_num_ctx)
                        result = _dispatch_llm(model_name, extra_body, effective_num_ctx)
                    else:
                        raise

                elapsed = time.monotonic() - started_at
                logger.info(f"Received response from LLM in {elapsed:.2f}s (num_ctx={effective_num_ctx})")
                logger.info(f"[Memory POST-LLM] {get_system_memory_snapshot()}")
                RESOURCE_AUDIT.capture_process_inventory("llm_post_call_inventory")
                RESOURCE_AUDIT.write_report()
                return result

            except Exception as e:
                last_error = e
                logger.error(f"LLM model attempt failed ({model_name}): {e}")

        elapsed = time.monotonic() - started_at
        logger.info(f"[Memory POST-LLM FAIL] {get_system_memory_snapshot()}")
        RESOURCE_AUDIT.capture_process_inventory("llm_post_call_failure_inventory")
        RESOURCE_AUDIT.write_report()
        logger.error(f"LLM Call Failed after {elapsed:.2f}s across all models: {last_error}")
        return "I am currently unable to access my language model, but I am listening."
    finally:
        _heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)

# Reuse a single thread-pool to avoid creating (and leaking) an executor per call.
_LLM_POOL = None
_LLM_POOL_LOCK = threading.Lock()


def _get_llm_pool():
    global _LLM_POOL
    if _LLM_POOL is None:
        with _LLM_POOL_LOCK:
            if _LLM_POOL is None:
                import concurrent.futures
                _LLM_POOL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="llm_async"
                )
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
    # Start LLM inference in background thread and engage interstitial bridge
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
