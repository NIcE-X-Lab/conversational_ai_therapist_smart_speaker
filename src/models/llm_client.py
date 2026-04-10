"""AI model wrapper abstracting communications with the primary LLM engine.

Uses Ollama's native /api/chat endpoint directly instead of the OpenAI-compat
/v1/chat/completions endpoint, which has a known runner crash on Jetson ARM64
with Ollama v0.20.x.
"""
import atexit
import json
import os
import time
import threading
import requests as _requests
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
    LLM_LOG_STREAMING,
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

# Derive Ollama base URL (strip /v1 suffix used by OpenAI compat layer)
_OLLAMA_BASE = OPENAI_BASE_URL.replace("/v1", "").rstrip("/")
_OLLAMA_CHAT_URL = f"{_OLLAMA_BASE}/api/chat"

_INTERMISSION_MANAGER = None


def _get_intermission_manager():
    global _INTERMISSION_MANAGER
    if _INTERMISSION_MANAGER is None:
        from src.services.response_bridge import IntermissionManager
        _INTERMISSION_MANAGER = IntermissionManager()
    return _INTERMISSION_MANAGER


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

    # ── Context Governance: enforce sliding window ──────────────────────
    # Rough token estimate: 1 token ≈ 4 chars.  If the combined prompt
    # exceeds 75% of num_ctx, trim the injected context from the TAIL of
    # system_content.  The core instructions live at the head and must be
    # preserved; the appended user history / context pack is expendable.
    _CHARS_PER_TOKEN = 4
    _CTX_BUDGET_RATIO = 0.75
    total_chars = len(system_content) + len(user_content)
    budget_chars = int(OLLAMA_NUM_CTX * _CTX_BUDGET_RATIO * _CHARS_PER_TOKEN)
    if total_chars > budget_chars:
        overshoot = total_chars - budget_chars
        logger.warning(
            f"[CONTEXT GOV] Prompt est. {total_chars // _CHARS_PER_TOKEN} tokens "
            f"exceeds 75% budget ({OLLAMA_NUM_CTX}). Trimming {overshoot} chars "
            "from system content tail (injected context)."
        )
        system_content = system_content[:len(system_content) - overshoot]

    logger.info(f"[LLM_CLIENT] Requesting GPU-accelerated inference for model: {LLM_MODEL}")
    logger.debug({"model": LLM_MODEL, "user": user_content})
    started_at = time.monotonic()
    clear_inference_cache("Before LLM phase")
    logger.info(f"[Memory PRE-LLM] {get_system_memory_snapshot()}")
    RESOURCE_AUDIT.capture_process_inventory("llm_pre_call_inventory")

    # ── Strict VRAM Budget ─────────────────────────────────────────────────
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

    def _build_options(num_ctx_val):
        """Build the Ollama options payload with the given num_ctx."""
        return {
            "num_gpu": OLLAMA_NUM_GPU,
            "num_thread": 4,
            "num_ctx": num_ctx_val,
            "num_predict": min(OLLAMA_NUM_PREDICT, num_ctx_val),
            "num_batch": 1,
            "f16_kv": False,
            "temperature": OPENAI_TEMPERATURE,
        }

    def _is_oom_error(exc: Exception) -> bool:
        """Detect Ollama / llama runner OOM crashes (HTTP 500, runner terminated)."""
        msg = str(exc).lower()
        return any(tok in msg for tok in ("500", "runner", "oom", "terminated", "killed"))

    def _dispatch_llm(model_name, options, num_ctx_val):
        """Execute the LLM call via Ollama's native /api/chat endpoint."""
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

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "stream": LLM_LOG_STREAMING,
            "keep_alive": 0,
            "options": options,
        }

        with RESOURCE_AUDIT.track_peak(
            f"LLM/{model_name}/handshake_and_generate",
            metadata={"num_ctx": num_ctx_val, "num_gpu": OLLAMA_NUM_GPU},
        ):
            with heavy_stage(f"LLM/{model_name}"):
                RESOURCE_AUDIT.capture_point(
                    f"LLM/{model_name}/pre_dispatch",
                    extra={"phase": "handshake", "num_ctx": num_ctx_val},
                )
                request_timeout = (
                    (10, LLM_REQUEST_TIMEOUT_SECONDS)
                    if LLM_LOG_STREAMING
                    else LLM_REQUEST_TIMEOUT_SECONDS
                )

                with _requests.post(
                    _OLLAMA_CHAT_URL,
                    json=payload,
                    timeout=request_timeout,
                    stream=LLM_LOG_STREAMING,
                ) as resp:
                    if resp.status_code != 200:
                        error_msg = resp.text[:200]
                        raise RuntimeError(
                            f"Ollama HTTP {resp.status_code}: {error_msg}"
                        )

                    if LLM_LOG_STREAMING:
                        logger.info("[LLM_STREAM] Streaming enabled for debug visibility.")
                        content_chunks = []
                        for raw_line in resp.iter_lines(decode_unicode=True):
                            if not raw_line:
                                continue

                            if isinstance(raw_line, bytes):
                                line = raw_line.decode("utf-8", errors="ignore").strip()
                            else:
                                line = str(raw_line).strip()
                            if not line:
                                continue
                            if line.startswith("data:"):
                                line = line[5:].strip()
                            if line == "[DONE]":
                                break

                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                logger.debug(f"[LLM_STREAM] Ignored non-JSON chunk: {line[:120]}")
                                continue

                            if event.get("error"):
                                raise RuntimeError(f"Ollama stream error: {event.get('error')}")

                            chunk = event.get("message", {}).get("content", "")
                            if chunk:
                                content_chunks.append(chunk)
                                safe_chunk = chunk.replace("\n", "\\n")
                                logger.info(f"[LLM_STREAM] {safe_chunk}")

                            if event.get("done"):
                                break

                        RESOURCE_AUDIT.capture_point(f"LLM/{model_name}/post_dispatch")
                        content = "".join(content_chunks).strip()
                        if not content:
                            raise RuntimeError("Empty response from Ollama streaming API.")
                        logger.info(f"[LLM_STREAM] Stream complete. chars={len(content)}")
                        return content

                    data = resp.json()
                    RESOURCE_AUDIT.capture_point(f"LLM/{model_name}/post_dispatch")
                    content = data.get("message", {}).get("content", "")
                    if not content:
                        raise RuntimeError(f"Empty response from Ollama: {data}")
                    return content

    try:
        for model_name in models_to_try:
            try:
                logger.info(f"LLM attempt using model: {model_name}")
                clear_inference_cache(f"Before LLM attempt ({model_name})")
                options = _build_options(effective_num_ctx)

                try:
                    result = _dispatch_llm(model_name, options, effective_num_ctx)
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
                        options = _build_options(effective_num_ctx)
                        result = _dispatch_llm(model_name, options, effective_num_ctx)
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
