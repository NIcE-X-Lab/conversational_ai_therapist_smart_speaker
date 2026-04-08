"""AI model wrapper abstracting communications with the primary LLM engine."""
import os
from openai import OpenAI
from src.utils.config_loader import OPENAI_BASE_URL, LLM_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, OLLAMA_KEEP_ALIVE
from src.utils.log_util import get_logger

logger = get_logger("LLMClient")

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

    logger.info("Sending request to LLM")
    logger.debug({"model": LLM_MODEL, "user": user_content})
    try:
        extra_body_params = {"keep_alive": OLLAMA_KEEP_ALIVE}
        try:
            resp = client.responses.create(
                model=LLM_MODEL,
                reasoning={"effort": "low"},
                instructions=system_content,
                input=user_content,
                extra_body=extra_body_params
            )
            logger.info("Received response from LLM (client.responses)")
            return resp.output_text
        except AttributeError:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=OPENAI_TEMPERATURE,
                extra_body=extra_body_params
            )
            logger.info("Received response from LLM (client.chat.completions)")
            return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM Call Failed: {e}")
        return "I am currently unable to access my language model, but I am listening."

def llm_complete_async(system_content: str, user_content: str):
    import concurrent.futures
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return pool.submit(llm_complete, system_content, user_content)


def _reset_screening():
    """Reset per-session interstitial state (screeners + meditation sequence)."""
    _get_intermission_manager().reset()


def llm_complete_with_interstitial(system_content: str, user_content: str, trigger_threshold: float = 3.0) -> str:
    """
    Async LLM wrapper with a looping filler engine.
    This keeps the user engaged during slow LLM inference turns (e.g. on Jetson Orin Nano).

    Sequence Flow:
    1. Fast Path (< threshold): wait silently.
    2. Slow Path (>= threshold):
       a. Check for unanswered Clinical Screening (PHQ-4 / GAD-2).
       b. If screening is done/refused, provide a guided Meditation.
       c. If the LLM is *still* thinking, fall back to neutral Waiting Music.

    LED Synchronization:
    The LED on/off logic is handled by the speech loop caller (SpeechInteractionLoop).
    We simple emit text via log_question().
    """
    # Start LLM inference in background and engage interstitial bridge while waiting
    future = llm_complete_async(system_content, user_content)
    _get_intermission_manager().engage_while_waiting(future=future, trigger_threshold=trigger_threshold)

    return future.result()


__all__ = ["llm_complete", "llm_complete_async", "llm_complete_with_interstitial", "_reset_screening"]
