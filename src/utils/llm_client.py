import os
from openai import OpenAI
from src.utils.config_loader import OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
from src.utils.log_util import get_logger

logger = get_logger("LLMClient")

_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    logger.warning("OPENAI_API_KEY is not set. Using dummy key for local LLM.")
    _api_key = "dummy"
client = OpenAI(api_key=_api_key, base_url=OPENAI_BASE_URL)


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
        from src.utils.io_record import get_user_context
        user_ctx = get_user_context()
        if user_ctx:
            system_content = f"{system_content}\n\n{user_ctx}"
    except ImportError:
        pass

    logger.info("Sending request to LLM")
    logger.debug({"model": OPENAI_MODEL, "user": user_content})
    try:
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                reasoning={"effort": "low"},
                instructions=system_content,
                input=user_content,
            )
            logger.info("Received response from LLM (client.responses)")
            return resp.output_text
        except AttributeError:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=OPENAI_TEMPERATURE,
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

def llm_complete_with_interstitial(system_content: str, user_content: str, trigger_threshold: float = 3.0) -> str:
    import time
    from src.utils.io_record import log_question
    import json
    
    future = llm_complete_async(system_content, user_content)
    
    start_time = time.time()
    while time.time() - start_time < trigger_threshold:
        if future.done():
            return future.result()
        time.sleep(0.1)
        
    meditations = [
        "While I'm processing that, let's try a quick breathing exercise. Breathe in deeply... and breathe out.",
        "It's taking me a moment to reflect. Let me invite you to close your eyes, and take a deep breath.",
        "While I think about your response, try noticing the contact between your feet and the ground.",
        "I'm still thinking. Let's practice a bit of counting: inhale for four seconds, and exhale for four seconds.",
        "Let's try a nostril awareness practice while I think. Focus on the sensation of air passing through your nose."
    ]
    import random
    if random.random() < 0.2:
        # Trigger GAD-2 Screening
        exercise = "While I'm processing, I'd like to do a quick check-in. Over the last 2 weeks, how often have you been bothered by feeling nervous, anxious, or not being able to stop worrying? Not at all, several days, or nearly every day?"
    else:
        exercise = random.choice(meditations)
    logger.info("LLM latency threshold exceeded. Triggering interstitial.")
    
    log_question(exercise)
    
    from src.utils.io_record import INPUT_QUEUE
    import queue
    
    while not future.done():
        try:
            user_input = INPUT_QUEUE.get_nowait()
            try:
                parsed = json.loads(user_input)
                text = parsed.get("transcript", "").lower()
            except:
                text = user_input.lower()
                
            if "don't want" in text or "stop" in text or "skip" in text:
                logger.info("Opt-out detected. Playing waiting music.")
                log_question("[PLAY_MUSIC] waiting_music.wav")
            elif "several" in text or "nearly" in text or "every day" in text or "half" in text or "often" in text or "always" in text:
                # GAD-2 naive scoring (score 1+ for any gives score >= 2 overall approx)
                from src.utils.io_record import append_to_csv
                append_to_csv("gad2_flag", "system", "[HIGH-PRIORITY-REVIEW] User hit GAD-2 criteria >= 2 during screening.")
        except queue.Empty:
            pass
        time.sleep(0.5)
        
    return future.result()

__all__ = ["llm_complete", "llm_complete_async", "llm_complete_with_interstitial"]
