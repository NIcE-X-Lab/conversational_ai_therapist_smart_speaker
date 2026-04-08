"""AI model wrapper abstracting communications with the primary LLM engine."""
import os
import itertools
from openai import OpenAI
from src.utils.config_loader import OPENAI_BASE_URL, LLM_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, OLLAMA_KEEP_ALIVE
from src.utils.log_util import get_logger
from src.core.therapy_content import (
    CLINICAL_SCREENING, GAD2_QUESTIONS, MEDITATIONS,
    WAITING_MUSIC_PATH, score_response,
    GAD2_THRESHOLD, PHQ4_THRESHOLD,
)

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

    # Inject latest PHQ-4 / GAD-2 scores so the LLM can adapt its next question
    if _anxiety_scores or _depression_scores:
        screening_ctx = (
            "\n[Clinical Screening — Current Session]\n"
            f"Anxiety sub-score (GAD-2): {sum(_anxiety_scores)} (items answered: {len(_anxiety_scores)})\n"
            f"Depression sub-score (PHQ-2): {sum(_depression_scores)} (items answered: {len(_depression_scores)})\n"
        )
        system_content = f"{system_content}\n{screening_ctx}"

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


# ── Per-session PHQ-4 state ────────────────────────────────────────────────────
_screening_index: int = 0          # which CLINICAL_SCREENING question is next
_anxiety_scores:  list = []        # GAD-2 sub-scores (indices 0-1)
_depression_scores: list = []      # PHQ-2 sub-scores (indices 2-3)
_screening_refused: bool = False   # user opted out of all screening

_meditation_cycle = itertools.cycle(MEDITATIONS)


def _reset_screening():
    """Reset the per-session PHQ-4 accumulators (call on new session start)."""
    global _screening_index, _anxiety_scores, _depression_scores, _screening_refused
    _screening_index = 0
    _anxiety_scores = []
    _depression_scores = []
    _screening_refused = False


def _next_screening_question(allow_phq2: bool) -> str | None:
    """
    Return the text of the next unanswered screening question, or None if
    all appropriate questions have been asked / refused.

    allow_phq2=False  → only offer GAD-2 (indices 0-1)
    allow_phq2=True   → offer full PHQ-4 (indices 0-3)
    """
    global _screening_index, _screening_refused
    if _screening_refused:
        return None
    max_idx = len(CLINICAL_SCREENING) if allow_phq2 else len(GAD2_QUESTIONS)
    if _screening_index >= max_idx:
        return None
    return CLINICAL_SCREENING[_screening_index]["text"]


def _store_screening_response(raw_text: str) -> bool:
    """
    Score a raw vocal response and store it against the current question index.
    Returns True if the response was an opt-out (refused).
    """
    global _screening_index, _anxiety_scores, _depression_scores, _screening_refused
    score = score_response(raw_text)

    if score == -1:
        # User refused this question; mark screening as refused and stop asking
        logger.info("Screening opt-out detected — skipping remaining screening questions.")
        _screening_refused = True
        return True

    q = CLINICAL_SCREENING[_screening_index]
    scale = q["scale"]
    if scale == "anxiety":
        _anxiety_scores.append(score)
    else:
        _depression_scores.append(score)

    logger.info(f"Screening [{q['id']}] scored {score} (scale={scale}). "
                f"anxiety={_anxiety_scores}, depression={_depression_scores}")
    _screening_index += 1
    return False


def _evaluate_and_flag(
    session_id,
    log_json_event,
    append_to_csv,
    db,
) -> None:
    """
    After all collected scores, compute sub-totals, persist them, and raise
    clinical flags if thresholds are breached.
    """
    anxiety_total    = sum(_anxiety_scores)
    depression_total = sum(_depression_scores)
    phq4_total       = anxiety_total + depression_total

    logger.info(f"Screening complete — anxiety={anxiety_total}, "
                f"depression={depression_total}, PHQ-4={phq4_total}")

    # Persist sub-scores to DB
    if db and session_id:
        try:
            db.log_screening_scores(
                session_id,
                anxiety_score=anxiety_total if _anxiety_scores else None,
                depression_score=depression_total if _depression_scores else None,
                phq4_total=phq4_total if (_anxiety_scores or _depression_scores) else None,
            )
        except Exception as e:
            logger.warning(f"Could not persist screening scores: {e}")

    # Log event for JSON audit trail
    log_json_event("screening_complete", {
        "anxiety_score": anxiety_total,
        "depression_score": depression_total,
        "phq4_total": phq4_total,
    })

    # Flagging
    if _anxiety_scores and anxiety_total >= GAD2_THRESHOLD:
        flag = f"[CLINICAL-FLAG] GAD2_POSITIVE — anxiety_score={anxiety_total} >= {GAD2_THRESHOLD}"
        logger.warning(flag)
        append_to_csv("clinical_flag", "system", flag)
        log_json_event("GAD2_POSITIVE", {"anxiety_score": anxiety_total})

    if (_anxiety_scores or _depression_scores) and phq4_total >= PHQ4_THRESHOLD:
        flag = f"[CLINICAL-FLAG] PHQ4_HIGH_RISK — total_score={phq4_total} >= {PHQ4_THRESHOLD}"
        logger.warning(flag)
        append_to_csv("clinical_flag", "system", flag)
        log_json_event("PHQ4_HIGH_RISK", {"phq4_total": phq4_total})


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
    import time
    import json
    import random
    import queue as _queue
    from src.utils.io_record import log_question, append_to_csv, log_json_event
    from src.utils.io_record import INPUT_QUEUE
    try:
        from src.utils import io_record as _io
        _session_id = _io.SESSION_ID
        _db = _io.DB
    except Exception:
        _session_id = None
        _db = None

    FILLERS = [
        "I'm still thinking, let's practice something together while I reflect...",
        "It's taking me a moment to reflect. Let me invite you to try this with me...",
        "While I'm processing that, let's try a quick breathing exercise...",
        "Let's try this while I work through your response...",
        "Give me just a moment. While we wait, let's try a little awareness practice together...",
    ]

    # Start the LLM inference in the background
    future = llm_complete_async(system_content, user_content)
    
    start_time = time.time()
    
    # ── Loop until LLM result arrives or user opts-out ──────────────────────────
    while not future.done():
        elapsed = time.time() - start_time
        
        # 1. Fast Path - Wait silently for the initial 3 seconds
        if elapsed < trigger_threshold:
            time.sleep(0.2)
            continue
            
        # 2. Slow Path - LLM is taking a while; start the interstitial sequence
        allow_phq2 = elapsed >= (trigger_threshold * 2)
        question_text = _next_screening_question(allow_phq2=allow_phq2)

        if question_text and not _screening_refused:
            is_screening = True
            # Prefix with a filler to smooth the transition
            filler = random.choice(FILLERS)
            play_text = f"{filler} {question_text}"
            logger.info(f"Looping Interstitial: asking clinical question ({CLINICAL_SCREENING[_screening_index]['id']})")
        else:
            is_screening = False
            # Pull a meditation from the cycle
            exercise_text = next(_meditation_cycle)
            # The meditations in therapy_content already have headers, check if we need to add more
            play_text = exercise_text
            logger.info("Looping Interstitial: playing guided meditation.")

        # Emit to the speech loop
        log_question(play_text)
        
        # ── Monitor response while LLM is thinking ────────────────────────────
        # (This block also serves as the wait-timer for the exercise)
        # For screening, we MUST wait for a response. For meditation, we just wait.
        
        turn_start = time.time()
        # Max wait for user response / meditation duration check
        while not future.done():
            try:
                # Poll for user response if we asked a question
                # (Timeout based on typical VAD recording + speech duration)
                raw_input = INPUT_QUEUE.get(timeout=0.5)
                
                try:
                    parsed = json.loads(raw_input)
                    text = parsed.get("transcript", "").lower().strip()
                except Exception:
                    text = str(raw_input).lower().strip()

                if not text:
                    continue

                # Handle global opt-out
                if any(kw in text for kw in ("don't want", "stop", "skip", "opt out", "no thanks")):
                    logger.info("Opt-out detected during interstitial.")
                    global _screening_refused
                    _screening_refused = True
                    log_question(f"[PLAY_MUSIC] {WAITING_MUSIC_PATH}")
                    break 

                # Process clinical response
                if is_screening:
                    refused = _store_screening_response(text)
                    if not refused:
                        log_json_event("screening_response", {
                            "question_id": CLINICAL_SCREENING[_screening_index - 1]["id"],
                            "text": text,
                            "score": score_response(text)
                        })
                    
                    # If screening index is complete, check if we should run analysis
                    if _screening_index >= len(CLINICAL_SCREENING):
                        _evaluate_and_flag(_session_id, log_json_event, append_to_csv, _db)
                    
                    # Question answered! break inner loop to see if LLM is done 
                    # or if we should ask the NEXT question.
                    break 

            except _queue.Empty:
                # For meditation (no response expected), we check if the duration has passed 
                # OR if the LLM finished. 
                # (Simulate ~15-20s per meditation turns if LLM is very slow)
                if not is_screening and (time.time() - turn_start > 15.0):
                    break
                continue

    return future.result()


__all__ = ["llm_complete", "llm_complete_async", "llm_complete_with_interstitial", "_reset_screening"]
