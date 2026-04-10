"""Service bridging data payloads between perception and orchestration."""
import re
import json
import time
import random
import itertools
import threading
import queue as _queue
from concurrent.futures import Future
from src.core.response_analyzer import classify_dimension_and_score
from src.core.therapy_content import (
    CLINICAL_SCREENING,
    GAD2_QUESTIONS,
    MEDITATIONS,
    WAITING_MUSIC_PATH,
    score_response,
    GAD2_THRESHOLD,
    PHQ4_THRESHOLD,
)
from src.utils.log_util import get_logger
logger = get_logger("ResponseBridge")


class IntermissionManager:
    """
    Non-blocking interstitial manager used while LLM inference is in-flight.
    Sequence priority:
      1) Clinical screening (GAD-2 first, optionally PHQ-2)
      2) Guided breathing / meditation prompts
      3) Waiting music fallback

    Thread-safety: a lock protects all mutable screening state so that
    concurrent calls to current_scores() / reset() / _store_screening_response()
    never see a torn snapshot.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._screening_index = 0
        self._anxiety_scores = []
        self._depression_scores = []
        self._screening_refused = False
        self._meditation_cycle = itertools.cycle(MEDITATIONS)

    def reset(self):
        with self._lock:
            self._screening_index = 0
            self._anxiety_scores = []
            self._depression_scores = []
            self._screening_refused = False

    def current_scores(self) -> dict:
        with self._lock:
            return {
                "anxiety": sum(self._anxiety_scores) if self._anxiety_scores else None,
                "depression": sum(self._depression_scores) if self._depression_scores else None,
                "total": (sum(self._anxiety_scores) + sum(self._depression_scores)) if (self._anxiety_scores or self._depression_scores) else None,
                "anxiety_items": len(self._anxiety_scores),
                "depression_items": len(self._depression_scores),
                "screening_refused": self._screening_refused,
            }

    def _next_screening_question(self, allow_phq2: bool):
        if self._screening_refused:
            return None
        max_idx = len(CLINICAL_SCREENING) if allow_phq2 else len(GAD2_QUESTIONS)
        if self._screening_index >= max_idx:
            return None
        return CLINICAL_SCREENING[self._screening_index]["text"]

    def _persist_scores(self, db, session_id, log_json_event, append_to_csv):
        anxiety_total = sum(self._anxiety_scores) if self._anxiety_scores else None
        depression_total = sum(self._depression_scores) if self._depression_scores else None
        phq4_total = None
        if anxiety_total is not None or depression_total is not None:
            phq4_total = (anxiety_total or 0) + (depression_total or 0)

        try:
            from src.utils import io_record as _io
            _io.set_latest_screening_scores(anxiety_total, depression_total, phq4_total)
        except Exception:
            pass

        if db and session_id and (anxiety_total is not None or depression_total is not None):
            try:
                db.log_screening_scores(
                    session_id,
                    anxiety_score=anxiety_total,
                    depression_score=depression_total,
                    phq4_total=phq4_total,
                )
            except Exception as e:
                logger.warning(f"Could not persist screening scores: {e}")

        log_json_event("screening_scores_snapshot", {
            "anxiety_score": anxiety_total,
            "depression_score": depression_total,
            "phq4_total": phq4_total,
        })

        if anxiety_total is not None and anxiety_total >= GAD2_THRESHOLD:
            flag = f"[CLINICAL-FLAG] GAD2_POSITIVE — anxiety_score={anxiety_total} >= {GAD2_THRESHOLD}"
            append_to_csv("clinical_flag", "system", flag)
            log_json_event("GAD2_POSITIVE", {"anxiety_score": anxiety_total})

        if phq4_total is not None and phq4_total >= PHQ4_THRESHOLD:
            flag = f"[CLINICAL-FLAG] PHQ4_HIGH_RISK — total_score={phq4_total} >= {PHQ4_THRESHOLD}"
            append_to_csv("clinical_flag", "system", flag)
            log_json_event("PHQ4_HIGH_RISK", {"phq4_total": phq4_total})

    def _store_screening_response(self, raw_text: str, db, session_id, log_json_event, append_to_csv):
        score = score_response(raw_text)
        with self._lock:
            if score == -1:
                self._screening_refused = True
                self._persist_scores(db, session_id, log_json_event, append_to_csv)
                return True

            q = CLINICAL_SCREENING[self._screening_index]
            if q["scale"] == "anxiety":
                self._anxiety_scores.append(score)
            else:
                self._depression_scores.append(score)

            self._screening_index += 1
        self._persist_scores(db, session_id, log_json_event, append_to_csv)
        return False

    # Default intermission trigger: 2.0 seconds of LLM silence before engaging
    # the user with clinical screening or breathing exercises.
    _DEFAULT_TRIGGER_THRESHOLD = 2.0

    def engage_while_waiting(
        self,
        future: Future,
        trigger_threshold: float | None = None,
        led_controller=None,
    ):
        """
        Non-blocking intermission engine.  Keeps the user therapeutically
        engaged while the LLM background thread is still processing.

        Args:
            future:            The LLM background thread future.
            trigger_threshold: Seconds to wait before triggering intermission.
            led_controller:    Optional callable(bool) to control Pin 18 LED.
                               LED is held LOW (off) for the entire duration of
                               this method — signalling "processing, not listening".

        State machine:
          WAITING   — silent fast-path (< trigger_threshold).
          EXERCISE  — clinical screening or breathing exercise active.
                      Stays in this state until the LLM thread signals done.
          MUSIC     — ambient fallback when exercises are exhausted.
          DONE      — LLM response received; current sentence allowed to finish.
        """
        if trigger_threshold is None:
            trigger_threshold = self._DEFAULT_TRIGGER_THRESHOLD

        # ── LED: Pin 18 LOW for entire thinking + intermission phase ───────
        # Resolve LED controller: prefer explicit arg, fall back to GPIO singleton.
        if led_controller is None:
            try:
                from src.drivers.gpio_manager import GPIOManager
                led_controller = GPIOManager().set_led
            except Exception:
                pass
        if led_controller is not None:
            led_controller(False)
            logger.info("[LED] Pin 18 LOW — device is processing, not listening.")

        from src.utils.io_record import log_question, append_to_csv, log_json_event, INPUT_QUEUE, END_SESSION_EVENT
        from src.utils import io_record as _io

        fillers = [
            "I'm still thinking, let's practice something together while I reflect...",
            "It's taking me a moment to reflect. Let me invite you to try this with me...",
            "While I'm processing that, let's try a quick breathing exercise...",
            "Let's try this while I work through your response...",
            "Give me just a moment. While we wait, let's try a little awareness practice together...",
        ]

        _HEARTBEAT_INTERVAL = 10.0
        start_time = time.time()
        last_heartbeat = start_time

        # ── State: WAITING — silent fast-path ──────────────────────────────
        while not future.done():
            if END_SESSION_EVENT.is_set():
                logger.info("[Intermission] END_SESSION_EVENT detected in WAITING state. Aborting.")
                return

            now = time.time()
            elapsed = now - start_time

            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                last_heartbeat = now
                logger.info(
                    f"[Heartbeat] LLM still thinking... "
                    f"Time elapsed: {elapsed:.0f}s. Waiting silently."
                )

            if elapsed >= trigger_threshold:
                logger.info(
                    f"Trigger threshold ({trigger_threshold}s) reached at {elapsed:.1f}s. "
                    "Entering EXERCISE state."
                )
                break  # transition → EXERCISE state

            time.sleep(0.2)

        if future.done():
            return  # fast path — LLM answered within threshold

        # ── State: EXERCISE — stay here until LLM signals done ─────────────
        _llm_ready = False

        while not _llm_ready:
            if END_SESSION_EVENT.is_set():
                logger.info("[Intermission] END_SESSION_EVENT detected in EXERCISE state. Breaking out.")
                break

            now = time.time()
            elapsed = now - start_time

            # Heartbeat
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                last_heartbeat = now
                logger.info(
                    f"[Heartbeat] LLM still thinking... "
                    f"Time elapsed: {elapsed:.0f}s. Intermission active."
                )

            # Select next interstitial content
            allow_phq2 = elapsed >= (trigger_threshold * 2)
            question_text = self._next_screening_question(allow_phq2=allow_phq2)

            if question_text and not self._screening_refused:
                is_screening = True
                play_text = f"{random.choice(fillers)} {question_text}"
                logger.info("Intermission: clinical screening question.")
            else:
                is_screening = False
                play_text = next(self._meditation_cycle)
                logger.info("Intermission: guided breathing / meditation.")

            log_question(play_text)

            # ── Collect user response for this exercise turn ───────────────
            turn_start = time.time()
            _llm_finished_during_turn = False

            while True:
                # Detect LLM completion — but let current sentence finish
                if future.done() and not _llm_finished_during_turn:
                    _llm_finished_during_turn = True
                    logger.info(
                        f"[Heartbeat] LLM response ready after "
                        f"{time.time() - start_time:.1f}s. "
                        "Allowing current interstitial turn to finish."
                    )

                # Heartbeat during turn
                now = time.time()
                if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                    last_heartbeat = now
                    logger.info(
                        f"[Heartbeat] LLM still thinking... "
                        f"Time elapsed: {now - start_time:.0f}s. Intermission active."
                    )

                try:
                    raw_input = INPUT_QUEUE.get(timeout=0.5)
                    try:
                        parsed = json.loads(raw_input)
                        text = parsed.get("transcript", "").lower().strip()
                    except Exception:
                        text = str(raw_input).lower().strip()

                    if not text:
                        continue

                    # Opt-out detection
                    if any(kw in text for kw in (
                        "don't want", "stop", "skip", "opt out", "no thanks"
                    )):
                        self._screening_refused = True
                        self._persist_scores(
                            _io.DB, _io.SESSION_ID,
                            log_json_event, append_to_csv,
                        )
                        if not _llm_finished_during_turn:
                            log_question(f"[PLAY_MUSIC] {WAITING_MUSIC_PATH}")
                        break

                    # Score screening response
                    if is_screening:
                        refused = self._store_screening_response(
                            text, _io.DB, _io.SESSION_ID,
                            log_json_event, append_to_csv,
                        )
                        if not refused:
                            log_json_event("screening_response", {
                                "question_id": CLINICAL_SCREENING[
                                    self._screening_index - 1
                                ]["id"],
                                "text": text,
                                "score": score_response(text),
                            })
                        break

                except _queue.Empty:
                    # Meditation timeout (15s) — queue waiting music and
                    # loop back for the next exercise.
                    if not is_screening and (time.time() - turn_start > 15.0):
                        if not _llm_finished_during_turn:
                            log_question(f"[PLAY_MUSIC] {WAITING_MUSIC_PATH}")
                        break
                    # If LLM already finished during a non-screening turn,
                    # no need to wait for user input — break immediately.
                    if _llm_finished_during_turn and not is_screening:
                        break
                    continue

            # ── Transition check ───────────────────────────────────────────
            if _llm_finished_during_turn or future.done():
                _llm_ready = True
                logger.info(
                    f"Intermission complete. LLM response ready after "
                    f"{time.time() - start_time:.1f}s total."
                )

def _normalize_dim_score(dim: str, score: int):
    """
    If dimension looks like DLA_digits_label or digits_label, strip prefix and keep only the label.
    Otherwise require DLA_ prefix.
    Always validate score range.
    """
    logger.debug(f"Normalizing dimension and score: dim={dim}, score={score}")
    # Check for patterns: DLA_digits_label or digits_label
    m = re.match(r"^(?:DLA_)?(\d+)_([A-Za-z_]+)$", dim)
    if m:
        # Only keep the label part
        dim = m.group(2)
        logger.debug(f"Normalized dimension to label only: {dim}")

    # Ensure score is an integer between 0 and 2 (inclusive)
    if not isinstance(score, int) or score < 0 or score > 2:
        logger.warning(f"Score {score} is invalid, must be int in 0-2")
        return None

    logger.debug(f"Dimension and score normalized: ({dim}, {score})")
    return dim, score

def _parse_dim_score_from_text(text: str):
    """
    Parse '[dim][sep][score]' from a free-form text line.
    Supports formats like 'talk, 1', '3_talk, 1', 'DLA_3_talk, 1', etc.
    Accept separators: comma, colon, hyphen, or whitespace.
    """
    logger.debug(f"Parsing dimension-score from text: {text}")
    # Regex to match general pattern: can match 'talk, 1', '3_talk:1', 'DLA_3_talk - 0', etc.
    m = re.search(
        r"\b((?:DLA_)?(?:\d+_)?[A-Za-z_]+)\s*[,:\-\s]\s*([0-2])\b",
        text
    )
    if m:
        dim = m.group(1).strip()
        score = int(m.group(2))
        logger.debug(f"Extracted with regex: dim={dim}, score={score}")
        norm = _normalize_dim_score(dim, score)
        if norm:
            logger.debug(f"Parsed dimension-score (text): {norm[0]}, {norm[1]}")
            return norm
        else:
            logger.warning(f"Normalization failed for: dim={dim}, score={score}")
    else:
        logger.debug("No matching pattern found for dimension-score in text.")
    return None

def _parse_from_json_like(raw: str):
    """
    If the model returns JSON-like content, try to extract:
    - {'res': '3_talk, 1'}
    - {'dimension': '3_talk', 'score': 1}
    """
    s = str(raw).strip()
    logger.debug(f"Trying to parse as JSON-like: {s}")
    # Only try if string looks like JSON object
    if s.startswith("{") and s.endswith("}"):
        try:
            # Parse JSON to dictionary
            data = json.loads(s)
            logger.debug(f"Parsed JSON classification: {data}")
            # Lowercase all keys for robust lookup
            kl = {str(k).lower(): v for k, v in data.items()}
            # Case 1: 'res' key contains a string like '3_talk, 1'
            if "res" in kl:
                val = str(kl["res"]).strip()
                logger.debug(f"Found 'res' in JSON: {val}")
                got = _parse_dim_score_from_text(val)
                if got:
                    logger.debug(f"Extracted from 'res': {got}")
                    return got
            # Case 2: JSON object has separate 'dimension' and 'score' keys
            if "dimension" in kl and "score" in kl:
                dim = str(kl["dimension"]).strip()
                sc = int(kl["score"])
                logger.debug(f"Found 'dimension' and 'score' in JSON: dim={dim}, score={sc}")
                norm = _normalize_dim_score(dim, sc)
                if norm:
                    logger.debug(f"Parsed dimension-score (json): {norm[0]}, {norm[1]}")
                    return norm
            # Fallback: try to extract from string form of the object in case the above failed
            logger.debug("Trying fallback: parsing content as text for dim-score extraction...")
            got = _parse_dim_score_from_text(s)
            if got:
                logger.debug(f"Extracted from stringified JSON: {got}")
                return got
        except Exception as e:
            logger.warning(f"Failed to parse JSON-like string: {e}")
    else:
        logger.debug("Input does not appear to be a JSON object.")
    # If not JSON object or any extraction method failed
    logger.debug("Could not parse dimension-score from JSON-like content.")
    return None

def get_openai_resp(user_input, original_question, dimension_label: str):
    """
    Main entry point to process model response or user input and extract a unified tuple.
    For general Yes/No/Stop/Maybe/Question answers, returns (dimension_label, Keyword).
    Otherwise, attempts to return (dimension, score:int) parsed from model output.
    Fallbacks to ('NA', 99) on parse failure.
    """
    # Strip the [Detected Emotion: ...] metadata tag before keyword analysis
    # so it doesn't inflate the token count and bypass shortcuts.
    _clean_input = re.sub(r"\[Detected Emotion:\s*\w+\]", "", user_input).strip()

    # Preprocess: get first 10 lowercased tokens after removing some punctuation for basic pattern catches
    tokens = _clean_input.replace(".", " ").replace(",", " ").replace("?", " ").split()
    lower = [t.lower() for t in tokens[:10]]

    # Detect easy and common cases up front, for quick handling.
    # We only apply these shortcuts if the user's response was very short (3 words or fewer).
    # If it's longer, they are likely providing context (e.g., "Yes, but I feel terrible") which the LLM should evaluate.
    if len(tokens) <= 3:
        if "stop" in lower:
            logger.debug(f"Quick token 'Stop' detected in short response; binding to dimension '{dimension_label}'")
            return dimension_label, "Stop"
        if "yes" in lower:
            logger.debug(f"Quick token 'Yes' detected in short response; binding to dimension '{dimension_label}'")
            return dimension_label, "Yes"
        if "no" in lower:
            logger.debug(f"Quick token 'No' detected in short response; binding to dimension '{dimension_label}'")
            return dimension_label, "No"
        if "maybe" in lower:
            logger.debug(f"Quick token 'Maybe' detected in short response; binding to dimension '{dimension_label}'")
            return dimension_label, "Maybe"
        if "question" in lower:
            logger.debug(f"Quick token 'Question' detected in short response; binding to dimension '{dimension_label}'")
            return dimension_label, "Question"

    try:
        # Use the response analyzer to try to classify the input
        raw = classify_dimension_and_score(user_input, original_question)
        # Take just the first line (in case of multi-line output)
        first = str(raw).strip().splitlines()[0].strip()
        logger.debug(f"OpenAI raw: {raw}")
        logger.debug(f"First line parsed: {first}")
    except Exception as e:
        # Log failure for diagnostics, fallback code
        logger.debug(f"classify_dimension_and_score exception: {e}")
        return "NA", 99

    # Try to match general words like Yes/No/Stop/Question/Maybe, possibly with a number after a comma
    m = re.match(r"^\s*(Yes|No|Stop|Question|Maybe)\s*,?\s*(\d+)?\s*$", first, flags=re.IGNORECASE)
    if m:
        # token is one of the general words
        token = m.group(1).strip().lower()
        # Return token capitalized if it's one of the general words
        if token in ("yes", "no", "maybe", "question", "stop"):
            logger.debug(f"Regex token '{token.capitalize()}' detected; binding to dimension '{dimension_label}'")
            return dimension_label, token.capitalize()

    # Maybe it's a plain-text dimension,score (e.g. 'talk, 1' or '3_talk, 1' or 'DLA_3_talk, 1')
    got = _parse_dim_score_from_text(first)
    logger.debug(f"Parsed from text: {got}")
    if got:
        logger.debug(f"Parsed from text: {got}")
        return got
    
    # Try to parse result in case it's JSON-ish (either first line or whole raw)
    got = _parse_from_json_like(first)
    logger.debug(f"Parsed first linefrom json-like: {got}")
    if not got:
        got = _parse_from_json_like(str(raw))
        logger.debug(f"Parsed whole raw answer from json-like: {got}")
    if got:
        return got

    # If response is 'Other, N', always fallback to NA,99
    m = re.match(r"^\s*(Other)\s*,\s*(\d+)\s*$", first, flags=re.IGNORECASE)
    if m:
        logger.debug(f"Response is 'Other, {m.group(2)}', fallback to NA,99")
        return "NA", 99

    # If all else fails, fallback
    logger.debug("Failed to parse classification, fallback to NA,99")
    return "NA", 99