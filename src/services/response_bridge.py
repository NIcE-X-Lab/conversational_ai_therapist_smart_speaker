"""Service bridging the Response Analyzer's raw output to typed (dim, score) tuples.

Legacy/paper pattern: the Analyzer LLM returns free-form text shaped like
"weight, 2" or '{"dim":"mood","score":1}'.  This module normalises every
shape into a (dimension_label, score|keyword) pair the RL handler and
questioner can consume.  Quick-path shortcuts for Yes/No/Stop/Maybe/Question
short replies avoid an LLM call entirely.
"""

import re
import json
from src.core.response_analyzer import classify_dimension_and_score
from src.models.llm_client import LLMError
from src.utils.log_util import get_logger
logger = get_logger("ResponseBridge")


# Bug-2 fix: soft-end phrases that must map to the paper's `Stop` keyword
# regardless of LLM classifier whim. Pre-LLM short-circuit for reliability
# since Gemma-4-E2B's analyzer sometimes misclassifies these as dimension
# answers (e.g. "That's enough for today" → care=0 instead of Stop).
# Matched against the whole lowercased utterance; any match terminates
# the screening loop via the questioner's Stop path, which still routes
# into CBT (handler_rl:run() calls run_cbt unless END_SESSION_EVENT fires).
_SOFT_END_STOP_PATTERNS = (
    re.compile(r"\bno\s+more\s+questions?\b", re.IGNORECASE),
    re.compile(r"\b(?:i\s+)?(?:don'?t|do\s+not)\s+want\s+to\s+answer\s+(?:any\s+)?(?:more\s+)?questions?\b", re.IGNORECASE),
    re.compile(r"\b(?:that'?s|thats|that\s+is)\s+enough\s+(?:for\s+today|for\s+now|questions?)\b", re.IGNORECASE),
    re.compile(r"\b(?:i'?m|i\s+am)\s+done\s+(?:with\s+)?(?:the\s+)?questions?\b", re.IGNORECASE),
    re.compile(r"\benough\s+questions?\b", re.IGNORECASE),
    re.compile(r"\bstop\s+(?:the\s+)?questions?\b", re.IGNORECASE),
    re.compile(r"\blet'?s\s+end\s+(?:the|this)\s+session\b", re.IGNORECASE),
    re.compile(r"\bi\s+want\s+to\s+end\s+(?:the|this)\s+session\b", re.IGNORECASE),
)


def _matches_soft_end_intent(text: str) -> bool:
    if not text:
        return False
    t = str(text).strip()
    return any(p.search(t) for p in _SOFT_END_STOP_PATTERNS)


def _normalize_dim_score(dim: str, score: int):
    """
    If dimension looks like DLA_digits_label or digits_label, strip prefix and keep only the label.
    Otherwise require DLA_ prefix.
    Always validate score range.
    """
    logger.debug(f"Normalizing dimension and score: dim={dim}, score={score}")
    m = re.match(r"^(?:DLA_)?(\d+)_([A-Za-z_]+)$", dim)
    if m:
        dim = m.group(2)
        logger.debug(f"Normalized dimension to label only: {dim}")

    if not isinstance(score, int) or score < 0 or score > 2:
        logger.warning(f"Score {score} is invalid, must be int in 0-2")
        return None

    return dim, score


def _parse_dim_score_from_text(text: str):
    """Parse '[dim][sep][score]' from a free-form text line.

    Supports formats like 'talk, 1', '3_talk, 1', 'DLA_3_talk, 1', etc.
    Accept separators: comma, colon, hyphen, or whitespace.
    """
    m = re.search(
        r"\b((?:DLA_)?(?:\d+_)?[A-Za-z_]+)\s*[,:\-\s]\s*([0-2])\b",
        text,
    )
    if not m:
        return None
    dim = m.group(1).strip()
    score = int(m.group(2))
    return _normalize_dim_score(dim, score)


def _parse_from_json_like(raw: str):
    """If the model returns JSON-like content, try to extract dim/score.

    Supports `{'res': '3_talk, 1'}` and `{'dimension': '3_talk', 'score': 1}`.
    """
    s = str(raw).strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        data = json.loads(s)
        kl = {str(k).lower(): v for k, v in data.items()}
        if "res" in kl:
            got = _parse_dim_score_from_text(str(kl["res"]).strip())
            if got:
                return got
        if "dimension" in kl and "score" in kl:
            dim = str(kl["dimension"]).strip()
            sc = int(kl["score"])
            norm = _normalize_dim_score(dim, sc)
            if norm:
                return norm
        return _parse_dim_score_from_text(s)
    except Exception as e:
        logger.warning(f"Failed to parse JSON-like string: {e}")
        return None


def get_openai_resp(user_input, original_question, dimension_label: str):
    """Classify a user utterance into (dimension, score|keyword).

    For general Yes/No/Stop/Maybe/Question answers, returns (dimension_label, Keyword).
    Otherwise, attempts to return (dimension, score:int) parsed from model output.
    Fallbacks to ('NA', 99) on parse failure.
    """
    _clean_input = user_input.strip()

    tokens = _clean_input.replace(".", " ").replace(",", " ").replace("?", " ").split()
    lower = [t.lower() for t in tokens[:10]]

    # Bug-2 fix: pre-LLM soft-end intent detection so phrases like
    # "I don't want to answer any more questions" / "that's enough for
    # today" / "let's end the session" reliably become the paper's Stop
    # keyword. The handler's screening loop will then terminate and
    # still route into CBT (run_cbt) — matching paper §5.1 / §5.3
    # semantics that Stop is "finish screening, proceed to CBT", not
    # "quit the whole app". This short-circuit runs BEFORE the ≤3-word
    # shortcut so multi-word soft-end utterances are still caught, and
    # BEFORE the LLM classifier so classification is deterministic.
    if _matches_soft_end_intent(_clean_input):
        logger.info(
            f"[SOFT-END] Matched soft-end intent in user utterance; "
            f"returning ({dimension_label}, Stop)."
        )
        return dimension_label, "Stop"

    # Short-response shortcuts: only trust these if the user said 3 words or fewer.
    # Longer replies (e.g. "Yes, but I feel terrible") need the LLM classifier.
    if len(tokens) <= 3:
        if "stop" in lower:
            return dimension_label, "Stop"
        if "yes" in lower:
            return dimension_label, "Yes"
        if "no" in lower:
            return dimension_label, "No"
        if "maybe" in lower:
            return dimension_label, "Maybe"
        if "question" in lower:
            return dimension_label, "Question"

    try:
        raw = classify_dimension_and_score(user_input, original_question)
        first = str(raw).strip().splitlines()[0].strip()
    except LLMError:
        # Engine-level failure: do NOT silently return ("NA", 99), which
        # would be indistinguishable from "user said something we couldn't
        # classify". Propagating lets the clinical hot path mark the turn
        # SKIPPED rather than scoring the user 0 by accident.
        raise
    except Exception as e:
        logger.debug(f"classify_dimension_and_score exception: {e}")
        return "NA", 99

    m = re.match(r"^\s*(Yes|No|Stop|Question|Maybe)\s*,?\s*(\d+)?\s*$", first, flags=re.IGNORECASE)
    if m:
        token = m.group(1).strip().lower()
        if token in ("yes", "no", "maybe", "question", "stop"):
            return dimension_label, token.capitalize()

    got = _parse_dim_score_from_text(first)
    if got:
        return got

    got = _parse_from_json_like(first) or _parse_from_json_like(str(raw))
    if got:
        return got

    if re.match(r"^\s*(Other)\s*,\s*(\d+)\s*$", first, flags=re.IGNORECASE):
        return "NA", 99

    return "NA", 99
