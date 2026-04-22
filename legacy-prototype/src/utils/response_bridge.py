import re
import json
from src.response_analyzer import classify_dimension_and_score
from src.utils.log_util import get_logger
logger = get_logger("ResponseBridge")

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
    # Preprocess: get first 10 lowercased tokens after removing some punctuation for basic pattern catches
    lower = [t.lower() for t in user_input.replace(".", " ").replace(",", " ").replace("?", " ").split()[:10]]

    # Detect easy and common cases up front, for quick handling:
    if "stop" in lower:
        logger.debug(f"Quick token 'Stop' detected; binding to dimension '{dimension_label}'")
        return dimension_label, "Stop"
    if "yes" in lower:
        logger.debug(f"Quick token 'Yes' detected; binding to dimension '{dimension_label}'")
        return dimension_label, "Yes"
    if "no" in lower:
        logger.debug(f"Quick token 'No' detected; binding to dimension '{dimension_label}'")
        return dimension_label, "No"
    if "maybe" in lower:
        logger.debug(f"Quick token 'Maybe' detected; binding to dimension '{dimension_label}'")
        return dimension_label, "Maybe"
    if "question" in lower:
        logger.debug(f"Quick token 'Question' detected; binding to dimension '{dimension_label}'")
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