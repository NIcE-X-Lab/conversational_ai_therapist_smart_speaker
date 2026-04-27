"""Domain logic responsible for generating dynamic therapy questions."""
from typing import List, Tuple, Dict, Any

import numpy as np

from src.services.response_bridge import get_openai_resp
from src.utils.text_generators import (
    generate_change,
    generate_change_positive,
    generate_change_negative,
)
from src.models.llm_client import llm_complete, LLMRole, LLMError

# Set up logger for this module
from src.utils.log_util import get_logger
from src.utils.io_record import get_answer, get_resp_log, log_question, set_question_prefix, log_reasoning
logger = get_logger("Questioner")

from src.core.reflection_validation import rv_consolidated, rv_validator_mi
from src.core.response_analyzer import (
    classify_multi_dimensions,
    reflective_summarizer,
    rephrase_question,
)
from src.utils.config_loader import (
    REPHRASE_AT_RUNTIME,
    REPHRASE_PROBABILITY,
    REWARD_MODE,
    REASK_DIMENSION_N,
    MULTI_DIM_BACKFILL_ENABLED,
    REFLECTIVE_SUMMARIZER_ENABLED,
)


_RESERVED_LABELS = frozenset({"yes", "no", "maybe", "question", "stop", "na", "other", ""})


def _record_clinical_score(
    question_lib: Dict[str, Any],
    i_key: str,
    score: int,
    *,
    evidence_text: str | None = None,
    source: str = "response_analyzer",
) -> None:
    """Phase B: mirror a `question_lib` score append into clinical_scores /
    clinical_score_attempts. Best-effort — DB write must never block the
    clinical pipeline on its own failure.
    """
    try:
        import src.utils.io_record as io_rec
        db = getattr(io_rec, "DB", None)
        session_id = getattr(io_rec, "SESSION_ID", None)
        if db is None or not session_id:
            return
        entry = question_lib.get(i_key, {}).get("1", {})
        dim_label = str(entry.get("label", "")).lower()
        dim_name = entry.get("name", dim_label)
        scores_so_far = entry.get("score", [])
        attempt_index = len(scores_so_far)
        db.record_clinical_score(
            session_id=session_id,
            dim_index=int(i_key),
            dim_label=dim_label,
            score=int(score),
            dim_name=dim_name,
            evidence_text=evidence_text,
            source=source,
            attempt_index=attempt_index,
        )
    except Exception as e:
        logger.warning(f"clinical_score persist failed (non-fatal): {e}")


def _build_label_index(question_lib: Dict[str, Any]) -> Dict[str, tuple]:
    """Label → (i_key, j_key, entry) index for O(1) back-fill lookups."""
    idx = {}
    for i_key in question_lib.keys():
        for j_key in question_lib[i_key].keys():
            entry = question_lib[i_key][j_key]
            idx[str(entry.get("label", "")).lower()] = (i_key, j_key, entry)
    return idx


def _apply_segment_level_backfill(
    question_lib: Dict[str, Any],
    dla_result: List[Tuple[str, Any]],
    primary_label: str,
    user_segments: List[str],
    original_question: str,
) -> List[Tuple[str, int]]:
    """Opportunistic catch-all back-fill using already-classified DLA_result.

    Paper §4.1 "minimal questioning": one user utterance can satisfy multiple
    dimensions simultaneously.  `classify_segments` has already run per-segment
    LLM classification and produced (label, score) pairs — for any pair whose
    label is a valid NON-primary dimension with an integer score in [0,1,2],
    record the score opportunistically.  No additional LLM call.

    Always runs, regardless of whether the primary answered the asked
    dimension.  Returns the list of (label, score) pairs that were applied.

    G8 — gated by rl.multi_dim_backfill_enabled in config.yaml. Default
    is OFF so the legacy flow (primary dim only) governs.
    """
    if not MULTI_DIM_BACKFILL_ENABLED:
        return []
    if not dla_result:
        return []

    label_to_entry = _build_label_index(question_lib)
    primary_lower = str(primary_label).strip().lower()
    applied: List[Tuple[str, int]] = []

    for seg_idx, (label, score) in enumerate(dla_result):
        label_l = str(label).strip().lower()
        if label_l == primary_lower or label_l in _RESERVED_LABELS:
            continue
        if not isinstance(score, int) or score not in (0, 1, 2):
            continue
        target = label_to_entry.get(label_l)
        if not target:
            continue
        i_key_matched, _, entry = target
        if entry.get("score"):
            continue  # already scored; never overwrite
        entry.setdefault("score", []).append(score)
        seg_text = user_segments[seg_idx] if seg_idx < len(user_segments) else ""
        entry.setdefault("notes", []).append([
            "segment_backfill: true",
            f"source_question: {original_question}",
            f"original_resp: {seg_text}",
            f"inferred_score: {score}",
        ])
        _record_clinical_score(
            question_lib, i_key_matched, score,
            evidence_text=seg_text, source="segment_backfill",
        )
        applied.append((label_l, score))

    if applied:
        logger.info(f"[DLA] Segment-level back-fill credited dims: {applied}")
        log_reasoning("segment_level_backfill", {
            "primary": primary_label,
            "applied": [{"dim": d, "score": s} for d, s in applied],
        })
    return applied


def _apply_multi_dim_updates(
    question_lib: Dict[str, Any],
    user_segments: List[str],
    original_question: str,
    primary_label: str,
) -> None:
    """LLM-based multi-dimension coverage for substantive utterances.

    Complements `_apply_segment_level_backfill`: when a single long segment
    implicitly covers multiple dimensions without the per-segment classifier
    picking them up, this extra LLM pass (on the joined utterance) back-fills
    them.  Gated by length (≥20 tokens, ≥2 segments) to amortise the extra
    LLM round-trip.

    G8 — gated by rl.multi_dim_backfill_enabled in config.yaml. Default
    is OFF so no extra LLM calls on long utterances.
    """
    if not MULTI_DIM_BACKFILL_ENABLED:
        return
    joined = " ".join(s for s in user_segments if s).strip()
    tokens = joined.split()
    if len(tokens) < 20 or len(user_segments) < 2:
        return

    try:
        pairs = classify_multi_dimensions(joined, original_question)
    except Exception as e:
        logger.debug(f"multi-dim classification failed (non-fatal): {e}")
        return

    if not pairs:
        return

    label_to_entry = _build_label_index(question_lib)

    applied = []
    for dim, score in pairs:
        if dim == primary_label.lower():
            continue  # primary handled separately
        target = label_to_entry.get(dim)
        if not target:
            continue
        i_key_matched, _, entry = target
        # Respect prior scores — segment-level back-fill may have already run.
        if entry.get("score"):
            continue
        entry.setdefault("score", []).append(score)
        entry.setdefault("notes", []).append([
            "multi_dim_backfill: true",
            f"source_question: {original_question}",
            f"original_resp: {joined}",
            f"inferred_score: {score}",
        ])
        _record_clinical_score(
            question_lib, i_key_matched, score,
            evidence_text=joined, source="multi_dim_backfill",
        )
        applied.append((dim, score))

    if applied:
        logger.info(f"[DLA] Multi-dim back-fill credited dims: {applied}")
        log_reasoning("multi_dim_backfill", {
            "primary": primary_label,
            "applied": [{"dim": d, "score": s} for d, s in applied],
        })

# System prompt for generating a retry guide when re-asking the same question.
RETRY_GUIDE_SYSTEM_PROMPT = '''You are a concise and supportive therapist-assistant.

You will be provided with:
1) The topic label of the question (Topic)
2) The original question (Original Question)
3) The user's original answer (Original Answer)

Your task is to generate a short guidance that helps the user retry answering the question.
Rules:
- If the Original Answer includes a sentence that shows the user does not understand the question (e.g., "I don't understand", "I don't get it", "what do you mean"), then CLARIFY the question directly in one sentence.
- If the Original Answer includes a sentence that shows doubt/unsure/maybe (e.g., "I'm not sure", "maybe", "unsure", "I doubt"), then ASK the SAME QUESTION from a DIFFERENT ANGLE/PERSPECTIVE in one sentence.
- Otherwise, briefly restate the essence of the Original Question and encourage a concise answer.

Output format (ONE line only):
GUIDE: <your guidance here>

Example A (not understand):
{"Topic": "DLA_1_mood", "Original Question": "How has your mood been?", "Original Answer": "I don't get it."}
GUIDE: I’m would like to know about your recent feelings and mood; could you describe how you’ve been feeling lately?

Example B (unsure/maybe):
{"Topic": "DLA_1_weight", "Original Question": "Have you experienced significant weight change recently?", "Original Answer": "I'm not sure."}
GUIDE: Let us try from a different perspective: have your clothes been fitting tighter or looser than usual lately?

Example C (neither):
{"Topic": "DLA_5_sleep", "Original Question": "Have you been sleeping enough recently?", "Original Answer": "I sleep sometimes."}
GUIDE: Let us focus on sleeping time: in the past week, have you generally slept enough hours most nights?
'''

def _chat_complete(system_content: str, user_content: str, role: LLMRole = LLMRole.GENERAL):
    """
    Unified LLM entry that delegates to llm_complete.
    """
    return llm_complete(system_content, user_content, role=role)

def retry_guide(topic: str, original_question: str, original_answer: str) -> str:
    """
    Generate a concise guide to help the user retry answering the same question.
    - Clarify if the user did not understand
    - Ask from a different angle if user is unsure/maybe/doubt
    - Otherwise, restate essence and invite concise answer

    Paper role: RV_GUIDE (same clarify-and-redirect semantics as R-V Guide;
    paper uses GPT-3.5-Turbo for this guidance style).
    """
    logger.info("[PIPELINE] Retry Guide — user response unclear, generating clarification.")
    payload = f'{{"Topic": {topic!r}, "Original Question": {original_question!r}, "Original Answer": {original_answer!r}}}'
    raw_resp = _chat_complete(RETRY_GUIDE_SYSTEM_PROMPT, payload, role=LLMRole.RV_GUIDE)
    if "GUIDE:" in raw_resp:
        return raw_resp.split("GUIDE:")[1].strip()
    return raw_resp

def classify_segments(user_segments: List[str], original_question: str, dimension_label: str) -> List[Tuple[str, int]]:
    """
    Classifies each user segment using the OpenAI response bridge.
    Returns a list of (dimension, keyword_or_score) tuples for each non-empty segment.
    - For general answers (Yes/No/Stop/Maybe/Question): (dimension_label, Keyword)
    - For scored outputs: (dimension, score:int in [0,1,2])
    """
    logger.debug("Classifying user segments. Total segments: %d", len(user_segments))
    result = []
    for seg in user_segments:
        if not seg:
            # Skip empty segments
            continue
        label, score = get_openai_resp(seg, original_question, dimension_label)
        logger.debug("Segment classified: '%s' -> (dim: %s, val: %s)", seg, label, str(score))
        result.append((label, score))
    logger.info("[DLA] Classification result: %s", str(result))
    return result

def _if_valid_response(
    dla_result: List[Tuple[str, Any]],
    item_index: int,
    question_index: str,
    user_segments: List[str],
    original_question: str,
    question_lib: Dict[str, Any],
    ) -> Tuple[int, int, str, Dict[str, Any], bool]:
    """
    Unified logic: iterate over all labels in dla_result,
    return as soon as an identifiable valid or command-like label is found.

    Returns (valid, terminate, followup_to_RV, question_lib, had_ambiguous).
    `had_ambiguous` is True only when every segment was Maybe/Question and no
    primary-valid token was found; callers use this to decide between
    re-asking Dimension_N (paper §4.2 "understand user input better") and
    the clarifying retry_guide path.
    """
    # Default to no follow-up; only set when we truly have a follow-up to ask
    followup_to_RV = ""
    if not dla_result:
        logger.debug("No DLA result provided. Returning default values.")
        return 0, 0, followup_to_RV, question_lib, False

    question_label = question_lib[str(item_index)][str(question_index)]["label"]

    for i, (label, score_val) in enumerate(dla_result):
        # Normalize label for robust match
        label_norm = str(label).strip()
        score_norm = score_val
        logger.debug(f"Processing dla_result entry: {label_norm}, {score_norm}")

        # Yes/No/Stop bound to the question's dimension (unified format)
        if str(score_norm) in ["Yes", "No", "Stop"]:
            logger.debug(f"Match special token: {score_norm}")
            if str(score_norm) == "Stop":
                logger.info("[QUESTIONER] User said 'Stop' — terminating screening.")
                return 1, 1, followup_to_RV, question_lib, False

            score = question_lib[str(item_index)][str(question_index)].get(str(score_norm), 99)
            question_lib[str(item_index)][str(question_index)]["score"].append(score)
            _seg_primary = user_segments[i] if i < len(user_segments) else (user_segments[0] if user_segments else "")
            _record_clinical_score(
                question_lib, str(item_index), score,
                evidence_text=_seg_primary, source=f"yes_no_{str(score_norm).lower()}",
            )
            logger.info(f"[SCORE] dim={question_label} score={score} (via {score_norm} keyword)")

            if score > 1:
                text = question_lib[str(item_index)][str(question_index)]["question"][0]
                if str(score_norm) == "Yes":
                    text = generate_change_positive(text)
                else:
                    text = generate_change_negative(text)
                # Fixed prompt — no LLM call needed for boilerplate paraphrase.
                followup_to_RV = "It seems that " + text + " Can you tell me more about it?"

            # Prepare note for follow-up, to be appended by caller after collecting follow-up
            _seg = user_segments[i] if i < len(user_segments) else (user_segments[0] if user_segments else "")
            original_resp = "original_resp: " + _seg
            note_resp = [
                "original_question: " + original_question,
                original_resp,
            ]
            question_lib[str(item_index)][str(question_index)]["notes"].append(note_resp)
            logger.debug("Appended note to question_lib[%s][%s]['notes'].", str(item_index), str(question_index))
            return 1, 0, followup_to_RV, question_lib, False

        # Valid response: Label matches question label & score in [0,1,2]
        if label_norm.lower() == str(question_label).lower() and score_norm in [0, 1, 2]:
            logger.info(f"[SCORE] dim={question_label} score={score_norm} (via Response Analyzer)")
            question_lib[str(item_index)][str(question_index)]["score"].append(score_norm)
            _seg_primary = user_segments[i] if i < len(user_segments) else (user_segments[0] if user_segments else "")
            _record_clinical_score(
                question_lib, str(item_index), int(score_norm),
                evidence_text=_seg_primary, source="response_analyzer",
            )
            if score_norm > 1:
                # Follow-up after a Score-2 answer: "You mentioned that X.
                # Can you tell me more?". Two implementations:
                #   - Legacy / demo path (default): regex-based generate_change()
                #     transform — zero extra LLM call.
                #   - Paper §5.2 ReflectiveSummarizer path: LLM restates the
                #     client's response in third person as an MI simple
                #     reflection before the "tell me more" prompt.
                # G9 — gated by rl.reflective_summarizer_enabled in config.yaml.
                # Default is OFF for legacy parity.
                seg = user_segments[i] if i < len(user_segments) else ""
                reflected = ""
                if REFLECTIVE_SUMMARIZER_ENABLED and seg:
                    try:
                        raw = reflective_summarizer(original_question, seg)
                        # Strip any "REFLECTIVE_SUMMERIZER:" label the prompt
                        # template specifies, keeping only the prose tail.
                        for line in (raw or "").splitlines():
                            stripped = line.strip()
                            if stripped.upper().startswith("REFLECTIVE_SUMMERIZER:"):
                                reflected = stripped.split(":", 1)[1].strip()
                                break
                        if not reflected:
                            reflected = (raw or "").strip()
                    except Exception as e:
                        logger.warning(f"reflective_summarizer failed, falling back: {e}")
                if reflected:
                    followup_to_RV = f"{reflected} Can you tell me more about it?"
                else:
                    # Legacy / fallback path: regex-based transform, zero LLM
                    # cost, matches the demo's wording pattern.
                    fallback = generate_change(seg).lower() if seg else ""
                    followup_to_RV = f"You mentioned that {fallback} Can you tell me more?"
            # Prepare note
            _seg = user_segments[i] if i < len(user_segments) else (user_segments[0] if user_segments else "")
            original_resp = "original_resp: " + _seg
            note_resp = [
                "original_question: " + original_question,
                original_resp,
            ]
            question_lib[str(item_index)][str(question_index)]["notes"].append(note_resp)
            logger.debug("Appended note to question_lib[%s][%s]['notes'].", str(item_index), str(question_index))
            return 1, 0, followup_to_RV, question_lib, False

        # Skip Maybe or Question, follow-up will be collected by caller
        if str(score_norm) in ["Maybe", "Question"]:
            logger.debug("Processing 'Maybe' or 'Question' token.")
            # return 0, 0, followup_to_RV, question_lib, False
            continue

    # No primary match. Distinguish ambiguous (user confused/uncertain) from
    # off-topic (user talked about a different dimension entirely). Paper
    # §4.2 routes these differently: ambiguous → clarify via retry_guide;
    # off-topic → re-ask the original Dimension_N question once.
    had_ambiguous = any(str(sc) in ("Maybe", "Question") for _, sc in dla_result)
    if had_ambiguous:
        logger.info("[QUESTIONER] Response was ambiguous (Maybe/Question) — will route to retry_guide.")
    else:
        logger.info("[QUESTIONER] No primary match — will re-ask the same dimension question.")
    return 0, 0, followup_to_RV, question_lib, had_ambiguous

def evaluate_result(question_lib, DLA_result, S, question_A, user_input, original_question_asked):
    """
    Evaluate the result of a user's response to a question.
    Updates the question library and last question as needed.
    ReflectionValidation three steps（topic = the dimension label of the current question）

    Returns (valid, terminate, previous_question, question_lib, had_ambiguous).
    `had_ambiguous` lets the caller decide between re-asking Dimension_N
    (off-topic user) and calling retry_guide (confused/uncertain user).
    """
    logger.debug(f"Evaluating result for item {S}, question {question_A}.")
    # If valid user response, update the question library and last question
    valid, terminate, followup_to_RV, updated, had_ambiguous = _if_valid_response(
        [(lbl, sc) for lbl, sc in DLA_result], S, question_A, user_input, original_question_asked, question_lib
    )
    question_lib = updated
    # Update previous_question if a new one is provided
    previous_question = followup_to_RV 
    if followup_to_RV:
        # If valid user response, log the last question and collect user response
        logger.debug(f"Logging AI follow-up question and collecting user response for item {S}, question {question_A}.")
        # Log the last AI question and get a user response
        log_question(followup_to_RV)
        user_response = get_resp_log()
        if user_response == "SESSION_END":
            logger.info("[SESSION] End signal received during Score-2 follow-up — closing session.")
            # Contract: evaluate_result returns a 5-tuple. Early-return paths
            # MUST preserve arity or callers in ask_question crash on unpack.
            return 1, 1, previous_question, question_lib, had_ambiguous

        # ReflectionValidation three steps (topic = dimension label)
        topic = question_lib[str(S)][str(question_A)]["label"]
        original_resp = user_input[0] if user_input else ""

        logger.info(f"[RV] Topic '{topic}' — Reasoner evaluating follow-up.")
        rv_decision_token, rv_guide_text, rv_validation_text = rv_consolidated(
            topic, original_question_asked, original_resp, user_response
        )

        logger.info(f"[RV] Decision: {'ON-TOPIC (Validator)' if rv_decision_token == '0' else 'OFF-TOPIC (Guide)'}")
        log_reasoning("reasoner_decision", {
            "component": "rv",
            "decision": "related" if rv_decision_token == "0" else "unrelated",
            "topic": topic,
        })

        user_response_0 = ""

        if rv_decision_token == "1":
            # Unrelated: speak the Guide, re-collect a new follow-up response,
            # THEN run the Validator on the new response. rv_consolidated()
            # returned guide_text only and empty validation_text for this
            # branch because the validation target changes (new user input).
            logger.info("[RV] Off-topic — speaking Guide redirect and re-collecting user response.")
            user_response_0 = user_response
            log_question(rv_guide_text)
            user_response = get_resp_log()
            if user_response == "SESSION_END":
                logger.info("[SESSION] End signal received during RV Guide — closing session.")
                return 1, 1, previous_question, question_lib, had_ambiguous

            # Run Validator ONCE on the new (post-Guide) response.
            logger.info("[RV] Running Validator on post-Guide response.")
            rv_validation_text = rv_validator_mi(
                topic, original_question_asked, original_resp, user_response
            )
        # else: rv_decision_token == "0" — rv_consolidated() already ran
        # Validator on `user_response` and returned its text. G4: reuse it
        # instead of duplicating the LLM call (halves RV hot-path latency).

        set_question_prefix(rv_validation_text)
        logger.debug("Queued RV validation to prepend before next question output.")

        log_reasoning("validation_flag", {
            "decision_token": rv_decision_token,
            "guide": rv_guide_text,
            "validation": rv_validation_text,
            "topic": topic,
        })

        therapist_resp = ""

        # Record notes
        logger.debug("Recording notes for this question/response.")
        note_resp = [
            "original_question: " + original_question_asked,
            "original_resp: " + (user_input[0] if user_input else ""),
            "followup_resp: " + (user_response_0 if user_response_0 else user_response),
            "rv_decision: " + rv_decision_token,
            "rv_guide: " + rv_guide_text,
            "followup_resp_1: " + (user_response if user_response_0 else ""),
            "rv_validation: " + rv_validation_text,
            "therapist_resp: " + therapist_resp,
        ]
        question_lib[str(S)][str(question_A)]["notes"].append(note_resp)

    return valid, terminate, previous_question, question_lib, had_ambiguous

def ask_question(question_lib, S: int) -> Tuple[float, int, str]:
        """
        Handles the RL loop for asking questions within a given item (S).
        Returns the total reward, termination flag, and the last question asked.
        """
        logger.info(f"[QUESTIONER] Starting turn for dim state S={S}.")
        question_reward = []
        DLA_terminate = 0
        
        previous_question = ""
        
        # If there is only one question for this item, ask it directly
        question_A = "1"
        # Check if the score list for this item is empty (i.e., not answered yet)
        if len(question_lib[str(S)][str(question_A)]["score"]) == 0:
            # if the item is not answered yet, ask it directly

            # Build the variant pool.  Paper p.11 specifies 7-11 variants per
            # dimension for the Rephraser to draw from; the active lib combines
            # the therapist-authored `question` array with any post-hoc
            # `question_synthetic` rephrases added by scripts/expand_question_lib.py.
            # Therapist-authored questions are preferred for clinical fidelity.
            # We allocate 60% of selection probability to the legacy pool and
            # 40% to synthetic regardless of their sizes — so a dimension with
            # 1 legacy question still has that question drawn 60% of the time.
            entry = question_lib[str(S)][str(question_A)]
            legacy_qs = list(entry.get("question", []))
            synth_qs = list(entry.get("question_synthetic", []))
            if legacy_qs and synth_qs:
                weights = (
                    [0.6 / len(legacy_qs)] * len(legacy_qs)
                    + [0.4 / len(synth_qs)] * len(synth_qs)
                )
                pool = legacy_qs + synth_qs
                question_text = str(np.random.choice(pool, p=weights))
            elif legacy_qs:
                question_text = legacy_qs[np.random.randint(len(legacy_qs))]
            elif synth_qs:
                question_text = synth_qs[np.random.randint(len(synth_qs))]
            else:
                # Defensive: no questions at all.  Fall back to the legacy
                # pool so ask_question's historic error path still fires.
                question_text = legacy_qs[0]
            # ── Paper §5.1 Rephraser ────────────────────────────────────────
            # After picking one of the therapist-authored variants, paper
            # §5.1 runs a GPT-4-based Rephraser at temp 0.7 to structurally
            # rewrite it before speaking — guaranteeing a fresh wording every
            # turn. Two config flags gate this to let deployments trade the
            # extra inference call for latency:
            #   rl.rephrase_at_runtime  — master switch (default true)
            #   rl.rephrase_probability — per-turn coin flip (default 0.95,
            #                              matching the legacy prototype)
            # With both at defaults, behaviour mirrors the paper: every turn
            # is rephrased ~95% of the time. On failure, the Rephraser is
            # designed to return the original string so the hot path never
            # breaks.
            question_text_ask = question_text
            if REPHRASE_AT_RUNTIME and np.random.uniform() < REPHRASE_PROBABILITY:
                try:
                    rephrased = rephrase_question(question_text)
                    if rephrased and rephrased.strip():
                        question_text_ask = rephrased.strip()
                        logger.debug(
                            f"[REPHRASER] '{question_text}' -> '{question_text_ask}'"
                        )
                except Exception as e:
                    logger.warning(f"Rephraser failed, using original: {e}")
            # Log the question being asked
            log_question(question_text_ask)
            # Get user input for the question
            _ , user_input = get_answer()
            if user_input and "SESSION_END" in user_input:
                logger.info("[SESSION] End signal received in Questioner — closing session.")
                return 0.0, 1, ""

            # Classify against the question the user actually heard (possibly
            # rephrased). Rephraser is structural-only so this preserves
            # clinical meaning while keeping analyzer context consistent
            # with what was asked out loud.
            #
            # If the Analyzer LLM is dead (LLMError), treat the turn as
            # SKIPPED rather than silently letting downstream ("NA", 99)
            # classifications corrupt the score distribution — the paper's
            # Response Analyzer is load-bearing for (Dim, Score), and
            # guessing 0 would be a clinical false-negative.
            dimension_label = question_lib[str(S)][str(question_A)]["label"]
            try:
                DLA_result = [[label, score] for (label, score) in classify_segments(user_input, question_text_ask, dimension_label)]
            except LLMError as e:
                logger.error(f"[ANALYZER_DEAD] Skipping turn for dim '{dimension_label}': {e}")
                log_reasoning("analyzer_skipped", {
                    "dimension": dimension_label,
                    "reason": "llm_error",
                    "error": str(e),
                })
                return 0.0, 0, ""

            # Log Semantic Scores to DB
            log_reasoning("semantic_scores", {"DLA_result": DLA_result, "dimension_label": dimension_label, "user_input": user_input})

            # ── Opportunistic back-fill (paper §4.1 "minimal questioning") ───
            # Runs on RAW DLA_result BEFORE primary evaluation so that other
            # dimensions the user actually talked about are credited even if
            # the primary dimension ends up unanswered and we re-ask below.
            # Segment-level back-fill is free (uses classifications we already
            # have). Multi-dim is LLM-gated on utterance length.
            _apply_segment_level_backfill(
                question_lib, DLA_result, dimension_label, user_input, question_text_ask
            )
            _apply_multi_dim_updates(
                question_lib, user_input, question_text_ask, dimension_label
            )

            # Evaluate primary dimension
            valid, DLA_terminate, previous_question, question_lib, had_ambiguous = evaluate_result(
                question_lib, DLA_result, S, question_A, user_input, question_text_ask
            )

            # ── Paper §4.2: "Understand user input better" ────────────────────
            # If the primary dimension got no score AND the user wasn't
            # confused/uncertain (no Maybe/Question tokens), they talked about
            # something else entirely. Re-ask the SAME Dimension_N question
            # once before falling back to retry_guide.
            # G5 — legacy/demo flow goes straight to retry_guide instead of
            # re-asking verbatim. Gated off by default (rl.reask_dimension_n
            # in config.yaml). Flip that flag to true to restore paper §4.2.
            if REASK_DIMENSION_N and valid == 0 and DLA_terminate == 0 and not had_ambiguous:
                logger.info(
                    f"[QUESTIONER] Re-asking dim '{dimension_label}' (primary unscored, no ambiguity)."
                )
                log_reasoning("reask_dimension_n", {
                    "dimension": dimension_label,
                    "reason": "primary_unscored_no_ambiguity",
                })
                log_question(question_text_ask)
                _, user_input = get_answer()
                if user_input and "SESSION_END" in user_input:
                    logger.info("[SESSION] End signal received during re-ask — closing session.")
                    return 0.0, 1, ""

                try:
                    DLA_result = [[label, score] for (label, score) in classify_segments(user_input, question_text_ask, dimension_label)]
                except LLMError as e:
                    logger.error(f"[ANALYZER_DEAD] Re-ask branch skipped for dim '{dimension_label}': {e}")
                    log_reasoning("analyzer_skipped", {
                        "dimension": dimension_label,
                        "reason": "llm_error_reask",
                        "error": str(e),
                    })
                    return 0.0, 0, ""
                log_reasoning("semantic_scores", {
                    "DLA_result": DLA_result,
                    "dimension_label": dimension_label,
                    "user_input": user_input,
                    "is_reask": True,
                })

                # Back-fill again on the re-ask response — the user may have
                # answered the primary AND touched new dimensions.
                _apply_segment_level_backfill(
                    question_lib, DLA_result, dimension_label, user_input, question_text_ask
                )
                _apply_multi_dim_updates(
                    question_lib, user_input, question_text_ask, dimension_label
                )

                valid, DLA_terminate, previous_question, question_lib, had_ambiguous = evaluate_result(
                    question_lib, DLA_result, S, question_A, user_input, question_text_ask
                )

            # ── retry_guide fallback (ambiguous or still-invalid after re-ask) ─
            # Triggers when the user was confused/uncertain from the start OR
            # when the Dimension_N re-ask also failed to produce a valid
            # primary score. retry_guide clarifies / asks from a different
            # angle rather than repeating the same question verbatim.
            if valid == 0 and DLA_terminate == 0:
                topic = question_lib[str(S)][str(question_A)]["label"]
                original_answer_text = " ".join(user_input) if user_input else ""
                guide_text = retry_guide(topic, question_text_ask, original_answer_text)
                log_question(guide_text)
                _, user_input = get_answer()
                if user_input and "SESSION_END" in user_input:
                    logger.info("[SESSION] End signal received during retry_guide — closing session.")
                    return 0.0, 1, ""

                try:
                    DLA_result = [[label, score] for (label, score) in classify_segments(user_input, question_text_ask, dimension_label)]
                except LLMError as e:
                    logger.error(f"[ANALYZER_DEAD] Retry-guide branch skipped for dim '{dimension_label}': {e}")
                    log_reasoning("analyzer_skipped", {
                        "dimension": dimension_label,
                        "reason": "llm_error_retry",
                        "error": str(e),
                    })
                    return 0.0, 0, ""
                log_reasoning("semantic_scores", {
                    "DLA_result": DLA_result,
                    "dimension_label": dimension_label,
                    "user_input": user_input,
                    "is_retry": True,
                })

                # Back-fill on retry response as well.
                _apply_segment_level_backfill(
                    question_lib, DLA_result, dimension_label, user_input, question_text_ask
                )
                _apply_multi_dim_updates(
                    question_lib, user_input, question_text_ask, dimension_label
                )

                valid, DLA_terminate, previous_question, question_lib, had_ambiguous = evaluate_result(
                    question_lib, DLA_result, S, question_A, user_input, question_text_ask
                )
        
        # ── Reward aggregation (paper §5.1 divergence) ──────────────────────
        # Paper p.11: "The Score, based on the analysis of the user's
        # response, represents the reward earned in that state." Legacy
        # aggregates multi-segment answers via np.mean:
        #     all_score = question_lib[S][Q]["score"]
        #     question_openai_res = np.mean(all_score) if all_score else 0.0
        # That's REWARD_MODE == "mean" (paper-strict).
        #
        # REWARD_MODE == "hybrid" (default) uses (max + mean) / 2 because
        # pure mean has a failure mode: scores like [2, 0, 0, 0] average
        # to 0.5 and barely move the Q-value, so a user turn containing
        # ONE severe segment and several benign ones does not bias the
        # RL agent toward revisiting that dimension next session.
        # Hybrid promotes it to 1.25 — still short of max, but enough to
        # shift revisit priority.
        #
        # Why this is safer than paper's mean (and still defensible):
        #   • Within-session clinical response is unchanged — the R-V
        #     pipeline already triggers on any single Score-2 segment.
        #   • Critical dims (sib/safe/risk/drug/alcohol) are routed to
        #     the crisis flow in handler_rl.py irrespective of reward.
        #   • Multi-dim back-fill writes scores for off-topic dims into
        #     the SAME score[] list; pure mean would dilute the asked
        #     dimension's signal even more, hybrid resists that.
        # Switch via config.yaml rl.reward_mode: mean to reproduce paper
        # numbers exactly.
        all_score = question_lib[str(S)][str(question_A)]["score"]
        int_scores = [s for s in all_score if isinstance(s, int) and 0 <= s <= 2]
        if not int_scores:
            question_reward_value = 0.0
            reward_desc = "empty"
        elif REWARD_MODE == "mean":
            question_reward_value = float(sum(int_scores)) / len(int_scores)
            reward_desc = "mean (paper §5.1 / legacy)"
        else:  # "hybrid" (default)
            mean_r = float(sum(int_scores)) / len(int_scores)
            max_r = float(max(int_scores))
            question_reward_value = (max_r + mean_r) / 2.0
            reward_desc = "(max+mean)/2 hybrid"

        logger.info(
            f"[QUESTIONER] Finished dim S={S}. Reward ({reward_desc}): {question_reward_value:.2f}"
        )
        return question_reward_value, int(DLA_terminate), previous_question
