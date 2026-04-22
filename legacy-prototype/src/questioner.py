from typing import List, Tuple, Dict, Any

import numpy as np

from src.utils.response_bridge import get_openai_resp
from src.utils.text_generators import (
    generate_change,
    generate_change_positive,
    generate_change_negative,
    generate_synonymous_sentences,
    generate_therapist_chat,
)
from src.utils.llm_client import llm_complete

# Set up logger for this module
from src.utils.log_util import get_logger
from src.utils.io_record import get_answer, get_resp_log, log_question, set_question_prefix
logger = get_logger("Questioner")

from src.reflection_validation import rv_reasoner, rv_guide, rv_validation

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

def _chat_complete(system_content: str, user_content: str):
    """
    Unified LLM entry that delegates to llm_complete.
    """
    return llm_complete(system_content, user_content)

def retry_guide(topic: str, original_question: str, original_answer: str) -> str:
    """
    Generate a concise guide to help the user retry answering the same question.
    - Clarify if the user did not understand
    - Ask from a different angle if user is unsure/maybe/doubt
    - Otherwise, restate essence and invite concise answer
    """
    logger.info("Generating retry guide for re-ask.")
    payload = f'{{"Topic": {topic!r}, "Original Question": {original_question!r}, "Original Answer": {original_answer!r}}}'
    return _chat_complete(RETRY_GUIDE_SYSTEM_PROMPT, payload)

def classify_segments(user_segments: List[str], original_question: str, dimension_label: str) -> List[Tuple[str, int]]:
    """
    Classifies each user segment using the OpenAI response bridge.
    Returns a list of (dimension, keyword_or_score) tuples for each non-empty segment.
    - For general answers (Yes/No/Stop/Maybe/Question): (dimension_label, Keyword)
    - For scored outputs: (dimension, score:int in [0,1,2])
    """
    logger.info("Classifying user segments. Total segments: %d", len(user_segments))
    result = []
    for seg in user_segments:
        if not seg:
            # Skip empty segments
            continue
        label, score = get_openai_resp(seg, original_question, dimension_label)
        logger.debug("Segment classified: '%s' -> (dim: %s, val: %s)", seg, label, str(score))
        result.append((label, score))
    logger.info("Classification complete. Results: %s", str(result))
    return result

def _if_valid_response(
    dla_result: List[Tuple[str, Any]],
    item_index: int,
    question_index: str,
    user_segments: List[str],
    original_question: str,
    question_lib: Dict[str, Any],
    ) -> Tuple[int, int, str, Dict[str, Any]]:
    """
    Unified logic: iterate over all labels in dla_result, 
    return as soon as an identifiable valid or command-like label is found.
    """
    # Default to no follow-up; only set when we truly have a follow-up to ask
    followup_to_RV = ""
    if not dla_result:
        logger.info("No DLA result provided. Returning default values.")
        return 0, 0, followup_to_RV, question_lib

    question_label = question_lib[str(item_index)][str(question_index)]["label"]

    for i, (label, score_val) in enumerate(dla_result):
        # Normalize label for robust match
        label_norm = str(label).strip()
        score_norm = score_val
        logger.info(f"Processing dla_result entry: {label_norm}, {score_norm}")

        # Yes/No/Stop bound to the question's dimension (unified format)
        if str(score_norm) in ["Yes", "No", "Stop"]:
            logger.info(f"Match special token: {score_norm}")
            if str(score_norm) == "Stop":
                logger.info("Received 'Stop' label. Terminating evaluation.")
                return 1, 1, followup_to_RV, question_lib

            score = question_lib[str(item_index)][str(question_index)].get(str(score_norm), 99)
            question_lib[str(item_index)][str(question_index)]["score"].append(score)
            logger.info("Appended score %s for keyword %s to question_lib[%s][%s].", str(score), str(score_norm), str(item_index), str(question_index))

            if score > 1:
                text = question_lib[str(item_index)][str(question_index)]["question"][0]
                if str(score_norm) == "Yes":
                    text = generate_change_positive(text)
                else:
                    text = generate_change_negative(text)
                followup = generate_synonymous_sentences(" Can you tell me more about it?")
                followup_to_RV = "It seems that " + text + " " + followup

            # Prepare note for follow-up, to be appended by caller after collecting follow-up
            original_resp = "original_resp: " + (user_segments[i] if i < len(user_segments) else user_segments[0])
            note_resp = [
                "original_question: " + original_question,
                original_resp,
            ]
            question_lib[str(item_index)][str(question_index)]["notes"].append(note_resp)
            logger.debug("Appended note to question_lib[%s][%s]['notes'].", str(item_index), str(question_index))
            return 1, 0, followup_to_RV, question_lib

        # Valid response: Label matches question label & score in [0,1,2]
        if label_norm.lower() == str(question_label).lower() and score_norm in [0, 1, 2]:
            logger.info("Valid response: label matches and score is in [0,1,2]")
            question_lib[str(item_index)][str(question_index)]["score"].append(score_norm)
            if score_norm > 1:
                text = generate_change(user_segments[i]).lower() if i < len(user_segments) else ''
                followup_to_RV = "You mentioned that " + text + " Can you tell me more?"
            # Prepare note
            original_resp = "original_resp: " + (user_segments[i] if i < len(user_segments) else user_segments[0])
            note_resp = [
                "original_question: " + original_question,
                original_resp,
            ]
            question_lib[str(item_index)][str(question_index)]["notes"].append(note_resp)
            logger.debug("Appended note to question_lib[%s][%s]['notes'].", str(item_index), str(question_index))
            return 1, 0, followup_to_RV, question_lib

        # Skip Maybe or Question, follow-up will be collected by caller
        if str(score_norm) in ["Maybe", "Question"]:
            logger.info("Processing 'Maybe' or 'Question' token.")
            # return 0, 0, followup_to_RV, question_lib
            continue

    # If nothing matched, fallback: invalid response
    logger.info("No valid, yes, no, or stop label found in results. Marking as invalid response.")
    return 0, 0, followup_to_RV, question_lib

def evaluate_result(question_lib, DLA_result, S, question_A, user_input, original_question_asked):
    """
    Evaluate the result of a user's response to a question.
    Updates the question library and last question as needed.
    ReflectionValidation three steps（topic = the dimension label of the current question）
    """
    logger.info(f"Evaluating result for item {S}, question {question_A}.")
    # If valid user response, update the question library and last question
    valid, terminate, followup_to_RV, updated = _if_valid_response(
        [(lbl, sc) for lbl, sc in DLA_result], S, question_A, user_input, original_question_asked, question_lib
    )
    question_lib = updated
    # Update previous_question if a new one is provided
    previous_question = followup_to_RV 
    if followup_to_RV:
        # If valid user response, log the last question and collect user response
        logger.info(f"Logging AI follow-up question and collecting user response for item {S}, question {question_A}.")
        # Log the last AI question and get a user response
        log_question(followup_to_RV)
        user_response = get_resp_log()

        # ReflectionValidation three steps（topic = the dimension label of the current question）
        topic = question_lib[str(S)][str(question_A)]["label"]
        original_resp = user_input[0] if user_input else ""

        logger.info(f"Running ReflectionValidation reasoner for topic '{topic}'.")
        rv_decision_raw = rv_reasoner(topic, original_question_asked, original_resp, user_response)
        # Simple parsing: extract '0/1', 0 means related, 1 means not related
        rv_decision_token = "0" if "0" in rv_decision_raw else "1"
        logger.info(f"ReflectionValidation decision: {rv_decision_token}")

        # If not related (1), give guidance, recollect follow-up
        rv_guide_text = ""
        user_response_0 = ""
        if rv_decision_token == "1":
            logger.info("Follow-up not related, generating guidance and recollecting follow-up.")
            user_response_0 = user_response
            rv_guide_text = rv_guide(topic, original_question_asked, original_resp, user_response)
            log_question(rv_guide_text)
            user_response = get_resp_log()

        # Empathic validation
        logger.info("Running ReflectionValidation empathic validation.")
        rv_validation_text = rv_validation(topic, original_question_asked, original_resp, user_response)
        # Set validation text to be prepended to the next user-facing question
        set_question_prefix(rv_validation_text)
        logger.info("Queued RV validation to prepend before next question output.")
        
        # Skip generating therapist response to avoid unnecessary LLM calls
        therapist_resp = ""
        
        # Record notes (expand RV fields)
        logger.info("Recording notes for this question/response.")
        note_resp = [
            "original_question: " + original_question_asked,
            "original_resp: " + (user_input[0] if user_input else ""),
            "followup_resp: " + user_response_0 if user_response_0 else "followup_resp: " + user_response,
            "rv_decision: " + rv_decision_token,
            ("rv_guide: " + rv_guide_text) if rv_guide_text else "rv_guide: ",
            "followup_resp_1: " + user_response if user_response_0 else "followup_resp_1: "
            "rv_validation: " + rv_validation_text,
            "therapist_resp: " + therapist_resp
        ]
        question_lib[str(S)][str(question_A)]["notes"].append(note_resp)
        
    return valid, terminate, previous_question, question_lib

def ask_question(question_lib, S: int) -> Tuple[float, int, str]:
        """
        Handles the RL loop for asking questions within a given item (S).
        Returns the total reward, termination flag, and the last question asked.
        """
        logger.info(f"Starting question RL loop for item S={S}.")
        question_reward = []
        DLA_terminate = 0
        
        previous_question = ""
        
        # If there is only one question for this item, ask it directly
        question_A = "1"
        # Check if the score list for this item is empty (i.e., not answered yet)
        if len(question_lib[str(S)][str(question_A)]["score"]) == 0:
            # if the item is not answered yet, ask it directly
            
            # Get the number of available question variants for this item
            number_of_questions = len(question_lib[str(S)][str(question_A)]["question"])
            # Randomly select one question variant to ask
            choice_of_question = np.random.randint(number_of_questions)
            question_text = question_lib[str(S)][str(question_A)]["question"][choice_of_question]
            # With probability, generate a synonymous version of the question
            if np.random.uniform() < 0.95:
                question_text = generate_synonymous_sentences(question_text)
            # Concatenate the last question (context) with the current question
            question_text_ask = question_text
            # Log the question being asked
            log_question(question_text_ask)
            # Get user input for the question
            _ , user_input = get_answer()
            # Classify the user response into DLA result segments
            dimension_label = question_lib[str(S)][str(question_A)]["label"]
            DLA_result = [[label, score] for (label, score) in classify_segments(user_input, question_text, dimension_label)]
            # Evaluate the result and update state
            valid, DLA_terminate, previous_question, question_lib = evaluate_result(
                question_lib, DLA_result, S, question_A, user_input, question_text
            )
            # If the answer is invalid (valid == 0) and the process has not been terminated (DLA_terminate == 0), 
            # we may want to give the user a chance to clarify their response.
            # Only retry if DLA_result is empty or every (label, score) pair suggests NA or an unclassified response (score==99 or label=="NA").
            if valid == 0 and DLA_terminate == 0:
                # Generate a concise retry guide based on topic, original question, and original answer
                topic = question_lib[str(S)][str(question_A)]["label"]
                original_answer_text = " ".join(user_input) if user_input else ""
                guide_text = retry_guide(topic, question_text, original_answer_text)
                # Show the guide to the user and collect a new response
                log_question(guide_text)
                _ , user_input = get_answer()
                # Classify the new user response
                dimension_label = question_lib[str(S)][str(question_A)]["label"]
                DLA_result = [[label, score] for (label, score) in classify_segments(user_input, question_text, dimension_label)]
                # Re-evaluate the new answer and update state accordingly
                valid, DLA_terminate, previous_question, question_lib = evaluate_result(
                    question_lib, DLA_result, S, question_A, user_input, question_text
                )
        
        # Retrieve all scores for this question after answering
        all_score = question_lib[str(S)][str(question_A)]["score"]
        # Calculate the mean score if available, otherwise set to 0.0
        question_openai_res = np.mean(all_score) if all_score else 0.0
        # Append the result to the question reward list
        question_reward.append(question_openai_res)

        # Return the total reward, termination flag, and last question
        logger.info(f"Finished question RL loop for item S={S}. Total reward: {float(sum(question_reward))}, DLA_terminate: {int(DLA_terminate)}")
        return float(sum(question_reward)), int(DLA_terminate), previous_question
