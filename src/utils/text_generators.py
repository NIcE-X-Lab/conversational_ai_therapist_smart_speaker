"""Utility helper wrapping boilerplate text prompts or templates."""
# src/text_generators.py
from src.models.llm_client import llm_complete, LLMRole

from src.utils.log_util import get_logger
logger = get_logger("TextGenerators")

# The following functions generate prompts and call OpenAI's API to generate various types of text transformations.
# Each function is commented to explain its purpose and logic.

def generate_prompt_synonymous_sentences(user_input):
    """
    Generate a prompt for the model to create synonymous sentences.
    The prompt provides several examples and then asks the model to generate a synonym for the user's input.
    """
    return """Generate synonymous sentences.

    User: I am sad.
    Answer: I feel sad.
    User: I really enjoy my work recently.
    Answer: I like my job a lot those days.
    User: I have problem hearing you well.
    Answer: I have problem understand you well.
    User:{}
    Answer:""".format(
        user_input.capitalize()
    )

def generate_synonymous_sentences(question_text):
    """
    Use LLM to generate a synonymous sentence for the given question_text.
    Paper describes 95% synonym rephrasing probability to reduce repetition.

    Paper role: REPHRASER (GPT-4 in paper; structural rephrase of screening
    questions).
    """
    user_input = question_text
    raw = llm_complete(
        "You generate synonymous sentences for a given text. Return only the rewritten sentence, without any prefixes.",
        generate_prompt_synonymous_sentences(user_input),
        role=LLMRole.REPHRASER,
    )
    results = raw.strip()
    lower = results.lower()
    if "answer:" in lower:
        idx = lower.rfind("answer:")
        results = results[idx+7:].strip()
    elif results.startswith("User:"):
        parts = [ln for ln in results.splitlines() if ln.strip().lower().startswith("answer:")]
        if parts:
            results = parts[-1].split(":", 1)[1].strip()
    logger.info(f"generate_synonymous_sentences: {results}")
    return results

def generate_prompt_change(user_input):
    """
    Generate a prompt for the model to convert a first-person sentence to a second-person sentence.
    The prompt provides several examples and then asks the model to convert the user's input.
    """
    return """　Change from first-person sentence to second-person.

    User: I feel so depressed daily.
    Answer: You feel so depressed daily.
    User: I am so happy.
    Answer: You are so happy.
    User: I am under a lot of pressure.
    Answer: You are under a lot of pressure.
    User:{}
    Answer:""".format(
        user_input.capitalize()
    )

def generate_change(user_input):
    """
    Use LLM to convert a first-person sentence to a second-person sentence.

    Paper role: REFLECTIVE_SUMMARIZER (1st→3rd person is the ReflectiveSummarizer
    task per paper §5.2; GPT-4 in paper).
    """
    resp = llm_complete(
        "Convert first-person to second-person statements.",
        generate_prompt_change(user_input),
        role=LLMRole.REFLECTIVE_SUMMARIZER,
    )
    logger.debug(resp)
    return resp

def generate_prompt_change_positive(user_input):
    """
    Generate a prompt for the model to convert a question to a positive declarative sentence.
    The prompt provides several examples and then asks the model to convert the user's input.
    """
    return """　Change from question to positive declarative sentence.

    User: Do you have coping skills to help you calm down.
    Answer: You have coping skills to help you calm down.
    User: Do you have self-harming behaviours?
    Answer: You have self-harming behaviours.
    User: Are you involved in any legal issues recently?
    Answer: You are involved in some legal issues recently.
    User:{}
    Answer:""".format(
        user_input.capitalize()
    )

def generate_change_positive(user_input):
    """
    Use OpenAI API to convert a question to a positive declarative sentence.

    Paper role: REPHRASER (structural rewrite of the asked question into a
    declarative for Yes-path follow-ups; GPT-4 in paper).
    """
    resp = llm_complete(
        "Turn a question into a positive declarative sentence.",
        generate_prompt_change_positive(user_input),
        role=LLMRole.REPHRASER,
    )
    logger.debug(resp)
    return resp

def generate_prompt_change_negative(user_input):
    """
    Generate a prompt for the model to convert a question to a negative declarative sentence.
    The prompt provides several examples and then asks the model to convert the user's input.
    """
    return """　Change from question to negative declarative sentence.

    User: Do you have coping skills to help you calm down.
    Answer: You don't have coping skills to help you calm down.
    User: Do you feel productive?
    Answer: You don't feel productive.
    User: Have you done anything creative recently?
    Answer: You haven't done anything creative recently.
    User:{}
    Answer:""".format(
        user_input.capitalize()
    )

def generate_change_negative(user_input):
    """
    Use OpenAI API to convert a question to a negative declarative sentence.

    Paper role: REPHRASER (structural rewrite of the asked question into a
    declarative for No-path follow-ups; GPT-4 in paper).
    """
    resp = llm_complete(
        "Turn a question into a negative declarative sentence.",
        generate_prompt_change_negative(user_input),
        role=LLMRole.REPHRASER,
    )
    logger.debug(resp)
    return resp