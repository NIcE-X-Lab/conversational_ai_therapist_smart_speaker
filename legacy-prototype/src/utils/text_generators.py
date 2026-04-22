# src/text_generators.py
from src.utils.llm_client import llm_complete

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
    Use OpenAI API to generate a synonymous sentence for the given question_text.
    """
    user_input = question_text
    
    raw = llm_complete(
        "You generate synonymous sentences for a given text. Return only the rewritten sentence, without any prefixes.",
        generate_prompt_synonymous_sentences(user_input)
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

def generate_prompt_therapist(user_input):
    """
    Generate a prompt for the model to act as a therapist in a conversation.
    The prompt provides several example exchanges and then asks the model to respond to the user's input.
    """
    return """Chat with people as a therapist.

    User: I feel so depressed daily.
    Answer: I am so sorry to hear that. It's OK to feel a little bit depressed but you need to figure out a way to makes you feel better. You can talk to a friend or family member. Or you can reach out to a therapist. And I am always here to support you.
    User: I don't want to talk.
    Answer: I get that you don’t want to have this conversation. But it's important to share your feelings with others and find out ways to make you feel better. 
    User: My partner wants to check my messages everyday.
    Answer: When you having a controlling partner, you might want to know the following items. Understand Controlling Personality Types. Recognize the Part of You That Accepts Another's Control. Take Back Responsibility for Your Life. Decide Whether You Need or Want Controlling Men in Your Life. Know What You Want Out of Life. Learn and Practice Assertiveness. Set Healthy Boundaries.
    User: I don't know what's going on with me.
    Answer: It's fine not to know the reason why you don't feel well now. Doing medication might help you understand yourself better. Or you can reach out to your family members, friends, or therapist to help you out.
    User:{}
    Answer:""".format(
        user_input.capitalize()
    )

def generate_therapist_chat(user_input):
    """
    Use OpenAI API to generate a therapist-like response to the user's input.
    """
    result = llm_complete(
        "Chat with people as a virtual AI therapist.",
        generate_prompt_therapist(user_input)
    )
    logger.info(f"generate_therapist_chat: {result}")
    return result

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
    Use OpenAI API to convert a first-person sentence to a second-person sentence.
    """
    resp = llm_complete(
        "Convert first-person to second-person statements.",
        generate_prompt_change(user_input)
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
    """
    resp = llm_complete(
        "Turn a question into a positive declarative sentence.",
        generate_prompt_change_positive(user_input)
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
    """
    resp = llm_complete(
        "Turn a question into a negative declarative sentence.",
        generate_prompt_change_negative(user_input)
    )
    logger.debug(resp)
    return resp