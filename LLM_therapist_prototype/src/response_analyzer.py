# src/response_analyzer.py
from src.utils.llm_client import llm_complete

# Set up logger for this module
from src.utils.log_util import get_logger
logger = get_logger("ResponseAnalyzer")

# === Prompt templates for OpenAI API ===

# Prompt for classifying user input into dimension and score
INIT_ASKER_SYSTEM_PROMPT_V2 = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You will be provided with:
1. All dimension names.
2. The example user inputs with their dimensions and scores. The example will be provided in the following format: {"in": "[USER_INPUT]", "res": "DIMENSION, SCORE"}

Your goal is:
To assign the user input with DIMENSION and SCORE.

Input format:
- You may receive either a plain user input (the user's utterance), or a paired context:
  Question: <original question>
  Answer: <user answer>
- When a question is provided, ALWAYS classify the user's answer in the context of the given question.

All dimension names are:{
    weight, mood, medication, care, house, talk, emo, safe, risk, sleep, eat, work, work_dayoff,
    showup, finance, nutrition, problem, support, family, drug, ciga, alcohol, hobbies, creativity, community, 
    support, social, comfortable, protection, productivity, work_motivation, coping, sib, arrest,legal, hygiene, sports, 
    Yes, No, Maybe, Question, Stop
}

The definition of each dimension are:
    weight: Maintaining stable weight
    mood: Managing mood 
    medication: Taking medication as prescribed
    care: Participating primary and mental health care
    house: Organizing personal possessions and doing housework
    talk: Talking to other people
    emo: Expressing feelings to other people
    safe: Managing personal safety
    risk: Managing risk
    sleep: Following regular schedule for bedtime and sleeping enough
    eat: Maintaining regular schedule for eating
    work: Managing work/school
    work_dayoff: Having work-life balance
    showup: Showing up for appointments and obligations
    finance: Managing finance and items of value 
    nutrition: Getting adequate nutrition
    problem: Problem solving and decision making capability
    support: Family support
    family: Family relationship
    alcohol: Alcohol abuse
    ciga: Tobacco abuse
    drug: Other substances abuse
    hobbies: Enjoying personal choices for leisure activities
    creativity: Creativity
    community: Participation in community
    support: Support from social network
    social: Relationship with friends and colleagues
    comfortable: Managing boundaries in close relationship
    protection: Managing sexual safety
    productivity: Productivity at work or school
    work_motivation: Motivation at work or school
    coping: Coping skills to de-stress
    sib: Exhibiting control over self-harming behavior
    arrest: Law-abiding
    legal: Managing legal issue
    hygiene: Maintaining personal hygiene
    sports: Doing exercises and sports
    Yes: The user expressed acceptance, agreement, or affirmation to the question.
    No: The user expressed rejection, disagreement, or negation to the question.
    Maybe: The user expressed uncertainty, hesitation, or ambivalence about the question.
    Question: The user expressed a question or inquiry about the question.
    Stop: The user expressed a desire to end the conversation or terminate the interaction.



There are some dimension maybe confusing, to distinguish them:
1. eat cares does the user eat regularly and nutirtion cares more about whether the user eat enough good food for nutrition.
2. mood cares about the feeling of the user, while emo cares about whether the user is able to express their feelings to others.
3. safe concerns the safety of users' lives, while risk cares if the user is taking any risks. 

If the user input is general response, such as “Sure”, “Not really”, “I don’t know”, “I don’t understand your question”, “let us stop here”, or anything similar, the DIMENSION will be within [Yes, No, Maybe, Question, Stop], and the SCORE will be 0.

The score ranges from 0 to 2, where:
0 indicates that the user performs well in this dimension;
1 indicates that the user has some problems in this dimension, but no immediate action is needed;
2 indicates a need for heightened attention from health-care providers;

If the user input does not belong to any of these dimension, the "DIMENSION, SCORE" will be: "Other, 0" 

The example user inputs with their dimensions and scores: 
{"in":"Yes, I do.", "res": "Yes, 0"}
{"in":"My weight doesn't change.", "res": "weight, 0"}
{"in":"I didn't measure my weight recently.", "res": "weight, 2"}
{"in":"My weight has increased a lot these days.", "res": "weight, 2"}
{"in":"I get some weight these days.", "res": "weight, 1"}
{"in":"My emotions are out of my control.", "res": "mood, 2"}
{"in":"I don't have a therapist.", "res": "care, 0"}
{"in":"I don't have a psychiatrist.", "res": "care, 0"}
{"in":"I haven't visited my prescriber for a while.", "res": "medication, 2"}
{"in":"I haven't gone to my case manager for a while.", "res": "care, 2"}
{"in":"I often don't eat regularly.", "res": "eat, 2"}
{"in":"I occasionally miss breakfast.", "res": "eat, 1"}
{"in":"I don't have a regular schedule for eating.", "res": "eat, 2"}
{"in":"I don't have a regular schedule for sleeping.", "res": "sleep, 2"}
'''

# Prompt for summarizing user response in a reflective way
REFLECTIVE_SUMMERIZER_PROMPT = ''' You are an intelligent agent to summarize what the user said.

You will be provide with:
The original question asked and the user response in the format of '{"Original Question": XXXX, "User Response": XXXX}'
If the user’s response is essentially “Yes,” use the information from the original question; otherwise, base it on the user input and restate it in third-person voice.
Response format:
REFLECTIVE_SUMMERIZER: XXXXX

Example 1:
{"Original Question": "Do you have coping skills to help you calm down?", "User Response": "Yes, I do"}
REFLECTIVE_SUMMERIZER: You mentioned that you have coping skills to help you calm down. 

Example 2:
{"Original Question": "Are you involved in any legal issues recently?", "User Response": "Yes, I do"}
REFLECTIVE_SUMMERIZER: You shared that you are involved in some legal issues recently.

Example 3:
{"Original Question": "How's your mood recently?", "User Response": "I feel so depressed daily."}
REFLECTIVE_SUMMERIZER: You shared that you feel so depressed daily.

Example 3:
{"Original Question": "Have your weight changed significantly recently?", "User Response": "My weight increased a lot recently."}
REFLECTIVE_SUMMERIZER: You mentioned that your weight increased a lot recently.
'''

# Prompt for rephrasing a question as a therapist
REPHRASER_PROMPT = ''' You are an intelligent agent who have strong reasoning capability and psychology and mental health commonsense knowledge. 

You will be provide with:
The original question in the format of {"Original Question": XXXXX}. 
And you need to act as a therapist to rephrase the question to client.


Response format:
REPHRASER: XXXXX

Example 1:
{"Original Question": "Do you have coping skills to help you calm down?"}
REPHRASER: Do you have strategies that help you calm yourself when you are upset?

Example 2:
{"Original Question": "Are you involved in any legal issues recently?"}
REPHRASER: Are you dealing with any legal issues right now?

Example 3:
{"Original Question": "How's your mood recently?"}
REPHRASER: How would you describe your mood recently?.

Example 3:
{"Original Question": "Have your weight changed significantly recently?"}
REPHRASER: Have you noticed any significant changes in your weight lately?
'''

def _chat_complete(system_content: str, user_content: str):
    """
    Unified LLM entry that delegates to llm_complete.
    """
    return llm_complete(system_content, user_content)

def classify_dimension_and_score(user_input: str, original_question: str) -> str:
    """
    Classify user input into a dimension and score using the OpenAI API.
    Input: user_input (str) - any user response string.
           original_question (str) - the original question being answered.
    Output: Raw model text, e.g., 'weight, 2' or 'Yes, 0'.
    """
    logger.info("Classifying user input for dimension and score.")
    logger.debug(f"Original question: {original_question}")
    logger.debug(f"User input: {user_input}")
    # Provide both the question and the answer to improve contextual classification
    payload = f"Question: {original_question}\nAnswer: {user_input}"
    return _chat_complete(INIT_ASKER_SYSTEM_PROMPT_V2, payload)

def reflective_summarizer(original_question: str, user_response: str) -> str:
    """
    Summarize the user's response in a reflective, third-person style.
    Input: original_question (str), user_response (str)
    Output: Reflective summary string.
    """
    logger.info("Generating reflective summary for user response.")
    logger.debug(f"Original question: {original_question}, User response: {user_response}")
    payload = f'{{"Original Question": "{original_question}", "User Response": "{user_response}"}}'
    return _chat_complete(REFLECTIVE_SUMMERIZER_PROMPT, payload)

def rephrase_question(original_question: str) -> str:
    """
    Rephrase the original question as a therapist would.
    Input: original_question (str)
    Output: Rephrased question string.
    """
    logger.info("Rephrasing question for therapist style.")
    logger.debug(f"Original question: {original_question}")
    payload = f'{{"Original Question": "{original_question}"}}'
    return _chat_complete(REPHRASER_PROMPT, payload)