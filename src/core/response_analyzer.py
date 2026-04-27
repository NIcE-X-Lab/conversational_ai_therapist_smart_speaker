"""Domain logic evaluating user answers to guide therapy state."""
# src/response_analyzer.py
import json
import re
from typing import List, Tuple

from src.models.llm_client import llm_complete, LLMRole

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
    social: Support from social network and relationships with friends and colleagues
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



There are some dimensions that may be confusing, to distinguish them:
1. eat cares if the user eats regularly and nutrition cares more about whether the user eats enough good food for nutrition.
2. mood cares about the feeling of the user, while emo cares about whether the user is able to express their feelings to others.
3. safe concerns the safety of users' lives, while risk cares if the user is taking any risks. 

If the user input is a general response, such as “Sure”, “Not really”, “I don’t know”, “I don’t understand your question”, “let us stop here”, or anything similar, the DIMENSION will be within [Yes, No, Maybe, Question, Stop], and the SCORE will be 0.

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

Example 4:
{"Original Question": "Have your weight changed significantly recently?", "User Response": "My weight increased a lot recently."}
REFLECTIVE_SUMMERIZER: You mentioned that your weight increased a lot recently.
'''

# Prompt for STRUCTURAL rephrasing of a therapist-validated question.
# The rephraser must preserve the clinical intent and the screening dimension
# being probed — only sentence structure and vocabulary may vary.
REPHRASER_PROMPT = '''You are a therapist-assistant with strong psychology and mental-health knowledge.

You will be provided with a therapist-validated screening question:
{"Original Question": "..."}

Your ONLY task is STRUCTURAL rephrasing:
- Vary sentence structure (word order, clause arrangement).
- Vary vocabulary (synonyms that preserve clinical meaning).
- PRESERVE the clinical intent, the screening dimension, and the time frame.

STRICT RULES:
- Do NOT change what the question is asking about.
- Do NOT add or remove conditions, timeframes, or scope.
- Do NOT reframe into a different clinical construct.
- Do NOT soften, broaden, or narrow the clinical target.
- Do NOT ask a different question, even if it seems more natural.
- Output exactly ONE rephrased question.

Response format:
REPHRASER: <rephrased question>

Example 1:
{"Original Question": "Do you have coping skills to help you calm down?"}
REPHRASER: Do you have strategies that help you calm yourself when you are upset?

Example 2:
{"Original Question": "Are you involved in any legal issues recently?"}
REPHRASER: Are you dealing with any legal issues right now?

Example 3:
{"Original Question": "How's your mood recently?"}
REPHRASER: How would you describe your mood recently?

Example 4:
{"Original Question": "Have your weight changed significantly recently?"}
REPHRASER: Have you noticed any significant changes in your weight lately?
'''

def _chat_complete(system_content: str, user_content: str, role: LLMRole = LLMRole.GENERAL):
    """
    Unified LLM entry that delegates to llm_complete.
    """
    return llm_complete(system_content, user_content, role=role)

def classify_dimension_and_score(user_input: str, original_question: str) -> str:
    """
    Classify user input into a dimension and score using the OpenAI API.
    Input: user_input (str) - any user response string.
           original_question (str) - the original question being answered.
    Output: Raw model text, e.g., 'weight, 2' or 'Yes, 0'.

    Paper role: ANALYZER (fine-tuned GPT-3.5-Turbo in paper).
    """
    logger.info("[PIPELINE] Response Analyzer — classifying user input (Dim, Score).")
    logger.debug(f"Original question: {original_question}")
    logger.debug(f"User input: {user_input}")
    # Provide both the question and the answer to improve contextual classification
    payload = f"Question: {original_question}\nAnswer: {user_input}"
    return llm_complete(INIT_ASKER_SYSTEM_PROMPT_V2, payload, role=LLMRole.ANALYZER)


# Prompt for multi-dimension mapping (paper's "minimal questioning" principle):
# a single user utterance may touch multiple clinical dimensions.
MULTI_DIM_SYSTEM_PROMPT = '''You are a clinical classifier.
Extract EVERY dimension the user discusses in a single utterance, along
with its score (0, 1, or 2).  The user's utterance may map to one OR many
dimensions.

Valid dimensions:
weight, mood, medication, care, house, talk, emo, safe, risk, sleep, eat,
work, work_dayoff, showup, finance, nutrition, problem, support, family,
drug, ciga, alcohol, hobbies, creativity, community, social, comfortable,
protection, productivity, work_motivation, coping, sib, arrest, legal,
hygiene, sports.

Score scale:
- 0: user performs well on this dimension
- 1: some concern, no immediate action needed
- 2: heightened clinical attention needed

Input format:
Question: <original question>
Answer: <user utterance>

Output format (STRICT JSON array, one line, no prose):
[{"dim": "<dim>", "score": <0|1|2>}, {"dim": "<dim>", "score": <0|1|2>}]

Rules:
- If the utterance covers ONE dimension, return a one-element array.
- If it covers MULTIPLE dimensions, include each with its own score.
- If none of the above dimensions apply, return: [{"dim": "Other", "score": 0}]
- Output ONLY the JSON array.  No markdown fences, no commentary.

Example 1:
Question: How is your eating?
Answer: I haven't been eating regularly because work is crushing me.
[{"dim": "eat", "score": 2}, {"dim": "work", "score": 2}]

Example 2:
Question: How is your mood?
Answer: I feel sad and I have stopped exercising.
[{"dim": "mood", "score": 2}, {"dim": "sports", "score": 2}]

Example 3:
Question: Have you been sleeping enough?
Answer: Yes, I sleep fine.
[{"dim": "sleep", "score": 0}]
'''


def classify_multi_dimensions(user_input: str, original_question: str) -> List[Tuple[str, int]]:
    """Return every (dimension, score) pair the user's utterance covers.

    Returns an empty list if the model output cannot be parsed.  Callers
    should always also validate with the existing Yes/No/Stop shortcuts
    first — this path is only used when the utterance is substantive.
    """
    payload = f"Question: {original_question}\nAnswer: {user_input}"
    # Paper role: ANALYZER (multi-dimension extension of Response Analyzer).
    raw = llm_complete(MULTI_DIM_SYSTEM_PROMPT, payload, role=LLMRole.ANALYZER)
    if not raw:
        return []

    # Extract the first JSON array found in the output — handles both
    # well-behaved outputs and ones with stray prose.
    m = re.search(r"\[[^\[\]]*\]", raw, flags=re.DOTALL)
    if not m:
        logger.debug(f"multi-dim: no JSON array found in output: {raw!r}")
        return []

    try:
        data = json.loads(m.group(0))
    except Exception as e:
        logger.debug(f"multi-dim: JSON parse failed: {e}; raw={m.group(0)!r}")
        return []

    if not isinstance(data, list):
        return []

    out: List[Tuple[str, int]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        dim = str(item.get("dim", "")).strip().lower()
        try:
            score = int(item.get("score"))
        except (TypeError, ValueError):
            continue
        if not dim or score not in (0, 1, 2):
            continue
        out.append((dim, score))
    return out

def reflective_summarizer(original_question: str, user_response: str) -> str:
    """
    Summarize the user's response in a reflective, third-person style.
    Input: original_question (str), user_response (str)
    Output: Reflective summary string.

    Paper role: REFLECTIVE_SUMMARIZER (GPT-4 in paper).
    """
    logger.info("[PIPELINE] Reflective Summarizer — 1st -> 3rd person rewrite for follow-up.")
    logger.debug(f"Original question: {original_question}, User response: {user_response}")
    payload = f'{{"Original Question": "{original_question}", "User Response": "{user_response}"}}'
    return llm_complete(REFLECTIVE_SUMMERIZER_PROMPT, payload, role=LLMRole.REFLECTIVE_SUMMARIZER)

def rephrase_question(original_question: str) -> str:
    """
    Rephrase the original question as a therapist would.
    Input: original_question (str)
    Output: Rephrased question string (label stripped, trimmed).

    Paper role: REPHRASER (GPT-4 in paper, structural rewrite only).

    On any empty / unparseable model output, returns the `original_question`
    unchanged so the screening hot path never breaks due to an LLM quirk.
    """
    logger.info("[PIPELINE] Rephraser — varying question wording for this turn.")
    logger.debug(f"Original question: {original_question}")
    payload = f'{{"Original Question": "{original_question}"}}'
    raw = llm_complete(REPHRASER_PROMPT, payload, role=LLMRole.REPHRASER)

    if not raw:
        return original_question

    # Extract the REPHRASER: ... line; tolerate stray preamble / code-fence.
    text = raw.strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("REPHRASER:"):
            candidate = stripped.split(":", 1)[1].strip()
            if candidate:
                return candidate

    # Fallback: first non-empty line if no labelled output was produced.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("```"):
            return stripped

    return original_question