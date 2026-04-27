"""Domain logic handling Cognitive Behavioral Therapy (CBT) protocols."""
import re

from src.models.llm_client import llm_complete, LLMRole
from src.core.therapy_content import CBT_ESCALATION_ENABLED, CBT_ESCALATION_MESSAGE


def _parse_decision(raw: str, default: str = "1") -> str:
    """Parse a Reasoner's DECISION output using legacy substring semantics.

    Demo-parity note: the legacy prototype / paper used plain substring
    detection (`"0" if "0" in raw else "1"`). Gemma-4-E2B on-device does
    not always emit the literal `DECISION:` header that the paper's
    GPT-4 Reasoner reliably produced; under a strict regex parser that
    drops us into unnecessary retry loops and the demo's single-shot
    CBT Stage-1/2/3 flow breaks.
    Kept here as a dedicated function (not inlined) so the trade-off is
    documented and easy to tighten again if a stricter backend is wired
    in via ROLE_MODEL_MAP.
    """
    return "0" if (raw and "0" in raw) else "1"


# ── Number-word parsing for CBT Stage 0 selection ────────────────────────────
# Maps cardinal + ordinal English number words 1-10 to their integer value,
# so a spoken reply like "Three." / "the third one" / "I'd like option two"
# resolves cleanly.  STT rarely emits numerals for small integers on
# Jetson-deployed faster-whisper, so without this mapping CBT Stage 0
# fails for voice users (observed in session 14 / Sally, 2026-04-25 —
# "Three." → regex-digit match fails → CBT aborted).
_NUMBER_WORDS: dict[str, int] = {
    # cardinals
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    # ordinals (informal spoken selections)
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


def _extract_choice_number(answer: str) -> int | None:
    """Parse a spoken CBT Stage 0 selection into an integer 1-based index.

    Accepts (in order of preference):
      1. Any embedded digit(s) via regex, e.g. "2" / "option 3" / "I pick 4."
      2. Any cardinal / ordinal word from ``_NUMBER_WORDS``, e.g. "three",
         "the third one", "number four please".
    Returns None when no number can be resolved.
    """
    if not answer:
        return None
    ans = str(answer).strip().lower()
    if not ans:
        return None

    # Prefer explicit digits when present — the user dictated a numeral.
    digit_match = re.findall(r"\d+", ans)
    if digit_match:
        try:
            return int(digit_match[0])
        except ValueError:
            pass

    # Word-form fallback: tokenize on non-alpha, match first known word.
    for tok in re.findall(r"[a-z]+", ans):
        if tok in _NUMBER_WORDS:
            return _NUMBER_WORDS[tok]
    return None


# ── CBT Guide output sanitizer ───────────────────────────────────────────────
# Gemma-4-E2B sometimes echoes the full few-shot template back (including
# the prompt labels STATEMENT / UNHELPFUL_THOUGHTS / CHALLENGE / REFRAME)
# instead of returning only the target label's content.  The legacy demo
# used GPT-3.5/4 which reliably emitted just the target label, so this
# sanitizer is a Gemma-specific cleanup step — not a divergence from the
# clinical contract, just a small-LLM output hygiene step.
#
# The few-shot examples are written in first-person ("I can challenge
# this thought by asking myself...") because they illustrate what the
# user could *internally* say.  When spoken aloud by CaiTI, those first-
# person lines confuse the listener — is CaiTI speaking as them?  A
# simple pronoun swap lifts the framing to second-person so the user
# hears it as an offered example.
_GUIDE_LABELS = ("UNHELPFUL_THOUGHTS", "CHALLENGE", "REFRAME")

_FIRST_TO_SECOND_PERSON = [
    # Order matters — longer/more-specific patterns first so "I can" doesn't
    # pre-empt bare "I".  Word boundaries (\b) prevent mangling names like
    # "Ian" or words like "India".
    # Contractions first.
    (r"\bI'm\b", "you're"),
    (r"\bI'd\b", "you'd"),
    (r"\bI'll\b", "you'll"),
    (r"\bI've\b", "you've"),
    # Modal / auxiliary + verb combinations commonly seen in CBT example
    # phrasing ("I can challenge", "I could ask myself", etc.).
    (r"\bI\s+can\b", "you can"),
    (r"\bI\s+could\b", "you could"),
    (r"\bI\s+might\b", "you might"),
    (r"\bI\s+should\b", "you should"),
    (r"\bI\s+would\b", "you would"),
    (r"\bI\s+need\b", "you need"),
    (r"\bI\s+want\b", "you want"),
    (r"\bI\s+have\b", "you have"),
    (r"\bI\s+am\b", "you are"),
    (r"\bI\s+was\b", "you were"),
    (r"\bI\s+will\b", "you will"),
    (r"\bI\s+do\b", "you do"),
    (r"\bI\s+did\b", "you did"),
    # Bare "I <verb>" — catches past-tense / present-tense verbs that
    # didn't match the modal patterns above.  Requires "I " followed by
    # an alphabetic word so "I." and "I?" are left alone.
    (r"\bI\s+(?=[a-z])", "you "),
    # Question-initial "Have I...", "Am I...", "Can I...", "Did I..." →
    # "Have you...", etc.  This must run AFTER the bare "I " rule so
    # "Have I noticed" becomes "Have you noticed" (via "I noticed" →
    # "you noticed") rather than a messy double-swap.
    (r"(^|[.!?]\s+)I\s", r"\1You "),
    # Object / possessive pronouns.
    (r"\bmyself\b", "yourself"),
    (r"\bmine\b", "yours"),
    (r"\bmy\b", "your"),
    (r"\bme\b", "you"),
]


def _sanitize_guide_text(raw: str, target_label: str) -> str:
    """Clean an LLM Guide output for user-facing speech.

    Handles three Gemma failure modes observed in Hudson's session 16
    (2026-04-26):
      1. The model echoes the prompt's STATEMENT / UNHELPFUL_THOUGHTS
         lines verbatim before the actual guidance.  These confuse the
         listener ("STATEMENT: I just feel like..." sounds like CaiTI is
         reading back what the user said, not offering help).
      2. The model includes the target label header as a prefix
         (e.g. "CHALLENGE: ...") — fine in the DB note but awkward in
         speech.
      3. The few-shot examples are first-person; Gemma mimics that
         register, so the offered guidance sounds like CaiTI is
         narrating the user's internal monologue.

    Strategy: find the target label (e.g. "CHALLENGE:") and keep only
    what follows.  If no target label is present, drop any stray
    STATEMENT / UNHELPFUL_THOUGHTS lines and use the remainder.  Then
    swap first-person pronouns to second-person.  Finally, prefix with
    "Here's an example you could try:" so the user knows this is offered
    guidance, not a statement or a question being asked of them.
    """
    if not raw:
        return ""

    text = str(raw).strip()

    # 1. Extract content after the target label when present.  Stage-N
    # Reasoner / Guide outputs may come as "CHALLENGE: <content>" or
    # with the label at the start of the last line after a prefix of
    # echoed STATEMENT / UNHELPFUL_THOUGHTS lines.
    label_upper = target_label.upper()
    match = re.search(
        rf"{label_upper}\s*:\s*(.+?)(?:\n\n|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        body = match.group(1).strip()
    else:
        # No target label — strip any leading STATEMENT / UNHELPFUL_THOUGHTS
        # line so the user doesn't hear their own words played back.
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if any(stripped.upper().startswith(f"{lbl}:") for lbl in _GUIDE_LABELS):
                continue
            if stripped.upper().startswith("STATEMENT:"):
                continue
            cleaned_lines.append(line)
        body = "\n".join(cleaned_lines).strip()

    if not body:
        # Sanitization dropped everything — fall back to the raw text
        # minus any labels, so the user at least hears *something*.
        body = re.sub(
            rf"^(?:{'|'.join(_GUIDE_LABELS + ('STATEMENT',))})\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        ).strip()

    # 2. First-person → second-person pronoun swap (Gemma mimics the
    # few-shot example register; lift to clinician voice).
    for pattern, replacement in _FIRST_TO_SECOND_PERSON:
        body = re.sub(pattern, replacement, body)

    # 2a. Recapitalise sentence-initial "you" / "your" / "you're" etc.
    # left lowercase by the pronoun swap.  `re.sub` with a lambda so we
    # don't have to enumerate every pronoun form.
    body = re.sub(
        r"(^|[.!?]\s+)(you|your|yours|you're|you'd|you'll|you've|yourself)\b",
        lambda m: m.group(1) + m.group(2)[0].upper() + m.group(2)[1:],
        body,
    )

    # 3. Prefix with a framing line so the user knows this is an
    # offered example and not the next CBT prompt.  The actual CBT
    # re-ask question is still spoken separately right after.
    framings = {
        "UNHELPFUL_THOUGHTS": "Here are some unhelpful thoughts you could consider: ",
        "CHALLENGE": "Here's an example challenge you could try: ",
        "REFRAME": "Here's an example reframe you could try: ",
    }
    prefix = framings.get(label_upper, "Here's an example: ")
    return prefix + body

# Set up logger for this module
from src.utils.log_util import get_logger
from src.utils.io_record import get_resp_log, log_question, set_question_prefix
import src.utils.io_record as io_rec

logger = get_logger("CBT")


def _log_cbt_intervention(
    stage: str,
    outcome: str,
    dim_label: str | None = None,
    detail: dict | None = None,
    technique: str = "cbt_stage",
):
    """Phase B: persist a CBT stage event into intervention_logs.

    Best-effort: never block the clinical pipeline on a DB write failure.
    """
    try:
        db = getattr(io_rec, "DB", None)
        session_id = getattr(io_rec, "SESSION_ID", None)
        if db is None or not session_id:
            return
        db.record_intervention_log(
            session_id=session_id,
            kind="CBT",
            stage=stage,
            technique=technique,
            outcome=outcome,
            dim_label=dim_label,
            detail=detail,
        )
    except Exception as e:
        logger.warning(f"intervention_log persist failed (non-fatal): {e}")


PROMPTER_CBT_STAGE0_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are reviewing the therapy session history and trying to ask the patient to choose a dimension that he/she would like to work on through this CBT process.
Only choose those dimensions that received a score of 2 in the conversation history.
Response format:
QUESTION: xxxx
'''

REASONER_CBT_STAGE1_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.



You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The user's answer towards the CBT question "Can you try to identify any unhelpful thoughts you have that contribute to this situation?". This is the step that the patient tries to recognize negative thoughts. These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.

You will be provideed with several examples in the format of STATEMENT: xxxxx; UNHELPFUL_THOUGHTS: xxxxx;


Usually the patient's statement and responses contain situation that is not valid or useful. As an AI assistant, you need to examine the validaity and utility of the patient's response.
There are 13 possible common cognitive distortions that the patient might encounter. And you might want to pay attention to.
1. Filtering: focusing on the negative but ignore the positive
2. Polarized thinking/extreme thinking: seeing everything in all-or-nothing terms.
3. Control fallacies: assumes only self or other takes all the responsibility and is to be blamed. Includes personalization (assuming self is responsible) and blaming (assuming others at fault). 
4. Fallacy of fairness: assumes life should be fair
5. Overgeneralization: assumes a rule from one experience, using one experience for all future experiences. 
6. Emotional reasoning: “if I feel it, it must be true.” Using emotional “terms” for all the situations. 
7. Fallacy of change: expects others to change
8. “shoulds”: using personal rules to judge self and others if the rules broken
9. Catastrophizing: expecting the worst case scenario.
10. Heaven’s reward fallacy: expecting to be rewarded in some way.
11. Always being right: being wrong is unacceptable, needs to be right all the time. 
12. Personalization (like control fallacies): assuming self is responsible.
13. Jumping to conclusions: make assumptions based on little evidence


If any of these cognitive distortion is included in the UNHELPFUL_THOUGHTS, the user may still properly identifies the unhelpful thoughts. But outline the cognitive distortions in analysis.

Your goal is:
Justify if the user is identify the unhelpful thoughts properly in the statement(0: identified properly, 1: not properly identified).
You also need to provide analysis to justify your decision. 


Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.





Example 1:
"STATEMENT: I have concern with your recent spending habits. I spent a lot of money. I spent a lot of money on clothing.; UNHELPFUL_THOUGHTS: I have issue on spending habits because I buy too much clothes.;"
DECISION: 0


Example 2:
"STATEMENT: I haven't done any creative work recently. I just don't know what are the creative things I can do.; UNHELPFUL THOUGHTS: I'm just not a creative person. I don't have any good ideas, and even if I did, they wouldn't be worth pursuing. "
DECISION:0


Example 3:
"STATEMENT: I have concern with my recent spending habits. I spent a lot of money. I spent a lot of money on clothing. RESPONSE: I like to go shopping "
DECISION:1

'''

REASONER_CBT_STAGE2_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.
3. The patient's answer to the CBT question "Can you challenge your thought?". This is the step when the patient begin to challenge the UNHELPFUL_THOUGHTS in the STATEMENT after recognizing and analyzing these thoughts. Challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?

You will be provideed with several examples in the format of STATEMENT: xxxxx; UNHELPFUL_THOUGHTS: xxxxx; CHALLENGE: xxxx;

Usually the patient's statement and responses contain situation that is not valid or useful. As an AI assistant, you need to examine the validaity and utility of the patient's response.
There are 13 possible common cognitive distortions that the patient might encounter. And you might want to pay attention to.
1. Filtering: focusing on the negative but ignore the positive
2. Polarized thinking/extreme thinking: seeing everything in all-or-nothing terms.
3. Control fallacies: assumes only self or other takes all the responsibility and is to be blamed. Includes personalization (assuming self is responsible) and blaming (assuming others at fault). 
4. Fallacy of fairness: assumes life should be fair
5. Overgeneralization: assumes a rule from one experience, using one experience for all future experiences. 
6. Emotional reasoning: “if I feel it, it must be true.” Using emotional “terms” for all the situations. 
7. Fallacy of change: expects others to change
8. “shoulds”: using personal rules to judge self and others if the rules broken
9. Catastrophizing: expecting the worst case scenario.
10. Heaven’s reward fallacy: expecting to be rewarded in some way.
11. Always being right: being wrong is unacceptable, needs to be right all the time. 
12. Personalization (like control fallacies): assuming self is responsible.
13. Jumping to conclusions: make assumptions based on little evidence


Your goal is:
Justify if the patient challenges the unhelpful thoughts (UNHELPFUL_THOUGHTS) properly. (0: properly challenge the unhelpful thoughts, 1: not challenge the unhelpful thoughts)
Note that:
1. The patient might identify the unhelpful thoughts in a wrong way (with cognitive distortions). In this case, the patient might challenge the STATEMENT or some unhelpful thoughts that related to this STATEMENT that is not explicitly identified, which is acceptable.
2. It would be acceptable if the patient not fully challenge the validity and usability of the unhelpful thoughts/situation. As long as the CHALLENGE is related to the STATEMENT and UNHELPFUL THOUGHTS, it is acceptable.
Make notes about the distortions in the analysis.
You also need to provide analysis to justify your decision. 


Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.



Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself.; CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?;"
DECISION: 0


Example 2:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself.; CHALLENGE: I don't know how to challenge my thoughts;"
DECISION: 1

Example 3:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself.; CHALLENGE: I am not cool to engage in the social events;"
DECISION: 1

Example 4:
"STATEMENT: I don't smoke cigarettes, but I vape every day. I vape when I am working hard or debugging.; UNHELPFUL_THOUGHTS: I can't work or solve problems effectively without vaping.; CHALLENGE: There are several potential health impacts of vaping. It's bad for my lung.;"
DECISION: 0

'''

REASONER_CBT_STAGE3_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.
3. The patient's response to challenge the UNHELPFUL_THOUGHTS after recognizing and analyzing these thoughts. Challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?
4. The patient's answer to the CBT question "What is another way of thinking about this situation?". This is the step that the patient tires to reframe your unhelpful thoughts into more balanced, realistic, and constructive ones. This process is about changing the way the patient thinks about the situation, which can lead to changes in emotions and behaviors.


You will be provideed with several examples in the format of STATEMENT: xxxxx; UNHELPFUL_THOUGHTS: xxxxx; CHALLENGE: xxxxx; REFRAME: xxxxx;

Your goal is:
Justify if the patient reframes the unhelpful thoughts properly. (0: properly reframe the unhelpful thoughts, 1: fail to reframe the unhelpful thoughts).
You also need to provide analysis to justify your decision. 


Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.




Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'; REFRAME: People may have their own concerns and may not be focused on me all the time and I've had good conversations in the past without embarrassing myself.;"
DECISION: 0

Example 2:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'; REFRAME: I'm finding it hard to reframe them. I still believe that: 'People are definitely judging me; I just know it.;"
DECISION: 1
'''

# Rephrase recap for Stage 3 based on user's CHALLENGE
RECAP_CBT_STAGE3_CHALLENGE_PROMPT = '''You are a concise and supportive therapist-assistant.

You will be provided with:
1) The patient's brief STATEMENT of the situation
2) The patient's identified UNHELPFUL_THOUGHTS
3) The patient's CHALLENGE to those thoughts

Your task is to rephrase the patient's CHALLENGE into a short recap that reminds the patient what they already did to challenge their thoughts. Be neutral, supportive, and concise.

Rules:
- 1-2 sentences only
- No extra headers or labels, output the recap directly
- Do not add new ideas beyond the user's content
- Use second-person neutral tone (you/your)
'''

GUIDE_CBT_STAGE1_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to answer the cognitive behavioural therapy (CBT) questions based-on patient's statement provided.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.


Your goal is:
Enumerate MULTIPLE distinct unhelpful thoughts that go through the patient's mind when they experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic. Address the patient in the second person ("you think ...", "you fear ...").

REQUIRED OUTPUT LENGTH: enumerate 3 to 5 distinct unhelpful thoughts, each as its own clause separated by semicolons. Do NOT output a single one-sentence guess. Each clause should start with a verb that names a cognitive distortion pattern: "you think ...", "you fear ...", "you worry ...", "you see ... as ...", "you assume ...". This matches the CaiTI clinical reference enumeration style.


Response format:
UNHELPFUL_THOUGHTS: <clause 1>; <clause 2>; <clause 3>; <clause 4>; <optional clause 5>.

You will be provided with several examples with the statement and example unhelpful thoughts in the format of "STATEMENT: xxxxx, UNHELPFUL_THOUGHTS: xxxxxx".



Example 1:
STATEMENT: I have not taken days off recently. Paper deadline is coming up! I don't even have time to sleep.
UNHELPFUL_THOUGHTS: You think taking a day off will set your paper back irrecoverably; you fear that anything less than nonstop effort means you are slacking; you worry that your colleagues will judge you if you rest while the deadline is close; you see sleep as competing with productivity rather than supporting it; you assume that one missed work day cannot be recovered later.

Example 2:
STATEMENT: I don't chat a lot with my colleagues. I can talk to them about work, but I can't talk to them about life. I can't seem to find common ground for life conversations with them. My personal life is quite dull and lacks the variety of personal and family activities that they have.
UNHELPFUL_THOUGHTS: You think your life is too dull to be interesting to anyone; you fear they will find you boring the moment the conversation leaves work topics; you worry that not having big family or activity stories means you have nothing worth sharing; you see small-talk skill as something you either have or lack entirely; you assume a few awkward exchanges prove you cannot connect socially at all.

Example 3 (TARGET SHAPE — enumerate this many clauses in this register):
STATEMENT: I just always forget to take my medication, and I don't wanna take too much of it, then not be able to get off of it.
UNHELPFUL_THOUGHTS: You think you always forget, so it feels hopeless to try to be consistent; you fear that if you take it regularly, you will get dependent and won't be able to stop; you worry that even taking the prescribed amount is too much and will harm you; you see needing medication as a loss of control or a personal weakness; you assume a few missed doses mean you can't do this at all.
'''

GUIDE_CBT_STAGE2_PROMPT  = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.

Your goal is:
Try to help the patient challenge the unhelpful thoughts (UNHELPFUL_THOUGHTS) properly. After recognizing and analyzing these UNHELPFUL_THOUGHTS, challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?

Response format:
CHALLENGE: xxxx

You will be provideed with several examples with the statement and example unhelpful thoughts in the format of "STATEMENT: xxxxx. UNHELPFUL_THOUGHTS: xxxxxx. CHALLENGE: xxxxxx". 



Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say. UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. 
CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'"


Example 2:
"STATEMENT: My issue is procrastination, especially when it comes to completing assignments for work or school. UNHELPFUL_THOUGHTS: When faced with a task I need to complete, I often have negative thoughts like: 'I'll never finish this on time; I'm so lazy.' 
CHALLENGE: asking myself: I have successfully complete a similar fairly challenging school project before. I might not be fair to label myself as lazy just because I'm struggling with this task."

'''

GUIDE_CBT_STAGE3_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.
3. The patient's response to challenge the UNHELPFUL_THOUGHTS after recognizing and analyzing these thoughts. Challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?
4. The patient's response to reframe the UNHELPFUL_THOUGHTS into more balanced, realistic, and constructive ones. This process is about changing the way the patient thinks about the situation, which can lead to changes in emotions and behaviors.


Your goal is:
Try to reframe the unhelpful thoughts (UNHELPFUL_THOUGHTS) for the patient. This is the step to reframe the patient's unhelpful thoughts into more balanced, realistic, and constructive ones. 


Response format:
REFRAME: xxxx




Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say. UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'."
REFRAME: People may have their own concerns and may not be focused on you all the time. You may had good conversations in the past without embarrassing your self.

Example 2:
"STATEMENT: I often avoid speaking up in meetings at work or in front of others. I’m afraid my ideas aren’t good enough. UNHELPFUL_THOUGHTS: If I speak up, people will think my ideas are silly. Others are much smarter than me, so my opinion doesn’t matter. CHALLENGE: I can challenge these thoughts by asking: ‘Have my colleagues ever reacted negatively when I spoke before?’ and ‘Do people usually respect different opinions, even if they’re not perfect?’."
REFRAME: My ideas have value, and sharing them can contribute to the discussion. Others are likely focused on the topic, not on judging me, and speaking up can help me grow more confident.
'''

def _chat_complete(system_content: str, user_content: str, role: LLMRole = LLMRole.GENERAL):
    return llm_complete(system_content, user_content, role=role)

def stage0_prompter(history: str) -> str:
    """Paper role: GENERAL (CBT entry prompt, not explicitly microbenchmarked)."""
    payload = f"HISTORY: {history}"
    return _chat_complete(PROMPTER_CBT_STAGE0_PROMPT, payload, role=LLMRole.GENERAL)

def stage1_reasoner(statement: str, unhelpful_thoughts: str) -> str:
    """Paper role: CBT_REASONER (GPT-4 in paper, hardest Stage 1 reasoning)."""
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts};"'
    return _chat_complete(REASONER_CBT_STAGE1_PROMPT, payload, role=LLMRole.CBT_REASONER)

def stage2_reasoner(statement: str, unhelpful_thoughts: str, challenge: str) -> str:
    """Paper role: CBT_REASONER (GPT-4 in paper)."""
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge};"'
    return _chat_complete(REASONER_CBT_STAGE2_PROMPT, payload, role=LLMRole.CBT_REASONER)

def stage3_reasoner(statement: str, unhelpful_thoughts: str, challenge: str, reframe: str) -> str:
    """Paper role: CBT_REASONER (GPT-4 in paper)."""
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge}; REFRAME: {reframe};"'
    return _chat_complete(REASONER_CBT_STAGE3_PROMPT, payload, role=LLMRole.CBT_REASONER)

def stage1_guide(statement: str) -> str:
    """Paper role: CBT_GUIDE (GPT-3.5-Turbo in paper; less "reads into feelings")."""
    payload = f"STATEMENT: {statement}"
    return _chat_complete(GUIDE_CBT_STAGE1_PROMPT, payload, role=LLMRole.CBT_GUIDE)

def stage2_guide(statement: str, unhelpful_thoughts: str) -> str:
    """Paper role: CBT_GUIDE (GPT-3.5-Turbo in paper)."""
    payload = f"STATEMENT: {statement}. UNHELPFUL_THOUGHTS: {unhelpful_thoughts}"
    return _chat_complete(GUIDE_CBT_STAGE2_PROMPT, payload, role=LLMRole.CBT_GUIDE)

def stage3_guide(statement: str, unhelpful_thoughts: str, challenge: str) -> str:
    """Paper role: CBT_GUIDE (GPT-3.5-Turbo in paper)."""
    payload = f"STATEMENT: {statement}. UNHELPFUL_THOUGHTS: {unhelpful_thoughts}. CHALLENGE: {challenge}"
    return _chat_complete(GUIDE_CBT_STAGE3_PROMPT, payload, role=LLMRole.CBT_GUIDE)

def recap_stage3_challenge(statement: str, unhelpful_thoughts: str, challenge: str) -> str:
    """Paper role: CBT_GUIDE (recap is part of Stage 3 guidance flow)."""
    payload = (
        f"STATEMENT: {statement}\n"
        f"UNHELPFUL_THOUGHTS: {unhelpful_thoughts}\n"
        f"CHALLENGE: {challenge}"
    )
    return _chat_complete(RECAP_CBT_STAGE3_CHALLENGE_PROMPT, payload, role=LLMRole.CBT_GUIDE)

__all__ = [
    "stage0_prompter",
    "stage1_reasoner",
    "stage2_reasoner",
    "stage3_reasoner",
    "stage1_guide",
    "stage2_guide",
    "stage3_guide",
]

def run_cbt(question_lib, crisis_callback=None):
    """Paper-aligned 3-stage CBT clinical loop: Recognize -> Challenge -> Reframe.

    Presents all Score-2 dimensions and awaits the user's selection (user
    autonomy).  Each subsequent stage uses a Reasoner (validity/utility
    check) and a Guide (clinical direction) with up to two retries.

    C7: `crisis_callback` is a no-arg callable supplied by the handler that
    (a) runs a fresh _crisis_scan on the current question_lib state and
    (b) delivers the SAFETY_RESOURCES_MESSAGE if a NEW critical dim just
    hit Score 2 (e.g. user mentions self-harm in their CHALLENGE).
    When the callback reports True, we pause CBT and exit — continuing
    clinical work while a crisis is unaddressed is not safe.
    """
    logger.info("[CBT] Starting CBT flow (3-stage: Recognize / Challenge / Reframe).")

    def _crisis_intervened() -> bool:
        """Call the handler's scan+deliver hook; True means pause CBT."""
        if crisis_callback is None:
            return False
        try:
            return bool(crisis_callback())
        except Exception as e:
            logger.warning(f"[CBT] crisis_callback raised: {e}")
            return False
    # Collect dimensions with score=2
    candidates = []
    idx = 1
    for i in range(1, len(question_lib) + 1):
        for j in range(1, len(question_lib[str(i)]) + 1):
            entry = question_lib[str(i)][str(j)]
            if any((isinstance(s, int) and s == 2) for s in entry.get("score", [])):
                candidates.append((
                    idx,
                    i,
                    j,
                    entry["label"],
                    entry.get("name", entry["label"]),
                ))
                idx += 1

    if not candidates:
        logger.info("[CBT] No Score-2 dimensions found — skipping CBT protocol.")
        log_question("We do not have a dimension at score 2 to work on today. We will conclude here.")
        return

    # Present Score-2 dimensions and await user selection.
    # G6 — legacy/demo wording restored verbatim: "you have issue in:",
    # "Which dimension would you like to work on today? Tell me the
    # dimension number. For example: 1". Matches mexa_llmtherapist_demo.mp4
    # at 4:40 and legacy-prototype/src/CBT.py:353-361.
    lines = [
        "Thank you for answering the questions.",
        "According to your previous responses, you have issue in:",
    ]
    for k, _, _, _, name0 in candidates:
        lines.append(f"{k}) {name0}")
    lines.append(
        "Which dimension would you like to work on today? "
        "Tell me the dimension number. For example: 1"
    )
    log_question(" \n".join(lines))
    resp = get_resp_log()
    if isinstance(resp, str) and "SESSION_END" in resp:
        logger.info("Session End signal received in CBT dimension selection.")
        return
    if isinstance(resp, str) and resp.strip().lower().find("stop") != -1:
        logger.info("User requested stop at CBT dimension selection.")
        return

    def _pick_candidate(answer: str):
        ans = str(answer).strip().lower()
        # Numeric path: accepts both digit ("3") and word ("three") forms.
        n = _extract_choice_number(ans)
        if n is not None:
            for (k0, i0, j0, lbl0, name0) in candidates:
                if k0 == n:
                    return (i0, j0, lbl0, name0)
        # Dimension-name fallback: user spoke the dim label / human name
        # instead of a number ("medication", "mood").
        for (_, i0, j0, lbl0, name0) in candidates:
            if name0.lower() in ans or lbl0.lower() in ans:
                return (i0, j0, lbl0, name0)
        return None

    chosen = _pick_candidate(resp)
    if chosen is None:
        opts = "; ".join([f"{k}) {name0}" for (k, _, _, _, name0) in candidates])
        log_question(
            f"Please reply with a single number between 1 and {len(candidates)}. "
            f"Example: 1. Options: {opts}"
        )
        resp = get_resp_log()
        if isinstance(resp, str) and "SESSION_END" in resp:
            logger.info("Session End signal received in CBT dimension retry.")
            return
        if isinstance(resp, str) and resp.strip().lower().find("stop") != -1:
            logger.info("User requested stop at CBT dimension retry.")
            return
        chosen = _pick_candidate(resp)
        if chosen is None:
            logger.info("Failed to parse user choice for CBT dimension. Exit CBT.")
            log_question("I could not determine your choice. We will stop CBT for now.")
            return

    i_sel, j_sel, label_sel, name_sel = chosen
    logger.info(f"[CBT] Dimension selected: {label_sel} ({name_sel}) — entering Stage 1 (Recognize).")
    _log_cbt_intervention(
        stage="dimension_selected",
        outcome="started",
        dim_label=label_sel,
        detail={"name": name_sel, "i_sel": i_sel, "j_sel": j_sel},
    )

    # Stage 1 — RECOGNIZE: derive statement from RV notes of the chosen dimension.
    # Prefer the latest RV follow-up response (followup_resp_1),
    # then fallback to followup_resp, then original_resp.
    statement = ""
    notes_list = question_lib[str(i_sel)][str(j_sel)].get("notes", [])
    for note_entry in reversed(notes_list):
        if not isinstance(note_entry, list):
            continue
        # Only consider RV note entries by checking the presence of rv fields
        has_rv_field = any((isinstance(x, str) and ("rv_decision:" in x or "rv_validation:" in x)) for x in note_entry)
        if not has_rv_field:
            continue
        # Try to extract in priority order
        for s in note_entry:
            if isinstance(s, str) and s.startswith("followup_resp_1: "):
                statement = s.split(": ", 1)[1]
                break
        if statement:
            break
        for s in note_entry:
            if isinstance(s, str) and s.startswith("followup_resp: "):
                statement = s.split(": ", 1)[1]
                break
        if statement:
            break
        for s in note_entry:
            if isinstance(s, str) and s.startswith("original_resp: "):
                statement = s.split(": ", 1)[1]
                break
        if statement:
            break

    # Phase A: Pre-stage crisis scan. Catch critical-dim Score=2 that may
    # have been set during the screening RV loop before we prompt the
    # user, rather than only after they respond.
    if _crisis_intervened():
        logger.warning("[CBT] Crisis intervention fired before Stage 1 entry; pausing CBT.")
        return

    # Add recap prefix (similar to RV), then ask to identify unhelpful thoughts
    recap = (
        f"Let us work on dimension '{name_sel}'. "
        f"From our record, you mentioned that: {statement}"
    )
    set_question_prefix(recap)
    logger.info("[CBT] Stage 1 (Recognize) — prompting user to identify unhelpful thoughts.")
    log_question("Can you try to identify any unhelpful thoughts you have that contribute to this situation?")
    unhelpful = get_resp_log()
    if isinstance(unhelpful, str) and "SESSION_END" in unhelpful:
        logger.info("[SESSION] End signal received in CBT Stage 1 — closing session.")
        return
    if isinstance(unhelpful, str) and unhelpful.strip().lower().find("stop") != -1:
        logger.info("[CBT] User requested stop at Stage 1 — pausing CBT.")
        return
    # C7: user may have just mentioned a critical-dim concern. Pause CBT if so.
    if _crisis_intervened():
        logger.warning("[CBT] Crisis intervention fired in Stage 1; pausing CBT.")
        return

    # Reason and guide up to two retries
    dec1_raw = stage1_reasoner(statement, unhelpful)
    dec1 = _parse_decision(dec1_raw)
    retry = 0
    while dec1 == "1" and retry < 2:
        guide1 = stage1_guide(statement)
        log_question(_sanitize_guide_text(guide1, "UNHELPFUL_THOUGHTS"))
        # Legacy prompt was "Please provide your UNHELPFUL_THOUGHTS again,
        # in one sentence." — the ALL-CAPS label with an underscore sounds
        # awful through Piper TTS ("UNHELPFUL underscore THOUGHTS").  Same
        # meaning, natural voice.
        log_question("Please share those unhelpful thoughts again, in one sentence.")
        unhelpful = get_resp_log()
        if isinstance(unhelpful, str) and "SESSION_END" in unhelpful:
            logger.info("Session End signal received in CBT stage 1 retry.")
            return
        if isinstance(unhelpful, str) and unhelpful.strip().lower().find("stop") != -1:
            logger.info("User requested stop during CBT stage 1 retry.")
            return
        dec1_raw = stage1_reasoner(statement, unhelpful)
        dec1 = _parse_decision(dec1_raw)
        retry += 1
    if dec1 == "1":
        # Paper p.15: direct the user to seek professional help after 3 failed
        # attempts at a CBT stage.
        # G7 — legacy/demo never speaks the 988/SAMHSA hotline on CBT
        # stage failure. Gated off by CBT_ESCALATION_ENABLED.
        if CBT_ESCALATION_ENABLED:
            log_question(CBT_ESCALATION_MESSAGE)
        log_question("It seems difficult to identify the unhelpful thoughts right now. Let's pause CBT and revisit later.")
        # record brief CBT notes
        question_lib[str(i_sel)][str(j_sel)]["notes"].append([
            f"CBT_dimension: {label_sel}",
            f"CBT_statement: {statement}",
            f"CBT_unhelpful_thoughts: {unhelpful}",
            "CBT_stage: 1_failed",
            f"CBT_escalation_delivered: {'true' if CBT_ESCALATION_ENABLED else 'false'}",
        ])
        _log_cbt_intervention(
            stage="recognize",
            outcome="failed",
            dim_label=label_sel,
            detail={"statement": statement, "unhelpful": unhelpful, "escalation_delivered": CBT_ESCALATION_ENABLED},
        )
        return

    # Stage 2 — CHALLENGE: challenge the unhelpful thoughts
    # Phase A: Pre-stage crisis scan, in case Stage 1's Reasoner/Guide
    # loop or any async path flipped a critical dim to Score=2.
    if _crisis_intervened():
        logger.warning("[CBT] Crisis intervention fired before Stage 2 entry; pausing CBT.")
        return
    logger.info("[CBT] Stage 2 (Challenge) — prompting user to challenge unhelpful thoughts.")
    log_question("Now, how could you challenge those unhelpful thoughts? Please write a brief challenge.")
    challenge = get_resp_log()
    if isinstance(challenge, str) and "SESSION_END" in challenge:
        logger.info("[SESSION] End signal received in CBT Stage 2 — closing session.")
        return
    if isinstance(challenge, str) and challenge.strip().lower().find("stop") != -1:
        logger.info("[CBT] User requested stop at Stage 2 — pausing CBT.")
        return
    if _crisis_intervened():
        logger.warning("[CBT] Crisis intervention fired in Stage 2; pausing CBT.")
        return

    dec2_raw = stage2_reasoner(statement, unhelpful, challenge)
    dec2 = _parse_decision(dec2_raw)
    retry = 0
    while dec2 == "1" and retry < 2:
        guide2 = stage2_guide(statement, unhelpful)
        log_question(_sanitize_guide_text(guide2, "CHALLENGE"))
        # Legacy said "Please try to CHALLENGE the unhelpful thoughts
        # again, in one sentence." — the ALL-CAPS label is awkward spoken
        # aloud.  Keep the clinical meaning; soften the surface form.
        log_question("Please try to challenge the unhelpful thoughts again, in one sentence.")
        challenge = get_resp_log()
        if isinstance(challenge, str) and "SESSION_END" in challenge:
            logger.info("Session End signal received in CBT stage 2 retry.")
            return
        if isinstance(challenge, str) and challenge.strip().lower().find("stop") != -1:
            logger.info("User requested stop during CBT stage 2 retry.")
            return
        dec2_raw = stage2_reasoner(statement, unhelpful, challenge)
        dec2 = _parse_decision(dec2_raw)
        retry += 1
    if dec2 == "1":
        # G7 — legacy/demo never speaks the 988/SAMHSA hotline on CBT
        # stage failure. Gated off by CBT_ESCALATION_ENABLED.
        if CBT_ESCALATION_ENABLED:
            log_question(CBT_ESCALATION_MESSAGE)
        log_question("Challenging the thought seems difficult now. Let's pause CBT and revisit later.")
        question_lib[str(i_sel)][str(j_sel)]["notes"].append([
            f"CBT_dimension: {label_sel}",
            f"CBT_statement: {statement}",
            f"CBT_unhelpful_thoughts: {unhelpful}",
            f"CBT_challenge: {challenge}",
            "CBT_stage: 2_failed",
            f"CBT_escalation_delivered: {'true' if CBT_ESCALATION_ENABLED else 'false'}",
        ])
        _log_cbt_intervention(
            stage="challenge",
            outcome="failed",
            dim_label=label_sel,
            detail={"statement": statement, "unhelpful": unhelpful, "challenge": challenge, "escalation_delivered": CBT_ESCALATION_ENABLED},
        )
        return

    # Stage 3 — REFRAME: reframe the thought (prepend an LLM-rephrased recap of user's CHALLENGE)
    # Phase A: Pre-stage crisis scan before the Reframe prompt.
    if _crisis_intervened():
        logger.warning("[CBT] Crisis intervention fired before Stage 3 entry; pausing CBT.")
        return
    recap3 = recap_stage3_challenge(statement, unhelpful, challenge)
    set_question_prefix(recap3.strip())
    logger.info("[CBT] Stage 3 (Reframe) — prompting user to reframe unhelpful thoughts into a balanced one.")
    log_question("Finally, can you reframe the unhelpful thought into a more balanced, constructive one?")
    reframe = get_resp_log()
    if isinstance(reframe, str) and "SESSION_END" in reframe:
        logger.info("[SESSION] End signal received in CBT Stage 3 — closing session.")
        return
    if isinstance(reframe, str) and reframe.strip().lower().find("stop") != -1:
        logger.info("[CBT] User requested stop at Stage 3 — pausing CBT.")
        return
    if _crisis_intervened():
        logger.warning("[CBT] Crisis intervention fired in Stage 3; pausing CBT.")
        return

    dec3_raw = stage3_reasoner(statement, unhelpful, challenge, reframe)
    dec3 = _parse_decision(dec3_raw)
    retry = 0
    while dec3 == "1" and retry < 2:
        guide3 = stage3_guide(statement, unhelpful, challenge)
        log_question(_sanitize_guide_text(guide3, "REFRAME"))
        # Legacy said "Please REFRAME again in one or two sentences."
        log_question("Please try to reframe that again, in one or two sentences.")
        reframe = get_resp_log()
        if isinstance(reframe, str) and "SESSION_END" in reframe:
            logger.info("Session End signal received in CBT stage 3 retry.")
            return
        if isinstance(reframe, str) and reframe.strip().lower().find("stop") != -1:
            logger.info("User requested stop during CBT stage 3 retry.")
            return
        dec3_raw = stage3_reasoner(statement, unhelpful, challenge, reframe)
        dec3 = _parse_decision(dec3_raw)
        retry += 1
    if dec3 == "1":
        # G7 — legacy/demo never speaks the 988/SAMHSA hotline on CBT
        # stage failure. Gated off by CBT_ESCALATION_ENABLED.
        if CBT_ESCALATION_ENABLED:
            log_question(CBT_ESCALATION_MESSAGE)
        log_question("Reframing seems hard right now. Let's pause CBT and revisit later.")
        question_lib[str(i_sel)][str(j_sel)]["notes"].append([
            f"CBT_dimension: {label_sel}",
            f"CBT_statement: {statement}",
            f"CBT_unhelpful_thoughts: {unhelpful}",
            f"CBT_challenge: {challenge}",
            f"CBT_reframe: {reframe}",
            "CBT_stage: 3_failed",
            f"CBT_escalation_delivered: {'true' if CBT_ESCALATION_ENABLED else 'false'}",
        ])
        _log_cbt_intervention(
            stage="reframe",
            outcome="failed",
            dim_label=label_sel,
            detail={"statement": statement, "unhelpful": unhelpful, "challenge": challenge, "reframe": reframe, "escalation_delivered": CBT_ESCALATION_ENABLED},
        )
        return

    # Success
    question_lib[str(i_sel)][str(j_sel)]["notes"].append([
        f"CBT_dimension: {label_sel}",
        f"CBT_statement: {statement}",
        f"CBT_unhelpful_thoughts: {unhelpful}",
        f"CBT_challenge: {challenge}",
        f"CBT_reframe: {reframe}",
        "CBT_stage: success"
    ])
    _log_cbt_intervention(
        stage="reframe",
        outcome="success",
        dim_label=label_sel,
        detail={"statement": statement, "unhelpful": unhelpful, "challenge": challenge, "reframe": reframe},
    )
    logger.info(f"[CBT] All 3 stages completed successfully on dim '{label_sel}'.")
    log_question("Great work today. We completed the CBT steps for this topic. Thank you for your effort.")


