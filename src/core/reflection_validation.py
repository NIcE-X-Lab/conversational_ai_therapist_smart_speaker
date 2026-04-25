"""Reflection-Validation (R-V) pipeline — Reasoner + OARS Validator + Guide.

Paper-aligned architecture: three distinct LLM tasks, each with a focused
prompt.  The Reasoner decides whether a follow-up response is on-topic.  If
it is, the OARS-based Validator produces an empathetic complex reflection
following Motivational Interviewing (MI) principles.  If it is not, the
Guide gently redirects the user back to the topic.

A thin `rv_consolidated()` wrapper is retained for call-site compatibility
but now dispatches to the split pipeline.
"""

from src.models.llm_client import llm_complete, LLMRole
from src.utils.log_util import get_logger
import src.utils.io_record as io_rec

logger = get_logger("ReflectionValidation")


def _log_mi_intervention(technique: str, detail: dict | None = None, dim_label: str | None = None):
    """Phase B: record an MI micro-event into intervention_logs."""
    try:
        db = getattr(io_rec, "DB", None)
        session_id = getattr(io_rec, "SESSION_ID", None)
        if db is None or not session_id:
            return
        db.record_intervention_log(
            session_id=session_id,
            kind="MI",
            stage="rv",
            technique=technique,
            outcome="delivered",
            dim_label=dim_label,
            detail=detail,
        )
    except Exception as e:
        logger.warning(f"intervention_log (MI) persist failed (non-fatal): {e}")


# ── Reasoner: decide if follow-up is topically related ───────────────────
RV_REASONER_SYSTEM_PROMPT = '''You are a clinical reasoning agent.
Decide whether the client's follow-up is related to the conversation topic
or their original response.

Input format:
{"Topic": "...", "Original Question": "...", "Original Response": "...", "Follow-up Response": "..."}

Output format (single line, no other text):
DECISION: 0
or
DECISION: 1

Rules:
- DECISION: 0 if the follow-up elaborates on the topic or original response,
  even loosely (e.g. describes cause, context, or an example).
- DECISION: 1 only if the follow-up is clearly off-topic or unrelated.
- Do NOT output analysis, questions, or any text other than the DECISION line.

Example 1:
{"Topic": "Managing mood", "Original Question": "How's your mood recently?", "Original Response": "I am sad recently.", "Follow-up Response": "I am sad because I am homesick. I haven't been back home for a few years due to Covid-19."}
DECISION: 0

Example 2:
{"Topic": "Family support", "Original Question": "Do you feel your family is supportive?", "Original Response": "I don't feel my family is supportive.", "Follow-up Response": "I live away from my parents and family. We are in two different countries. We don't usually talk a lot."}
DECISION: 0

Example 3:
{"Topic": "Taking medication as prescribed", "Original Question": "Are you taking your medication as prescribed?", "Original Response": "I don't want to follow the prescription.", "Follow-up Response": "I have been trying to exercise more and eat healthier. I want to try and handle my symptoms naturally before resorting to medication."}
DECISION: 0

Example 4:
{"Topic": "Participating primary and mental health care", "Original Question": "Have you been going to your prescriber regularly?", "Original Response": "I haven't gone to my prescriber for a long time.", "Follow-up Response": "I've been trying to pick up running as a hobby. I find it helps clear my mind and relieve stress. Plus, it's a great way to stay fit and healthy."}
DECISION: 1

Example 5:
{"Topic": "Organizing personal possessions and doing housework", "Original Question": "Have you been keeping up with your housework?", "Original Response": "I never mop the floor.", "Follow-up Response": "Recently, I started learning how to cook. I'm trying to make dishes from different cuisines. Yesterday, I made pasta for the first time and it turned out really good."}
DECISION: 1
'''


# ── Validator: empathic validation & support (paper p.13, Fig.9) ─────────
RV_VALIDATOR_OARS_SYSTEM_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are in the conversation with a client. You need to provide empathic validation and support to the client.

You will be provided with:
1. The conversation topic.
2. The original response from the client.
3. The follow-up response from the client to the question "Can you tell me more about it?".
These will be provided in the format of '{"Topic": XXXX, "Original Question": XXXX, "Original Response": XXXX, "Follow-up Response": XXXX}'

Goal:
Provide empathic validation and support to the client based on the conversation topic, original response, and follow-up response.
First express understanding of the client's follow-up response, then offer grounded, concrete support or strategies connected to what they said.
Use the phrases and wording the client used rather than heavily rephrasing, so your response feels anchored to their experience.
Do not read into the client's mind or make strong assumptions. Do not output any open-ended questions or invitations for further follow-up.

Formatting:
- ASCII characters only. Replace smart quotes, en/em dashes, and ellipses with ASCII equivalents (' " - ...).
- Aim for a substantive reflection: typically 3-5 sentences. Shorter is fine when the client's input is short; go longer when the situation genuinely warrants deeper support.
- Output format (one block of prose, prefixed with the label):
VALIDATION: <your validation and support text here>

Example 1:
{"Topic": "Maintaining stable weight", "Original Response": "My weight increased a lot recently.", "Follow-up Response": "I am a ISFP. I like to follow my heart. My personality leads me to just eat whenever I want. And I usually don't control how much I eat."}
VALIDATION: I completely understand that as an ISFP, rigid rules may not be your preference. Nonetheless, it is worth being aware of how eating whenever you want can affect your health over time. If the recent weight gain is tied to this pattern, small adjustments to your diet and daily routine can make a meaningful difference without feeling like strict rules. Talking with a healthcare professional or dietitian can also help you build a more tailored, sustainable approach that still honours how you naturally operate.

Example 2:
{"Topic": "Maintaining stable weight", "Original Response": "My weight increased a lot recently.", "Follow-up Response": "I have upcoming deadlines. So I often do stress eating."}
VALIDATION: It sounds like the pressure from your upcoming deadlines is really weighing on you, and the stress eating is one of the ways your body has been responding to that pressure. This is a very common reaction, and it does not make you weak; it makes sense that a stressed body reaches for comfort. You might notice it helps to identify the moments when stress peaks, keep some healthier snacks nearby, and protect regular meal times even during busy weeks. Leaning on a friend, family member, or therapist when deadlines pile up can also give you a healthier outlet than food alone.

Example 3:
{"Topic": "Managing mood", "Original Response": "I am sad recently.", "Follow-up Response": "My sadness stems from a variety of factors. I have been dealing with a lot of stress at work, and it is affecting my mood. I am also finding it hard to connect with my friends due to the pandemic, and this isolation has been making me feel quite depressed."}
VALIDATION: It sounds like you are going through a really challenging time, with work stress and pandemic isolation compounding each other and weighing on your mood. It is important to acknowledge these feelings and to understand that feeling overwhelmed in these circumstances is completely understandable. Staying connected during a pandemic can be hard, but small steps such as scheduled video calls with people you trust, or joining online communities around interests you enjoy, can ease the isolation over time. Sharing what you are going through with a friend, family member, or a mental health professional can also offer relief and help you cope more sustainably.
'''


# ── Guide: redirect off-topic follow-up (paper p.13, Fig.9) ──────────────
RV_GUIDE_SYSTEM_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are in the conversation with a client.

You will be provided with:
1. The conversation topic.
2. The original response from the client.
3. The follow-up response from the client to the question "Can you tell me more about it?". This follow-up is off-topic or unclear, and the client needs guidance to provide a more relevant continuation.
These will be provided in the format of '{"Topic": XXXX, "Original Question": XXXX, "Original Response": XXXX, "Follow-up Response": XXXX}'

Goal:
Guide the client to produce a valid follow-up response that adds detail to the original response or the topic.
First express understanding of the client's follow-up response, then gently lead them back toward the right direction.
Use the phrases the client already used when redirecting, rather than heavily rephrasing. Do not make assumptions about them.
Do not output open-ended invitations like "feel free to share anything"; instead, ask one concrete, focused question that ties back to the topic.

Formatting:
- ASCII characters only. Replace smart quotes, en/em dashes, and ellipses with ASCII equivalents.
- Aim for a substantive redirect: typically 2-4 sentences. Include a brief acknowledgement, a connective such as "However, since we are discussing ...", and a focused redirect question that uses the client's own phrasing.
- Output format (one block of prose, prefixed with the label):
Guide: <your guidance text here>

Example 1:
{"Topic": "Managing mood", "Original Response": "I am sad recently.", "Follow-up Response": "I love to go out for movie alone."}
Guide: It is good to know about your habit of going out for movies on your own. However, since we are focusing on mood management and you mentioned you have been sad recently, could you tell me more about what might be contributing to your sadness? For example, are there specific situations, people, or times of day where you notice the sadness getting heavier?

Example 2:
{"Topic": "Maintaining stable weight", "Original Response": "My weight increased a lot recently.", "Follow-up Response": "I am a ISFP. I like to follow my heart."}
Guide: It is interesting to hear about your personality type and how you like to follow your heart. However, since we are talking about the recent weight increase, could you share more about how your day-to-day routine might have contributed to this change? In particular, I would love to hear about your eating habits, activity levels, or any recent changes in your schedule.

Example 3:
{"Topic": "Maintaining stable weight", "Original Response": "My weight increased a lot recently.", "Follow-up Response": "Besides that, I have been finding it hard to concentrate at work. I have been making a lot of errors and it is not like me at all. It is been stressing me out."}
Guide: I understand that you are dealing with difficulties concentrating at work and that the stress has been weighing on you. However, since our current focus is the recent weight increase, could you elaborate on any recent lifestyle changes that might be contributing to it? For instance, have your eating patterns, sleep, or activity levels shifted along with the work stress you are describing?

Example 4:
{"Topic": "Managing mood", "Original Response": "My mood swings a lot these days.", "Follow-up Response": "The weather has been really unpredictable lately. It is been raining heavily for the past few days and then suddenly it is sunny. I really enjoy going for a walk when it is sunny outside."}
Guide: It is nice to hear how you enjoy going for a walk when the weather is sunny. However, since we are looking at your recent mood swings, could you share more about the situations or triggers that tend to move your mood? For example, are there specific events, thoughts, or moments in the day when you notice your mood shifting the most?

Example 5:
{"Topic": "Managing mood", "Original Response": "I cannot control my mood.", "Follow-up Response": "Apart from this, I also struggle with maintaining a regular sleep schedule. I often stay up late and feel tired the next day."}
Guide: It sounds like your sleep schedule has also been a challenge, and feeling tired the next day is understandably difficult. However, since we are focusing on mood management right now, could you elaborate on the moments when you feel you cannot control your mood? Do you notice a pattern connecting those moments with your sleep or with specific triggers during the day?
'''


def _payload(topic: str, original_question: str, original_response: str, follow_up_response: str) -> str:
    return (
        f'{{"Topic": {topic!r}, '
        f'"Original Question": {original_question!r}, '
        f'"Original Response": {original_response!r}, '
        f'"Follow-up Response": {follow_up_response!r}}}'
    )


def _extract_labelled(raw: str, *labels: str) -> str:
    """Return every character after the first matching label up to end-of-output.

    Validation and Guide responses are paragraphs (3-5 sentences); the old
    first-line-only parser truncated everything after the newline following
    the label. This captures the full block and strips whitespace.
    """
    if not raw:
        return ""
    text = raw.strip()
    lower = text.lower()
    for label in labels:
        lbl_lower = label.lower()
        idx = lower.find(lbl_lower)
        if idx != -1:
            tail = text[idx + len(lbl_lower):]
            # Drop an optional leading colon + whitespace: "Guide:" or "GUIDE: "
            if tail.startswith(":"):
                tail = tail[1:]
            return tail.strip()
    return text


def rv_reasoner(topic: str, original_question: str, original_response: str, follow_up_response: str) -> str:
    """Return '0' (related) or '1' (unrelated).

    Paper role: RV_REASONER (GPT-4 in paper; best validity judgement).

    Fail-closed default: if the LLM output is unparseable, return '1'
    (unrelated) so the caller runs the Guide path. Matches the symmetric
    default in `CBT._parse_decision` and the paper's intent: when in doubt,
    guide the user back to the topic rather than emit an OARS validation
    that might land on an off-topic reply.
    """
    payload = _payload(topic, original_question, original_response, follow_up_response)
    raw = llm_complete(RV_REASONER_SYSTEM_PROMPT, payload, role=LLMRole.RV_REASONER)
    for line in (raw or "").splitlines():
        line = line.strip()
        if line.upper().startswith("DECISION:"):
            token = line.split(":", 1)[1].strip()
            if token in ("0", "1"):
                return token
    logger.warning("R-V Reasoner returned no parseable DECISION; defaulting to 1 (unrelated, guide path).")
    return "1"


def rv_validator_mi(topic: str, original_question: str, original_response: str, follow_up_response: str) -> str:
    """Produce a paper-length empathic validation (3-5 sentences).

    Paper role: RV_VALIDATOR (GPT-3.5-Turbo in paper; therapists flagged
    GPT-4 as "reads into feelings" for empathic validation).
    """
    payload = _payload(topic, original_question, original_response, follow_up_response)
    raw = llm_complete(RV_VALIDATOR_OARS_SYSTEM_PROMPT, payload, role=LLMRole.RV_VALIDATOR)
    return _extract_labelled(raw, "VALIDATION")


def rv_guide(topic: str, original_question: str, original_response: str, follow_up_response: str) -> str:
    """Produce a paper-length redirect (2-4 sentences) when follow-up is off-topic.

    Paper role: RV_GUIDE (GPT-3.5-Turbo in paper; fewer "read-mind" drifts).
    """
    payload = _payload(topic, original_question, original_response, follow_up_response)
    raw = llm_complete(RV_GUIDE_SYSTEM_PROMPT, payload, role=LLMRole.RV_GUIDE)
    return _extract_labelled(raw, "GUIDE", "FOLLOW-UP")


def rv_consolidated(
    topic: str,
    original_question: str,
    original_response: str,
    follow_up_response: str,
) -> tuple[str, str, str]:
    """Split R-V pipeline matching paper/legacy: Reasoner -> Guide/Validator.

    Returns (decision_token, guide_text, validation_text):
      - decision_token '0': guide_text is empty, validation_text has OARS reflection.
      - decision_token '1': guide_text has redirect, validation_text is empty
        (caller must run Validation separately on the user's new response).
    """
    logger.info("Running split R-V pipeline (Reasoner -> Validator/Guide).")
    decision = rv_reasoner(topic, original_question, original_response, follow_up_response)
    if decision == "1":
        guide_text = rv_guide(topic, original_question, original_response, follow_up_response)
        # Phase B: record the MI guide-redirect event.
        _log_mi_intervention(
            "guide_redirect",
            detail={"topic": topic, "original_response": original_response,
                    "follow_up_response": follow_up_response,
                    "guide_text": guide_text},
            dim_label=str(topic),
        )
        return decision, guide_text, ""
    else:
        validation_text = rv_validator_mi(topic, original_question, original_response, follow_up_response)
        # Phase B: record the MI OARS simple-reflection event.
        _log_mi_intervention(
            "oars_validation",
            detail={"topic": topic, "original_response": original_response,
                    "follow_up_response": follow_up_response,
                    "validation_text": validation_text},
            dim_label=str(topic),
        )
        return decision, "", validation_text
