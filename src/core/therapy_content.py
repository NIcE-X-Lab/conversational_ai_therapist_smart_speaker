"""
Domain module for PHQ-4 and GAD-2 clinical metrics.
Questions 1-2 = GAD-2 (Anxiety)
Questions 3-4 = PHQ-2 (Depression)
Total 1-4     = PHQ-4 (Composite)
"""

# ── Unified PHQ-4 / GAD-2 clinical screening ──────────────────────────────────
CLINICAL_SCREENING = [
    {
        "id": "gad_1",
        "text": "Over the last 2 weeks, how often have you been bothered by feeling nervous, anxious, or on edge?",
        "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        "scale": "anxiety"
    },
    {
        "id": "gad_2",
        "text": "Over the last 2 weeks, how often have you been bothered by not being able to stop or control worrying?",
        "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        "scale": "anxiety"
    },
    {
        "id": "phq_1",
        "text": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
        "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        "scale": "depression"
    },
    {
        "id": "phq_2",
        "text": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
        "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        "scale": "depression"
    }
]

# Clinical alerting thresholds
GAD2_THRESHOLD = 3   # Anxiety sub-score >= 3 → GAD2_POSITIVE
PHQ4_THRESHOLD  = 6  # Total PHQ-4 score   >= 6 → PHQ4_HIGH_RISK

# Convenience slices
GAD2_QUESTIONS = [q for q in CLINICAL_SCREENING if q["scale"] == "anxiety"]
PHQ2_QUESTIONS = [q for q in CLINICAL_SCREENING if q["scale"] == "depression"]

# ── Meditation interstitials (cycling, used when all screening is done / refused) ──
MEDITATIONS = [
    (
        "While I'm processing that, let's try a quick breathing exercise together. "
        "Simple Breath Awareness. "
        "Sit comfortably with your spine relaxed but upright. Gently close your eyes or soften your gaze. "
        "Bring your attention to your breathing. Notice the air moving in through your nose… and out again. "
        "There is no need to change the breath. Simply observe it. "
        "Feel the inhale filling the body slightly… and the exhale releasing. "
        "If your mind wanders, gently guide your attention back to the next breath. "
        "Let each inhale arrive naturally. Let each exhale soften the body a little more. "
        "Continue resting your awareness on the rhythm of breathing."
    ),
    (
        "It's taking me a moment to reflect — let me invite you to try something with me. "
        "Counting the Breath. "
        "Sit comfortably and bring attention to your breathing. "
        "As you inhale, silently count one. As you exhale, count two. "
        "Inhale three… exhale four. "
        "Continue counting up to ten, then begin again at one. "
        "If the mind drifts or you lose track, simply return to one without judgment. "
        "Allow the counting to anchor your attention to the steady rhythm of breathing."
    ),
    (
        "I'm still thinking — let's use this moment well. "
        "Expanding Breath Through the Body. "
        "Sit or lie down comfortably. "
        "Take a slow breath in and notice the chest gently expand. Exhale and feel the body soften. "
        "Now imagine the breath spreading through the body. "
        "As you inhale, feel the breath reaching the ribs… the back… the belly. "
        "As you exhale, allow the shoulders and jaw to release any tension. "
        "Each breath expands awareness slightly through the body. Each exhale invites a sense of ease."
    ),
    (
        "Let's try this while I work through your response. "
        "Short Body Scan. "
        "Sit or lie down comfortably and bring attention to the body. "
        "Notice the sensation of your feet touching the floor or surface beneath you. "
        "Move your awareness slowly up to the legs… simply noticing any sensations. "
        "Bring attention to the belly and chest, noticing the gentle movement of breathing. "
        "Now notice the shoulders, letting them soften if they are holding tension. "
        "Finally, bring awareness to the face — jaw, cheeks, forehead — allowing them to relax. "
        "Rest for a few breaths, feeling the body as a whole."
    ),
    (
        "Give me just a moment — and while we wait, let's try a little awareness practice together. "
        "Breath at the Nostrils. "
        "Sit comfortably and bring attention to the tip of your nose. "
        "Notice the subtle sensation of air entering the nostrils as you inhale. It may feel slightly cool. "
        "As you exhale, notice the warmth of the air leaving the body. "
        "Allow your attention to stay with these small sensations of breathing. "
        "When the mind wanders, gently return to the feeling of the breath at the nostrils. "
        "Remain with this simple awareness for the next few breaths."
    ),
]

# ── Safe fallback asset paths ──────────────────────────────────────────────────
WAITING_MUSIC_PATH = "assets/waiting_music.wav"


def score_response(text: str) -> int:
    """
    Map a verbal Likert response to the PHQ-4 / GAD-2 integer scale (0–3).
    Accepts free-form speech like 'several days', 'almost every day', etc.
    Returns -1 if the user refuses or opts out.
    """
    t = text.lower().strip()

    # Opt-out / refusal
    if any(kw in t for kw in ("skip", "don't want", "opt out", "[opt_out]", "refuse", "pass")):
        return -1

    # Scale anchors (most-specific first)
    if any(k in t for k in ("nearly every day", "almost every day", "every day", "always", "3")):
        return 3
    if any(k in t for k in ("more than half", "most days", "most of the", "often", "frequently", "2")):
        return 2
    if any(k in t for k in ("several days", "several", "sometimes", "a few", "some days", "1")):
        return 1
    return 0   # "not at all", "never", "no", or unmatched → 0
