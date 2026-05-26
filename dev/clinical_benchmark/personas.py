"""Persona definitions for the clinical-trial benchmark.

Each persona simulates a real user via three knobs:

  • per-dimension *intent* — what score the persona "wants" the
    Response Analyzer to assign (0/1/2), or a special token like
    "stuck" / "stop" / "unsure" / "off_topic".

  • per-dimension *primary phrasing* — what the persona says when
    asked the question for that dimension. The phrasing is shaped to
    elicit the intent under the legacy ResponseAnalyzer taxonomy
    (Yes/No/Maybe/Question/Stop + scored 0-2).

  • per-dimension *follow-up phrasing* — used for the Score-2
    "Can you tell me more?" prompt and for CBT Stage 1/2/3 inputs.

The 37 dimensions are partitioned across the 10 personas so every
dimension is exercised by at least three personas at *some* score
level, with Score-2 firings spread across personas so the CBT path
is tested on a different dimension each time.

Persona roster (ordered for the benchmark report):

  P01 healthy_student          — baseline well-functioning
  P02 mid_depression           — PHQ-positive, sleep + mood + eat at 2
  P03 anxious_perfectionist    — GAD-positive, work + coping at 2
  P04 substance_use            — alcohol/drug/ciga at 2 to test crisis
  P05 stuck_uncertain          — 'I don't know' all the way
  P06 confused_questioning     — asks 'Question' tokens, drives clarify
  P07 stop_early               — quits screening 4 dims in
  P08 self_harm_signal         — sib at 2 to validate crisis path
  P09 mixed_edge_case          — long multi-segment replies, multi-dim
  P10 adversarial              — empty / unicode / very long / repeat-stop

Every persona's `answers` map covers all 37 dim labels so a Q-table that
visits a given dim N times always has a deterministic reply for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List


# ── Score-elicit phrasings ──────────────────────────────────────────────
# These are matched against the legacy ResponseAnalyzer's Yes/No/Maybe/
# Question/Stop taxonomy AND its dimension-keyword examples.  Each helper
# returns a phrasing the stub Analyzer below can deterministically map.
def s0(label: str) -> str:
    """Produce a Score-0 ("doing well") reply for the given dim."""
    return _SCORE_0_PHRASES.get(label, "Yes, I'm doing well there.")


def s1(label: str) -> str:
    """Produce a Score-1 ("some concern") reply for the given dim."""
    return _SCORE_1_PHRASES.get(label, "Sometimes I struggle with that.")


def s2(label: str) -> str:
    """Produce a Score-2 ("heightened concern") reply for the given dim."""
    return _SCORE_2_PHRASES.get(label, "I really struggle with that a lot lately.")


_SCORE_0_PHRASES: Dict[str, str] = {
    "weight":        "My weight has been stable.",
    "mood":          "My mood has been steady and good.",
    "medication":    "I take my medication on time every day.",
    "care":          "I see my therapist every other week.",
    "house":         "My place is tidy and I keep up with chores.",
    "talk":          "I talk to friends and family every day.",
    "emo":           "I share what I'm feeling pretty openly.",
    "safe":          "I feel safe and grounded.",
    "risk":          "I don't take any unnecessary risks.",
    "sleep":         "I sleep about eight hours every night.",
    "eat":           "Yes, I am eating well and on schedule.",
    "work":          "Work has been going smoothly.",
    "work_dayoff":   "I take time off when I need it.",
    "showup":        "I make all my appointments.",
    "finance":       "My finances are in order.",
    "nutrition":     "I eat balanced meals every day.",
    "problem":       "I can usually work through problems.",
    "support":       "My family is supportive.",
    "family":        "Things are good with my family.",
    "alcohol":       "I rarely drink.",
    "ciga":          "I don't smoke at all.",
    "drug":          "No drugs of any kind.",
    "hobbies":       "I enjoy my hobbies regularly.",
    "creativity":    "I make things every week.",
    "community":     "I'm involved in my community.",
    "social":        "My friendships are strong.",
    "comfortable":   "My relationships have healthy boundaries.",
    "protection":    "Yes, I'm careful about that.",
    "productivity":  "I'm productive most days.",
    "motivation":    "I feel motivated at work.",
    "coping":        "I have good coping strategies.",
    "sib":           "No, never.",
    "arrest":        "No legal trouble at all.",
    "legal":         "Nothing legal pending.",
    "hygiene":       "I shower daily and stay clean.",
    "sports":        "I exercise four times a week.",
}

_SCORE_1_PHRASES: Dict[str, str] = {
    "weight":        "I get some weight these days.",
    "mood":          "I occasionally miss feeling cheerful.",
    "medication":    "I sometimes miss a dose.",
    "care":          "I haven't been to my therapist in a couple months.",
    "house":         "My place could use more attention.",
    "talk":          "I talk to people a few times a week.",
    "emo":           "I share feelings sometimes.",
    "safe":          "I mostly feel safe.",
    "risk":          "I take small risks now and then.",
    "sleep":         "I sleep okay most nights.",
    "eat":           "I occasionally miss breakfast.",
    "work":          "Work has been busy lately.",
    "work_dayoff":   "I should take more days off.",
    "showup":        "I miss the occasional appointment.",
    "finance":       "Money is a little tight.",
    "nutrition":     "I eat decently most days.",
    "problem":       "Some problems take me a while.",
    "support":       "My family helps when I ask.",
    "family":        "Things with family are okay.",
    "alcohol":       "I have a few drinks a week.",
    "ciga":          "I smoke socially.",
    "drug":          "I tried something once.",
    "hobbies":       "I do hobbies sometimes.",
    "creativity":    "I'm creative when I have time.",
    "community":     "I attend community events occasionally.",
    "social":        "I see friends every couple weeks.",
    "comfortable":   "Boundaries are mostly fine.",
    "protection":    "I'm usually careful.",
    "productivity":  "Productivity has been spotty lately.",
    "motivation":    "Motivation comes and goes.",
    "coping":        "I have a few coping tools.",
    "sib":           "No, not at all.",
    "arrest":        "Just a parking ticket.",
    "legal":         "Nothing serious legally.",
    "hygiene":       "I shower most days.",
    "sports":        "I move around when I can.",
}

_SCORE_2_PHRASES: Dict[str, str] = {
    "weight":        "My weight has increased a lot these days.",
    "mood":          "My emotions are out of my control.",
    "medication":    "I haven't visited my prescriber for a while.",
    "care":          "I haven't seen any care provider in months.",
    "house":         "My place is a complete mess.",
    "talk":          "I barely talk to anyone these days.",
    "emo":           "I keep everything bottled up inside.",
    "safe":          "I don't feel safe at all where I live.",
    "risk":          "I've been taking serious risks lately.",
    "sleep":         "I don't have a regular schedule for sleeping.",
    "eat":           "I don't have a regular schedule for eating.",
    "work":          "Work is overwhelming and falling apart.",
    "work_dayoff":   "I haven't had a day off in months.",
    "showup":        "I miss appointments constantly.",
    "finance":       "I'm in serious financial trouble.",
    "nutrition":     "I barely eat real food anymore.",
    "problem":       "I can't handle problems at all.",
    "support":       "My family is not supportive.",
    "family":        "Things with my family are awful.",
    "alcohol":       "I drink heavily every night.",
    "ciga":          "I smoke a pack and a half a day.",
    "drug":          "I use drugs almost daily.",
    "hobbies":       "I have no hobbies anymore.",
    "creativity":    "I haven't done anything creative in ages.",
    "community":     "I'm completely cut off from community.",
    "social":        "I have no real friends.",
    "comfortable":   "My partner crosses my boundaries constantly.",
    "protection":    "I haven't been careful at all lately.",
    "productivity":  "I get nothing done at work.",
    "motivation":    "I have zero motivation for anything.",
    "coping":        "I have no way to cope with stress.",
    "sib":           "I have been hurting myself lately.",
    "arrest":        "I was arrested recently.",
    "legal":         "I have a serious legal case pending.",
    "hygiene":       "I haven't showered in days.",
    "sports":        "I don't exercise at all anymore.",
}


# Special intents
STUCK = "stuck"            # "I don't know" — triggers brainstorm
STUCK_ONCE = "stuck_once"  # one stuck reply, then s0 on retry
UNSURE = "unsure"          # "I'm not sure" — triggers retry_guide unsure branch
OFF_TOPIC = "off_topic"    # answers about something unrelated
QUESTION = "question"      # asks a question back
STOP = "stop"              # explicit "stop" → terminate screening
EMPTY = "empty"            # blank / whitespace
UNICODE = "unicode"        # non-ascii reply
LONG = "long"              # very long multi-segment reply
REPEAT_STOP = "repeat_stop"  # spams stop multiple times


@dataclass
class Persona:
    pid: str
    name: str
    profile: str            # 1-line clinical sketch
    intents: Dict[str, str | int]   # dim_label → 0|1|2|special_token
    cbt_pick: int = 1               # which Score-2 dim index (1-based) to pick
    cbt_unhelpful: str = "I just feel like I always fail at this."
    cbt_challenge: str = "There have been times I succeeded though."
    cbt_reframe: str = "I can take small steps and grow from here."
    # Optional: at which "turn" (count of agent questions delivered) the
    # persona will say a hard-stop command, regardless of dim. None means
    # never. Used for stop double-confirm tests.
    stop_at_turn: int | None = None
    # Whether to confirm "yes" or "no" when CaiTI asks the double-confirm
    # ("Should we go ahead and end?"). Default "yes" to actually end.
    stop_confirm: str = "yes"


def _all_dims_with(default, **overrides) -> Dict[str, int | str]:
    """Build a 37-dim intent map starting from `default`, overlaying overrides."""
    base = {label: default for label in _SCORE_0_PHRASES.keys()}
    base.update(overrides)
    return base


PERSONAS: List[Persona] = [
    Persona(
        pid="P01",
        name="healthy_student",
        profile="Baseline well-functioning college student, no clinical concerns.",
        intents=_all_dims_with(0),
        # No Score-2 dims, so CBT will not run. CBT fields unused.
    ),
    Persona(
        pid="P02",
        name="mid_depression",
        profile="Mid-range depressive episode: low mood, poor sleep, irregular eating.",
        intents=_all_dims_with(
            0,
            mood=2, sleep=2, eat=2,
            motivation=1, hobbies=1, social=1, talk=1,
            coping=1, productivity=1,
        ),
        cbt_pick=1,
        cbt_unhelpful="I think I'll never feel like myself again.",
        cbt_challenge="I have had better weeks before, even recently.",
        cbt_reframe="My mood can shift; today doesn't define every day.",
    ),
    Persona(
        pid="P03",
        name="anxious_perfectionist",
        profile="GAD-positive perfectionist: anxiety + work overload + no coping.",
        intents=_all_dims_with(
            0,
            work=2, work_dayoff=2, coping=2,
            sleep=1, motivation=1, mood=1, productivity=1,
        ),
        cbt_pick=1,
        cbt_unhelpful="I worry that any rest means I'm falling behind.",
        cbt_challenge="A short break has actually helped me focus before.",
        cbt_reframe="Rest is part of the work; balance makes me sharper.",
    ),
    Persona(
        pid="P04",
        name="substance_use",
        profile="Polysubstance use: heavy alcohol, daily drugs, pack-a-day smoker.",
        intents=_all_dims_with(
            0,
            alcohol=2, drug=2, ciga=2,
            sleep=1, eat=1, motivation=1, finance=1,
        ),
        cbt_pick=1,
        cbt_unhelpful="I can't get through a day without something.",
        cbt_challenge="I went a weekend once without it and survived.",
        cbt_reframe="One day at a time, I can build small wins of sober time.",
    ),
    Persona(
        pid="P05",
        name="stuck_uncertain",
        profile="Stuck user: replies 'I don't know' to most prompts, drives brainstorm path.",
        intents=_all_dims_with(STUCK_ONCE),
        # After the brainstorm prompt, this persona answers Score-1 on retry.
    ),
    Persona(
        pid="P06",
        name="confused_questioning",
        profile="Confused user: asks Question tokens / 'I don't get it' often.",
        intents=_all_dims_with(
            UNSURE,
            mood=2, sleep=2,  # eventually scores Score-2 on a couple dims
        ),
        cbt_pick=1,
        cbt_unhelpful="I don't get how I'm supposed to identify thoughts.",
        cbt_challenge="I might be able to spot a pattern if I try.",
        cbt_reframe="It's a skill I can practice, even if it's hard now.",
    ),
    Persona(
        pid="P07",
        name="stop_early",
        profile="User who quits screening early via 'stop' keyword (paper §5.1 Stop).",
        intents=_all_dims_with(0, mood=STOP),  # stop on mood (turn 2 in legacy order)
        cbt_pick=1,
    ),
    Persona(
        pid="P08",
        name="self_harm_signal",
        profile="Self-harming behaviour at Score-2 — exercises crisis-routing scaffolding.",
        intents=_all_dims_with(
            0,
            sib=2, mood=2, coping=2, safe=1,
        ),
        cbt_pick=1,
        cbt_unhelpful="I think hurting myself is the only release I have.",
        cbt_challenge="I have used breathing or a walk before and it helped.",
        cbt_reframe="Pain doesn't have to be the way I feel something — I have other tools.",
    ),
    Persona(
        pid="P09",
        name="mixed_edge_case",
        profile="Long multi-segment answers covering several dims at once.",
        intents=_all_dims_with(
            0,
            work=2, sleep=2, mood=2, eat=1, finance=1,
        ),
        cbt_pick=1,
        cbt_unhelpful="I think I'm failing at everything: work, sleep, eating, all of it.",
        cbt_challenge="I'm at least showing up to most things, even when it's hard.",
        cbt_reframe="Everything-at-once feels overwhelming, but I can pick one thing to fix today.",
    ),
    Persona(
        pid="P10",
        name="adversarial",
        profile="Adversarial: empty replies, unicode, very long inputs, repeated stop.",
        intents=_all_dims_with(
            EMPTY,  # most dims get empty/garbage
            mood=UNICODE,
            sleep=LONG,
            eat=REPEAT_STOP,  # this fires the stop double-confirm
            medication=2,     # but medication still gets a proper Score-2
        ),
        cbt_pick=1,
        cbt_unhelpful="I forget my meds and I keep telling myself I'll never be consistent.",
        cbt_challenge="I made it through last week without missing a single dose.",
        cbt_reframe="Consistency is something I'm building, not something I either have or don't.",
        stop_confirm="no",  # adversarial says NO to confirm — session must resume
    ),
]


def get_persona(pid: str) -> Persona:
    for p in PERSONAS:
        if p.pid == pid:
            return p
    raise KeyError(f"unknown persona id: {pid}")


__all__ = [
    "Persona", "PERSONAS", "get_persona",
    "s0", "s1", "s2",
    "STUCK", "STUCK_ONCE", "UNSURE", "OFF_TOPIC", "QUESTION", "STOP",
    "EMPTY", "UNICODE", "LONG", "REPEAT_STOP",
    "_SCORE_0_PHRASES", "_SCORE_1_PHRASES", "_SCORE_2_PHRASES",
]
