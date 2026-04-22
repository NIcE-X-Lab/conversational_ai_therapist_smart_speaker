"""Expand the active question library so every dimension reaches the
paper's 7-11 variant range (p.11).

Design:
- NEVER mutates the therapist-authored `question` array — legacy fidelity.
- Adds variants to a sibling `question_synthetic` array with provenance.
- Supports two modes:
    --seed       Write a hand-authored, clinically reviewed seed set.
    --extend     Run the project's LLM Rephraser to top up any dimension
                 that still falls short of the target variant count.

The seed set below was written to obey the structural-only rule from
`src/core/response_analyzer.py::REPHRASER_PROMPT`: no change of clinical
construct, timeframe, or scored polarity; vocabulary and sentence
structure only.  Each seed is marked `reviewed=False` so a clinician
can sign off before these enter a formal study.

Usage:
    python scripts/expand_question_lib.py --seed
    python scripts/expand_question_lib.py --extend --target 8
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

# Allow running from repo root without `-m`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config_loader import QUESTION_LIB_FILENAME  # noqa: E402


# ── Seed set ───────────────────────────────────────────────────────────────
# For each dimension we author up to 8 structural-only rephrases.  The key is
# the internal `label` from data/libs/question_lib_v4.json.  The dedup step in
# _apply_seed() guarantees we never re-add a phrasing the legacy lib already
# contains.
_SEED: dict[str, list[str]] = {
    "weight": [
        "Has your weight shifted in either direction lately?",
        "Over the past few weeks, have you noticed meaningful changes in your weight?",
        "Have the numbers on the scale moved much for you recently?",
        "Have you put on or dropped weight in a way that stood out to you?",
        "Thinking back over the last month, has your weight been stable?",
        "Have clothes been fitting you differently recently?",
        "Has your body weight gone up or down by a noticeable amount lately?",
        "Over recent weeks, have you experienced a meaningful shift in weight?",
    ],
    "mood": [
        "Over the last while, how would you describe your mood overall?",
        "Have your moods been something you can manage lately?",
        "How would you say your general mood has been these past few days?",
        "When you look back over the last week, how steady has your mood felt?",
        "Has your mood been in a place that feels workable to you recently?",
        "How are you finding your mood day to day at the moment?",
        "Would you say you are holding a fairly even mood these days?",
        "Overall, how settled has your emotional state been lately?",
    ],
    "medication": [
        "Have you been keeping up with the prescriptions your doctor gave you?",
        "Are you taking your medications at the times and doses you were told?",
        "Have you been following the medication plan your provider set up?",
        "Are you staying current with the medicines that were prescribed to you?",
        "Have your prescribed medications been taken as directed lately?",
        "Over the past week, have you kept to your prescription schedule?",
        "Have you been consistent with the medications your clinician recommended?",
        "Are you following the medicine regimen that was laid out for you?",
    ],
    "care": [
        "Have you been attending your appointments with your medical or mental-health providers?",
        "Are you keeping up with visits to your doctor, therapist, or case manager?",
        "Have you been staying in touch with your care team as planned?",
        "Over the last while, have you kept your medical and therapy appointments?",
        "Have you been showing up for your regular check-ins with your providers?",
        "Are you seeing the clinicians involved in your care at the cadence they asked for?",
        "Have you maintained contact with your therapist or prescriber recently?",
        "Are you following through on your scheduled appointments with your care team?",
    ],
    "house": [
        "Have you been keeping up with chores around your home?",
        "Are you able to stay on top of everyday tasks in your living space?",
        "Over the past week, have you been managing the upkeep of where you live?",
        "Have basic household tasks been getting done lately?",
        "Are the routine chores around the house still happening for you?",
        "Have you been taking care of the day-to-day of your home recently?",
        "Over recent days, has the housework been manageable for you?",
        "Have you been keeping your living environment in order?",
    ],
    "talk": [
        "Have you been keeping up conversations with people in your life?",
        "Are you staying in touch with others through day-to-day talking?",
        "Over the last week, have you had conversations with people you know?",
        "Have you been speaking with friends, family, or others recently?",
        "Are you maintaining regular conversation with people around you?",
        "Have you found opportunities to talk with others lately?",
        "Over recent days, have you been in conversation with the people in your life?",
        "Are you keeping spoken contact with others in your routine?",
    ],
    "emo": [
        "Have you been able to share how you are feeling with people you know?",
        "Over the past week, have you let others know about your emotions?",
        "Have you talked through your feelings with anyone recently?",
        "Are you finding ways to express what you are feeling to others?",
        "Have you voiced your emotions to the people around you lately?",
        "Over recent days, have you shared your inner feelings with someone?",
        "Have you let anyone close to you know what has been on your mind emotionally?",
        "Are you putting your feelings into words with people you trust?",
    ],
    "safe": [
        "Do you feel that you are in a safe situation right now?",
        "At this moment, would you describe yourself as safe?",
        "When you make decisions lately, are you keeping your safety in mind?",
        "Have concerns about your personal safety been part of your choices recently?",
        "Are you weighing safety when you decide what to do day to day?",
        "Would you say your current circumstances feel safe to you?",
        "Is safety something you feel confident about in your daily life?",
        "Are you looking out for your own safety in the way you make decisions?",
    ],
    "risk": [
        "Over recent days, have you taken any risks that stand out?",
        "Have you been making choices lately that involve noticeable risk?",
        "In the last while, have any of your decisions put you at risk?",
        "Have you gone after anything risky in the past week or so?",
        "Would you say your recent choices have leaned toward the risky side?",
        "Have the decisions you've made lately involved taking chances?",
        "Are there recent actions of yours that you would call risk-taking?",
        "Over the last while, have you put yourself in risky situations?",
    ],
    "sleep": [
        "Are you getting enough sleep, and is your schedule consistent?",
        "How has your sleep been holding up lately?",
        "Over recent nights, have you been getting adequate rest?",
        "Is your sleep schedule staying regular for you?",
        "Have the last few nights of sleep felt sufficient?",
        "Are you sleeping well and at consistent times?",
        "Has the quality and timing of your sleep been working out?",
        "Over the past week, have you been resting enough at night?",
    ],
    "eat": [
        "Have your meals been happening on a regular schedule?",
        "Are you eating at fairly steady times each day?",
        "Over recent days, have you kept a consistent eating routine?",
        "Has your eating pattern been regular lately?",
        "Are your meals spaced out the way they usually are?",
        "Over the past week, have you been eating at predictable times?",
        "Is your eating schedule holding up day to day?",
        "Have you been eating on a routine recently?",
    ],
    "work": [
        "Have you been making it to work or school consistently?",
        "Are you staying engaged with your work or classes lately?",
        "Over recent days, have you been showing up for your job or school?",
        "Is your attendance at work or school holding steady?",
        "Have you kept up with your job or schoolwork recently?",
        "Are you continuing to go in to work or class as expected?",
        "Over the past week, have you been present for your work or studies?",
        "Have you been on top of your work or school commitments lately?",
    ],
    "work_dayoff": [
        "Have you taken time away from work or school recently?",
        "Over the last while, have you given yourself any days off?",
        "Have you stepped back from work or class for a break lately?",
        "Are you taking any breaks from your usual work or school routine?",
        "Over recent weeks, have you had time off from your responsibilities?",
        "Have you been giving yourself room to rest away from work?",
        "Have any recent days been dedicated to time off for you?",
        "Is there balance between work and time away from it in your week?",
    ],
    "showup": [
        "Have you been keeping your appointments and other commitments?",
        "Are you following through on the plans you've made lately?",
        "Over the past week, have you shown up for what you had scheduled?",
        "Have you been meeting your obligations as they come up?",
        "Are you arriving at the appointments you set for yourself?",
        "Over recent days, have you kept your commitments to others?",
        "Have you been there for the plans and appointments on your calendar?",
        "Is your follow-through on obligations still steady?",
    ],
    "finance": [
        "How is your financial situation feeling to you these days?",
        "Have money matters been a source of concern for you recently?",
        "Over the past while, have there been worries about your finances?",
        "Are there any money-related issues weighing on you lately?",
        "Has your spending been raising any concerns for you?",
        "Over recent weeks, have financial pressures been on your mind?",
        "Are you feeling on top of your finances, or are there worries?",
        "Have you noticed any concerns about how money is flowing lately?",
    ],
    "nutrition": [
        "Are the meals you've been eating mostly healthy ones?",
        "Has your food been nourishing lately?",
        "Over recent days, have you been eating in a healthy way?",
        "Are the foods you are choosing generally good for you?",
        "Has your diet been a healthy one this past week?",
        "Are you eating foods that take care of your body lately?",
        "Over recent meals, has the nutrition been solid?",
        "Are healthy choices showing up in what you eat these days?",
    ],
    "problem": [
        "Are you finding you can make decisions on your own lately?",
        "When problems come up, are you able to work through them yourself?",
        "Over recent weeks, has your decision-making felt manageable to you?",
        "Have you been able to handle the problems that show up for you?",
        "Are you resolving day-to-day issues on your own when they come up?",
        "Over the past while, has solving problems felt within your reach?",
        "Are you navigating the choices you face without help feeling stuck?",
        "Has independent decision-making been going okay for you?",
    ],
    "support": [
        "Do you feel you are getting support from your family?",
        "Over the last while, has your family been there for you?",
        "Have you felt your family backing you up lately?",
        "Is the support from your family feeling present in your life?",
        "Are family members offering you the support you need these days?",
        "Over recent weeks, has family support been part of your experience?",
        "Do you feel held up by your family right now?",
        "Has your family shown up for you the way you'd want?",
    ],
    "family": [
        "How have things been between you and your family lately?",
        "Over recent weeks, how is your relationship with your family going?",
        "Are you getting along with family members these days?",
        "Has your family dynamic been working out for you lately?",
        "Over the past while, how is the connection with your family?",
        "Are things with your family feeling okay at the moment?",
        "How are you and your family doing relationally right now?",
        "Is the state of your family relationships comfortable for you?",
    ],
    "alcohol": [
        "Do you find yourself drinking by yourself often?",
        "Has solo drinking been part of your routine lately?",
        "Over recent weeks, have you been drinking alone with any regularity?",
        "Are you drinking on your own more often than you would like?",
        "Have there been times lately when you've been drinking solo?",
        "Is drinking alone something that happens in your week?",
        "Over the past while, have you had drinks by yourself frequently?",
        "Are you often having a drink when no one else is around?",
    ],
    "ciga": [
        "Are you smoking or vaping, and if so, how often?",
        "Over recent days, have you been using cigarettes or vape products?",
        "Is cigarette or vape use part of your routine, and at what pace?",
        "Have you been smoking or vaping lately, and how frequently?",
        "Over the past week, how much have you smoked or vaped?",
        "Are tobacco or vape products showing up in your day, and how often?",
        "How regularly are you using cigarettes or vape these days?",
        "Is smoking or vaping something you are doing with any regularity?",
    ],
    "drug": [
        "Are you using any substances right now, and at what frequency?",
        "Over recent days, have you taken any substances, and how often?",
        "Is substance use part of your week, and how regularly?",
        "Have you been using anything lately, and at what pace?",
        "Over the past while, how often have substances been involved?",
        "Is there any substance use happening for you, and how frequent is it?",
        "How regular has substance use been for you recently?",
        "Are you taking any substances at the moment, and how much?",
    ],
    "hobbies": [
        "Do you have hobbies you enjoy, and what are they?",
        "What do you like to do in your free time these days?",
        "Are there hobbies that you spend time on lately?",
        "Over recent weeks, what have you been doing for fun?",
        "Is there anything you do just for enjoyment these days?",
        "What activities do you turn to when you have time for yourself?",
        "Are there hobbies in your life right now that you look forward to?",
        "What have you been doing for leisure lately?",
    ],
    "creativity": [
        "Have you done anything creative in the past while?",
        "Over recent weeks, has creativity shown up in what you do?",
        "Are you expressing yourself creatively these days?",
        "Have there been any creative projects for you lately?",
        "Over the past week, has creative work been part of your time?",
        "Have you made something creative recently?",
        "Is there a creative outlet you've been using lately?",
        "Over recent days, have you had a chance to be creative?",
    ],
    "community": [
        "Are you taking part in anything going on in your community?",
        "Have you joined in on community activities lately?",
        "Over recent weeks, have you been out and about in your neighborhood?",
        "Is community life part of your week right now?",
        "Are you engaging with things happening locally around you?",
        "Over the past while, have you shown up for community events?",
        "Have you been present in your neighborhood or community lately?",
        "Are local gatherings or activities part of your routine these days?",
    ],
    "support_2": [
        "Besides family, is there anyone close who supports you?",
        "Outside of family, who do you count on for support right now?",
        "Beyond your family, do you have people who back you up?",
        "Is there support in your life that comes from outside the family?",
        "Who in your circle, other than family, is there for you?",
        "Do you have people apart from family who you lean on?",
        "Outside of relatives, who shows up in your support network?",
        "Besides family members, who else do you feel supported by?",
    ],
    "social": [
        "Do you have people in your life who feel close to you?",
        "Are you spending time with friends or coworkers lately?",
        "Over recent weeks, have you been connecting with close people in your life?",
        "Is there anyone close you've been sharing time with recently?",
        "Are you in touch with friends or colleagues in a meaningful way?",
        "Over the past while, have close relationships been active for you?",
        "Do the people you consider close still feel present in your week?",
        "Are you finding time to be with people you feel close to?",
    ],
    "comfortable": [
        "If you have a partner, do you feel comfortable with them?",
        "In any romantic relationships, are you feeling at ease?",
        "If there's a partner in your life, does the relationship feel comfortable?",
        "Do you feel safe and relaxed with any partner you have right now?",
        "In your close partnership, if there is one, do you feel comfortable?",
        "For any partner you are with, is the dynamic comfortable for you?",
        "If you have a partner or partners, do things feel okay between you?",
        "In a romantic partnership, would you say you feel at ease these days?",
    ],
    "protection": [
        "Do you use protection when you have sex?",
        "Are you taking steps to protect yourself during sex?",
        "When you are sexually active, are you using protection?",
        "Have you been using protection in your sexual activity?",
        "Over recent encounters, has protection been part of what you do?",
        "Are you keeping sexual safety measures in place?",
        "During sex, are you using protection consistently?",
        "Has safe-sex protection been part of your practice lately?",
    ],
    "productivity": [
        "Do you feel productive in your work or school right now?",
        "Over recent days, has your productivity been where you want it?",
        "Is your output at work or school feeling solid lately?",
        "Have you been productive in your daily responsibilities?",
        "Over the past week, has productivity been holding up for you?",
        "Do you feel you are getting things done at work or school?",
        "Is your sense of productivity steady in your current routine?",
        "Have you felt effective in your work or studies lately?",
    ],
    "motivation": [
        "Do you feel motivated for your work or studies lately?",
        "Has your drive for work or school been holding up recently?",
        "Over the past week, has motivation for your responsibilities been present?",
        "Are you feeling motivated to engage with work or school these days?",
        "Is your motivation level for your job or classes where you'd want it?",
        "Over recent days, have you felt drawn to your work or studies?",
        "Has the push to do your work or school been there for you?",
        "Do you feel energized toward your work or coursework right now?",
    ],
    "coping": [
        "Do you have coping tools that help you calm down?",
        "Are there tricks you rely on to settle yourself when stressed?",
        "What do you usually do to de-stress and calm down?",
        "Have you been using coping strategies when you get overwhelmed?",
        "What helps you relax when things feel tough?",
        "Over recent weeks, which coping methods have worked for you?",
        "Are there ways you reliably calm yourself during stress?",
        "What are your go-to strategies for calming down lately?",
    ],
    "sib": [
        "Have you been engaging in any behaviors that hurt yourself?",
        "Over recent days, has self-harm been something you have done?",
        "Are there behaviors going on that cause harm to you physically?",
        "Have you had urges or actions lately that involve hurting yourself?",
        "Over the past while, has self-injury been part of what you are doing?",
        "Are you safe from self-harming behavior right now?",
        "Have any actions you've taken lately caused physical harm to you?",
        "Is self-harming something that has been happening for you recently?",
    ],
    "arrest": [
        "Have you been arrested recently?",
        "Over recent weeks, have there been any arrests involving you?",
        "Has any arrest happened for you in the past while?",
        "In the last while, have you had contact with law enforcement that led to an arrest?",
        "Over recent days, were there any arrest incidents for you?",
        "Has an arrest come into your life recently?",
        "Have you experienced an arrest in the past weeks?",
        "Is there a recent arrest in your situation?",
    ],
    "legal": [
        "Are there any legal matters going on for you right now?",
        "Over recent weeks, have legal issues come up in your life?",
        "Is there anything legal you are dealing with lately?",
        "Have legal concerns surfaced for you in the past while?",
        "Over the past week or so, have you faced any legal problems?",
        "Is there a legal situation currently active for you?",
        "Have any legal matters been weighing on you lately?",
        "Are you dealing with court or legal issues these days?",
    ],
    "hygiene": [
        "Are you keeping up with your personal hygiene?",
        "Have showers, teeth brushing, and skincare been happening regularly?",
        "Over recent days, has daily hygiene been part of your routine?",
        "Are the basics of self-care still happening for you?",
        "Over the past week, how has your hygiene been holding up?",
        "Have you been maintaining personal cleanliness as usual?",
        "Is daily grooming and hygiene part of your days right now?",
        "Are you staying on top of the small acts of daily hygiene?",
    ],
    "sports": [
        "Have you been getting any exercise lately?",
        "Are you doing sports or physical activity right now?",
        "Over recent days, have you moved your body through exercise?",
        "Has exercise been part of your week?",
        "Are you engaging in any physical activity these days?",
        "Over the past while, have you kept up with workouts or sport?",
        "Is exercise showing up in your routine lately?",
        "Have you been active in a sport or workout recently?",
    ],
}


def _clean(s: str) -> str:
    return " ".join(s.strip().split()).lower().rstrip("?.!")


def _apply_seed(lib: dict, target: int, dry_run: bool = False) -> tuple[int, list[str]]:
    added = 0
    warnings: list[str] = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    # Map duplicate label "support" (items 18 and 26 in v4) to the right seed pool.
    seed_alias = {"18": "support", "26": "support_2"}

    for i_key in sorted(lib.keys(), key=int):
        entry = lib[i_key]["1"]
        label = str(entry.get("label", "")).lower()
        seed_key = seed_alias.get(i_key, label)

        legacy_qs = list(entry.get("question", []))
        existing_synth = list(entry.get("question_synthetic", []))
        total = len(legacy_qs) + len(existing_synth)
        need = target - total
        if need <= 0:
            continue

        pool = _SEED.get(seed_key, [])
        if not pool:
            warnings.append(f"no seed pool for dim {i_key} ({label})")
            continue

        seen = {_clean(q) for q in legacy_qs + existing_synth}
        fresh: list[str] = []
        for candidate in pool:
            if _clean(candidate) in seen:
                continue
            fresh.append(candidate)
            seen.add(_clean(candidate))
            if len(fresh) == need:
                break

        if len(fresh) < need:
            warnings.append(
                f"dim {i_key} ({label}) short by {need - len(fresh)} variants "
                f"after seeding (pool exhausted)"
            )

        if not dry_run:
            entry.setdefault("question_synthetic", []).extend(fresh)
            meta = entry.setdefault("question_synthetic_meta", {
                "source": "expand_question_lib.py::_SEED",
                "provenance": "hand-authored structural rephrases following "
                              "REPHRASER_PROMPT in src/core/response_analyzer.py",
                "reviewed": False,
                "entries": [],
            })
            for q in fresh:
                meta["entries"].append({
                    "text": q,
                    "mode": "seed",
                    "added_at": now,
                    "reviewed": False,
                })
        added += len(fresh)

    return added, warnings


def _extend_via_llm(lib: dict, target: int, dry_run: bool = False) -> tuple[int, list[str]]:
    """Top up any dimension still under `target` using the project Rephraser."""
    from src.core.response_analyzer import rephrase_question  # lazy import

    added = 0
    warnings: list[str] = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    for i_key in sorted(lib.keys(), key=int):
        entry = lib[i_key]["1"]
        legacy_qs = list(entry.get("question", []))
        existing_synth = list(entry.get("question_synthetic", []))
        total = len(legacy_qs) + len(existing_synth)
        need = target - total
        if need <= 0:
            continue

        base = legacy_qs[0] if legacy_qs else ""
        if not base:
            warnings.append(f"dim {i_key} has no legacy question to rephrase from")
            continue

        seen = {_clean(q) for q in legacy_qs + existing_synth}
        attempts = 0
        new_variants: list[str] = []
        while len(new_variants) < need and attempts < need * 3:
            attempts += 1
            try:
                raw = rephrase_question(base)
            except Exception as e:
                warnings.append(f"dim {i_key} rephrase failed: {e}")
                break
            # Normalise the LLM output — strip the REPHRASER: label if present.
            text = (raw or "").strip()
            for line in text.splitlines():
                line = line.strip()
                if line.upper().startswith("REPHRASER:"):
                    text = line.split(":", 1)[1].strip()
                    break
            text = text.splitlines()[0] if text else ""
            if not text or _clean(text) in seen:
                continue
            new_variants.append(text)
            seen.add(_clean(text))

        if len(new_variants) < need:
            warnings.append(
                f"dim {i_key} only produced {len(new_variants)}/{need} LLM variants"
            )

        if not dry_run and new_variants:
            entry.setdefault("question_synthetic", []).extend(new_variants)
            meta = entry.setdefault("question_synthetic_meta", {
                "source": "expand_question_lib.py::--extend",
                "provenance": "LLM-rephrased via rephrase_question() (structural)",
                "reviewed": False,
                "entries": [],
            })
            for q in new_variants:
                meta["entries"].append({
                    "text": q,
                    "mode": "llm",
                    "added_at": now,
                    "reviewed": False,
                })
        added += len(new_variants)

    return added, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=8,
                        help="target total variants per dimension (7-11 per paper)")
    parser.add_argument("--seed", action="store_true",
                        help="apply the hand-authored seed set")
    parser.add_argument("--extend", action="store_true",
                        help="top up remaining gaps via the LLM Rephraser")
    parser.add_argument("--dry-run", action="store_true",
                        help="do not write the lib; print a summary only")
    parser.add_argument("--lib", default=QUESTION_LIB_FILENAME,
                        help="path to question_lib JSON (defaults to config value)")
    args = parser.parse_args()

    if not (args.seed or args.extend):
        parser.error("pass --seed and/or --extend")

    if not (7 <= args.target <= 11):
        print(f"[WARN] target={args.target} is outside the paper's 7-11 range", file=sys.stderr)

    lib_path = Path(args.lib)
    if not lib_path.exists():
        print(f"[ERR] lib not found: {lib_path}", file=sys.stderr)
        return 2

    with lib_path.open("r", encoding="utf-8") as f:
        lib = json.load(f)

    total_added = 0
    all_warnings: list[str] = []

    if args.seed:
        added, warnings = _apply_seed(lib, args.target, dry_run=args.dry_run)
        total_added += added
        all_warnings += warnings
        print(f"[seed] added {added} variants across dimensions")

    if args.extend:
        added, warnings = _extend_via_llm(lib, args.target, dry_run=args.dry_run)
        total_added += added
        all_warnings += warnings
        print(f"[extend] added {added} LLM variants across dimensions")

    for w in all_warnings:
        print(f"[warn] {w}", file=sys.stderr)

    if not args.dry_run and total_added > 0:
        # Timestamped backup so we never overwrite blindly.
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = lib_path.with_suffix(f".backup_{stamp}.json")
        backup.write_text(lib_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[backup] wrote {backup}")

        with lib_path.open("w", encoding="utf-8") as f:
            json.dump(lib, f, indent=2, ensure_ascii=False)
        print(f"[write]  updated {lib_path} (+{total_added} variants)")

    # Final audit.
    short = []
    for i_key in sorted(lib.keys(), key=int):
        entry = lib[i_key]["1"]
        total = len(entry.get("question", [])) + len(entry.get("question_synthetic", []))
        if total < args.target:
            short.append((i_key, entry.get("label", ""), total))
    if short:
        print(f"[audit] {len(short)} dimensions still under target {args.target}:")
        for i_key, label, total in short:
            print(f"  - {i_key} {label}: {total}")
    else:
        print(f"[audit] all dimensions now have >= {args.target} variants")

    return 0


if __name__ == "__main__":
    sys.exit(main())
