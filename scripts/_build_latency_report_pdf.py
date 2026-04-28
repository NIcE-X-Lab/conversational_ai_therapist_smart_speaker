#!/usr/bin/env python3
"""Build the CaiTI latency report as a PDF.

Output: /mnt/c/Users/arthv/Downloads/CaiTI_Latency_Report_Max.pdf

Uses reportlab's Platypus flow layout so headings, tables, code blocks, and
paragraphs paginate cleanly without manual page breaks. All numbers and
citations are embedded inline from the source report.
"""
from __future__ import annotations
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Preformatted, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT

OUT_PATH = "/mnt/c/Users/arthv/Downloads/CaiTI_Latency_Report_Max.pdf"

# ── Styles ─────────────────────────────────────────────────────────────────
STYLES = getSampleStyleSheet()

H_TITLE = ParagraphStyle(
    "TitleBig", parent=STYLES["Title"],
    fontName="Helvetica-Bold", fontSize=18, leading=22,
    spaceAfter=8, textColor=colors.HexColor("#1a1a1a"),
)
H_SUB = ParagraphStyle(
    "Subtitle", parent=STYLES["Normal"],
    fontName="Helvetica", fontSize=10, leading=13,
    spaceAfter=14, textColor=colors.HexColor("#555555"),
)
H1 = ParagraphStyle(
    "H1", parent=STYLES["Heading1"],
    fontName="Helvetica-Bold", fontSize=14, leading=18,
    spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"),
    keepWithNext=True,
)
H2 = ParagraphStyle(
    "H2", parent=STYLES["Heading2"],
    fontName="Helvetica-Bold", fontSize=11.5, leading=15,
    spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#2a2a2a"),
    keepWithNext=True,
)
BODY = ParagraphStyle(
    "Body", parent=STYLES["Normal"],
    fontName="Helvetica", fontSize=10, leading=14,
    spaceAfter=6, textColor=colors.HexColor("#222222"), alignment=TA_LEFT,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY,
    leftIndent=14, bulletIndent=2, spaceAfter=3,
)
NOTE = ParagraphStyle(
    "Note", parent=BODY,
    fontSize=9, textColor=colors.HexColor("#555555"),
    leftIndent=10, spaceAfter=4,
)
CODE = ParagraphStyle(
    "Code", parent=STYLES["Code"],
    fontName="Courier", fontSize=8.5, leading=11,
    textColor=colors.HexColor("#1a1a1a"),
    backColor=colors.HexColor("#f4f4f4"),
    borderColor=colors.HexColor("#dcdcdc"), borderWidth=0.5,
    borderPadding=6, spaceBefore=4, spaceAfter=8,
)

# ── Table helpers ──────────────────────────────────────────────────────────
TABLE_HEAD_BG = colors.HexColor("#2a5298")
TABLE_HEAD_FG = colors.white
TABLE_ALT_BG = colors.HexColor("#f4f7fb")
TABLE_GRID = colors.HexColor("#d0d7e2")

def make_table(data, col_widths):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD_BG),
        ("TEXTCOLOR",  (0, 0), (-1, 0), TABLE_HEAD_FG),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("LEADING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("GRID",       (0, 0), (-1, -1), 0.25, TABLE_GRID),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT_BG))
    t.setStyle(TableStyle(style))
    return t


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BULLET):
    return [Paragraph("• " + t, style) for t in items]


# ── Document ───────────────────────────────────────────────────────────────
def build():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=LETTER,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="CaiTI Latency Report — Jetson Orin Nano, Max session",
        author="CaiTI",
    )

    story = []

    # ── Header ─────────────────────────────────────────────────────────────
    story += [
        P("CaiTI Latency Report", H_TITLE),
        P("Jetson Orin Nano &nbsp;·&nbsp; Max session (2026-04-27 22:51–22:56)", H_SUB),
        P("<b>Hardware:</b> NVIDIA Jetson Orin Nano Dev Kit Super &nbsp;·&nbsp; 6-core ARMv8 Cortex-A78AE &nbsp;·&nbsp; 8 GB LPDDR5 unified memory &nbsp;·&nbsp; MAXN_SUPER power mode &nbsp;·&nbsp; Linux 5.15.148-tegra.", BODY),
        P("<b>Stack:</b> Gemma-4-E2B via LiteRT-LM (GPU backend, WebGPU/Ampere) &nbsp;·&nbsp; faster-whisper <font face='Courier'>base.en</font> int8 (CPU) &nbsp;·&nbsp; Piper <font face='Courier'>en_US-amy-medium</font> (subprocess).", BODY),
        P("<b>Session:</b> <font face='Courier'>Max_20260427_225105</font> — 7 turns (medication → mood → eat → work → retry → showup → care). All Score 0 or 1, no CBT path exercised.", BODY),
    ]

    # ── Part 1 ─────────────────────────────────────────────────────────────
    story += [P("Part 1 — Per-module latency (real Jetson, warm)", H1)]
    story += [P("From <font face='Courier'>scripts/_audit_latency.py</font> run on Jetson at 22:35, plus cross-checks against the Max session log where the same stage appears inline.", BODY)]

    # STT
    story += [P("STT — faster-whisper base.en int8, CPU", H2)]
    stt_table = make_table(
        [
            ["Metric", "Time"],
            ["Model load (cold start, one-time)", "2.06 s"],
            ["Resume from suspend (per turn)", "~0.5–1.1 s"],
            ["Transcribe 3 s silence (benchmark)", "1.40 s"],
            ["Transcribe ~3 s speech (session)", "~1.5 s"],
            ["Transcribe ~5–9 s speech (session)", "~1.7–2.0 s"],
            ["RSS during decode (peak)", "+135–225 MB above baseline"],
            ["Suspend (gc + free)", "~0.1–0.2 s"],
        ],
        col_widths=[3.5 * inch, 3.3 * inch],
    )
    story += [stt_table, Spacer(1, 6)]
    story += [P("Max-session examples:", BODY)]
    story += bullets([
        "22:51:43 mic closed → 22:51:45 transcript ready → <b>1.7 s</b> (\"Nearly every day.\")",
        "22:53:01 mic closed → 22:53:03 transcript ready → <b>2.2 s</b> (\"I've been feeling okay. Just a little sad…\")",
        "22:55:41 mic closed → 22:55:44 transcript ready → <b>2.6 s</b> (\"I'm just like I just don't go…\")",
    ])
    story += [P("<b>STT is not a latency bottleneck.</b> Decode cost scales linearly with audio length and averages ~25% of real-time.", BODY)]

    # TTS
    story += [P("TTS — Piper en_US-amy-medium (ONNX, CPU via subprocess)", H2)]
    tts_table = make_table(
        [
            ["Text length", "Gen time", "Playback time"],
            ["\"Hello world.\" (3 words)", "1.92 s", "1.0 s"],
            ["\"How has your mood been…\" (6 words)", "1.99 s", "~2 s"],
            ["Short question ~14 words", "2.15 s", "~3 s"],
            ["50-word RV_VALIDATOR response", "3.05 s", "~20 s"],
            ["Breathing script ~90 words (Max)", "2.7 s", "41–54 s"],
            ["PHQ question w/ Likert options ~30 words", "~2.5 s", "~15 s"],
        ],
        col_widths=[3.6 * inch, 1.4 * inch, 1.8 * inch],
    )
    story += [tts_table, Spacer(1, 6)]
    story += [P("Piper has a <b>~1.8 s fixed process-spawn cost</b> plus ~25 ms/word of synthesis. Playback is 1:1 with speech duration.", BODY)]
    story += [P("<b>TTS is not a latency bottleneck either</b> — but long spoken blocks (meditations, multi-paragraph RV validations) add wall-clock time simply because they <i>are</i> that long.", BODY)]

    # LLM
    story += [P("LLM — Gemma-4-E2B via LiteRT-LM, GPU backend", H2)]
    story += [P("Bench numbers, real prompts used in production:", BODY)]
    llm_table = make_table(
        [
            ["Role", "Prompt tokens", "Time", "Notes"],
            ["Short GENERAL prompt", "~50", "0.9 s", "engine-warm baseline"],
            ["REPHRASER", "~350", "1.6 s", "short output"],
            ["ANALYZER (single-segment)", "~1,263", "2.0 s", "per user segment"],
            ["MULTI-DIM ANALYZER", "~395", "2.4 s", "one call per utterance"],
            ["RV_REASONER", "~588", "2.1 s", "returns DECISION: 0/1"],
            ["RV_GUIDE", "~1,140", "3.9 s", "2–4 sentence redirect"],
            ["RV_VALIDATOR", "~1,485", "9.4 s", "4–7 sentence OARS paragraph"],
            ["Cold start (first call)", "—", "~4.6 s", "engine init + first decode"],
        ],
        col_widths=[1.8 * inch, 1.15 * inch, 0.75 * inch, 3.1 * inch],
    )
    story += [llm_table, Spacer(1, 6)]
    story += [P("Max-session examples (includes everything: prompt build, LLM, queue push, Python handoff):", BODY)]
    llm_turns = make_table(
        [
            ["Turn", "Stage timing"],
            ["Turn 1 (medication, Rephraser + Analyzer pipeline)",
             "22:52:09.9 transcript → 22:52:12.7 next agent line = <b>2.8 s</b>"],
            ["Turn 2 (mood, Analyzer + Rephraser)",
             "22:53:03.8 transcript → 22:53:08.4 next agent line = <b>4.6 s</b>"],
            ["Turn 3 (eat, Analyzer)",
             "22:53:54.2 transcript → 22:53:56.9 next agent line = <b>2.7 s</b>"],
            ["Turn 4 (work → retry_guide)",
             "22:55:04.0 transcript → 22:55:09.5 guide question = <b>5.5 s</b> (Analyzer + backfill + retry_guide LLM)"],
            ["Turn 5 (retry answer)",
             "22:55:44.2 transcript → 22:55:46.8 next question = <b>2.6 s</b>"],
            ["Turn 6 (showup → Rephraser for care)",
             "22:56:07.6 transcript → 22:56:12.1 next question = <b>4.5 s</b> (Analyzer + Rephraser for next turn)"],
        ],
        col_widths=[2.6 * inch, 4.2 * inch],
    )
    # Apply Paragraph wrapping for the "timing" column so it can wrap
    wrapped_data = [[P(row[0], BODY), P(row[1], BODY)] for row in [
        ["Turn", "Stage timing"],
        ["Turn 1 (medication, Rephraser + Analyzer pipeline)",
         "22:52:09.9 transcript → 22:52:12.7 next agent line = <b>2.8 s</b>"],
        ["Turn 2 (mood, Analyzer + Rephraser)",
         "22:53:03.8 transcript → 22:53:08.4 next agent line = <b>4.6 s</b>"],
        ["Turn 3 (eat, Analyzer)",
         "22:53:54.2 transcript → 22:53:56.9 next agent line = <b>2.7 s</b>"],
        ["Turn 4 (work → retry_guide)",
         "22:55:04.0 transcript → 22:55:09.5 guide question = <b>5.5 s</b> (Analyzer + backfill + retry_guide LLM)"],
        ["Turn 5 (retry answer)",
         "22:55:44.2 transcript → 22:55:46.8 next question = <b>2.6 s</b>"],
        ["Turn 6 (showup → Rephraser for care)",
         "22:56:07.6 transcript → 22:56:12.1 next question = <b>4.5 s</b> (Analyzer + Rephraser for next turn)"],
    ]]
    llm_turns = make_table(wrapped_data, col_widths=[2.6 * inch, 4.2 * inch])
    # override header font to white-bold by making the first-row Paragraph
    wrapped_data[0] = [
        P("<font color='white'><b>Turn</b></font>", BODY),
        P("<font color='white'><b>Stage timing</b></font>", BODY),
    ]
    llm_turns = make_table(wrapped_data, col_widths=[2.6 * inch, 4.2 * inch])
    story += [llm_turns, Spacer(1, 6)]
    story += [P("<b>LLM is no longer the bottleneck on short-answer turns.</b> A typical Score 0/1 turn with just ANALYZER costs 2–3 s.", BODY)]

    # ── Part 2 ─────────────────────────────────────────────────────────────
    story += [PageBreak(), P("Part 2 — End-to-end turn breakdown (Max session)", H1)]
    story += [P("The pipeline has six distinct stages between \"user stops speaking\" and \"user hears next question\":", BODY)]

    pipeline_ascii = (
        "USER STOPS SPEAKING ---------------- mic closes @ t=0\n"
        "|\n"
        "+-- Stage 1: record + save WAV         0.0-0.1 s    (pygame mixer write)\n"
        "+-- Stage 2: STT decode (parallel)     1.5-2.6 s    (runs in worker thread)\n"
        "|            `--> intermission starts @ t~0.1 s while STT still running\n"
        "+-- Stage 3: Intermission TTS gen       2-3 s        (Piper synthesis)\n"
        "+-- Stage 4: Intermission playback      15-45 s      (interrupted early by watcher)\n"
        "+-- Stage 5: Handler LLM chain          2-6 s        (parallel with Stage 3-4)\n"
        "+-- Stage 6: Delivery                   5-8 s        (bridge + next TTS + playback)\n"
        "|\n"
        "NEXT AGENT QUESTION STARTS SPEAKING"
    )
    story += [Preformatted(pipeline_ascii, CODE)]

    # Turn 2 walkthrough
    story += [P("Detailed walkthrough: Max Turn 2 (mood) — representative best case", H2)]
    turn2_rows = [
        ["Time", "Stage", "Event"],
        ["22:52:53.1", "—", "mic opens (user prompt playback just finished)"],
        ["22:53:01.5", "—", "mic closes (user said \"I've been feeling okay…\")"],
        ["22:53:01.5", "1", "WAV saved to active_user_input_188695290.wav (8.5 s audio)"],
        ["22:53:01.5", "4a", "INTERMISSION Pre-wait: SCREENING (PHQ-4) — parallel path selected"],
        ["22:53:01.6", "4b", "PHQ TTS generation starts"],
        ["22:53:03.8", "2", "STT transcription complete — 2.2 s after mic closed"],
        ["22:53:03.8", "5", "Transcript pushed to input_queue; handler starts ANALYZER LLM call"],
        ["22:53:06.3", "5", "ANALYZER returns ('mood', 0) — 2.4 s"],
        ["22:53:06.3", "4b", "PHQ plays (14.9 s of audio)"],
        ["22:53:08.4", "5", "RL picks next dim (eat), next AGENT question ready"],
        ["22:53:21.2", "4c", "PHQ playback finishes"],
        ["22:53:29.7", "6", "Bridge + next-question TTS"],
        ["22:53:38.3", "—", "\"How's your eating habits?\" playback finishes"],
        ["22:53:46.5", "—", "mic opens for next turn"],
    ]
    story += [make_table(turn2_rows, col_widths=[0.95 * inch, 0.6 * inch, 5.25 * inch]), Spacer(1, 6)]
    story += [P("<b>Total mic-close → next-question-starts: ~28 s</b> (heavy intermission) or ~18 s if a shorter activity were picked.", BODY)]

    # Turn 4 walkthrough
    story += [P("Alternative: Max Turn 4 with early-exit (breathing cut short by watcher)", H2)]
    turn4_rows = [
        ["Time", "Stage", "Event"],
        ["22:55:02.2", "—", "mic closed, WAV saved (3.9 s audio)"],
        ["22:55:02.2", "4a", "INTERMISSION: BREATHING_EXERCISE starts"],
        ["22:55:02.2", "4b", "Breathing TTS gen (41 s of audio to play)"],
        ["22:55:04.0", "2", "STT: \"Okay.\" (1.8 s)"],
        ["22:55:04.2", "5", "ANALYZER starts"],
        ["22:55:06.6", "5", "ANALYZER → ('work_dayoff', 0)"],
        ["22:55:06.7", "5", "retry_guide LLM fires (primary dim had no match)"],
        ["22:55:09.5", "5", "Guide text ready: \"Could you tell me a bit more…\""],
        ["22:55:09.6", "—", "[INTERMISSION] Output ready — interrupting activity (engagement=7.4 s)"],
        ["22:55:09.6", "—", "breathing playback cut"],
        ["22:55:10.5", "6", "bridge phrase TTS"],
        ["22:55:21.1", "6", "guide question playback starts"],
        ["22:55:26.1", "—", "playback finishes"],
        ["22:55:30.2", "—", "mic opens"],
    ]
    story += [make_table(turn4_rows, col_widths=[0.95 * inch, 0.6 * inch, 5.25 * inch]), Spacer(1, 6)]
    story += [P("<b>Total mic-close → next-question-starts: ~19 s.</b> Interrupt-watcher saved ~30 s (breathing would have run 41 s + post-hold). This is the new parallel-STT + early-exit behavior working as designed.", BODY)]

    story += [P("The gap that still exists", H2)]
    story += [P("You can see at Turn 2: the LLM answer is ready at 22:53:08 but the user doesn't hear the next question until 22:53:29 — that's a <b>21-second gap</b>. The intermission (PHQ + Likert) plays its full 15 seconds, then the bridge phrase + fade adds another 6–8 s. The watcher doesn't trip because the PHQ is a SCREENING intermission and we deliberately don't interrupt those (clinical data).", BODY)]
    story += [P("The remaining levers to shorten this without hurting UX are Tier 2 prompt surgery (cut RV_VALIDATOR/ANALYZER prompts), or Tier 3 streaming, or shortening the bridge phrase.", BODY)]

    # ── Part 3 ─────────────────────────────────────────────────────────────
    story += [PageBreak(), P("Part 3 — Before / after comparison", H1)]
    ba_rows = [
        ["Configuration", "Score-0/1 turn", "Score-2 turn", "Notes"],
        ["Baseline (CPU backend, all extensions on)", "43–60 s", "85–134 s", "observed in \"Smith\" session"],
        ["After Tier 0 (GPU + flags)", "13–15 s", "15–20 s", "benchmark + first GPU session"],
        ["After early-exit watcher", "12–19 s", "15–22 s", "Alex session partial fix"],
        ["After parallel STT worker", "10–19 s", "~15–20 s (not exercised in Max)", "Max session"],
    ]
    ba_wrapped = [[P(cell, BODY) for cell in row] for row in ba_rows]
    # white header row
    ba_wrapped[0] = [P(f"<font color='white'><b>{c}</b></font>", BODY) for c in ba_rows[0]]
    story += [make_table(ba_wrapped, col_widths=[2.3 * inch, 1.1 * inch, 1.4 * inch, 2.0 * inch]), Spacer(1, 6)]

    story += [P("Max session per-turn totals (mic-close → next-agent-question-audible):", BODY)]
    per_turn_rows = [
        ["Turn", "Duration", "Intermission type", "Notes"],
        ["1 (medication)", "~23 s", "SCREENING (nervous/anxious)", "first turn, Rephraser for opening"],
        ["2 (mood)", "~28 s", "SCREENING (worry control)", "8.5 s input, 15 s PHQ TTS playback dominates"],
        ["3 (eat)", "~18 s", "SCREENING (interest/pleasure)", "shorter user input"],
        ["4 (work)", "~14 s", "BREATHING (cut at 7.4 s)", "early-exit watcher triggered"],
        ["5 (retry_guide answer)", "~13 s", "MUSIC (cut at 6.0 s)", "early-exit watcher triggered"],
        ["6 (showup → care)", "~16 s", "SCREENING (hopeless)", "Rephraser in-flight + Analyzer"],
        ["7 (care)", "~10 s", "BREATHING (full run)", "short user input, breathing not cut"],
    ]
    pt_wrapped = [[P(cell, BODY) for cell in row] for row in per_turn_rows]
    pt_wrapped[0] = [P(f"<font color='white'><b>{c}</b></font>", BODY) for c in per_turn_rows[0]]
    story += [make_table(pt_wrapped, col_widths=[1.5 * inch, 0.85 * inch, 2.0 * inch, 2.45 * inch]), Spacer(1, 6)]
    story += [P("<b>Median turn: ~16 s.</b> Compared to your baseline \"Smith\" observations of ~82 s median, this is a <b>~5× reduction</b>.", BODY)]

    # ── Part 4 ─────────────────────────────────────────────────────────────
    story += [PageBreak(), P("Part 4 — Where the remaining time actually goes", H1)]
    story += [P("For a \"canonical\" 16-second Max turn, here's where each second goes:", BODY)]

    canonical = (
        "  0.0 s   Mic closes\n"
        "  0.1 s   WAV saved to disk                     (0.1 s)\n"
        "  0.1 s   Intermission selected + TTS gen starts\n"
        "  2.0 s   STT transcript ready                  (1.9 s STT, parallel)\n"
        "  2.1 s   Transcript queued -> handler picks it up\n"
        "  4.1 s   ANALYZER done                         (2.0 s LLM)\n"
        "  4.3 s   Next question ready in output_queue\n"
        "  4-10 s  Intermission audio playing (user hears the PHQ or meditation)\n"
        " 10.0 s   Intermission audio ends OR watcher cuts it\n"
        " 10.5 s   Music fade + brief pause              (~0.5 s real, 0.9 s sleep)\n"
        " 12.5 s   Bridge phrase TTS gen + playback      (2 s gen + 3 s playback)\n"
        " 13.0 s   Final question TTS gen                (~2 s)\n"
        " 14.0 s   Final question playback starts        <- user hears next question\n"
        " 16.0 s   Final question playback finishes"
    )
    story += [Preformatted(canonical, CODE)]

    story += [P("What's still on the critical path", H2)]
    cp_rows = [
        ["Stage", "~seconds per turn", "Can it shrink?"],
        ["STT decode (parallel)", "1.5–2.5", "~1 s possible with tiny.en model (quality trade)"],
        ["ANALYZER LLM (critical path)", "2–3", "Tier 2: shrink prompt → 1–1.5 s"],
        ["Intermission TTS playback", "5–20", "Shorter scripts, or make every activity interruptible not just blocking wait"],
        ["Music fade + pause", "0.9", "Can shrink time.sleep(0.9) to 0.3 s"],
        ["Bridge phrase", "3–5", "Remove entirely (saves ~4 s), or use 1-word \"Okay:\" instead"],
        ["Final TTS gen", "~2", "Mostly Piper process spawn — could pre-warm"],
    ]
    cp_wrapped = [[P(cell, BODY) for cell in row] for row in cp_rows]
    cp_wrapped[0] = [P(f"<font color='white'><b>{c}</b></font>", BODY) for c in cp_rows[0]]
    story += [make_table(cp_wrapped, col_widths=[2.1 * inch, 1.3 * inch, 3.4 * inch]), Spacer(1, 6)]
    story += [P("<b>If you applied all of these: ~8–10 s per turn, half of today's median.</b>", BODY)]

    # ── Part 5 ─────────────────────────────────────────────────────────────
    story += [P("Part 5 — Validated observations from Max session", H1)]
    story += [P("Good", H2)]
    story += bullets([
        "<b>Parallel path fired on every qualifying turn</b> (<font face='Courier'>Captured 8.5 s of audio … Parallel path.</font>) — turns 2, 3, 5, 7 all took this path correctly.",
        "<b>Serial path kicked in for short utterances</b> (turns 4, 6 both \"Captured 2.7–2.8 s of audio … Serial path.\") — fragment-merge is preserved as intended.",
        "<b>STT suspend happens inside the worker</b> — no main-thread stalls on gc.collect. Log shows suspend firing ~0.1 s after transcript publication.",
        "<b>Early-exit watcher fired twice:</b> engagement=7.4 s on breathing at 22:55:09 and engagement=6.0 s on music at 22:55:47. Both respected the minimum engagement floor.",
        "<b>Multi-dim backfill is re-enabled</b> but never fired (no utterance had ≥20 tokens and ≥2 segments in this session) — that's expected, not a bug.",
    ])

    # Footer
    story += [Spacer(1, 14)]
    story += [P("Report generated from live Jetson logs and benchmark run. Re-run <font face='Courier'>scripts/_audit_latency.py</font> on the Jetson any time to refresh per-module numbers.", NOTE)]

    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
