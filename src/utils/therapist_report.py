"""Phase B: Therapist report generator.

Joins relational SQL data into a single flat CSV suitable for clinical
review and therapist validation.  Triggered automatically by
`mark_session_finalised("normal")` and, as a backstop, by the `atexit`
hook in io_record.

Output columns (one row per evidentiary event):
    Timestamp           — when the event occurred (ISO-8601).
    RowType             — {"Crisis", "ScoredDim", "Intervention", "Turn"}.
    Dimension           — 37-dim label + human name for ScoredDim and
                          Intervention rows; empty for Turn rows.
    FinalScore          — 0 | 1 | 2 | "" (empty for non-ScoredDim rows).
    UserEvidence        — the raw user utterance that sourced the score
                          or intervention.
    InterventionDetail  — a compact "stage/technique/outcome" summary for
                          MI/CBT rows; empty otherwise.
    Notes               — free-form notes (e.g. agent text for Turn rows).

Crisis rows (Score=2 on a CRITICAL dim) are emitted first so a
reviewer sees heightened-attention items before anything else.
"""
from __future__ import annotations

import csv
import datetime
import os
from typing import Optional

from src.utils.log_util import get_logger

logger = get_logger("TherapistReport")


# Track sessions we've already exported so repeated calls (normal exit
# followed by atexit backstop) don't generate duplicate files.
_EXPORTED_SESSIONS: set[int] = set()


def _clinical_dir() -> str:
    base = os.path.join(os.path.abspath("."), "data", "clinical")
    os.makedirs(base, exist_ok=True)
    return base


def _resolve_subject_id(db, session_id: int) -> Optional[str]:
    """Resolve the subject_id for a session by joining sessions → users."""
    try:
        with db._connect() as conn:
            c = conn.cursor()
            c.execute(
                """SELECT u.subject_id FROM sessions s
                   JOIN users u ON s.user_id = u.id
                   WHERE s.id = ?""",
                (session_id,),
            )
            row = c.fetchone()
        return row[0] if row else None
    except Exception as e:
        logger.warning(f"subject_id lookup failed: {e}")
        return None


def _intervention_detail(i: dict) -> str:
    parts = []
    if i.get("kind"):
        parts.append(str(i["kind"]))
    if i.get("stage"):
        parts.append(str(i["stage"]))
    if i.get("technique"):
        parts.append(str(i["technique"]))
    if i.get("outcome"):
        parts.append(f"outcome={i['outcome']}")
    return " / ".join(parts)


def _intervention_notes(i: dict) -> str:
    detail = i.get("detail")
    if not isinstance(detail, dict):
        return ""
    # Prefer the two most therapist-relevant fields if present.
    for key in ("validation_text", "guide_text", "reframe", "challenge", "unhelpful", "statement", "name"):
        if detail.get(key):
            return f"{key}: {detail[key]}"
    return ""


def generate_therapist_report(session_id: int, db=None) -> Optional[str]:
    """Generate `data/clinical/clinical_report_{SUBJECT_ID}_{SESSION_ID}.csv`.

    Returns the path to the written CSV (string) on success, or None if
    the session has already been exported or the export failed.  Safe to
    call multiple times for the same session — second and later calls
    are no-ops.
    """
    if session_id in _EXPORTED_SESSIONS:
        return None

    if db is None:
        try:
            import src.utils.io_record as io_rec
            db = getattr(io_rec, "DB", None)
        except Exception:
            db = None
    if db is None:
        logger.warning(f"generate_therapist_report({session_id}): no DB handle; skipping.")
        return None

    try:
        final_scores = db.get_clinical_scores(session_id)
        attempts = db.get_clinical_score_attempts(session_id)
        interventions = db.get_intervention_logs(session_id)
        turns = db.get_session_history(session_id)
        safety_deliveries = db.get_safety_deliveries(session_id)
    except Exception as e:
        logger.error(f"generate_therapist_report: DB read failed: {e}")
        return None

    subject_id = _resolve_subject_id(db, session_id) or "unknown"
    out_dir = _clinical_dir()
    out_path = os.path.join(
        out_dir, f"clinical_report_{subject_id}_{session_id}.csv"
    )

    # Index attempts by dim_label so the FinalScore row can carry the
    # attempt count as a Note.
    attempts_by_dim: dict[str, list[dict]] = {}
    for a in attempts:
        attempts_by_dim.setdefault(str(a["dim_label"]), []).append(a)

    header = [
        "Timestamp", "RowType", "Dimension", "FinalScore",
        "UserEvidence", "InterventionDetail", "Notes",
    ]

    rows: list[list[str]] = []

    # 1) Crisis rows first — any final score of 2 is heightened attention.
    crisis_scores = [s for s in final_scores if s.get("score") == 2]
    for s in crisis_scores:
        rows.append([
            str(s.get("updated_at") or ""),
            "Crisis",
            f"{s.get('dim_label','')} ({s.get('dim_name') or s.get('dim_label','')})",
            str(s.get("score")),
            str(s.get("evidence_text") or ""),
            "",
            f"Dim index {s.get('dim_index')}; flagged Score=2",
        ])

    # 2) All scored dimensions (including crisis, repeated in canonical
    #    order so the full picture is in one place).
    for s in sorted(final_scores, key=lambda r: r.get("dim_index") or 0):
        attempt_list = attempts_by_dim.get(str(s.get("dim_label")), [])
        note_bits = [f"attempts={len(attempt_list)}"]
        if len(attempt_list) > 1:
            trail = ",".join(str(a.get("score")) for a in attempt_list)
            note_bits.append(f"trail=[{trail}]")
        rows.append([
            str(s.get("updated_at") or ""),
            "ScoredDim",
            f"{s.get('dim_label','')} ({s.get('dim_name') or s.get('dim_label','')})",
            str(s.get("score")),
            str(s.get("evidence_text") or ""),
            "",
            "; ".join(note_bits),
        ])

    # 3) Intervention events (MI + CBT) in chronological order.
    for i in interventions:
        rows.append([
            str(i.get("created_at") or ""),
            "Intervention",
            str(i.get("dim_label") or ""),
            "",
            "",
            _intervention_detail(i),
            _intervention_notes(i),
        ])

    # 4) Safety-message delivery audit — proves the crisis resources
    #    message was actually spoken (method=tts_queue) or fell back.
    for d in safety_deliveries:
        rows.append([
            str(d.get("created_at") or ""),
            "SafetyDelivery",
            str(d.get("critical_dim") or ""),
            "",
            "",
            f"method={d.get('method')}; success={d.get('success')}",
            str(d.get("error_text") or d.get("message_text") or ""),
        ])

    # 5) Raw dialogue turns last, so a clinician can drop back into the
    #    transcript after reviewing the scored metrics.
    for idx, t in enumerate(turns):
        speaker = str(t.get("speaker", "unknown")).lower()
        rows.append([
            "",
            "Turn",
            "",
            "",
            str(t.get("text") or "") if speaker == "user" else "",
            "",
            f"{speaker}: {t.get('text','')}",
        ])

    try:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# Generated", datetime.datetime.now().isoformat()])
            w.writerow(["# Subject", subject_id])
            w.writerow(["# Session", session_id])
            w.writerow([])
            w.writerow(header)
            w.writerows(rows)
    except Exception as e:
        logger.error(f"generate_therapist_report: CSV write failed: {e}")
        return None

    _EXPORTED_SESSIONS.add(session_id)
    logger.info(
        f"[REPORT] Wrote clinical report: {out_path} "
        f"({len(final_scores)} scored dims, {len(interventions)} interventions, "
        f"{len(turns)} turns, {len(crisis_scores)} crisis rows)"
    )
    return out_path
