"""Intermission ladder state manager for speech-session waiting periods."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Dict

from src.core.therapy_content import CLINICAL_SCREENING, MEDITATIONS


class IntermissionStage(str, Enum):
    """Ordered intermission stages used while the LLM is still thinking."""

    SCREENING = "SCREENING"
    BREATHING_EXERCISE = "BREATHING_EXERCISE"
    MUSIC = "MUSIC"


@dataclass(frozen=True)
class ScreeningQuestion:
    """Immutable question descriptor returned to callers."""

    index: int
    question_id: str
    text: str


class IntermissionTracker:
    """Tracks screening question status so questions are never repeated."""

    STATUS_PENDING = "PENDING"
    STATUS_ANSWERED = "ANSWERED"
    STATUS_SKIPPED = "SKIPPED"

    def __init__(self):
        self._questions = list(CLINICAL_SCREENING)
        self._state: Dict[str, dict] = {}
        self._asked_questions: list[str] = []
        self.reset()

    def reset(self):
        self._asked_questions = []
        self._state = {
            q["id"]: {
                "status": self.STATUS_PENDING,
                "score": None,
                "response": "",
                "reason": "",
            }
            for q in self._questions
        }

    def mark_answered(self, question_id: str, score: int, response: str = ""):
        if question_id not in self._state:
            return
        if question_id not in self._asked_questions:
            self._asked_questions.append(question_id)
        self._state[question_id]["status"] = self.STATUS_ANSWERED
        self._state[question_id]["score"] = int(score)
        self._state[question_id]["response"] = str(response or "")
        self._state[question_id]["reason"] = ""

    def mark_skipped(self, question_id: str, reason: str = ""):
        if question_id not in self._state:
            return
        if question_id not in self._asked_questions:
            self._asked_questions.append(question_id)
        self._state[question_id]["status"] = self.STATUS_SKIPPED
        self._state[question_id]["score"] = None
        self._state[question_id]["reason"] = str(reason or "")

    def restore_from_status_map(self, status_map: Dict[str, dict] | None):
        """Load persisted question checkpoints from DB for this session."""
        self.reset()
        if not status_map:
            return

        for q in self._questions:
            qid = q["id"]
            row = status_map.get(qid)
            if not row:
                continue

            status = str(row.get("status") or "").upper().strip()
            if status == self.STATUS_ANSWERED:
                self.mark_answered(
                    qid,
                    score=int(row.get("score") or 0),
                    response=str(row.get("response_text") or ""),
                )
            elif status == self.STATUS_SKIPPED:
                self.mark_skipped(qid, reason=str(row.get("reason") or "restored"))

    def next_unanswered(self) -> ScreeningQuestion | None:
        for idx, q in enumerate(self._questions):
            item = self._state.get(q["id"], {})
            if item.get("status") == self.STATUS_PENDING:
                return ScreeningQuestion(index=idx, question_id=q["id"], text=q["text"])
        return None

    def is_complete(self) -> bool:
        return all(v.get("status") != self.STATUS_PENDING for v in self._state.values())

    def stage_snapshot(self) -> Dict[str, dict]:
        snapshot = {k: dict(v) for k, v in self._state.items()}
        snapshot["asked_questions"] = list(self._asked_questions)
        return snapshot


class IntermissionLadderManager:
    """Strict SCREENING -> BREATHING -> MUSIC stage progression."""

    def __init__(self):
        self._tracker = IntermissionTracker()
        self._breathing_count = 0
        self._rng = random.Random()
        self._last_breathing_idx = None

    @property
    def tracker(self) -> IntermissionTracker:
        return self._tracker

    def reset(self):
        self._tracker.reset()
        self._breathing_count = 0
        self._last_breathing_idx = None

    def load_checkpoint(self, status_map: Dict[str, dict] | None):
        self._tracker.restore_from_status_map(status_map)

    def current_stage(self) -> IntermissionStage:
        if not self._tracker.is_complete():
            return IntermissionStage.SCREENING
        if self._breathing_count <= 0:
            return IntermissionStage.BREATHING_EXERCISE
        return IntermissionStage.MUSIC

    def next_screening_question(self) -> ScreeningQuestion | None:
        return self._tracker.next_unanswered()

    def record_screening_answer(self, question_id: str, score: int, response: str = ""):
        self._tracker.mark_answered(question_id, score, response=response)

    def skip_screening_question(self, question_id: str, reason: str = ""):
        self._tracker.mark_skipped(question_id, reason=reason)

    def next_breathing_exercise(self) -> str:
        if not MEDITATIONS:
            return "Let's take a gentle breath together while I continue thinking."

        candidate_indices = list(range(len(MEDITATIONS)))
        if self._last_breathing_idx is not None and len(candidate_indices) > 1:
            candidate_indices.remove(self._last_breathing_idx)
        idx = self._rng.choice(candidate_indices)
        self._last_breathing_idx = idx
        return MEDITATIONS[idx]

    def mark_breathing_complete(self):
        self._breathing_count += 1

    def stage_snapshot(self) -> dict:
        return {
            "stage": self.current_stage().value,
            "breathing_count": self._breathing_count,
            "breathing_done": self._breathing_count > 0,
            "screening": self._tracker.stage_snapshot(),
        }
