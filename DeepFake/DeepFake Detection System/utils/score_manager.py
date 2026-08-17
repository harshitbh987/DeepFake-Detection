
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import sys, os
sys.path.insert(0, "/home/claude/DeepSentinel")
from config import FAKE_DECISION_BOUNDARY, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM

logger = logging.getLogger(__name__)

@dataclass
class FrameScore:
    frame_index: int
    fake_prob:   float
    has_face:    bool = True

    @property
    def certainty(self) -> float:
        return abs(self.fake_prob - 0.5) * 2.0

    @property
    def label(self) -> str:
        return "FAKE" if self.fake_prob >= FAKE_DECISION_BOUNDARY else "REAL"


@dataclass
class DetectionVerdict:
    frame_scores:     List[FrameScore]
    weighted_score:   float
    simple_score:     float
    verdict:          str
    confidence_tier:  str
    frames_with_face: int
    frames_analysed:  int
    metadata:         Dict = field(default_factory=dict)

    @property
    def is_fake(self) -> bool:
        return self.verdict == "FAKE"

    @property
    def display_score(self) -> float:
        return self.weighted_score

    def summary_str(self) -> str:
        return (f"Verdict: {self.verdict} | Score: {self.weighted_score:.3f} | "
                f"Confidence: {self.confidence_tier} | "
                f"Faces: {self.frames_with_face}/{self.frames_analysed}")


class ScoreManager:
    def __init__(self) -> None:
        self._records: List[FrameScore] = []

    def add(self, record: FrameScore) -> None:
        self._records.append(record)

    def add_batch(self, indices, probs) -> None:
        for i, p in zip(indices, probs):
            self._records.append(FrameScore(frame_index=i, fake_prob=p))

    def reset(self) -> None:
        self._records.clear()

    @property
    def frame_count(self) -> int:
        return len(self._records)

    def compute_verdict(self) -> DetectionVerdict:
        if not self._records:
            raise ValueError("No scores recorded.")
        valid = [r for r in self._records if r.has_face] or self._records
        probs       = np.array([r.fake_prob  for r in valid])
        certainties = np.array([r.certainty  for r in valid])
        w_sum = certainties.sum()
        weighted_score = float(np.dot(probs, certainties) / w_sum) if w_sum > 1e-9 else float(probs.mean())
        simple_score   = float(probs.mean())
        verdict        = "FAKE" if weighted_score >= FAKE_DECISION_BOUNDARY else "REAL"
        mean_cert      = float(certainties.mean())
        tier = "High" if mean_cert >= 0.30 else ("Medium" if mean_cert >= 0.10 else "Low")
        return DetectionVerdict(
            frame_scores=list(self._records), weighted_score=weighted_score,
            simple_score=simple_score, verdict=verdict, confidence_tier=tier,
            frames_with_face=len(valid), frames_analysed=len(self._records),
            metadata={"min": round(float(probs.min()),4), "max": round(float(probs.max()),4),
                      "std": round(float(probs.std()),4), "median": round(float(np.median(probs)),4)},
        )

    def export_scores(self) -> List[Dict]:
        return [{"frame_index": r.frame_index, "fake_prob": round(r.fake_prob,4),
                 "certainty": round(r.certainty,4), "has_face": r.has_face,
                 "label": r.label} for r in self._records]
