"""
datatrust/scoring/models.py

Score containers. The core rule of this phase:

    EVERY POINT LOST IS A `Deduction` THAT NAMES ITS CAUSE.

So for any score you can answer "why isn't this 100?" by listing the deductions,
and "what should I fix first?" by sorting them by impact on the final score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Deduction:
    dimension: str            # e.g. "validity"
    points: float             # points removed from that dimension (0-100 scale)
    reason: str               # human-readable cause
    recommendation: str = ""  # what to do about it
    column: str | None = None
    issue_type: str | None = None
    impact: float = 0.0       # points removed from the FINAL score = points x dimension weight
    trust_impact: float = 0.0  # for trust deductions carried into ML readiness: impact on the trust score

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DimensionScore:
    name: str
    weight: float
    score: float                       # 0..100 after deductions (never below 0)
    deductions: list[Deduction] = field(default_factory=list)
    note: str = ""                     # e.g. how the score was derived when not deduction-based

    def to_dict(self) -> dict:
        return {
            "name": self.name, "weight": self.weight, "score": self.score, "note": self.note,
            "deductions": [d.to_dict() for d in self.deductions],
        }


@dataclass
class ScoreResult:
    name: str                          # "Data Trust Score" / "ML Readiness Score"
    score: float                       # 0..100
    grade: str                         # A..F
    dimensions: list[DimensionScore] = field(default_factory=list)
    caps: list[str] = field(default_factory=list)    # rules that limited the score, if any
    notes: list[str] = field(default_factory=list)   # informational, no points involved

    def dimension(self, name: str) -> DimensionScore:
        for d in self.dimensions:
            if d.name == name:
                return d
        raise KeyError(f"No dimension '{name}'")

    def all_deductions(self) -> list[Deduction]:
        return [d for dim in self.dimensions for d in dim.deductions]

    def top_actions(self, n: int = 5) -> list[Deduction]:
        """Biggest wins first: the deductions that cost the final score the most."""
        actions = [d for d in self.all_deductions() if d.impact > 0]
        return sorted(actions, key=lambda d: -d.impact)[:n]

    def to_dict(self) -> dict:
        return {
            "name": self.name, "score": self.score, "grade": self.grade,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "caps": list(self.caps), "notes": list(self.notes),
            "top_actions": [d.to_dict() for d in self.top_actions()],
        }


@dataclass
class ScoringReport:
    trust: ScoreResult
    ml_readiness: ScoreResult

    def to_dict(self) -> dict:
        return {"trust": self.trust.to_dict(), "ml_readiness": self.ml_readiness.to_dict()}


# ---------------------------------------------------------------- helpers
GRADE_BANDS: tuple[tuple[float, str], ...] = ((90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F"))
GRADE_MEANING = {
    "A": "excellent", "B": "good", "C": "usable with fixes", "D": "poor", "F": "not usable as-is",
}


def grade_for(score: float) -> str:
    for threshold, grade in GRADE_BANDS:
        if score >= threshold:
            return grade
    return "F"


def finalize_dimension(name: str, weight: float, deductions: list[Deduction], note: str = "") -> DimensionScore:
    """Score = 100 - sum(deductions), floored at 0; records each deduction's impact on the final score."""
    for d in deductions:
        d.impact = round(d.points * weight, 2)
    score = max(0.0, 100.0 - sum(d.points for d in deductions))
    return DimensionScore(name=name, weight=weight, score=round(score, 1), deductions=deductions, note=note)


def combine(dimensions: list[DimensionScore]) -> float:
    """Weighted average of dimension scores; weights are renormalised to sum to 1."""
    total_weight = sum(d.weight for d in dimensions)
    if total_weight == 0:
        return 0.0
    return round(sum(d.score * d.weight for d in dimensions) / total_weight, 1)
