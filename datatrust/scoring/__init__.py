"""Scoring layer: Data Trust Score and ML Readiness Score, with traceable deductions."""

from datatrust.scoring.ml_readiness import score_ml_readiness
from datatrust.scoring.models import (
    GRADE_MEANING,
    Deduction,
    DimensionScore,
    ScoreResult,
    ScoringReport,
    grade_for,
)
from datatrust.scoring.trust import score_trust


def score_dataset(quality, stats, profile, target=None, settings=None) -> ScoringReport:
    """Convenience: compute both scores from the Phase 1-3 reports."""
    from datatrust.config import DEFAULT_SETTINGS
    settings = settings or DEFAULT_SETTINGS
    trust = score_trust(quality, settings)
    ml = score_ml_readiness(trust, quality, stats, profile, target, settings)
    return ScoringReport(trust=trust, ml_readiness=ml)


__all__ = [
    "GRADE_MEANING", "Deduction", "DimensionScore", "ScoreResult", "ScoringReport",
    "grade_for", "score_dataset", "score_ml_readiness", "score_trust",
]
