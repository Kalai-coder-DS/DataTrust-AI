"""
tests/test_scoring_integration.py

Scores on the generated datasets. The dirty file's dimension scores are derived by
hand from its known defects (dirty_manifest.json) and the severity points:

  completeness 76 = 100 - 10 (income missing, MEDIUM) - 4 (city missing, LOW) - 10 ('Unknown', MEDIUM)
  uniqueness   90 = 100 - 10 (duplicate rows, MEDIUM); repeated IDs charged once
  validity     70 = 100 - 10 ('abc' ages) - 10 (impossible ages) - 10 (impossible dates)
  consistency  90 = 100 - 10 ('Chennai' / 'chennai')
  trust = 0.3*76 + 0.2*90 + 0.3*70 + 0.2*90 = 79.8  (grade C)
"""
from __future__ import annotations

import pytest

from datatrust.ingestion import CSVLoader
from datatrust.profiling import DataProfiler
from datatrust.quality import QualityAnalyzer
from datatrust.scoring import score_dataset
from datatrust.stats import StatisticsAnalyzer


def _score(df, target=None):
    profile = DataProfiler().profile(df)
    quality = QualityAnalyzer().analyze(df, profile)
    stats = StatisticsAnalyzer().analyze(df, profile, target=target)
    return score_dataset(quality, stats, profile, target)


@pytest.fixture(scope="module")
def dirty_df(sample_dir):
    return CSVLoader().load(sample_dir / "customers_dirty.csv").df


def test_clean_data_scores_a(sample_dir):
    r = _score(CSVLoader().load(sample_dir / "customers_clean.csv").df, target="churned")
    assert r.trust.score == 100 and r.trust.grade == "A"
    assert r.ml_readiness.grade == "A"


def test_dirty_dimensions_match_hand_calculation(dirty_df):
    t = _score(dirty_df, target="churned").trust
    scores = {d.name: d.score for d in t.dimensions}
    assert scores == {"completeness": 76.0, "uniqueness": 90.0, "validity": 70.0, "consistency": 90.0}
    assert t.score == pytest.approx(79.8)
    assert t.grade == "C"


def test_dirty_ml_readiness_capped_at_trust(dirty_df):
    r = _score(dirty_df, target="churned")
    assert r.ml_readiness.score == r.trust.score
    assert r.ml_readiness.caps


def test_fixing_problems_raises_the_score(dirty_df):
    """Removing duplicates and the constant column must improve the score, never lower it."""
    before = _score(dirty_df, target="churned")
    fixed = dirty_df.drop_duplicates().drop(columns=["country"])
    after = _score(fixed, target="churned")
    assert after.trust.dimension("uniqueness").score == 100
    assert after.trust.score > before.trust.score
    assert after.ml_readiness.score >= before.ml_readiness.score
