"""
tests/test_stats_integration.py

Phase 3 on the generated datasets, checked against dirty_manifest.json and
against the known design of the generator (churn depends on tenure;
monthly spend is derived from income).
"""
from __future__ import annotations

import pytest

from datatrust.ingestion import CSVLoader
from datatrust.stats import StatisticsAnalyzer


def _stats(path, target=None):
    return StatisticsAnalyzer().analyze(CSVLoader().load(path).df, target=target)


@pytest.fixture(scope="module")
def dirty(sample_dir):
    return _stats(sample_dir / "customers_dirty.csv", target="churned")


@pytest.fixture(scope="module")
def clean(sample_dir):
    return _stats(sample_dir / "customers_clean.csv", target="churned")


def test_invalid_ages_excluded_exactly(dirty, dirty_manifest):
    inv = dirty_manifest["invalid"]
    expected_excluded = inv["age_non_numeric"] + inv["age_negative"] + inv["age_over_120"]
    age = dirty.numeric_summary("age")
    assert age.excluded == expected_excluded
    assert age.count == dirty_manifest["total_rows"] - expected_excluded
    assert 18 <= age.min and age.max <= 75  # generator range: no impossible values leaked in


def test_invalid_dates_excluded_exactly(dirty, dirty_manifest):
    signup = next(d for d in dirty.datetime if d.column == "signup_date")
    assert signup.excluded == dirty_manifest["invalid"]["signup_date_impossible"]


def test_outliers_inflate_mean_not_median(dirty, clean):
    d, c = dirty.numeric_summary("monthly_spend"), clean.numeric_summary("monthly_spend")
    assert d.mean > 2 * c.mean                          # 7 extreme values drag the mean
    assert d.median == pytest.approx(c.median, rel=0.05)  # the median barely moves


def test_income_spend_relationship_robust_to_outliers(dirty, clean):
    d = dirty.association("annual_income", "monthly_spend")
    c = clean.association("annual_income", "monthly_spend")
    assert d.value == pytest.approx(c.value, abs=0.05)   # Spearman: same in both files
    assert d.details["pearson"] < 0.2 < c.details["pearson"]  # Pearson: wrecked by outliers


def test_churn_depends_on_tenure(clean):
    """The generator makes churn more likely at low tenure; the analysis must find it."""
    assoc = clean.association("tenure_months", "churned", method="correlation_ratio")
    assert assoc is not None and assoc.value >= 0.4


def test_inconsistent_city_spellings_show_as_extra_categories(dirty, clean, dirty_manifest):
    assert clean.categorical_summary("city").n_categories == 7
    assert dirty.categorical_summary("city").n_categories > 7


def test_churn_class_balance(clean):
    t = clean.target
    assert t.task == "classification" and t.n_classes == 2
    assert set(t.class_counts) == {"0", "1"}
    assert t.balance in {"balanced", "moderate"}


def test_clean_data_has_no_redundant_columns(clean):
    assert not [a for a in clean.associations if a.redundant]
