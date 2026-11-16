"""
tests/test_scoring.py

Phase 4 scoring. The most important property is TRACEABILITY: every point lost
must be a Deduction, and the deductions must add up exactly to the score.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from datatrust.profiling import DataProfiler
from datatrust.quality import IssueType, QualityAnalyzer, QualityIssue, QualityReport, Severity
from datatrust.scoring import grade_for, score_dataset, score_ml_readiness, score_trust
from datatrust.stats import StatisticsAnalyzer


def issue(issue_type, severity, column="c", **details) -> QualityIssue:
    return QualityIssue(check="x", issue_type=issue_type, severity=severity, column=column,
                        affected_rows=1, affected_pct=1.0, message=f"{issue_type.value} in {column}",
                        recommendation="fix it", details=details)


def report(*issues) -> QualityReport:
    return QualityReport(n_rows=100, n_columns=5, issues=list(issues))


def full(df: pd.DataFrame, target=None):
    profile = DataProfiler().profile(df)
    quality = QualityAnalyzer().analyze(df, profile)
    stats = StatisticsAnalyzer().analyze(df, profile, target=target)
    return score_dataset(quality, stats, profile, target)


# ------------------------------------------------------------------ grades
@pytest.mark.parametrize("score, grade", [(100, "A"), (90, "A"), (89.9, "B"), (80, "B"),
                                          (70, "C"), (60, "D"), (59.9, "F"), (0, "F")])
def test_grade_bands(score, grade):
    assert grade_for(score) == grade


# ------------------------------------------------------------- trust score
def test_no_issues_is_perfect():
    t = score_trust(report())
    assert t.score == 100 and t.grade == "A"


def test_single_medium_validity_issue():
    t = score_trust(report(issue(IssueType.OUT_OF_RANGE, Severity.MEDIUM)))
    assert t.dimension("validity").score == 90           # 100 - 10
    assert t.score == pytest.approx(100 - 10 * 0.30)     # validity weight 30%


@pytest.mark.parametrize("issue_type, dimension", [
    (IssueType.MISSING, "completeness"), (IssueType.DISGUISED_MISSING, "completeness"),
    (IssueType.DUPLICATE_ROWS, "uniqueness"), (IssueType.DUPLICATES_HIDDEN_BY_ID, "uniqueness"),
    (IssueType.TYPE_MISMATCH, "validity"), (IssueType.INVALID_DATE, "validity"),
    (IssueType.INCONSISTENT_FORMAT, "consistency"), (IssueType.NUMBERS_AS_WORDS, "consistency"),
])
def test_issue_charged_to_correct_dimension(issue_type, dimension):
    t = score_trust(report(issue(issue_type, Severity.LOW)))
    assert [d.dimension for d in t.all_deductions()] == [dimension]


def test_outliers_and_constant_columns_do_not_reduce_trust():
    t = score_trust(report(issue(IssueType.OUTLIERS, Severity.MEDIUM),
                           issue(IssueType.CONSTANT_COLUMN, Severity.MEDIUM)))
    assert t.score == 100


def test_dimension_floors_at_zero():
    t = score_trust(report(*[issue(IssueType.TYPE_MISMATCH, Severity.HIGH, column=f"c{i}") for i in range(8)]))
    assert t.dimension("validity").score == 0


def test_duplicate_ids_from_exact_duplicates_charged_once():
    t = score_trust(report(issue(IssueType.DUPLICATE_ROWS, Severity.MEDIUM, column=None),
                           issue(IssueType.DUPLICATE_IDS, Severity.MEDIUM, conflicting_records=0)))
    assert len(t.dimension("uniqueness").deductions) == 1
    assert any("charged once" in n for n in t.notes)


def test_conflicting_duplicate_ids_are_charged():
    t = score_trust(report(issue(IssueType.DUPLICATE_IDS, Severity.HIGH, conflicting_records=3)))
    assert t.dimension("uniqueness").score == 80


def test_critical_issue_caps_trust_at_grade_d():
    t = score_trust(report(issue(IssueType.EMPTY_COLUMN, Severity.CRITICAL)))
    # Uncapped: completeness 65 -> 100 - 35*0.3 = 89.5 (B); a critical issue must not hide in an average.
    assert t.score == 69 and t.grade == "D"
    assert t.caps


def test_failed_check_reduces_its_dimension():
    failed = QualityIssue(check="validity", issue_type=IssueType.CHECK_FAILED, severity=Severity.HIGH,
                          column=None, affected_rows=0, affected_pct=0, message="boom", recommendation="")
    assert score_trust(report(failed)).dimension("validity").score == 80


def test_trust_traceability_on_dirty_data(sample_dir):
    from datatrust.ingestion import CSVLoader
    df = CSVLoader().load(sample_dir / "customers_dirty.csv").df
    t = score_trust(QualityAnalyzer().analyze(df))
    for dim in t.dimensions:
        assert dim.score == pytest.approx(max(0, 100 - sum(d.points for d in dim.deductions)))
    assert t.score == pytest.approx(100 - sum(d.impact for d in t.all_deductions()), abs=0.11)


# ---------------------------------------------------------- ML readiness
def _clean_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(50, 10, n)
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "x": x.round(2),
        "z": rng.normal(0, 1, n).round(3),
        "group": rng.choice(["a", "b", "c"], n),
        "label": np.where(x + rng.normal(0, 10, n) > 50, "yes", "no"),
    })


def test_ml_readiness_never_exceeds_trust():
    df = _clean_frame()
    df["x"] = df["x"].astype(object)   # like a CSV column mixing numbers and text
    df.loc[:59, "x"] = "abc"           # validity problems lower trust
    r = full(df, target="label")
    assert r.ml_readiness.score <= r.trust.score


def test_when_capped_only_trust_fixes_have_impact(sample_dir):
    from datatrust.ingestion import CSVLoader
    r = full(CSVLoader().load(sample_dir / "customers_dirty.csv").df, target="churned")
    assert r.ml_readiness.caps
    for dim in r.ml_readiness.dimensions[1:]:
        assert all(d.impact == 0 for d in dim.deductions)
    assert all(a.dimension == "data_trust" for a in r.ml_readiness.top_actions())


def test_no_target_skips_target_dimension_and_says_so():
    r = full(_clean_frame())
    names = [d.name for d in r.ml_readiness.dimensions]
    assert "target" not in names
    assert sum(d.weight for d in r.ml_readiness.dimensions) == pytest.approx(1.0)
    assert any("No target given" in n for n in r.ml_readiness.notes)


def test_small_dataset_loses_sample_size_points():
    r = full(_clean_frame(n=60), target="label")
    assert r.ml_readiness.dimension("sample_size").score < 60


def test_redundancy_charged_per_droppable_column_not_per_pair():
    """Three near-identical columns form 3 redundant pairs but only 2 columns can be dropped."""
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 500)
    df = pd.DataFrame({"a": base, "b": base * 2 + rng.normal(0, 0.01, 500),
                       "c": base * 3 + rng.normal(0, 0.01, 500), "noise": rng.normal(0, 1, 500)})
    r = full(df)
    redundant = [d for d in r.ml_readiness.dimension("feature_quality").deductions
                 if d.issue_type == "redundant_feature"]
    assert len(redundant) == 2


def test_target_leakage_detected():
    df = _clean_frame()
    df["label_copy"] = df["label"].map({"yes": "Y", "no": "N"})   # the answer, renamed
    r = full(df, target="label")
    leaks = [d for d in r.ml_readiness.dimension("target").deductions if d.issue_type == "target_leakage"]
    assert [d.column for d in leaks] == ["label_copy"]


def test_severe_class_imbalance():
    df = _clean_frame()
    df["label"] = ["yes"] * 15 + ["no"] * (len(df) - 15)
    t = full(df, target="label").ml_readiness.dimension("target")
    assert any(d.issue_type == "class_imbalance" and d.points == 65 for d in t.deductions)


def test_single_class_target_is_capped_low():
    df = _clean_frame()
    df["label"] = "yes"
    r = full(df, target="label")
    assert r.ml_readiness.score <= 40 and r.ml_readiness.grade == "F"


def test_missing_target_values_penalised():
    df = _clean_frame()
    df.loc[:99, "label"] = None
    t = full(df, target="label").ml_readiness.dimension("target")
    assert any(d.issue_type == "missing_values" for d in t.deductions)


def test_free_text_column_penalised_and_identifier_noted():
    df = _clean_frame()
    df["comment"] = [f"customer {i} said the delivery was late again" for i in range(len(df))]
    r = full(df, target="label")
    fq = r.ml_readiness.dimension("feature_quality").deductions
    assert any(d.issue_type == "free_text" and d.column == "comment" for d in fq)
    assert any("customer_id" in n and "identifier" in n for n in r.ml_readiness.notes)


def test_scores_are_json_serialisable():
    json.dumps(full(_clean_frame(), target="label").to_dict())
