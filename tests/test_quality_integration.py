"""
tests/test_quality_integration.py

End-to-end: load the generated datasets and check the detectors find EXACTLY the
defects recorded in dirty_manifest.json - no more, no less - and raise no false
alarms of medium severity or above on the clean dataset.
"""
from __future__ import annotations

import pytest

from datatrust.ingestion import CSVLoader
from datatrust.quality import IssueType, QualityAnalyzer, Severity


def _analyze(path):
    return QualityAnalyzer().analyze(CSVLoader().load(path).df)


@pytest.fixture(scope="module")
def dirty_report(sample_dir):
    return _analyze(sample_dir / "customers_dirty.csv")


@pytest.fixture(scope="module")
def clean_report(sample_dir):
    return _analyze(sample_dir / "customers_clean.csv")


def count(report, issue_type, column=None) -> int:
    issues = report.filter(issue_type, column)
    assert len(issues) <= 1
    return issues[0].affected_rows if issues else 0


def test_missing_matches_manifest(dirty_report, dirty_manifest):
    for column, expected in dirty_manifest["missing"].items():
        assert count(dirty_report, IssueType.MISSING, column) == expected, column


def test_disguised_missing_matches_manifest(dirty_report, dirty_manifest):
    expected = dirty_manifest["disguised_missing"]["gender"]
    issue = dirty_report.filter(IssueType.DISGUISED_MISSING, "gender")[0]
    assert issue.affected_rows == expected["count"]
    assert issue.examples == [expected["value"]]


def test_duplicates_match_manifest(dirty_report, dirty_manifest):
    n = dirty_manifest["exact_duplicate_rows"]
    assert count(dirty_report, IssueType.DUPLICATE_ROWS) == n
    ids = dirty_report.filter(IssueType.DUPLICATE_IDS, "customer_id")[0]
    assert ids.affected_rows == n
    assert ids.details["conflicting_records"] == 0


def test_invalid_values_match_manifest(dirty_report, dirty_manifest):
    inv = dirty_manifest["invalid"]
    assert count(dirty_report, IssueType.TYPE_MISMATCH, "age") == inv["age_non_numeric"]
    rng = dirty_report.filter(IssueType.OUT_OF_RANGE, "age")[0]
    assert rng.details["below_min"] == inv["age_negative"]
    assert rng.details["above_max"] == inv["age_over_120"]
    assert count(dirty_report, IssueType.INVALID_DATE, "signup_date") == inv["signup_date_impossible"]


def test_inconsistent_city_matches_manifest(dirty_report, dirty_manifest):
    assert count(dirty_report, IssueType.INCONSISTENT_FORMAT, "city") == dirty_manifest["inconsistent_case_city"]


def test_constant_column_matches_manifest(dirty_report, dirty_manifest):
    found = {i.column for i in dirty_report.filter(IssueType.CONSTANT_COLUMN)}
    assert found == set(dirty_manifest["constant_columns"])


def test_every_injected_outlier_is_detected(dirty_report, dirty_manifest):
    issue = dirty_report.filter(IssueType.OUTLIERS, "monthly_spend")[0]
    flagged = set(issue.details["outlier_row_indices"])
    assert set(dirty_manifest["outlier_rows"]["monthly_spend"]) == flagged  # all found, no extras


def test_no_unexpected_issue_types_in_dirty_data(dirty_report):
    expected_types = {
        IssueType.MISSING, IssueType.DISGUISED_MISSING, IssueType.DUPLICATE_ROWS,
        IssueType.DUPLICATE_IDS, IssueType.TYPE_MISMATCH, IssueType.OUT_OF_RANGE,
        IssueType.INVALID_DATE, IssueType.INCONSISTENT_FORMAT, IssueType.CONSTANT_COLUMN,
        IssueType.OUTLIERS,
    }
    assert {i.issue_type for i in dirty_report.issues} == expected_types


def test_clean_data_has_no_real_problems(clean_report):
    """Only natural statistical outliers (LOW) are acceptable in the clean file."""
    serious = [i for i in clean_report.issues if i.severity.rank >= Severity.MEDIUM.rank]
    assert serious == []
    assert {i.issue_type for i in clean_report.issues} <= {IssueType.OUTLIERS}
