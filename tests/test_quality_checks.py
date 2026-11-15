"""
tests/test_quality_checks.py

Unit tests for each Phase 2 quality check, using small hand-built DataFrames
where the correct answer is obvious by reading the test.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from datatrust.profiling import DataProfiler
from datatrust.quality import IssueType, QualityAnalyzer, Severity
from datatrust.quality.base import QualityCheck
from datatrust.quality.columns import LowInformationCheck
from datatrust.quality.consistency import ConsistencyCheck
from datatrust.quality.duplicates import DuplicateCheck
from datatrust.quality.missing import MissingValueCheck
from datatrust.quality.models import severity_from_pct
from datatrust.quality.outliers import OutlierCheck
from datatrust.quality.rules import name_tokens, valid_range
from datatrust.quality.validity import InvalidValueCheck
from datatrust.config import DEFAULT_SETTINGS


def run(check_cls, df: pd.DataFrame):
    return check_cls().run(df, DataProfiler().profile(df))


def only(issues, issue_type):
    found = [i for i in issues if i.issue_type == issue_type]
    assert len(found) == 1, f"expected exactly one {issue_type}, got {found}"
    return found[0]


# ------------------------------------------------------------------- helpers
@pytest.mark.parametrize("p, expected", [
    (0.0, Severity.LOW), (4.99, Severity.LOW), (5.0, Severity.MEDIUM),
    (19.9, Severity.MEDIUM), (20.0, Severity.HIGH), (50.0, Severity.CRITICAL), (100, Severity.CRITICAL),
])
def test_severity_bands(p, expected):
    assert severity_from_pct(p, (5.0, 20.0, 50.0)) == expected


def test_severity_floor():
    assert severity_from_pct(1.0, (5.0, 20.0, 50.0), floor=Severity.MEDIUM) == Severity.MEDIUM


def test_name_tokens_do_not_match_substrings():
    assert "age" not in name_tokens("mileage")
    assert "age" not in name_tokens("percentage")
    assert "age" in name_tokens("Customer Age")


@pytest.mark.parametrize("name, expected", [
    ("age", (0.0, 120.0)),
    ("discount_pct", (0.0, 100.0)),
    ("annual_income", (0.0, np.inf)),
    ("temperature", None),
])
def test_valid_range_from_column_name(name, expected):
    assert valid_range(name, DEFAULT_SETTINGS) == expected


# ------------------------------------------------------------------- missing
def test_missing_values_counted():
    df = pd.DataFrame({"score": [1.0, None, 3.0, None, 5.0] * 4})
    issue = only(run(MissingValueCheck, df), IssueType.MISSING)
    assert issue.affected_rows == 8
    assert issue.affected_pct == 40.0
    assert issue.severity == Severity.HIGH


def test_empty_column_is_critical_and_not_double_reported():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    issues = run(MissingValueCheck, df)
    assert only(issues, IssueType.EMPTY_COLUMN).severity == Severity.CRITICAL
    assert not [i for i in issues if i.issue_type == IssueType.MISSING]


def test_disguised_missing_text_tokens_case_insensitive():
    df = pd.DataFrame({"gender": ["Male", "Female", "UNKNOWN", "?", " - ", "Male"] * 5})
    issue = only(run(MissingValueCheck, df), IssueType.DISGUISED_MISSING)
    assert issue.affected_rows == 15
    assert issue.severity.rank >= Severity.MEDIUM.rank


def test_disguised_missing_numeric_sentinels():
    df = pd.DataFrame({"reading": [12.5, -999, 13.1, -9999, 14.0] * 4})
    assert only(run(MissingValueCheck, df), IssueType.DISGUISED_MISSING).affected_rows == 8


def test_no_missing_issues_on_complete_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert run(MissingValueCheck, df) == []


# ---------------------------------------------------------------- duplicates
def test_exact_duplicate_rows():
    df = pd.DataFrame({"a": [1, 2, 1, 1], "b": ["x", "y", "x", "x"]})
    issue = only(run(DuplicateCheck, df), IssueType.DUPLICATE_ROWS)
    assert issue.affected_rows == 2  # first occurrence is the original
    assert issue.examples == [2, 3]


def _ids_with_one_repeat(repeat_city: str) -> pd.DataFrame:
    """30 unique customers, then C0 appears again (97% unique: still an ID column)."""
    ids = [f"C{i}" for i in range(30)] + ["C0"]
    cities = ["Pune"] * 30 + [repeat_city]
    return pd.DataFrame({"customer_id": ids, "city": cities})


def test_duplicate_ids_with_conflicting_data_are_high():
    df = _ids_with_one_repeat("Chennai")  # C0 is in Pune AND Chennai: conflicting
    issue = only(run(DuplicateCheck, df), IssueType.DUPLICATE_IDS)
    assert issue.affected_rows == 1
    assert issue.details["conflicting_records"] == 1
    assert issue.severity == Severity.HIGH


def test_duplicate_ids_from_exact_duplicates_are_not_conflicting():
    df = _ids_with_one_repeat("Pune")  # second C0 row is an exact copy
    issue = only(run(DuplicateCheck, df), IssueType.DUPLICATE_IDS)
    assert issue.details["conflicting_records"] == 0


def _insurance_like(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "id": range(1, n + 1),
        "age": rng.integers(18, 65, n),
        "smoker": rng.choice(["yes", "no"], n),
        "charges": rng.lognormal(9, 0.8, n).round(2),  # precise amounts: mostly unique
    })


def test_duplicates_hidden_by_id_column():
    """Real insurance.csv case: same person entered twice, each copy got a new id."""
    df = _insurance_like()
    copies = df.iloc[[3, 10]].copy()
    copies["id"] = [101, 102]
    df = pd.concat([df, copies], ignore_index=True)

    issues = run(DuplicateCheck, df)
    hidden = only(issues, IssueType.DUPLICATES_HIDDEN_BY_ID)
    assert hidden.affected_rows == 2
    assert hidden.examples == [100, 101]
    assert hidden.details["ignored_id_columns"] == ["id"]
    assert not [i for i in issues if i.issue_type == IssueType.DUPLICATE_ROWS]  # ids differ


def test_exact_duplicates_not_double_counted_as_hidden():
    df = _insurance_like()
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # full copy, same id
    issues = run(DuplicateCheck, df)
    assert only(issues, IssueType.DUPLICATE_ROWS).affected_rows == 1
    assert not [i for i in issues if i.issue_type == IssueType.DUPLICATES_HIDDEN_BY_ID]


def test_no_hidden_duplicates_when_columns_are_low_cardinality():
    """With only sex/smoker/region, different people match by chance: not duplicates."""
    rng = np.random.default_rng(8)
    n = 200
    df = pd.DataFrame({
        "id": range(n),
        "sex": rng.choice(["male", "female"], n),
        "smoker": rng.choice(["yes", "no"], n),
        "region": rng.choice(["n", "s", "e", "w"], n),
    })
    issues = run(DuplicateCheck, df)
    assert not [i for i in issues if i.issue_type == IssueType.DUPLICATES_HIDDEN_BY_ID]


# ------------------------------------------------------------------ validity
def test_type_mismatch_in_numeric_column():
    df = pd.DataFrame({"age": [str(a) for a in range(20, 60)] + ["abc", "xyz"]})
    issue = only(run(InvalidValueCheck, df), IssueType.TYPE_MISMATCH)
    assert issue.affected_rows == 2
    assert set(issue.examples) == {"abc", "xyz"}


def test_placeholders_are_not_counted_as_type_mismatch():
    df = pd.DataFrame({"age": [str(a) for a in range(20, 60)] + ["Unknown", "?"]})
    assert not [i for i in run(InvalidValueCheck, df) if i.issue_type == IssueType.TYPE_MISMATCH]


def test_out_of_range_age():
    df = pd.DataFrame({"age": [25, 30, -5, 210, 45, 50, 33, 41, 38, 29]})
    issue = only(run(InvalidValueCheck, df), IssueType.OUT_OF_RANGE)
    assert issue.affected_rows == 2
    assert issue.details["below_min"] == 1 and issue.details["above_max"] == 1


def test_negative_price_is_out_of_range():
    df = pd.DataFrame({"price": [100.0, 250.5, -20.0, 80.0, 99.9]})
    assert only(run(InvalidValueCheck, df), IssueType.OUT_OF_RANGE).affected_rows == 1


def test_no_range_rule_for_unknown_column_names():
    df = pd.DataFrame({"temperature": [-10.5, 3.2, -40.0, 25.0, 12.0]})
    assert run(InvalidValueCheck, df) == []


def test_invalid_dates():
    dates = pd.date_range("2024-01-01", periods=28).strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"signup": dates + ["2023-02-30", "2024-13-01"]})
    issue = only(run(InvalidValueCheck, df), IssueType.INVALID_DATE)
    assert issue.affected_rows == 2


# --------------------------------------------------------------- consistency
def test_inconsistent_case_and_spacing():
    cities = ["Chennai"] * 10 + ["chennai"] * 2 + [" Chennai "] + ["Pune"] * 8 + ["PUNE"]
    issue = only(run(ConsistencyCheck, pd.DataFrame({"city": cities})), IssueType.INCONSISTENT_FORMAT)
    assert issue.affected_rows == 4  # 2 'chennai' + 1 ' Chennai ' + 1 'PUNE'
    assert set(issue.details["variant_groups"]) == {"chennai", "pune"}


def test_consistent_labels_have_no_issue():
    df = pd.DataFrame({"plan": ["Basic", "Premium", "Standard"] * 10})
    assert run(ConsistencyCheck, df) == []


def test_numbers_as_words_carprice_style():
    df = pd.DataFrame({"doornumber": ["two", "four", "four", "two"] * 10})
    issue = only(run(ConsistencyCheck, df), IssueType.NUMBERS_AS_WORDS)
    assert issue.severity == Severity.LOW
    assert issue.examples == ["four", "two"]


# ------------------------------------------------------------------ outliers
def test_outliers_found_by_consensus():
    rng = np.random.default_rng(0)
    values = list(rng.normal(50, 5, 200)) + [500.0, -400.0]
    issue = only(run(OutlierCheck, pd.DataFrame({"score": values})), IssueType.OUTLIERS)
    assert {500.0, -400.0} <= set(issue.examples)
    assert issue.severity == Severity.LOW


def test_very_few_outliers_in_normal_data():
    """Pure normal data can still produce a rare extreme value, so 'zero' is too strict.
    The consensus method should flag well under 1% of values."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"score": rng.normal(50, 5, 1000)})
    flagged = sum(i.affected_rows for i in run(OutlierCheck, df))
    assert flagged / 1000 < 0.01


def test_skewed_money_column_uses_log_scale():
    """Right-skewed data (like medical charges) has a long natural tail; those values
    must not be flooded with outlier flags, but a truly absurd value must still be found."""
    rng = np.random.default_rng(3)
    charges = list(rng.lognormal(9.1, 0.9, 1000)) + [5_000_000.0]
    issue = only(run(OutlierCheck, pd.DataFrame({"charges": charges})), IssueType.OUTLIERS)
    assert issue.details["log_scale"] is True
    assert 5_000_000.0 in issue.examples
    assert issue.affected_rows <= 10  # raw-scale IQR would flag ~80 natural values


def test_outliers_skip_discrete_codes():
    """symboling (-2..3) in CarPrice has 6 distinct values: a rating, not a distribution."""
    df = pd.DataFrame({"symboling": [-2, -1, 0, 1, 2, 3] * 20 + [3] * 2})
    assert run(OutlierCheck, df) == []


def test_impossible_values_excluded_from_outliers():
    """age=210 is reported by the validity check, so it must not ALSO be an outlier."""
    ages = list(range(20, 70)) * 3 + [210]
    issues = run(OutlierCheck, pd.DataFrame({"age": ages}))
    assert all(210 not in i.examples for i in issues)


# ------------------------------------------------------------------- columns
def test_constant_column():
    df = pd.DataFrame({"country": ["India"] * 20, "x": range(20)})
    issue = only(run(LowInformationCheck, df), IssueType.CONSTANT_COLUMN)
    assert issue.column == "country"


def test_near_constant_column():
    df = pd.DataFrame({"flag": ["N"] * 199 + ["Y"]})
    issue = only(run(LowInformationCheck, df), IssueType.NEAR_CONSTANT_COLUMN)
    assert issue.details["dominant_share_pct"] == 99.5


def test_identifier_is_not_low_information():
    df = pd.DataFrame({"customer_id": [f"C{i}" for i in range(50)]})
    assert run(LowInformationCheck, df) == []


# ------------------------------------------------------------------ analyzer
def test_analyzer_rejects_non_dataframe():
    with pytest.raises(TypeError):
        QualityAnalyzer().analyze("not a dataframe")


def test_analyzer_isolates_a_crashing_check():
    class Broken(QualityCheck):
        name = "broken"

        def run(self, df, profile):
            raise RuntimeError("boom")

    df = pd.DataFrame({"country": ["India"] * 5})
    report = QualityAnalyzer(checks=[Broken(), LowInformationCheck()]).analyze(df)
    types = {i.issue_type for i in report.issues}
    assert IssueType.CHECK_FAILED in types          # failure is reported...
    assert IssueType.CONSTANT_COLUMN in types       # ...and other checks still ran


def test_analyzer_does_not_modify_input():
    df = pd.DataFrame({"age": ["25", "abc", "-5"], "city": ["Pune", "pune", None]})
    before = df.copy()
    QualityAnalyzer().analyze(df)
    pd.testing.assert_frame_equal(df, before)


def test_report_sorted_and_serialisable():
    df = pd.DataFrame({"a": [None] * 10, "country": ["India"] * 10})
    report = QualityAnalyzer().analyze(df)
    ranks = [i.severity.rank for i in report.sorted_issues()]
    assert ranks == sorted(ranks, reverse=True)
    json.dumps(report.to_dict())
