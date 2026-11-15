"""
tests/test_profiler.py

Unit tests for datatrust.profiling.profiler.DataProfiler.
Small hand-built DataFrames make expected results obvious; the generated sample
files check behaviour on realistic data against the injected-defect manifest.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from datatrust.ingestion import CSVLoader
from datatrust.profiling import DataProfiler, SemanticType


@pytest.fixture
def profiler() -> DataProfiler:
    return DataProfiler()


def infer(profiler: DataProfiler, values, name: str = "col"):
    return profiler.infer_semantic_type(pd.Series(values), name)


# ------------------------------------------------------------ input validation
def test_rejects_non_dataframe(profiler):
    with pytest.raises(TypeError):
        profiler.profile([1, 2, 3])


# ------------------------------------------------------------- dataset level
def test_dataset_level_counts(profiler):
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": ["x", "y", None, None]})
    p = profiler.profile(df)
    assert (p.n_rows, p.n_columns, p.total_cells) == (4, 2, 8)
    assert p.total_missing_cells == 3
    assert p.missing_cells_pct == 37.5
    assert p.memory_bytes > 0


def test_column_level_counts(profiler):
    df = pd.DataFrame({"city": ["Pune", "Pune", "Delhi", None]})
    col = profiler.profile(df).get_column("city")
    assert col.null_count == 1
    assert col.null_pct == 25.0
    assert col.unique_count == 2
    assert col.unique_pct == pytest.approx(66.67)
    assert set(col.sample_values) == {"Pune", "Delhi"}


def test_get_column_unknown_raises(profiler):
    p = profiler.profile(pd.DataFrame({"a": [1]}))
    with pytest.raises(KeyError):
        p.get_column("missing")


def test_profile_is_json_serialisable(profiler):
    df = pd.DataFrame({
        "n": np.array([1, 2, 3], dtype=np.int64),
        "f": [1.5, np.nan, 2.5],
        "d": pd.to_datetime(["2024-01-01", "2024-01-02", None]),
        "b": [True, False, True],
    })
    json.dumps(profiler.profile(df).to_dict())


# ---------------------------------------------------------- semantic types
@pytest.mark.parametrize(
    "values, name, expected",
    [
        ([None, None, None], "c", SemanticType.EMPTY),
        ([True, False, True], "c", SemanticType.BOOLEAN),
        ([0, 1, 1, 0], "churned", SemanticType.BOOLEAN),
        (["Yes", "No", "yes", "No"], "active", SemanticType.BOOLEAN),
        ([1.5, 2.7, 3.1, 4.4], "price", SemanticType.NUMERIC),
        ([2019, 2020, 2021, 2020, 2019], "year", SemanticType.NUMERIC),
        (["Basic", "Premium", "Basic", "Standard"] * 10, "plan", SemanticType.CATEGORICAL),
        (["C001", "C002", "C003", "C004"], "code", SemanticType.IDENTIFIER),
        ([101, 102, 103, 104], "customer_id", SemanticType.IDENTIFIER),
        ([101, 102, 103, 104], "CustomerID", SemanticType.IDENTIFIER),
    ],
)
def test_semantic_type_inference(profiler, values, name, expected):
    stype, _, _ = infer(profiler, values, name)
    assert stype == expected


def test_all_unique_small_integers_without_id_name_are_numeric(profiler):
    """Five people with five different ages is NOT an ID column."""
    stype, _, _ = infer(profiler, [23, 45, 31, 52, 38], "age")
    assert stype == SemanticType.NUMERIC


def test_sequential_row_numbers_are_identifier(profiler):
    stype, _, _ = infer(profiler, list(range(1, 51)), "sno")
    assert stype == SemanticType.IDENTIFIER


def test_numeric_stored_as_text_is_flagged(profiler):
    values = [str(v) for v in range(20, 60)] + ["abc"]  # 40 numbers + 1 bad value
    stype, as_text, pct = infer(profiler, values, "age")
    assert stype == SemanticType.NUMERIC
    assert as_text is True
    assert pct == pytest.approx(40 / 41 * 100, abs=0.01)


def test_mostly_text_column_is_not_numeric(profiler):
    values = ["abc", "def", "1", "ghi", "jkl"] * 4
    stype, _, _ = infer(profiler, values, "c")
    assert stype != SemanticType.NUMERIC


def test_dates_stored_as_text(profiler):
    dates = pd.date_range("2024-01-01", periods=30).strftime("%Y-%m-%d").tolist()
    stype, as_text, pct = infer(profiler, dates, "signup")
    assert stype == SemanticType.DATETIME
    assert as_text is True
    assert pct == 100.0


def test_impossible_dates_lower_parse_rate(profiler):
    dates = pd.date_range("2024-01-01", periods=29).strftime("%Y-%m-%d").tolist() + ["2023-02-30"]
    stype, _, pct = infer(profiler, dates, "signup")
    assert stype == SemanticType.DATETIME
    assert pct < 100.0


def test_free_text_is_text(profiler):
    sentences = [f"Customer {i} reported a delay in delivery and asked for a refund" for i in range(80)]
    stype, _, _ = infer(profiler, sentences, "feedback")
    assert stype == SemanticType.TEXT


def test_high_cardinality_short_strings_are_text(profiler):
    # Mostly unique (not 100%), not ID-named, too many values for categorical.
    values = [f"name{i}" for i in range(200)] + ["name1"] * 5
    stype, _, _ = infer(profiler, values, "full_name")
    assert stype == SemanticType.TEXT


# ---------------------------------------------------------- realistic data
@pytest.fixture(scope="module")
def dirty_profile(sample_dir):
    df = CSVLoader().load(sample_dir / "customers_dirty.csv").df
    return DataProfiler().profile(df)


def test_dirty_missing_cells_match_manifest(dirty_profile, dirty_manifest):
    assert dirty_profile.total_missing_cells == sum(dirty_manifest["missing"].values())
    assert dirty_profile.get_column("annual_income").null_count == dirty_manifest["missing"]["annual_income"]


def test_dirty_id_with_duplicates_still_identifier(dirty_profile, dirty_manifest):
    col = dirty_profile.get_column("customer_id")
    assert col.semantic_type == SemanticType.IDENTIFIER
    assert col.unique_count == dirty_manifest["total_rows"] - dirty_manifest["exact_duplicate_rows"]


def test_dirty_age_is_numeric_stored_as_text(dirty_profile):
    col = dirty_profile.get_column("age")
    assert col.semantic_type == SemanticType.NUMERIC
    assert col.stored_as_text is True
    assert col.parse_success_pct < 100


def test_dirty_expected_types(dirty_profile):
    expected = {
        "gender": SemanticType.CATEGORICAL,
        "city": SemanticType.CATEGORICAL,
        "plan": SemanticType.CATEGORICAL,
        "annual_income": SemanticType.NUMERIC,
        "monthly_spend": SemanticType.NUMERIC,
        "signup_date": SemanticType.DATETIME,
        "is_active": SemanticType.BOOLEAN,
        "churned": SemanticType.BOOLEAN,
        "country": SemanticType.CATEGORICAL,
    }
    for name, stype in expected.items():
        assert dirty_profile.get_column(name).semantic_type == stype, name


def test_constant_column_has_one_unique_value(dirty_profile):
    assert dirty_profile.get_column("country").unique_count == 1
