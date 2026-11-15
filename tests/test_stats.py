"""
tests/test_stats.py

Unit tests for Phase 3 statistics. Wherever possible the expected answer can be
worked out by hand (e.g. the mean of 1..5 is 3), so a failure points to a real bug.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from datatrust.profiling import DataProfiler
from datatrust.stats import StatisticsAnalyzer
from datatrust.stats.associations import correlation_ratio, cramers_v, find_associations
from datatrust.stats.descriptive import (
    describe_shape,
    summarize_categorical,
    summarize_datetime,
    summarize_numeric,
)
from datatrust.stats.models import strength_label
from datatrust.stats.target import summarize_target


def numeric(values, name="x"):
    df = pd.DataFrame({name: values})
    col = DataProfiler().profile(df).get_column(name)
    return summarize_numeric(df[name], col)


def categorical(values, name="c"):
    df = pd.DataFrame({name: values})
    col = DataProfiler().profile(df).get_column(name)
    return summarize_categorical(df[name], col)


# ------------------------------------------------------------- numeric
def test_numeric_summary_hand_computed():
    s = numeric([1, 2, 3, 4, 5] * 3)   # repeated so it isn't "discrete"-labelled noise
    assert s.mean == 3 and s.median == 3
    assert (s.min, s.q1, s.q3, s.max, s.iqr) == (1, 2, 4, 5, 2)
    assert s.std == pytest.approx(1.4639, abs=1e-4)   # sample std of 1..5 repeated 3 times
    assert s.skewness == 0 and s.quartile_skewness == 0


def test_invalid_values_excluded_and_counted():
    ages = [str(a) for a in range(20, 60)] + ["abc", "210", "-5"]
    s = numeric(ages, name="age")
    assert s.count == 40
    assert s.excluded == 3
    assert s.max == 59  # 210 did not leak into the statistics


def test_placeholders_excluded():
    s = numeric([10.0, 12.0, -999.0, 11.0, 13.0, 9.0, 10.5, 11.5, 12.5, 9.5, 10.2])
    assert s.count == 10 and s.excluded == 1
    assert s.min == 9.0


def test_cv_is_none_when_mean_is_zero():
    assert numeric([-2.0, -1.0, 0.0, 1.0, 2.0] * 3).cv is None


def test_zeros_and_negatives_pct():
    s = numeric([0, 0, -1, 1, 2, 3, 4, 5, 6, 7])
    assert s.zeros_pct == 20.0 and s.negatives_pct == 10.0


@pytest.mark.parametrize("skew, trimmed, kurt, distinct, expected", [
    (0.1, 0.1, 0.0, 100, "approximately symmetric"),
    (0.7, 0.6, 0.0, 100, "moderately right-skewed"),
    (-1.5, -1.2, 0.0, 100, "highly left-skewed"),
    (2.0, 0.1, 0.0, 100, "highly right-skewed (caused by extreme values; symmetric once outliers are removed)"),
    (0.2, 0.2, 5.0, 100, "approximately symmetric, heavy-tailed"),
    (1.0, 1.0, 0.0, 6, "discrete (6 distinct values)"),
])
def test_describe_shape(skew, trimmed, kurt, distinct, expected):
    assert describe_shape(skew, trimmed, kurt, distinct) == expected


def test_lognormal_data_is_right_skewed():
    rng = np.random.default_rng(0)
    s = numeric(rng.lognormal(9, 0.9, 2000))
    assert "right-skewed" in s.shape
    assert "caused by extreme values" not in s.shape  # genuinely skewed throughout
    assert s.mean > s.median  # long right tail pulls the mean up


def test_quartile_skewness_ignores_extreme_values():
    """Symmetric bulk + a few huge values: moment skewness explodes, Bowley stays ~0."""
    rng = np.random.default_rng(1)
    values = list(rng.normal(100, 10, 2000)) + [10_000.0] * 3
    s = numeric(values)
    assert s.skewness > 5
    assert abs(s.quartile_skewness) < 0.05
    assert abs(s.skewness_without_outliers) < 0.5
    assert "caused by extreme values" in s.shape


# --------------------------------------------------------- categorical
def test_categorical_counts_and_entropy():
    s = categorical(["a", "b", "c", "d"] * 25)
    assert s.n_categories == 4
    assert s.top_pct == 25.0
    assert s.entropy_norm == pytest.approx(1.0)


def test_single_category_has_zero_entropy():
    assert categorical(["India"] * 30).entropy_norm == 0.0


def test_rare_categories():
    s = categorical(["common"] * 995 + ["rare1"] * 3 + ["rare2"] * 2)
    assert s.rare_categories == 2  # each below 1% of rows


def test_placeholders_not_counted_as_category():
    s = categorical(["Male", "Female", "Unknown"] * 10)
    assert s.n_categories == 2


# ------------------------------------------------------------ datetime
def test_datetime_span_and_exclusions():
    dates = pd.date_range("2024-01-01", periods=30).strftime("%Y-%m-%d").tolist() + ["2023-02-30"]
    df = pd.DataFrame({"d": dates})
    s = summarize_datetime(df["d"], DataProfiler().profile(df).get_column("d"))
    assert s.min.startswith("2024-01-01") and s.max.startswith("2024-01-30")
    assert s.span_days == 29
    assert s.excluded == 1


# -------------------------------------------------------- associations
def test_cramers_v_identical_and_independent():
    rng = np.random.default_rng(2)
    a = pd.Series(rng.choice(["x", "y", "z"], 1000))
    assert cramers_v(a, a.copy()) > 0.95
    b = pd.Series(rng.choice(["p", "q"], 1000))
    assert cramers_v(a, b) < 0.1


def test_correlation_ratio_extremes():
    groups = pd.Series(["a"] * 50 + ["b"] * 50)
    separated = pd.Series([10.0] * 50 + [20.0] * 50)   # group explains everything
    assert correlation_ratio(separated, groups) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    unrelated = pd.Series(rng.normal(0, 1, 100))
    assert correlation_ratio(unrelated, groups) < 0.3


def _assocs(df):
    return find_associations(df, DataProfiler().profile(df))


def test_spearman_captures_curved_monotonic_relationship():
    x = np.linspace(1, 10, 200)
    assoc = _assocs(pd.DataFrame({"x": x, "y": np.exp(x)}))[0]
    assert assoc.method == "spearman"
    assert assoc.value == pytest.approx(1.0)
    assert assoc.details["pearson"] < 0.8  # Pearson underestimates a curved relationship


def test_redundant_numeric_pair_flagged():
    rng = np.random.default_rng(4)
    x = rng.normal(50, 10, 300)
    assoc = _assocs(pd.DataFrame({"length_cm": x, "length_mm": x * 10 + rng.normal(0, 1, 300)}))[0]
    assert assoc.redundant is True
    assert assoc.strength == "very strong"


def test_outliers_break_pearson_but_not_spearman():
    """Mirrors the dirty sample file: a few huge values destroy the linear correlation."""
    rng = np.random.default_rng(5)
    x = rng.normal(100, 15, 500)
    y = x * 2 + rng.normal(0, 10, 500)
    y[:5] = 1e6
    assoc = _assocs(pd.DataFrame({"x": x, "y": y}))[0]
    assert assoc.value > 0.8
    assert abs(assoc.details["pearson"]) < 0.3


def test_redundant_label_and_mixed_pairs_carprice_style():
    """Found in the real CarPrice data: fuelsystem 'idi' <=> diesel, and compression ratio
    ~21-23 for diesel vs ~7-10 for gas. Both pairs must be flagged as redundant."""
    rng = np.random.default_rng(10)
    n = 205
    fueltype = rng.choice(["gas", "diesel"], n, p=[0.9, 0.1])
    fuelsystem = np.where(fueltype == "diesel", "idi", rng.choice(["mpfi", "2bbl", "spdi"], n))
    compression = np.where(fueltype == "diesel", rng.uniform(21, 23, n), rng.uniform(7, 10, n))
    df = pd.DataFrame({"fueltype": fueltype, "fuelsystem": fuelsystem, "compressionratio": compression})
    report = StatisticsAnalyzer().analyze(df)
    assert report.association("fueltype", "fuelsystem", "cramers_v").redundant is True
    assert report.association("compressionratio", "fueltype", "correlation_ratio").redundant is True


def test_moderate_label_association_is_not_redundant():
    rng = np.random.default_rng(11)
    a = rng.choice(["x", "y"], 500)
    b = np.where(rng.uniform(size=500) < 0.8, a, rng.choice(["x", "y"], 500))  # agrees ~90%
    report = StatisticsAnalyzer().analyze(pd.DataFrame({"a": a, "b": b}))
    assoc = report.association("a", "b")
    assert assoc is not None and assoc.value < 0.9 and assoc.redundant is False


def test_target_association_is_signal_not_redundancy():
    rng = np.random.default_rng(9)
    size = rng.normal(0, 1, 300)
    df = pd.DataFrame({"weight": 2500 + 500 * size, "price": 10000 + 3000 * size + rng.normal(0, 100, 300)})
    no_target = StatisticsAnalyzer().analyze(df).association("weight", "price")
    with_target = StatisticsAnalyzer().analyze(df, target="price").association("weight", "price")
    assert no_target.redundant is True            # two features: one is redundant
    assert with_target.redundant is False          # feature vs target: predictive signal
    assert with_target.involves_target is True


def test_unrelated_columns_not_reported():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300),
                       "g": rng.choice(["x", "y"], 300)})
    assert _assocs(df) == []


@pytest.mark.parametrize("value, label", [(0.95, "very strong"), (-0.75, "strong"),
                                          (0.45, "moderate"), (0.1, "weak")])
def test_strength_labels(value, label):
    assert strength_label(value) == label


# -------------------------------------------------------------- target
def _target(values, name="label"):
    df = pd.DataFrame({name: values, "x": range(len(values))})
    return summarize_target(df, DataProfiler().profile(df), name)


@pytest.mark.parametrize("counts, ratio, balance", [
    ({"yes": 50, "no": 50}, 1.0, "balanced"),
    ({"yes": 70, "no": 30}, 2.33, "moderate"),
    ({"yes": 90, "no": 10}, 9.0, "high"),
    ({"yes": 97, "no": 3}, 32.33, "severe"),
])
def test_class_balance(counts, ratio, balance):
    values = [k for k, n in counts.items() for _ in range(n)]
    t = _target(values)
    assert t.task == "classification"
    assert t.imbalance_ratio == ratio
    assert t.balance == balance
    if len(set(counts.values())) > 1:  # with a 50/50 tie either class is "the minority"
        assert t.minority_class == min(counts, key=counts.get)


def test_numeric_target_with_many_values_is_regression():
    rng = np.random.default_rng(7)
    t = _target(rng.normal(1000, 200, 200).round(2), name="price")
    assert t.task == "regression"
    assert t.class_counts == {}


def test_target_must_exist():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="not found"):
        summarize_target(df, DataProfiler().profile(df), "missing")


def test_identifier_cannot_be_target():
    df = pd.DataFrame({"customer_id": [f"C{i}" for i in range(50)], "x": range(50)})
    with pytest.raises(ValueError, match="identifier"):
        summarize_target(df, DataProfiler().profile(df), "customer_id")


# ------------------------------------------------------------ analyzer
def test_analyzer_rejects_non_dataframe():
    with pytest.raises(TypeError):
        StatisticsAnalyzer().analyze([1, 2, 3])


def test_analyzer_does_not_modify_input():
    df = pd.DataFrame({"age": ["25", "abc", "40"] * 5, "city": ["Pune", "pune", None] * 5})
    before = df.copy()
    StatisticsAnalyzer().analyze(df)
    pd.testing.assert_frame_equal(df, before)


def test_report_is_json_serialisable():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50),
                       "g": rng.choice(["a", "b"], 50),
                       "d": pd.date_range("2024-01-01", periods=50).strftime("%Y-%m-%d")})
    json.dumps(StatisticsAnalyzer().analyze(df, target="g").to_dict())
