"""
datatrust/stats/associations.py

How strongly are pairs of columns related? Three measures, one per pair type:

  numeric  x numeric      Spearman rank correlation (primary) + Pearson (for comparison)
      Pearson measures a straight-line relationship and is pulled around by outliers.
      Spearman correlates the RANKS, so it captures any steadily increasing/decreasing
      relationship and is robust to outliers. A big gap between them is itself a clue
      (curved relationship, or outliers driving the Pearson value).

  label    x label        Cramér's V (bias-corrected, Bergsma 2013), 0..1
      Based on the chi-square test of independence. The correction stops it from
      reporting spurious association when there are many categories and few rows.

  numeric  x label        Correlation ratio (eta), 0..1
      "How much of the numeric column's variation is explained by the groups?"
      eta^2 = between-group sum of squares / total sum of squares.

Why this matters for ML:
  - Two features associated >= 0.9 are near-duplicates (redundant): keep one, or
    the model's coefficients become unstable (multicollinearity). This applies to
    every pair type. Real example from CarPrice: fuelsystem 'idi' is used only by
    diesel cars (Cramér's V 0.985 with fueltype), and compression ratio is almost
    fully explained by fuel type (eta 0.984). Each pair says the same thing twice.
  - Strong association with the target is good news (predictive signal);
    suspiciously perfect association can mean target leakage (Phase 4).
"""
from __future__ import annotations

import itertools
import math

import pandas as pd

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.profiling import DatasetProfile, SemanticType
from datatrust.quality.rules import usable_values
from datatrust.stats.descriptive import valid_numbers
from datatrust.stats.models import Association, strength_label

_MIN_PAIRED_ROWS = 10
_LABEL_TYPES = (SemanticType.CATEGORICAL, SemanticType.BOOLEAN)


def find_associations(df: pd.DataFrame, profile: DatasetProfile,
                      settings: Settings = DEFAULT_SETTINGS) -> list[Association]:
    cap = settings.max_columns_for_associations
    numeric_cols = [c.name for c in profile.columns if c.semantic_type == SemanticType.NUMERIC][:cap]
    label_cols = [c.name for c in profile.columns if c.semantic_type in _LABEL_TYPES][:cap]

    numbers = pd.DataFrame({c: valid_numbers(df[c], c, settings) for c in numeric_cols}, index=df.index)
    labels = pd.DataFrame({c: usable_values(df[c], settings).astype(str) for c in label_cols}, index=df.index)

    found: list[Association] = []
    found += _numeric_pairs(numbers, settings)
    found += _label_pairs(labels, settings)
    found += _numeric_label_pairs(numbers, labels, settings)
    return sorted(found, key=lambda a: -abs(a.value))


# ------------------------------------------------------------ numeric x numeric
def _numeric_pairs(numbers: pd.DataFrame, s: Settings) -> list[Association]:
    if numbers.shape[1] < 2:
        return []
    spearman = numbers.corr(method="spearman", min_periods=_MIN_PAIRED_ROWS)
    pearson = numbers.corr(method="pearson", min_periods=_MIN_PAIRED_ROWS)
    out = []
    for a, b in itertools.combinations(numbers.columns, 2):
        rho, r = spearman.loc[a, b], pearson.loc[a, b]
        if pd.isna(rho) or abs(rho) < s.association_report_threshold:
            continue
        out.append(Association(
            column_a=a, column_b=b, method="spearman",
            value=round(float(rho), 4),
            strength=strength_label(rho),
            redundant=bool(max(abs(rho), abs(r) if not pd.isna(r) else 0) >= s.redundancy_threshold),
            details={"pearson": None if pd.isna(r) else round(float(r), 4)},
        ))
    return out


# ---------------------------------------------------------------- label x label
def cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Bias-corrected Cramér's V for two label columns (rows missing in either are dropped)."""
    paired = pd.DataFrame({"a": a, "b": b}).dropna()
    n = len(paired)
    if n < _MIN_PAIRED_ROWS:
        return None
    table = pd.crosstab(paired["a"], paired["b"]).to_numpy(dtype=float)
    r, k = table.shape
    if r < 2 or k < 2:
        return None
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return math.sqrt(phi2_corr / denom) if denom > 0 else 0.0


def _label_pairs(labels: pd.DataFrame, s: Settings) -> list[Association]:
    out = []
    for a, b in itertools.combinations(labels.columns, 2):
        v = cramers_v(labels[a], labels[b])
        if v is None or v < s.association_report_threshold:
            continue
        out.append(Association(column_a=a, column_b=b, method="cramers_v",
                               value=round(v, 4), strength=strength_label(v),
                               redundant=v >= s.redundancy_threshold))
    return out


# -------------------------------------------------------------- numeric x label
def correlation_ratio(values: pd.Series, groups: pd.Series) -> float | None:
    """eta: share of the numeric column's spread explained by group membership (0..1)."""
    paired = pd.DataFrame({"x": values, "g": groups}).dropna()
    if len(paired) < _MIN_PAIRED_ROWS or paired["g"].nunique() < 2:
        return None
    grand_mean = paired["x"].mean()
    ss_total = float(((paired["x"] - grand_mean) ** 2).sum())
    if ss_total == 0:
        return None
    stats = paired.groupby("g")["x"].agg(["count", "mean"])
    ss_between = float((stats["count"] * (stats["mean"] - grand_mean) ** 2).sum())
    return math.sqrt(ss_between / ss_total)


def _numeric_label_pairs(numbers: pd.DataFrame, labels: pd.DataFrame, s: Settings) -> list[Association]:
    out = []
    for num, lab in itertools.product(numbers.columns, labels.columns):
        eta = correlation_ratio(numbers[num], labels[lab])
        if eta is None or eta < s.association_report_threshold:
            continue
        out.append(Association(column_a=num, column_b=lab, method="correlation_ratio",
                               value=round(eta, 4), strength=strength_label(eta),
                               redundant=eta >= s.redundancy_threshold))
    return out


__all__ = ["cramers_v", "correlation_ratio", "find_associations"]

