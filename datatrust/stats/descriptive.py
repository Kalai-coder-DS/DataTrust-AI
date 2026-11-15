"""
datatrust/stats/descriptive.py

Descriptive statistics per column, chosen by semantic type:

  NUMERIC                 -> centre, spread, quartiles, shape (skewness, kurtosis)
  CATEGORICAL / BOOLEAN   -> category counts, dominance, rare categories, entropy
  DATETIME                -> earliest, latest, span

Statistics are computed on VALID values only: placeholders ('Unknown'),
unparseable text ('abc') and impossible values (age 210) are excluded and
counted in `excluded`. Otherwise a single age of 210 would drag the mean and
standard deviation, and the summary would describe errors, not the data.
Phase 2 already reports those cells as issues.

Skewness is reported three ways on purpose:
  - moment skewness (the textbook one) reacts strongly to a few extreme values;
  - quartile (Bowley) skewness = (Q3 + Q1 - 2*median) / (Q3 - Q1) uses only the
    middle 50% of data, so outliers can't move it (but it is noisy in small samples);
  - skewness without outliers: recomputed after dropping values outside the
    IQR fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
If the data is skewed but the skew disappears once outliers are removed, the
bulk of the data is symmetric and a few extreme values cause the skew. That is a
direct test of the claim, rather than an inference from quartiles.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.profiling import ColumnProfile
from datatrust.quality.models import to_python
from datatrust.quality.rules import to_datetime, to_numeric, usable_values, valid_range
from datatrust.stats.models import CategoricalSummary, DatetimeSummary, NumericSummary

_DISCRETE_MAX_DISTINCT = 10


def valid_numbers(series: pd.Series, name: str, settings: Settings = DEFAULT_SETTINGS) -> pd.Series:
    """Numeric values that are present, parseable and inside the column's valid range."""
    numbers = to_numeric(usable_values(series, settings)).dropna()
    rng = valid_range(name, settings)
    if rng is not None:
        numbers = numbers[(numbers >= rng[0]) & (numbers <= rng[1])]
    return numbers


def summarize_numeric(series: pd.Series, col: ColumnProfile,
                      settings: Settings = DEFAULT_SETTINGS) -> NumericSummary | None:
    x = valid_numbers(series, col.name, settings)
    if x.empty:
        return None
    n = len(x)
    q1, median, q3 = (float(v) for v in np.percentile(x, [25, 50, 75]))
    iqr = q3 - q1
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0       # sample standard deviation
    skew = _safe(x.skew()) if n > 2 else 0.0            # pandas: adjusted Fisher-Pearson
    kurt = _safe(x.kurt()) if n > 3 else 0.0            # pandas: excess kurtosis
    bowley = (q3 + q1 - 2 * median) / iqr if iqr > 0 else 0.0
    inside = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    skew_trimmed = _safe(inside.skew()) if len(inside) > 2 else skew

    return NumericSummary(
        column=col.name,
        count=n,
        excluded=col.non_null_count - n,
        mean=round(mean, 4),
        median=round(median, 4),
        std=round(std, 4),
        min=round(float(x.min()), 4),
        q1=round(q1, 4),
        q3=round(q3, 4),
        max=round(float(x.max()), 4),
        iqr=round(iqr, 4),
        cv=round(std / abs(mean), 4) if mean != 0 else None,
        skewness=round(skew, 4),
        quartile_skewness=round(bowley, 4),
        skewness_without_outliers=round(skew_trimmed, 4),
        kurtosis=round(kurt, 4),
        zeros_pct=round(float((x == 0).mean() * 100), 2),
        negatives_pct=round(float((x < 0).mean() * 100), 2),
        shape=describe_shape(skew, skew_trimmed, kurt, int(x.nunique())),
    )


def describe_shape(skew: float, skew_without_outliers: float, kurt: float, n_distinct: int) -> str:
    """Plain-language distribution shape, using common rule-of-thumb thresholds."""
    if n_distinct <= _DISCRETE_MAX_DISTINCT:
        return f"discrete ({n_distinct} distinct values)"
    if abs(skew) < 0.5:
        shape = "approximately symmetric"
    else:
        side = "right" if skew > 0 else "left"
        level = "moderately" if abs(skew) < 1 else "highly"
        shape = f"{level} {side}-skewed"
        if abs(skew_without_outliers) < 0.5:
            shape += " (caused by extreme values; symmetric once outliers are removed)"
    if kurt > 3:
        shape += ", heavy-tailed"
    return shape


def summarize_categorical(series: pd.Series, col: ColumnProfile,
                          settings: Settings = DEFAULT_SETTINGS) -> CategoricalSummary | None:
    values = usable_values(series, settings)
    if values.empty:
        return None
    counts = values.astype(str).value_counts()
    n, k = int(counts.sum()), len(counts)
    shares = counts / n
    entropy = float(-(shares * np.log(shares)).sum())
    rare = int((shares * 100 < settings.rare_category_pct).sum())
    return CategoricalSummary(
        column=col.name,
        count=n,
        n_categories=k,
        top_value=to_python(counts.index[0]),
        top_pct=round(float(shares.iloc[0] * 100), 2),
        rare_categories=rare,
        entropy_norm=round(entropy / math.log(k), 4) if k > 1 else 0.0,
        value_counts={str(key): int(v) for key, v in counts.head(10).items()},
    )


def summarize_datetime(series: pd.Series, col: ColumnProfile,
                       settings: Settings = DEFAULT_SETTINGS) -> DatetimeSummary | None:
    values = usable_values(series, settings)
    dates = to_datetime(values).dropna()
    if dates.empty:
        return None
    lo, hi = dates.min(), dates.max()
    return DatetimeSummary(
        column=col.name,
        count=len(dates),
        excluded=col.non_null_count - len(dates),
        min=lo.isoformat(),
        max=hi.isoformat(),
        span_days=int((hi - lo).days),
    )


def _safe(value) -> float:
    """pandas returns NaN for constant columns; report 0 instead."""
    v = float(value)
    return 0.0 if math.isnan(v) else v
