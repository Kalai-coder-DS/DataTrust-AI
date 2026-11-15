"""
datatrust/quality/outliers.py

Statistical outlier detection for numeric columns, using TWO methods:

  1. IQR fence (Tukey):
        Q1, Q3 = 25th and 75th percentiles,  IQR = Q3 - Q1
        outlier if  x < Q1 - 1.5*IQR   or   x > Q3 + 1.5*IQR

  2. Modified z-score (Iglewicz & Hoaglin, 1993):
        MAD = median(|x - median(x)|)
        z_mod = 0.6745 * (x - median) / MAD
        outlier if |z_mod| > 3.5
     Uses the median and MAD instead of mean and standard deviation, so the
     outliers themselves can't distort the yardstick used to find them
     (a few huge values inflate the std-dev and hide each other - 'masking').

Skewed data: money-like columns (income, charges, prices) are usually right-skewed,
with a long natural tail of big values. On the raw scale those look like outliers
even though they are normal for that kind of data. So if a column is strictly
positive and strongly skewed (skewness > 1), detection runs on log(x), where
'10x the median' becomes '+2.3' instead of a huge jump. Fences are converted back
to the original units for the report.

A value is reported only if BOTH methods agree (consensus). IQR alone flags many
legitimate values in naturally skewed data like income; requiring agreement keeps
false alarms low. If MAD is 0 (over half the values identical), only IQR is used.

Outliers are NOT errors - a very rich customer is real. That's why severity
starts at LOW and the advice is to investigate, not delete.

Values already reported as impossible by the validity check (e.g. age = 210) are
excluded first, so the same cell is never reported twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from datatrust.profiling import DatasetProfile, SemanticType
from datatrust.quality.base import QualityCheck
from datatrust.quality.models import IssueType, QualityIssue, pct, severity_from_pct
from datatrust.quality.rules import to_numeric, usable_values, valid_range

_MAD_SCALE = 0.6745  # makes MAD comparable to a standard deviation for normal data
_MIN_DISTINCT = 10   # fewer distinct values = discrete codes (e.g. ratings), not a distribution
_SKEW_THRESHOLD = 1.0  # |skewness| above 1 is conventionally "highly skewed"


class OutlierCheck(QualityCheck):
    name = "outliers"

    def run(self, df: pd.DataFrame, profile: DatasetProfile) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        for col in profile.columns:
            if col.semantic_type != SemanticType.NUMERIC:
                continue
            numbers = self._clean_numbers(df[col.name], col.name)
            if len(numbers) < self.settings.min_rows_for_outliers or numbers.nunique() < _MIN_DISTINCT:
                continue
            issue = self._detect(col.name, numbers, len(df), valid_range(col.name, self.settings))
            if issue:
                issues.append(issue)
        return issues

    def _clean_numbers(self, series: pd.Series, name: str) -> pd.Series:
        numbers = to_numeric(usable_values(series, self.settings)).dropna()
        rng = valid_range(name, self.settings)
        if rng is not None:
            numbers = numbers[(numbers >= rng[0]) & (numbers <= rng[1])]
        return numbers

    def _detect(self, name: str, x: pd.Series, n: int,
                rng: tuple[float, float] | None = None) -> QualityIssue | None:
        s = self.settings
        original = x
        log_scale = bool(x.min() > 0 and x.skew() > _SKEW_THRESHOLD)
        if log_scale:
            x = np.log(x)
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - s.iqr_multiplier * iqr, q3 + s.iqr_multiplier * iqr
        iqr_flag = (x < lower) | (x > upper)

        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        if mad > 0:
            z_mod = _MAD_SCALE * (x - median) / mad
            z_flag = z_mod.abs() > s.modified_z_threshold
            flagged = iqr_flag & z_flag
            method = "consensus (IQR and modified z-score)"
        elif iqr > 0:
            z_flag = pd.Series(False, index=x.index)
            flagged = iqr_flag
            method = "IQR only (MAD is 0)"
        else:
            return None  # no spread at all; nothing can be an outlier

        count = int(flagged.sum())
        if not count:
            return None

        # Report fences and median in the column's original units.
        if log_scale:
            lower, upper, median = float(np.exp(lower)), float(np.exp(upper)), float(np.exp(median))
            method += ", on log scale (right-skewed data)"
        if rng is not None:
            # A fence below a column's valid minimum (e.g. negative money) is meaningless; clip it.
            lower, upper = max(lower, rng[0]), min(upper, rng[1])

        p = pct(count, n)
        extreme = original[flagged].sort_values(key=lambda v: (v - median).abs(), ascending=False)
        return QualityIssue(
            check=self.name,
            issue_type=IssueType.OUTLIERS,
            severity=severity_from_pct(p, s.severity_thresholds),
            column=name,
            affected_rows=count,
            affected_pct=p,
            message=(f"{count} outlier(s) in '{name}' outside the typical range "
                     f"[{lower:,.2f}, {upper:,.2f}] (median {median:,.2f})."),
            recommendation="Investigate whether these are real extremes or entry errors. If real, "
                           "consider capping (winsorising) or a log transform before modelling.",
            examples=[float(v) for v in extreme.head(s.max_examples)],
            details={
                "method": method,
                "log_scale": log_scale,
                "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                "lower_fence": float(lower), "upper_fence": float(upper),
                "median": median, "mad": mad,  # q1/q3/iqr/mad are on the log scale when log_scale is True
                "iqr_flagged": int(iqr_flag.sum()), "modified_z_flagged": int(z_flag.sum()),
                "outlier_row_indices": [int(i) for i in x.index[flagged]],
            },
        )
