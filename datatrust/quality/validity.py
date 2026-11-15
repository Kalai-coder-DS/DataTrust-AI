"""
datatrust/quality/validity.py

Invalid-value detection. A value is invalid when it CANNOT be correct:

  1. Type mismatch  - 'abc' in a column that is otherwise numeric.
  2. Invalid date   - '2023-02-30' in a date column (no such day).
  3. Out of range   - age = 210, age = -5, price = -100.

Range rules come from column names (config.range_rules / non_negative_tokens).
This is deliberately simple and explainable; Phase 6 can let users define
their own rules per column through the web UI.

Placeholders like 'Unknown' are excluded here: the missing-value check already
reports them, and counting the same cell twice would inflate the problem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from datatrust.profiling import DatasetProfile, SemanticType
from datatrust.quality.base import QualityCheck
from datatrust.quality.models import IssueType, QualityIssue, Severity, pct, severity_from_pct, to_python
from datatrust.quality.rules import to_datetime, to_numeric, usable_values, valid_range


class InvalidValueCheck(QualityCheck):
    name = "validity"

    def run(self, df: pd.DataFrame, profile: DatasetProfile) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        for col in profile.columns:
            values = usable_values(df[col.name], self.settings)
            if values.empty:
                continue
            if col.semantic_type == SemanticType.NUMERIC:
                issues += self._check_numeric(col.name, values, len(df))
            elif col.semantic_type == SemanticType.DATETIME:
                issues += self._check_dates(col.name, values, len(df))
        return issues

    # ---------------------------------------------------------------- numeric
    def _check_numeric(self, name: str, values: pd.Series, n: int) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        numbers = to_numeric(values)

        not_numeric = numbers.isna()
        if not_numeric.any():
            issues.append(self._issue(
                name, IssueType.TYPE_MISMATCH, values[not_numeric], n,
                message=f"{int(not_numeric.sum())} non-numeric value(s) in numeric column '{name}'.",
                recommendation="Correct these at the source, or convert them to missing values "
                               "with pd.to_numeric(errors='coerce') and impute.",
            ))

        rng = valid_range(name, self.settings)
        if rng is not None:
            lo, hi = rng
            parsed = numbers.dropna()
            outside = (parsed < lo) | (parsed > hi)
            if outside.any():
                bad = values.loc[parsed.index[outside]]
                bound = f"between {lo:g} and {hi:g}" if np.isfinite(hi) else f"at least {lo:g}"
                issues.append(self._issue(
                    name, IssueType.OUT_OF_RANGE, bad, n,
                    message=f"{int(outside.sum())} value(s) in '{name}' are impossible (must be {bound}).",
                    recommendation="Treat impossible values as missing and impute, or correct them at the source.",
                    details={"valid_min": lo, "valid_max": hi if np.isfinite(hi) else None,
                             "below_min": int((parsed < lo).sum()), "above_max": int((parsed > hi).sum())},
                ))
        return issues

    # ------------------------------------------------------------------ dates
    def _check_dates(self, name: str, values: pd.Series, n: int) -> list[QualityIssue]:
        parsed = to_datetime(values)
        bad = parsed.isna()
        if not bad.any():
            return []
        return [self._issue(
            name, IssueType.INVALID_DATE, values[bad], n,
            message=f"{int(bad.sum())} value(s) in date column '{name}' are not valid dates.",
            recommendation="Check for impossible dates (e.g. 30 February) or mixed formats; "
                           "correct them or set them to missing.",
        )]

    # ---------------------------------------------------------------- helpers
    def _issue(self, name, issue_type, bad_values: pd.Series, n: int,
               message: str, recommendation: str, details: dict | None = None) -> QualityIssue:
        count = int(len(bad_values))
        p = pct(count, n)
        counts = bad_values.astype(str).value_counts()
        return QualityIssue(
            check=self.name,
            issue_type=issue_type,
            # An impossible value is an error, never just noise: minimum MEDIUM.
            severity=severity_from_pct(p, self.settings.severity_thresholds, floor=Severity.MEDIUM),
            column=name,
            affected_rows=count,
            affected_pct=p,
            message=message,
            recommendation=recommendation,
            examples=[to_python(v) for v in counts.index[: self.settings.max_examples]],
            details={**(details or {}), "value_counts": {str(k): int(v) for k, v in counts.head(20).items()}},
        )
