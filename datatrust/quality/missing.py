"""
datatrust/quality/missing.py

Missing-value detection.

Three kinds of "missing":
  1. Empty column   - every value is missing. Column is useless.
  2. True missing   - NaN / blank (pandas already converted '', 'NA', 'null', ... to NaN).
  3. Disguised      - a value that LOOKS present but means 'no data':
                      'Unknown', '?', '-', or numeric codes like -999.
                      These are dangerous because df.isna() reports 0% missing,
                      and a model would treat 'Unknown' as a real category
                      or -999 as a real (very small) number.
"""
from __future__ import annotations

import pandas as pd

from datatrust.profiling import DatasetProfile, SemanticType
from datatrust.quality.base import QualityCheck
from datatrust.quality.models import IssueType, QualityIssue, Severity, pct, severity_from_pct, to_python
from datatrust.quality.rules import disguised_missing_mask


class MissingValueCheck(QualityCheck):
    name = "missing"

    def run(self, df: pd.DataFrame, profile: DatasetProfile) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        n = len(df)
        thresholds = self.settings.severity_thresholds

        for col in profile.columns:
            series = df[col.name]

            if col.semantic_type == SemanticType.EMPTY:
                issues.append(QualityIssue(
                    check=self.name,
                    issue_type=IssueType.EMPTY_COLUMN,
                    severity=Severity.CRITICAL,
                    column=col.name,
                    affected_rows=n,
                    affected_pct=100.0,
                    message=f"Column '{col.name}' is completely empty.",
                    recommendation="Drop this column; it contains no information.",
                ))
                continue  # nothing else to check in an empty column

            if col.null_count:
                p = pct(col.null_count, n)
                issues.append(QualityIssue(
                    check=self.name,
                    issue_type=IssueType.MISSING,
                    severity=severity_from_pct(p, thresholds),
                    column=col.name,
                    affected_rows=col.null_count,
                    affected_pct=p,
                    message=f"{col.null_count} missing value(s) ({p}%) in '{col.name}'.",
                    recommendation=_missing_recommendation(p, col.semantic_type),
                ))

            disguised = disguised_missing_mask(series, self.settings)
            n_disguised = int(disguised.sum())
            if n_disguised:
                p = pct(n_disguised, n)
                values = series[disguised].astype(str).value_counts()
                issues.append(QualityIssue(
                    check=self.name,
                    issue_type=IssueType.DISGUISED_MISSING,
                    # At least MEDIUM: these are invisible to normal missing-value checks.
                    severity=severity_from_pct(p, thresholds, floor=Severity.MEDIUM),
                    column=col.name,
                    affected_rows=n_disguised,
                    affected_pct=p,
                    message=(
                        f"{n_disguised} placeholder value(s) in '{col.name}' that mean 'missing' "
                        f"({', '.join(repr(v) for v in values.index[:3])})."
                    ),
                    recommendation="Convert these placeholders to real missing values (NaN), "
                                   "then treat them like other missing data.",
                    examples=[to_python(v) for v in values.index[: self.settings.max_examples]],
                    details={"value_counts": {str(k): int(v) for k, v in values.items()}},
                ))
        return issues


def _missing_recommendation(p: float, stype: SemanticType) -> str:
    if p >= 50:
        return "More than half the values are missing; consider dropping the column."
    if stype == SemanticType.NUMERIC:
        return "Impute with the median (robust to outliers) or drop the affected rows if few."
    if stype in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN):
        return "Impute with the most frequent value or add an explicit 'Missing' category."
    if stype == SemanticType.IDENTIFIER:
        return "Rows without an identifier cannot be traced; investigate the source system."
    return "Investigate why values are missing before choosing to impute or drop."
