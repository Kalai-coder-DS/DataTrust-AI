"""
datatrust/quality/columns.py

Low-information column detection.

  Constant column      - one distinct value (e.g. country = 'India' everywhere).
                         Carries zero information: a model cannot learn from it.
  Near-constant column - one value covers >= 99% of rows. Almost no information,
                         and models can overfit to the rare exceptions.

Identifiers are skipped (they are unique by design) and empty columns are skipped
(the missing-value check already reports them as CRITICAL).
"""
from __future__ import annotations

import pandas as pd

from datatrust.profiling import DatasetProfile, SemanticType
from datatrust.quality.base import QualityCheck
from datatrust.quality.models import IssueType, QualityIssue, Severity, pct, to_python


class LowInformationCheck(QualityCheck):
    name = "columns"

    def run(self, df: pd.DataFrame, profile: DatasetProfile) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        n = len(df)
        for col in profile.columns:
            if col.semantic_type in (SemanticType.EMPTY, SemanticType.IDENTIFIER):
                continue
            values = df[col.name].dropna()
            if values.empty:
                continue
            counts = values.value_counts()
            top_value, top_count = counts.index[0], int(counts.iloc[0])

            if len(counts) == 1:
                issues.append(QualityIssue(
                    check=self.name,
                    issue_type=IssueType.CONSTANT_COLUMN,
                    severity=Severity.MEDIUM,
                    column=col.name,
                    affected_rows=n,
                    affected_pct=100.0,
                    message=f"'{col.name}' has the same value in every row ({to_python(top_value)!r}).",
                    recommendation="Drop this column before modelling; it carries no information.",
                    examples=[to_python(top_value)],
                ))
            elif top_count / len(values) >= self.settings.near_constant_ratio:
                share = pct(top_count, len(values))
                issues.append(QualityIssue(
                    check=self.name,
                    issue_type=IssueType.NEAR_CONSTANT_COLUMN,
                    severity=Severity.LOW,
                    column=col.name,
                    affected_rows=len(values) - top_count,
                    affected_pct=pct(len(values) - top_count, n),
                    message=f"'{col.name}' is {share}% one value ({to_python(top_value)!r}).",
                    recommendation="Check whether the rare values matter; otherwise consider dropping the column.",
                    examples=[to_python(v) for v in counts.index[1: 1 + self.settings.max_examples]],
                    details={"dominant_value": to_python(top_value), "dominant_share_pct": share},
                ))
        return issues
