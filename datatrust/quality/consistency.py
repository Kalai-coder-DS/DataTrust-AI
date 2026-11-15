"""
datatrust/quality/consistency.py

Consistency checks for label-like columns (categorical / boolean):

  1. Inconsistent format - the same label written different ways:
     'Chennai', 'chennai', ' Chennai '. Humans read these as one city;
     pandas and ML models treat them as three different categories.
     Method: normalise (strip, lower-case, collapse spaces), group raw values
     by their normalised form, and count rows NOT using the most common spelling.

  2. Numbers as words - 'two', 'four', 'six'. Valid, but an ML model can't
     know 'four' > 'two' unless they're converted to 2 and 4.
"""
from __future__ import annotations

import pandas as pd

from datatrust.profiling import DatasetProfile, SemanticType
from datatrust.quality.base import QualityCheck
from datatrust.quality.models import IssueType, QualityIssue, Severity, pct, severity_from_pct
from datatrust.quality.rules import usable_values

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "twenty", "thirty", "forty", "fifty", "hundred",
}
_LABEL_TYPES = {SemanticType.CATEGORICAL, SemanticType.BOOLEAN}


class ConsistencyCheck(QualityCheck):
    name = "consistency"

    def run(self, df: pd.DataFrame, profile: DatasetProfile) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        for col in profile.columns:
            if col.semantic_type not in _LABEL_TYPES:
                continue
            series = df[col.name]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                continue  # numbers/bools have no spelling variants
            values = usable_values(series, self.settings).astype(str)
            if values.empty:
                continue
            issue = self._inconsistent_format(col.name, values, len(df))
            if issue:
                issues.append(issue)
            issue = self._numbers_as_words(col.name, values, len(df))
            if issue:
                issues.append(issue)
        return issues

    def _inconsistent_format(self, name: str, values: pd.Series, n: int) -> QualityIssue | None:
        normalised = values.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        frame = pd.DataFrame({"raw": values, "norm": normalised})
        variant_counts = frame.groupby(["norm", "raw"]).size()

        groups: dict[str, dict[str, int]] = {}
        affected = 0
        for norm_value, raw_counts in variant_counts.groupby(level=0):
            if len(raw_counts) < 2:
                continue
            raw_counts = raw_counts.droplevel(0).sort_values(ascending=False)
            affected += int(raw_counts.iloc[1:].sum())  # everything except the dominant spelling
            groups[str(norm_value)] = {str(k): int(v) for k, v in raw_counts.items()}

        if not affected:
            return None
        p = pct(affected, n)
        examples = [" / ".join(repr(v) for v in variants) for variants in list(groups.values())[:3]]
        return QualityIssue(
            check=self.name,
            issue_type=IssueType.INCONSISTENT_FORMAT,
            severity=severity_from_pct(p, self.settings.severity_thresholds, floor=Severity.MEDIUM),
            column=name,
            affected_rows=affected,
            affected_pct=p,
            message=(f"{len(groups)} label(s) in '{name}' are written in more than one way "
                     f"(case/spacing); {affected} row(s) use a non-standard spelling."),
            recommendation="Standardise labels, e.g. df[col].str.strip().str.title(), "
                           "so each category has exactly one spelling.",
            examples=examples,
            details={"variant_groups": groups},
        )

    def _numbers_as_words(self, name: str, values: pd.Series, n: int) -> QualityIssue | None:
        distinct = set(values.str.strip().str.lower().unique())
        if not distinct or not distinct.issubset(_NUMBER_WORDS):
            return None
        return QualityIssue(
            check=self.name,
            issue_type=IssueType.NUMBERS_AS_WORDS,
            severity=Severity.LOW,
            column=name,
            affected_rows=int(len(values)),
            affected_pct=pct(len(values), n),
            message=f"'{name}' stores numbers as words ({', '.join(sorted(distinct)[:5])}).",
            recommendation="Map words to numbers (e.g. {'two': 2, 'four': 4}) so models can use their order.",
            examples=sorted(distinct)[: self.settings.max_examples],
        )
