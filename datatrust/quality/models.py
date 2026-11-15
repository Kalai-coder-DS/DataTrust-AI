"""
datatrust/quality/models.py

Data structures shared by every quality check.

Every detector returns the SAME shape - a list of QualityIssue objects - so the
analyzer, the CLI, the scoring engine (Phase 4) and the HTML report (Phase 5)
never need to know which detector produced an issue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    LOW = "low"            # cosmetic or minor; fix when convenient
    MEDIUM = "medium"      # will distort analysis if ignored
    HIGH = "high"          # likely to produce wrong results or models
    CRITICAL = "critical"  # column/dataset unusable until fixed

    @property
    def rank(self) -> int:
        return ["low", "medium", "high", "critical"].index(self.value)


class IssueType(str, Enum):
    MISSING = "missing_values"
    DISGUISED_MISSING = "disguised_missing"
    EMPTY_COLUMN = "empty_column"
    DUPLICATE_ROWS = "duplicate_rows"
    DUPLICATE_IDS = "duplicate_ids"
    DUPLICATES_HIDDEN_BY_ID = "duplicates_hidden_by_id"  # same data, different row id
    TYPE_MISMATCH = "type_mismatch"          # 'abc' in a numeric column
    INVALID_DATE = "invalid_date"            # '2023-02-30'
    OUT_OF_RANGE = "out_of_range"            # age = 210, price = -5
    INCONSISTENT_FORMAT = "inconsistent_format"  # 'Chennai' vs 'chennai'
    NUMBERS_AS_WORDS = "numbers_as_words"    # 'four', 'six'
    OUTLIERS = "outliers"
    CONSTANT_COLUMN = "constant_column"
    NEAR_CONSTANT_COLUMN = "near_constant_column"
    CHECK_FAILED = "check_failed"            # a detector crashed; reported, not hidden


@dataclass
class QualityIssue:
    check: str                    # which detector found it, e.g. "missing"
    issue_type: IssueType
    severity: Severity
    column: str | None            # None = dataset-level issue (e.g. duplicate rows)
    affected_rows: int
    affected_pct: float
    message: str                  # one-line, human-readable
    recommendation: str           # what the user should do about it
    examples: list[Any] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "column": self.column,
            "affected_rows": self.affected_rows,
            "affected_pct": self.affected_pct,
            "message": self.message,
            "recommendation": self.recommendation,
            "examples": list(self.examples),
            "details": dict(self.details),
        }


@dataclass
class QualityReport:
    n_rows: int
    n_columns: int
    issues: list[QualityIssue] = field(default_factory=list)

    def sorted_issues(self) -> list[QualityIssue]:
        """Most severe first, then most rows affected."""
        return sorted(self.issues, key=lambda i: (-i.severity.rank, -i.affected_rows))

    def filter(self, issue_type: IssueType | None = None, column: str | None = None) -> list[QualityIssue]:
        return [
            i for i in self.issues
            if (issue_type is None or i.issue_type == issue_type)
            and (column is None or i.column == column)
        ]

    def severity_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for i in self.issues:
            counts[i.severity.value] += 1
        return counts

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "total_issues": len(self.issues),
            "severity_counts": self.severity_counts(),
            "issues": [i.to_dict() for i in self.sorted_issues()],
        }


# ------------------------------------------------------------------ helpers
def pct(part: int | float, whole: int | float) -> float:
    return round(float(part) / float(whole) * 100, 2) if whole else 0.0


def severity_from_pct(
    affected_pct: float,
    thresholds: tuple[float, float, float],
    floor: Severity = Severity.LOW,
) -> Severity:
    """
    Map '% of rows affected' to a severity band, never going below `floor`.
    The floor exists because some problems are errors regardless of how rare they are:
    a single impossible age is still wrong, so validity issues use floor=MEDIUM.
    """
    low, medium, high = thresholds
    if affected_pct < low:
        sev = Severity.LOW
    elif affected_pct < medium:
        sev = Severity.MEDIUM
    elif affected_pct < high:
        sev = Severity.HIGH
    else:
        sev = Severity.CRITICAL
    return sev if sev.rank >= floor.rank else floor


def to_python(value: Any) -> Any:
    """numpy/pandas scalars -> plain Python, so issues are JSON-serialisable."""
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
