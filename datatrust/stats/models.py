"""
datatrust/stats/models.py

Result containers for statistical analysis. Plain dataclasses with to_dict(),
so the CLI, the JSON export, the scoring engine (Phase 4) and the web app
(Phase 6) all read the same structures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NumericSummary:
    column: str
    count: int                 # valid values used for the statistics
    excluded: int              # non-null values left out (placeholders, unparseable, impossible)
    mean: float
    median: float
    std: float
    min: float
    q1: float
    q3: float
    max: float
    iqr: float
    cv: float | None           # coefficient of variation = std / |mean|; None if mean is 0
    skewness: float            # moment-based (sensitive to extreme values)
    quartile_skewness: float   # Bowley: uses only quartiles (robust to extreme values)
    skewness_without_outliers: float  # recomputed inside the IQR fences
    kurtosis: float            # excess kurtosis: 0 = normal-like tails, > 0 = heavier tails
    zeros_pct: float
    negatives_pct: float
    shape: str                 # plain-language description of the distribution

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CategoricalSummary:
    column: str
    count: int
    n_categories: int
    top_value: Any
    top_pct: float
    rare_categories: int       # categories each covering < rare_category_pct of rows
    entropy_norm: float        # 0 = one value only, 1 = perfectly even spread
    value_counts: dict[str, int] = field(default_factory=dict)  # top 10

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatetimeSummary:
    column: str
    count: int
    excluded: int              # values that could not be parsed as dates
    min: str
    max: str
    span_days: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Association:
    column_a: str
    column_b: str
    method: str                # pearson | spearman | cramers_v | correlation_ratio
    value: float               # -1..1 for correlations, 0..1 for cramers_v / correlation_ratio
    strength: str              # weak | moderate | strong | very strong
    redundant: bool = False    # two FEATURES so correlated that one adds little (never set for the target)
    involves_target: bool = False  # one side is the target: strength here = predictive signal
    details: dict[str, Any] = field(default_factory=dict)  # e.g. the Pearson value for numeric pairs

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TargetSummary:
    column: str
    task: str                  # "classification" or "regression"
    n_classes: int | None = None
    class_counts: dict[str, int] = field(default_factory=dict)
    minority_class: Any = None
    minority_pct: float | None = None
    imbalance_ratio: float | None = None   # majority count / minority count
    balance: str | None = None             # balanced | moderate | high | severe

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StatisticsReport:
    numeric: list[NumericSummary] = field(default_factory=list)
    categorical: list[CategoricalSummary] = field(default_factory=list)
    datetime: list[DatetimeSummary] = field(default_factory=list)
    associations: list[Association] = field(default_factory=list)
    target: TargetSummary | None = None

    def numeric_summary(self, column: str) -> NumericSummary:
        return _find(self.numeric, column)

    def categorical_summary(self, column: str) -> CategoricalSummary:
        return _find(self.categorical, column)

    def association(self, a: str, b: str, method: str | None = None) -> Association | None:
        for assoc in self.associations:
            if {assoc.column_a, assoc.column_b} == {a, b} and (method is None or assoc.method == method):
                return assoc
        return None

    def to_dict(self) -> dict:
        return {
            "numeric": [s.to_dict() for s in self.numeric],
            "categorical": [s.to_dict() for s in self.categorical],
            "datetime": [s.to_dict() for s in self.datetime],
            "associations": [a.to_dict() for a in self.associations],
            "target": self.target.to_dict() if self.target else None,
        }


def _find(items: list, column: str):
    for item in items:
        if item.column == column:
            return item
    raise KeyError(f"No summary for column '{column}'")


def strength_label(value: float) -> str:
    """Conventional rule-of-thumb bands for |r|, Cramér's V and eta."""
    v = abs(value)
    if v >= 0.9:
        return "very strong"
    if v >= 0.7:
        return "strong"
    if v >= 0.4:
        return "moderate"
    return "weak"
