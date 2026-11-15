"""
Statistics layer: descriptive summaries, associations between columns, target balance.

Named `stats` rather than `statistics` so it can never be confused with Python's
built-in `statistics` module.
"""

from datatrust.stats.analyzer import StatisticsAnalyzer
from datatrust.stats.models import (
    Association,
    CategoricalSummary,
    DatetimeSummary,
    NumericSummary,
    StatisticsReport,
    TargetSummary,
)

__all__ = [
    "Association", "CategoricalSummary", "DatetimeSummary", "NumericSummary",
    "StatisticsAnalyzer", "StatisticsReport", "TargetSummary",
]
