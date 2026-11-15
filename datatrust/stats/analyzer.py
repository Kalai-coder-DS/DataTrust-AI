"""
datatrust/stats/analyzer.py

Runs all statistical analyses and returns one StatisticsReport.

    report = StatisticsAnalyzer().analyze(df)                    # no target
    report = StatisticsAnalyzer().analyze(df, target="churned")  # + class balance
"""
from __future__ import annotations

import pandas as pd

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.logger import get_logger
from datatrust.profiling import DataProfiler, DatasetProfile, SemanticType
from datatrust.stats.associations import find_associations
from datatrust.stats.descriptive import summarize_categorical, summarize_datetime, summarize_numeric
from datatrust.stats.models import StatisticsReport
from datatrust.stats.target import summarize_target

logger = get_logger(__name__)


class StatisticsAnalyzer:
    def __init__(self, settings: Settings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    def analyze(self, df: pd.DataFrame, profile: DatasetProfile | None = None,
                target: str | None = None) -> StatisticsReport:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
        profile = profile or DataProfiler(self.settings).profile(df)
        report = StatisticsReport()

        for col in profile.columns:
            series = df[col.name]
            if col.semantic_type == SemanticType.NUMERIC:
                summary = summarize_numeric(series, col, self.settings)
                if summary:
                    report.numeric.append(summary)
            elif col.semantic_type in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN):
                summary = summarize_categorical(series, col, self.settings)
                if summary:
                    report.categorical.append(summary)
            elif col.semantic_type == SemanticType.DATETIME:
                summary = summarize_datetime(series, col, self.settings)
                if summary:
                    report.datetime.append(summary)

        report.associations = find_associations(df, profile, self.settings)
        if target is not None:
            report.target = summarize_target(df, profile, target, self.settings)
            for assoc in report.associations:
                if target in (assoc.column_a, assoc.column_b):
                    # Strong feature-target association is predictive signal, not redundancy.
                    assoc.involves_target = True
                    assoc.redundant = False

        logger.info(
            "Statistics: %d numeric, %d categorical, %d datetime summaries; %d notable associations",
            len(report.numeric), len(report.categorical), len(report.datetime), len(report.associations),
        )
        return report
