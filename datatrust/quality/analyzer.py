"""
datatrust/quality/analyzer.py

Runs all quality checks and combines their findings into one QualityReport.

Design notes:
- Checks are independent. If one crashes on unusual data, the others still run
  and the failure is recorded as a CHECK_FAILED issue (never silently swallowed).
- The default list is defined once here; tests or future users can pass their own.
"""
from __future__ import annotations

import pandas as pd

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.logger import get_logger
from datatrust.profiling import DataProfiler, DatasetProfile
from datatrust.quality.base import QualityCheck
from datatrust.quality.columns import LowInformationCheck
from datatrust.quality.consistency import ConsistencyCheck
from datatrust.quality.duplicates import DuplicateCheck
from datatrust.quality.missing import MissingValueCheck
from datatrust.quality.models import IssueType, QualityIssue, QualityReport, Severity
from datatrust.quality.outliers import OutlierCheck
from datatrust.quality.validity import InvalidValueCheck

logger = get_logger(__name__)

DEFAULT_CHECKS: tuple[type[QualityCheck], ...] = (
    MissingValueCheck,
    DuplicateCheck,
    InvalidValueCheck,
    ConsistencyCheck,
    OutlierCheck,
    LowInformationCheck,
)


class QualityAnalyzer:
    """Usage: report = QualityAnalyzer().analyze(df)"""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS,
                 checks: list[QualityCheck] | None = None) -> None:
        self.settings = settings
        self.checks = checks if checks is not None else [c(settings) for c in DEFAULT_CHECKS]

    def analyze(self, df: pd.DataFrame, profile: DatasetProfile | None = None) -> QualityReport:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
        profile = profile or DataProfiler(self.settings).profile(df)
        report = QualityReport(n_rows=len(df), n_columns=df.shape[1])

        for check in self.checks:
            try:
                found = check.run(df, profile)
                logger.info("Check '%s' found %d issue(s)", check.name, len(found))
                report.issues.extend(found)
            except Exception as exc:  # noqa: BLE001 - isolate failures per check on purpose
                logger.exception("Check '%s' failed", check.name)
                report.issues.append(QualityIssue(
                    check=check.name,
                    issue_type=IssueType.CHECK_FAILED,
                    severity=Severity.HIGH,
                    column=None,
                    affected_rows=0,
                    affected_pct=0.0,
                    message=f"The '{check.name}' check could not complete: {exc}",
                    recommendation="Results for this check are missing; see logs/datatrust.log.",
                ))
        logger.info("Quality analysis complete: %d issue(s)", len(report.issues))
        return report
