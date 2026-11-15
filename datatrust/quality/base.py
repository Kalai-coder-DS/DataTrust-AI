"""
datatrust/quality/base.py

The contract every quality check follows.

Why an abstract base class?
- The analyzer can run ANY list of checks without knowing what they do
  (open/closed principle: add a new check without editing the analyzer).
- Every check receives the Phase 1 DatasetProfile, so it can use semantic types
  (e.g. only run outlier detection on NUMERIC columns) instead of re-inferring them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.profiling import DatasetProfile
from datatrust.quality.models import QualityIssue


class QualityCheck(ABC):
    name: str = "base"

    def __init__(self, settings: Settings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    @abstractmethod
    def run(self, df: pd.DataFrame, profile: DatasetProfile) -> list[QualityIssue]:
        """Inspect the data and return zero or more issues. Must not modify `df`."""
