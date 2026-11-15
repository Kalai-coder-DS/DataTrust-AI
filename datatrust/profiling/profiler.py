"""
datatrust/profiling/profiler.py

Structural data profiling.

Answers "what is in this dataset?" BEFORE we judge its quality:
  - dataset level: rows, columns, memory, how many columns of each kind
  - column level : pandas dtype, inferred *semantic* type, null count, cardinality, samples

Key idea - pandas dtype vs semantic type:
  pandas only knows storage types (int64, float64, object/str, bool, datetime64).
  A column of '2024-01-05' strings is stored as text but is semantically a DATE;
  a column of '45', '52', 'abc' is stored as text but is semantically NUMERIC with
  one bad value. Later phases (invalid-data detection, ML readiness) rely on the
  semantic type, which is why this module infers it explicitly.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.logger import get_logger

logger = get_logger(__name__)

_BOOLEAN_TOKENS = {"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"}


class SemanticType(str, Enum):
    EMPTY = "empty"            # all values missing
    BOOLEAN = "boolean"        # two-valued flag (true/false, yes/no, 0/1)
    NUMERIC = "numeric"        # continuous or discrete numbers
    DATETIME = "datetime"      # dates / timestamps
    CATEGORICAL = "categorical"  # small set of repeating labels
    IDENTIFIER = "identifier"  # unique key per row (customer_id, order no.)
    TEXT = "text"              # free text / high-cardinality strings


@dataclass
class ColumnProfile:
    name: str
    pandas_dtype: str
    semantic_type: SemanticType
    non_null_count: int
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float              # distinct values / non-null values * 100
    sample_values: list
    stored_as_text: bool = False   # True when semantic type is numeric/datetime/boolean but pandas holds strings
    parse_success_pct: float | None = None  # % of non-null values that parsed to the semantic type

    def to_dict(self) -> dict:
        d = asdict(self)
        d["semantic_type"] = self.semantic_type.value
        return d


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    memory_bytes: int
    total_cells: int
    total_missing_cells: int
    missing_cells_pct: float
    semantic_type_counts: dict[str, int]
    columns: list[ColumnProfile] = field(default_factory=list)

    def get_column(self, name: str) -> ColumnProfile:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"No profile for column '{name}'")

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "memory_bytes": self.memory_bytes,
            "total_cells": self.total_cells,
            "total_missing_cells": self.total_missing_cells,
            "missing_cells_pct": self.missing_cells_pct,
            "semantic_type_counts": dict(self.semantic_type_counts),
            "columns": [c.to_dict() for c in self.columns],
        }


class DataProfiler:
    """Usage: profile = DataProfiler().profile(df)"""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    # ------------------------------------------------------------------ public
    def profile(self, df: pd.DataFrame) -> DatasetProfile:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
        logger.info("Profiling dataset with %d rows x %d columns", *df.shape)

        columns = [self.profile_column(df[col], str(col)) for col in df.columns]

        type_counts = {t.value: 0 for t in SemanticType}
        for c in columns:
            type_counts[c.semantic_type.value] += 1

        total_cells = int(df.shape[0] * df.shape[1])
        missing = int(df.isna().sum().sum())
        return DatasetProfile(
            n_rows=int(df.shape[0]),
            n_columns=int(df.shape[1]),
            memory_bytes=int(df.memory_usage(deep=True).sum()),
            total_cells=total_cells,
            total_missing_cells=missing,
            missing_cells_pct=_pct(missing, total_cells),
            semantic_type_counts=type_counts,
            columns=columns,
        )

    def profile_column(self, series: pd.Series, name: str) -> ColumnProfile:
        non_null = series.dropna()
        n_total = len(series)
        n_non_null = len(non_null)
        n_unique = int(non_null.nunique())

        semantic_type, stored_as_text, parse_pct = self.infer_semantic_type(series, name)

        return ColumnProfile(
            name=name,
            pandas_dtype=str(series.dtype),
            semantic_type=semantic_type,
            non_null_count=n_non_null,
            null_count=n_total - n_non_null,
            null_pct=_pct(n_total - n_non_null, n_total),
            unique_count=n_unique,
            unique_pct=_pct(n_unique, n_non_null),
            sample_values=[_to_python(v) for v in non_null.unique()[: self.settings.sample_values_count]],
            stored_as_text=stored_as_text,
            parse_success_pct=parse_pct,
        )

    # --------------------------------------------------------- type inference
    def infer_semantic_type(
        self, series: pd.Series, name: str = ""
    ) -> tuple[SemanticType, bool, float | None]:
        """
        Rule-based inference, evaluated in priority order.
        Returns (semantic_type, stored_as_text, parse_success_pct).
        """
        s = self.settings
        non_null = series.dropna()
        if non_null.empty:
            return SemanticType.EMPTY, False, None

        n_unique = non_null.nunique()
        unique_ratio = n_unique / len(non_null)

        # 1. Native boolean dtype
        if ptypes.is_bool_dtype(series):
            return SemanticType.BOOLEAN, False, 100.0

        # 2. Native datetime dtype
        if ptypes.is_datetime64_any_dtype(series):
            return SemanticType.DATETIME, False, 100.0

        # 3. Native numeric dtype
        if ptypes.is_numeric_dtype(series):
            if n_unique <= 2 and set(non_null.unique()).issubset({0, 1}):
                return SemanticType.BOOLEAN, False, 100.0
            if self._looks_like_numeric_id(non_null, name, unique_ratio):
                return SemanticType.IDENTIFIER, False, 100.0
            return SemanticType.NUMERIC, False, 100.0

        # ---- Text-stored column from here on ----
        text = non_null.astype(str).str.strip()
        sample = self._sample(text)

        # 4. Boolean words (yes/no, true/false, y/n). Compared case-insensitively:
        #    'Yes' vs 'yes' is a consistency problem (Phase 2), not a third category.
        lowered = set(sample.str.lower().unique())
        if len(lowered) <= 2 and lowered.issubset(_BOOLEAN_TOKENS):
            return SemanticType.BOOLEAN, True, 100.0

        # 5. Numbers stored as text (often because a few values are invalid, e.g. 'abc', 'N/A')
        numeric_ratio = pd.to_numeric(sample, errors="coerce").notna().mean()
        if numeric_ratio >= s.type_inference_threshold:
            return SemanticType.NUMERIC, True, round(float(numeric_ratio) * 100, 2)

        # 6. Dates stored as text. Pure numbers are excluded above, so '2023' won't be read as a date.
        date_ratio = self._datetime_parse_ratio(sample)
        if date_ratio >= s.type_inference_threshold:
            return SemanticType.DATETIME, True, round(date_ratio * 100, 2)

        # 7. Identifier: (almost) every value unique, short, and not sentence-like.
        #    Near-unique columns named like an ID are still IDs - their duplicates are a
        #    data-quality problem for Phase 2 to report, not a reason to change the type.
        near_unique_id = unique_ratio >= s.id_min_unique_ratio and _has_id_name(name)
        if (unique_ratio == 1.0 or near_unique_id) and len(non_null) > 1:
            avg_len = sample.str.len().mean()
            has_spaces = sample.str.contains(" ", regex=False).mean() > 0.5
            if avg_len < s.free_text_min_avg_length and not has_spaces:
                return SemanticType.IDENTIFIER, False, None

        # 8. Categorical: few distinct values relative to rows
        if n_unique <= s.categorical_max_unique or unique_ratio <= s.categorical_max_unique_ratio:
            return SemanticType.CATEGORICAL, False, None

        # 9. Everything else
        return SemanticType.TEXT, False, None

    # ---------------------------------------------------------------- helpers
    def _sample(self, s: pd.Series) -> pd.Series:
        n = self.settings.inference_sample_size
        return s.sample(n=n, random_state=42) if len(s) > n else s

    @staticmethod
    def _datetime_parse_ratio(sample: pd.Series) -> float:
        # Quick reject: dates need at least one digit; avoids slow parsing of pure-word columns.
        if not sample.str.contains(r"\d", regex=True).any():
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pandas warns when it has to guess formats
            try:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            except (ValueError, TypeError, OverflowError):
                return 0.0
        return float(parsed.notna().mean())

    def _looks_like_numeric_id(self, non_null: pd.Series, name: str, unique_ratio: float) -> bool:
        """
        An all-unique integer column is an ID only if its name says so (customer_id, CustomerID)
        or it is a perfect 1-step sequence (row numbers). Otherwise 'age' in a tiny file
        with all-different ages would be misclassified.
        """
        if len(non_null) < 2:
            return False
        values = non_null.to_numpy()
        if not np.all(np.mod(values, 1) == 0):
            return False
        if _has_id_name(name) and unique_ratio >= self.settings.id_min_unique_ratio:
            return True
        if unique_ratio < 1.0 or len(values) < 10:  # too few rows to trust a sequence pattern
            return False
        sorted_vals = np.sort(values)
        return bool(np.all(np.diff(sorted_vals) == 1))


# -------------------------------------------------------------- module utils
def _has_id_name(name: str) -> bool:
    """customer_id, CustomerID, order-id, id, rownum -> True; 'paid', 'valid' -> False."""
    tokens = re.split(r"[^a-z0-9]+", name.lower())
    return "id" in tokens or name.endswith(("Id", "ID")) or name.lower() in {"index", "rownum"}


def _pct(part: int | float, whole: int | float) -> float:
    return round(float(part) / float(whole) * 100, 2) if whole else 0.0


def _to_python(value):
    """Convert numpy/pandas scalars to plain Python so profiles are JSON-serialisable."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
