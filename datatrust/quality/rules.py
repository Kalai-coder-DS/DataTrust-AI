"""
datatrust/quality/rules.py

Small helper functions shared by several checks.

Keeping them here guarantees that every check agrees on the same definitions.
For example, a value like 'Unknown' is counted ONCE as disguised-missing, and is
NOT also counted as an invalid number by the validity check.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from datatrust.config import Settings


def name_tokens(column_name: str) -> set[str]:
    """'annual_income' -> {'annual', 'income'};  'Monthly Spend' -> {'monthly', 'spend'}.

    Whole tokens are matched on purpose: substring matching would make 'age'
    match 'mileage', 'average' and 'percentage'.
    """
    return {t for t in re.split(r"[^a-z0-9]+", column_name.lower()) if t}


def disguised_missing_mask(series: pd.Series, settings: Settings) -> pd.Series:
    """True where a non-null value is really a placeholder for 'missing'."""
    mask = pd.Series(False, index=series.index)
    non_null = series.notna()
    if not non_null.any():
        return mask

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.isin(settings.sentinel_numbers) & non_null

    # Phase 10 speed-up: judge each DISTINCT value once, then map the answer back to the rows.
    # A 300,000-row column usually has a handful of distinct labels; the string work (str, strip,
    # lower) on every row was the single slowest step of the whole analysis (3.7 s of 9.7 s).
    values = series[non_null]
    codes, uniques = pd.factorize(values)
    text = pd.Index(uniques).astype(str).str.strip().str.lower()
    tokens = set(settings.disguised_missing_tokens)
    is_placeholder = np.asarray(text.isin(tokens) | (text == ""))   # whitespace-only strings strip to ""
    mask.loc[values.index] = is_placeholder[codes]
    return mask


def usable_values(series: pd.Series, settings: Settings) -> pd.Series:
    """Non-null values that are not disguised-missing placeholders."""
    return series[series.notna() & ~disguised_missing_mask(series, settings)]


def to_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numbers; anything unparseable becomes NaN."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def to_datetime(series: pd.Series) -> pd.Series:
    """Coerce to datetimes; anything unparseable (including '2023-02-30') becomes NaT."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series.astype(str).str.strip(), errors="coerce", format="mixed")


def valid_range(column_name: str, settings: Settings) -> tuple[float, float] | None:
    """
    The (min, max) a column's values must fall within, based on its name, or None
    if no rule applies. Explicit range rules win over the generic non-negative rule.
    """
    tokens = name_tokens(column_name)
    for token, lo, hi in settings.range_rules:
        if token in tokens:
            return float(lo), float(hi)
    if tokens & set(settings.non_negative_tokens):
        return 0.0, float(np.inf)
    return None
