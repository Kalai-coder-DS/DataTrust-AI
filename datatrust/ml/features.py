"""
datatrust/ml/features.py

Turns the uploaded table into model input (X, y), in one of two ways:

  raw      the data as uploaded, with only the conversions any model needs to run at all
           (text that should be numbers -> numbers, dates -> numbers). Placeholders like
           'Unknown' or -999, impossible values, spelling variants, duplicate rows, constant,
           redundant and leaking columns all stay in - exactly what a hurried analyst would do.

  cleaned  the same, after the fixes the earlier phases RECOMMENDED for this file:
           duplicate rows removed (including duplicates hidden by an ID), placeholders and
           impossible values set to missing, label spellings merged, constant and redundant
           columns dropped. Every step is recorded with what it changed.

           Columns flagged as POSSIBLE target leakage are NOT dropped automatically: Phase 4 only
           says "confirm it is known before the outcome", and a strong genuine predictor looks the
           same (found in testing: price = 3x + noise flagged x, and dropping it took R² from 0.97
           to 0). baseline.py measures the model with and without them instead (drop_columns).

Both versions use the same rows of the same file, the same exclusions (target, IDs, free
text) and the same model, so any score difference comes from the cleaning alone.

The fixes are driven by the analysis dict (the Phase 2-4 findings), not re-detected here,
so the model is trained on exactly what the report told the user to fix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.quality.rules import disguised_missing_mask, to_datetime, to_numeric, valid_range

EXCLUDED_TYPES = {
    "identifier": "identifier (unique per row, no pattern to learn)",
    "text": "free text (needs text processing, out of scope for a baseline)",
    "empty": "empty column",
}
NUMERIC_TYPES = {"numeric"}
DATETIME_TYPES = {"datetime"}
CATEGORY_TYPES = {"categorical", "boolean"}


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    numeric: list[str]
    categorical: list[str]
    steps: list[str] = field(default_factory=list)          # what cleaning changed (cleaned only)
    excluded: list[dict] = field(default_factory=list)      # columns not used, with the reason
    rows_in: int = 0
    rows_used: int = 0


def _column_types(analysis: dict) -> dict[str, str]:
    return {c["name"]: c["semantic_type"] for c in analysis["profile"]["columns"]}


def _issues(analysis: dict, *types: str) -> list[dict]:
    return [i for i in analysis["quality"]["issues"] if i["issue_type"] in types]


def _deduction_columns(analysis: dict, issue_type: str) -> list[str]:
    out = []
    for dim in analysis["scores"]["ml_readiness"]["dimensions"]:
        for d in dim["deductions"]:
            if d["issue_type"] == issue_type and d["column"] and d["column"] not in out:
                out.append(d["column"])
    return out


def task_of(analysis: dict) -> str | None:
    t = analysis["statistics"].get("target")
    return t["task"] if t else None


def _target_values(series: pd.Series, task: str, settings: Settings) -> pd.Series:
    """Usable target values: missing and placeholder values removed (never imputed)."""
    usable = series.notna() & ~disguised_missing_mask(series, settings)
    if task == "regression":
        y = to_numeric(series).where(usable)
    else:
        y = series.where(usable).astype("string").str.strip()
    return y


def _merge_spellings(s: pd.Series) -> tuple[pd.Series, int]:
    """'Chennai' / 'chennai ' -> the most common spelling of that label. Returns (series, cells changed)."""
    text = s.astype("string")
    key = text.str.strip().str.casefold()
    most_common = (pd.DataFrame({"key": key, "value": text}).dropna()
                   .groupby("key")["value"].agg(lambda v: v.value_counts().index[0]))
    merged = key.map(most_common).where(text.notna())
    changed = int((merged.notna() & (merged != text)).fillna(False).sum())
    return merged, changed


def prepare(df: pd.DataFrame, analysis: dict, target: str, clean: bool,
            settings: Settings = DEFAULT_SETTINGS, drop_columns: tuple[str, ...] = ()) -> PreparedData:
    task = task_of(analysis)
    if task is None:
        raise ValueError("The analysis has no target summary; run it with a target column.")
    types = _column_types(analysis)
    steps: list[str] = []
    data = df.copy()
    rows_in = len(data)

    # ---- columns: same exclusions for raw and cleaned
    excluded = [{"column": c, "reason": EXCLUDED_TYPES[t]}
                for c, t in types.items() if c != target and t in EXCLUDED_TYPES and c in data]
    drop = {e["column"] for e in excluded}

    if clean:
        flagged = (
            [(i["column"], "almost no variation (constant column)")
             for i in _issues(analysis, "constant_column", "near_constant_column")]
            + [(c, "redundant with another column") for c in _deduction_columns(analysis, "redundant_feature")]
        )
        dropped_now = []
        for col, reason in flagged:
            if col and col != target and col not in drop and col in data:
                excluded.append({"column": col, "reason": f"dropped by cleaning: {reason}"})
                drop.add(col)
                dropped_now.append(col)
        if dropped_now:
            steps.append(f"Dropped {len(dropped_now)} column(s) the report flagged as constant or "
                         f"redundant: {', '.join(dropped_now)}.")
    for col in drop_columns:
        if col in data and col != target and col not in drop:
            excluded.append({"column": col, "reason": "left out to measure its effect: possible target leakage"})
            drop.add(col)

    features = [c for c in data.columns if c != target and c not in drop]

    # ---- rows: duplicates (cleaned only). IDs are ignored, so duplicates hidden by an ID are caught too.
    if clean:
        keys = [c for c in data.columns if types.get(c) != "identifier"]
        before = len(data)
        data = data.drop_duplicates(subset=keys)
        if before - len(data):
            steps.append(f"Removed {before - len(data)} duplicate row(s) (ID columns ignored when comparing).")

    # ---- feature values
    X = pd.DataFrame(index=data.index)
    numeric, categorical = [], []
    placeholder_cells = range_cells = spelling_cells = 0
    variant_cols = {i["column"] for i in _issues(analysis, "inconsistent_format")}
    for col in features:
        s = data[col]
        kind = types.get(col, "categorical")
        if clean:
            mask = disguised_missing_mask(s, settings)
            placeholder_cells += int(mask.sum())
            s = s.mask(mask)
        if kind in NUMERIC_TYPES:
            values = to_numeric(s)
            if clean and (rng := valid_range(col, settings)) is not None:
                bad = values.notna() & ((values < rng[0]) | (values > rng[1]))
                range_cells += int(bad.sum())
                values = values.mask(bad)
            X[col] = values
            numeric.append(col)
        elif kind in DATETIME_TYPES:
            dates = to_datetime(s)
            X[col] = (dates - pd.Timestamp("1970-01-01")).dt.days.astype(float)   # NaT -> NaN
            numeric.append(col)
        else:
            if clean and col in variant_cols:
                s, changed = _merge_spellings(s)
                spelling_cells += changed
            X[col] = s.map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
            categorical.append(col)

    if clean:
        if placeholder_cells:
            steps.append(f"Set {placeholder_cells} placeholder value(s) (e.g. 'Unknown', -999) to missing.")
        if range_cells:
            steps.append(f"Set {range_cells} impossible value(s) (outside the valid range) to missing.")
        if spelling_cells:
            steps.append(f"Merged {spelling_cells} label(s) written in a different case/spacing.")

    # ---- target: rows without a usable target are dropped in BOTH versions (a target is never imputed)
    y = _target_values(data[target], task, settings)
    keep = y.notna()
    if int((~keep).sum()):
        if clean:
            steps.append(f"Dropped {int((~keep).sum())} row(s) with no usable target value.")
    X, y = X[keep], y[keep]

    return PreparedData(X=X, y=y, numeric=numeric, categorical=categorical, steps=steps,
                        excluded=excluded, rows_in=rows_in, rows_used=len(X))
