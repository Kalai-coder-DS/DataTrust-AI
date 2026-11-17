"""
tests/test_speedups.py

Phase 10 made two checks work on DISTINCT values instead of every row. These tests compare them
with the simple per-row versions they replaced, on random messy data, so the speed-up can never
change a result. (Measured on 300,000 rows: whole analysis 4.95 s -> 3.58 s.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datatrust.config import DEFAULT_SETTINGS
from datatrust.quality.consistency import ConsistencyCheck
from datatrust.quality.rules import disguised_missing_mask

LABELS = ["Chennai", "chennai", " Chennai ", "CHENNAI", "Mumbai", "mumbai  ", "Delhi", "Unknown", "unknown ",
          "N/A", "", "   ", "na", "-", "two", "Two", None, np.nan]


def reference_mask(series: pd.Series, settings) -> pd.Series:
    """The pre-Phase-10 implementation, row by row."""
    mask = pd.Series(False, index=series.index)
    non_null = series.notna()
    if not non_null.any():
        return mask
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.isin(settings.sentinel_numbers) & non_null
    text = series[non_null].astype(str).str.strip().str.lower()
    hits = text.isin(set(settings.disguised_missing_tokens)) | (text == "")
    mask.loc[hits.index] = hits
    return mask


@pytest.mark.parametrize("seed", range(6))
def test_mask_matches_the_row_by_row_version(seed):
    rng = np.random.default_rng(seed)
    n = 2000
    columns = [
        pd.Series(rng.choice(np.array(LABELS, dtype=object), n), index=rng.permutation(n) + 10),   # shuffled index
        pd.Series(rng.choice([1, 2, -999, 999, 0, np.nan], n)),
        pd.Series(rng.choice(np.array(["a", 1, 2.5, "Unknown", None, True], dtype=object), n)),
        pd.Series(rng.choice(LABELS[:7], n), dtype="string"),
        pd.Series([None] * 5, dtype=object),
    ]
    for s in columns:
        got = disguised_missing_mask(s, DEFAULT_SETTINGS)
        want = reference_mask(s, DEFAULT_SETTINGS)
        pd.testing.assert_series_equal(got, want, check_names=False)


def reference_variants(values: pd.Series) -> dict:
    norm = values.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    counts = pd.DataFrame({"raw": values, "norm": norm}).groupby(["norm", "raw"]).size()
    out = {}
    for key, raw in counts.groupby(level=0):
        if len(raw) >= 2:
            raw = raw.droplevel(0).sort_values(ascending=False)
            out[str(key)] = {str(k): int(v) for k, v in raw.items()}
    return out


@pytest.mark.parametrize("seed", range(6))
def test_spelling_variants_match_the_row_by_row_version(seed):
    rng = np.random.default_rng(seed)
    values = pd.Series(rng.choice([l for l in LABELS if isinstance(l, str) and l.strip()], 3000))
    issue = ConsistencyCheck(DEFAULT_SETTINGS)._inconsistent_format("city", values, len(values))
    want = reference_variants(values)
    assert (issue.details["variant_groups"] if issue else {}) == want
    if issue:
        assert issue.affected_rows == sum(sum(sorted(g.values())[:-1]) for g in want.values())
