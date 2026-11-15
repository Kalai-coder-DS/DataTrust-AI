"""
datatrust/stats/target.py

Target-column analysis: the column a future ML model should predict
(e.g. 'churned', 'price', 'charges').

  - Few distinct values (labels, yes/no, 0/1)  -> CLASSIFICATION -> check class balance
  - Many distinct numbers                      -> REGRESSION     -> distribution is in the numeric summary

Class imbalance (e.g. 95% 'No', 5% 'Yes') matters because a model can score 95%
accuracy by always predicting 'No' and still be useless. Imbalance ratio =
majority class count / minority class count:

    <= 1.5   balanced
    <= 3     moderate  (usually fine)
    <= 10    high      (use stratified splits, class weights, and metrics like F1/recall)
    > 10     severe    (consider resampling; accuracy is misleading)

The target is chosen by the user. Guessing it would be unreliable, and a
wrong guess would silently produce a misleading ML Readiness Score.
"""
from __future__ import annotations

import pandas as pd

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.profiling import DatasetProfile, SemanticType
from datatrust.quality.models import to_python
from datatrust.quality.rules import usable_values
from datatrust.stats.models import TargetSummary

_UNUSABLE_TARGET_TYPES = (SemanticType.EMPTY, SemanticType.IDENTIFIER, SemanticType.TEXT, SemanticType.DATETIME)


def summarize_target(df: pd.DataFrame, profile: DatasetProfile, target: str,
                     settings: Settings = DEFAULT_SETTINGS) -> TargetSummary:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {', '.join(map(str, df.columns))}")
    col = profile.get_column(target)
    if col.semantic_type in _UNUSABLE_TARGET_TYPES:
        raise ValueError(
            f"Column '{target}' is {col.semantic_type.value}; a prediction target must be "
            f"numeric, categorical or boolean."
        )

    values = usable_values(df[target], settings)
    n_distinct = int(values.nunique())
    if col.semantic_type == SemanticType.NUMERIC and n_distinct > settings.max_classes_for_classification:
        return TargetSummary(column=target, task="regression")

    counts = values.astype(str).value_counts()
    if counts.empty:
        raise ValueError(f"Target column '{target}' has no usable values.")
    minority_value, minority_count = counts.index[-1], int(counts.iloc[-1])
    ratio = round(int(counts.iloc[0]) / minority_count, 2)
    return TargetSummary(
        column=target,
        task="classification",
        n_classes=len(counts),
        class_counts={str(k): int(v) for k, v in counts.items()},
        minority_class=to_python(minority_value),
        minority_pct=round(minority_count / int(counts.sum()) * 100, 2),
        imbalance_ratio=ratio,
        balance=_balance_label(ratio, len(counts)),
    )


def _balance_label(ratio: float, n_classes: int) -> str:
    if n_classes < 2:
        return "single class (cannot train a classifier)"
    if ratio <= 1.5:
        return "balanced"
    if ratio <= 3:
        return "moderate"
    if ratio <= 10:
        return "high"
    return "severe"
