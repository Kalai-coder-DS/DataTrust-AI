"""
datatrust/ml/importance.py

Phase 8: which columns does the baseline model rely on, and do they have data problems?

Method: permutation importance, cross-validated.
  For each of the k folds: train the Phase 7 model on the other folds, then shuffle ONE column
  at a time in the test fold and measure how much the score drops. A big drop = the model
  relies on that column; ~0 = it doesn't. Averaged over folds; the spread gives the noise.

Why permutation importance and not SHAP (interview talking points):
  - One number per ORIGINAL column ('city'), not per one-hot piece ('city_Chennai', ...):
    that is the level at which the report lists data problems, so the two can be joined.
  - Measured on held-out rows, in the model's own metric (balanced accuracy / R²):
    "shuffling 'smoker' costs 0.40 R²" is directly comparable with the Phase 7 scores.
  - Needs only scikit-learn (already required). SHAP explains individual predictions, which
    this tool does not need; it can be added later as an optional extra.
  - Known limit: two strongly correlated columns share importance (shuffling one, the model
    falls back on the other), so both can look unimportant. Redundant columns are dropped by
    cleaning before this runs, which removes most of that effect, and the report says so.

The join with Phase 2 is the useful part: "the 2nd most important column has 12% placeholders"
says which fix to do FIRST; "this column has problems but the model ignores it" says which
fix can wait (for this model).
"""
from __future__ import annotations

import warnings

import numpy as np

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.ml.features import PreparedData

# A column counts as "relied on" only if its importance is BOTH
#   - clearly above zero: > 3 standard errors across folds (folds share training rows, so the
#     usual 2 understates the noise), and > MIN_IMPORTANCE in absolute terms, and
#   - big enough to matter: >= RELATIVE_FLOOR x the most important column.
# Found in testing: on a few hundred rows, columns drawn at random ('plan' in the sample, a pure
# noise column in a synthetic file) reached 0.01-0.017 importance at 2.5-3.4 standard errors,
# because the whole sample happens to correlate a little with the target and cross-validation
# cannot see a pattern every fold shares. The relative floor removes those false alarms.
MIN_IMPORTANCE = 0.005
SE_FACTOR = 3
RELATIVE_FLOOR = 0.05
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
# Issues that are about the column's VALUES (what a model reads). Whole-dataset issues and
# "this column is an ID / constant" are excluded: those columns are not features anyway.
VALUE_ISSUES = {
    "missing_values", "disguised_missing", "type_mismatch", "invalid_date", "out_of_range",
    "inconsistent_format", "numbers_as_words", "outliers",
}


def column_problems(analysis: dict, column: str) -> list[dict]:
    """Phase 2 issues on this column's values, most severe first."""
    out = []
    for i in analysis["quality"]["issues"]:
        if i.get("column") != column or i["issue_type"] not in VALUE_ISSUES:
            continue
        out.append({"issue_type": i["issue_type"], "severity": i["severity"],
                    "affected_pct": i.get("affected_pct"), "message": i["message"]})
    return sorted(out, key=lambda p: SEVERITY_RANK.get(p["severity"], 9))


def compute_importance(prep: PreparedData, task: str, scoring: str, pipeline_factory,
                       settings: Settings = DEFAULT_SETTINGS) -> dict:
    """
    Cross-validated permutation importance of every feature column in prep.
    pipeline_factory() returns a fresh, unfitted Phase 7 pipeline (same model as the baseline).
    """
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold, StratifiedKFold

    X, y = prep.X, prep.y
    columns = list(X.columns)
    if task == "classification":
        counts = y.value_counts()
        keep = y.isin(counts[counts >= 2].index)
        X, y = X[keep], y[keep]
        folds = int(min(settings.ml_cv_folds, y.value_counts().min()))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=settings.ml_random_state)
    else:
        folds = int(max(2, min(settings.ml_cv_folds, len(X) // 10)))
        cv = KFold(n_splits=folds, shuffle=True, random_state=settings.ml_random_state)

    per_fold = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train, test in cv.split(X, y):
            model = pipeline_factory().fit(X.iloc[train], y.iloc[train])
            r = permutation_importance(model, X.iloc[test], y.iloc[test], scoring=scoring,
                                       n_repeats=settings.ml_importance_repeats,
                                       random_state=settings.ml_random_state)
            per_fold.append(r.importances_mean)
    per_fold = np.array(per_fold)                       # folds x columns
    mean = per_fold.mean(axis=0)
    se = per_fold.std(axis=0, ddof=1) / np.sqrt(len(per_fold)) if len(per_fold) > 1 else np.zeros(len(columns))
    top = float(max(mean.max(), 0.0))
    return {
        "folds": folds,
        "repeats": settings.ml_importance_repeats,
        "columns": [
            {"column": c, "importance": round(float(m), 4), "se": round(float(s), 4),
             "relied_on": bool(m > max(MIN_IMPORTANCE, SE_FACTOR * s) and m >= RELATIVE_FLOOR * top)}
            for c, m, s in sorted(zip(columns, mean, se), key=lambda t: -t[1])
        ],
    }


def explain(importance: dict, analysis: dict, metric_label: str, suspects: list[str]) -> dict:
    """Join importance with the Phase 2 issues and write the verdict."""
    cols = importance["columns"]
    for rank, c in enumerate(cols, start=1):
        c["rank"] = rank
        c["problems"] = column_problems(analysis, c["column"])
        c["suspect"] = c["column"] in suspects

    relied = [c for c in cols if c["relied_on"]]
    fix_first = [c for c in relied if c["problems"]]
    can_wait = [c for c in cols if c["problems"] and not c["relied_on"]]
    verdict = []
    if not relied:
        verdict.append("No single column makes a measurable difference on its own: the model has little to "
                       "rely on (see the baseline verdict), or the signal is spread over many columns.")
    else:
        total = sum(c["importance"] for c in relied)
        if len(relied) == 1:
            verdict.append(f"The model relies on one column only, '{relied[0]['column']}'; "
                           f"shuffling any of the other {len(cols) - 1} makes no measurable difference.")
        else:
            top = relied[:3]
            share = sum(c["importance"] for c in top) / total if total > 0 else 0
            names = ", ".join(f"'{c['column']}'" for c in top)
            which = "Together they" if len(top) == len(relied) else f"The top {len(top)} ({names})"
            verdict.append(f"The model relies on {len(relied)} of {len(cols)} columns"
                           + (f": {names}." if len(top) == len(relied) else f". {which} account for {share:.0%} of the importance."))
    if fix_first:
        parts = [f"'{c['column']}' (#{c['rank']}: {c['problems'][0]['message'].rstrip('.')})" for c in fix_first[:3]]
        verdict.append("Fix first: columns the model relies on that have data problems: " + "; ".join(parts) + ".")
    elif relied:
        verdict.append("None of the columns the model relies on has a data problem flagged on its values.")
    if can_wait:
        names = ", ".join(f"'{c['column']}'" for c in can_wait[:5])
        if len(can_wait) > 5:
            names += f" and {len(can_wait) - 5} more"
        verb = "has" if len(can_wait) == 1 else "have"
        verdict.append(f"Can wait (for this model): {names} {verb} data problems but no measurable effect on "
                       "the predictions. They still matter for reports and averages that read the values.")
    for c in cols:
        if c["suspect"] and c["relied_on"] and c["rank"] == 1:
            verdict.append(f"'{c['column']}' is both the most important column and flagged as possible target "
                           "leakage: confirm it is known before the outcome (see the baseline section).")
    return {
        **importance,
        "metric_label": metric_label,
        "verdict": verdict,
        "note": ("Importance = how much " + metric_label + " drops when the column is shuffled in held-out rows. "
                 "'Relied on' = clearly above the noise (3 standard errors) and at least 5% of the top column. "
                 "Two columns that carry the same information share it, so each can look less important "
                 "than the information really is."),
    }
