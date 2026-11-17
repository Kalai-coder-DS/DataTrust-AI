"""
datatrust/ml/baseline.py

One fixed, simple model, trained and scored three ways on the same rows:

    guessing   a model that ignores the features (most frequent class / the average value)
    raw        gradient boosting on the data as uploaded
    cleaned    gradient boosting after the report's fixes (see features.py)
    without    only if Phase 4 flagged possible target leakage: cleaned, minus those columns.
               Shows how much of the model's skill depends on them; a person decides whether
               they are genuine (known before the outcome) or leakage.

Why these choices (interview talking points):
  - Gradient boosting (scikit-learn HistGradientBoosting, 100 trees, fixed settings): strong on
    tables, handles missing values natively, needs no scaling, fast. NOT tuned: this is a
    measuring instrument, so it must be the same every time.
  - k-fold cross-validation (5 folds, stratified for classification), repeated 3 times with new
    shuffles on files up to 5,000 rows: every row is tested several times and the spread of the
    15 scores gives the noise level. Same random seed every run.
  - A change (or a claim of signal) counts only if it is bigger than the noise (about 2 standard
    errors over one round of folds). Found while
    testing: on the 515-row sample, the raw-vs-cleaned gap moved between 0.001 and 0.034 just by
    changing the random seed; a fixed threshold would have reported noise as an effect.
  - Metrics that are fair when classes are imbalanced: balanced accuracy (guessing = 1 / number
    of classes, e.g. 0.5 for yes/no) and macro F1. For regression: R2 (guessing the average = 0)
    and mean absolute error in the target's own units.
  - Big files are sampled (default 20,000 rows, class-stratified) so the web page stays fast.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.logger import get_logger
from datatrust.ml.features import PreparedData, _deduction_columns, prepare, task_of
from datatrust.ml.present import f3

logger = get_logger(__name__)

MODEL_NAME = "Gradient boosting (scikit-learn HistGradientBoosting, 100 trees, not tuned)"
METRICS = {
    "classification": {
        "primary": {"key": "balanced_accuracy", "label": "Balanced accuracy",
                    "about": "share of each class predicted correctly, averaged over classes; "
                             "guessing scores 1 / number of classes"},
        "secondary": {"key": "f1_macro", "label": "Macro F1", "about": "F1 averaged over classes"},
        "weak_lift": 0.05,
    },
    "regression": {
        "primary": {"key": "r2", "label": "R²",
                    "about": "share of the target's variation explained; guessing the average scores 0"},
        "secondary": {"key": "neg_mean_absolute_error", "label": "Mean absolute error",
                      "about": "average size of the error, in the target's units (lower is better)"},
        "weak_lift": 0.10,
    },
}
MEANINGFUL_CHANGE = 0.01      # smallest change ever reported as real, even when the noise is lower


def _sample(df: pd.DataFrame, target: str, task: str, settings: Settings) -> tuple[pd.DataFrame, int | None]:
    if len(df) <= settings.ml_max_rows:
        return df, None
    if task == "classification":
        labels = df[target].astype("string").fillna("(missing)")
        # Stratified sample: every class keeps its share. groupby().sample needs a fraction.
        frac = settings.ml_max_rows / len(df)
        sample = df.groupby(labels, group_keys=False).sample(frac=frac, random_state=settings.ml_random_state)
    else:
        sample = df.sample(n=settings.ml_max_rows, random_state=settings.ml_random_state)
    return sample, len(sample)


def _pipeline(prep: PreparedData, task: str, settings: Settings):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    transformers = []
    if prep.numeric:
        transformers.append(("numeric", "passthrough", prep.numeric))   # boosting handles NaN itself
    if prep.categorical:
        transformers.append(("categorical", make_pipeline(
            SimpleImputer(strategy="constant", fill_value="(missing)"),
            OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=settings.ml_max_categories,
                          sparse_output=False),
        ), prep.categorical))
    common = dict(max_iter=100, early_stopping=False, random_state=settings.ml_random_state)
    model = (HistGradientBoostingClassifier(class_weight="balanced", **common) if task == "classification"
             else HistGradientBoostingRegressor(**common))
    return Pipeline([("prepare", ColumnTransformer(transformers)), ("model", model)])


def _evaluate(prep: PreparedData, task: str, settings: Settings) -> dict:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_validate

    base = {"rows": prep.rows_used, "features": len(prep.numeric) + len(prep.categorical),
            "excluded": prep.excluded, "steps": prep.steps}
    X, y = prep.X, prep.y
    if base["features"] == 0:
        return {**base, "status": "skipped", "reason": "No usable feature columns are left."}
    if len(X) < settings.ml_min_rows:
        return {**base, "status": "skipped",
                "reason": f"Only {len(X)} usable row(s); at least {settings.ml_min_rows} are needed."}

    notes = []
    if task == "classification":
        counts = y.value_counts()
        rare = counts[counts < 2].index
        if len(rare):
            notes.append(f"{len(rare)} class(es) with a single row left out (can't be both trained and tested).")
            X, y = X[~y.isin(rare)], y[~y.isin(rare)]
            counts = y.value_counts()
        if len(counts) < 2:
            return {**base, "status": "skipped", "reason": "The target has fewer than two classes with 2+ rows."}
    repeats = settings.ml_cv_repeats if len(X) <= settings.ml_repeat_max_rows else 1
    if task == "classification":
        folds = int(min(settings.ml_cv_folds, counts.min()))
        cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=settings.ml_random_state)
        dummy = DummyClassifier(strategy="prior")
    else:
        folds = int(max(2, min(settings.ml_cv_folds, len(X) // 10)))
        cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=settings.ml_random_state)
        dummy = DummyRegressor(strategy="mean")
    if folds < settings.ml_cv_folds:
        notes.append(f"{folds}-fold cross-validation (smallest class is too small for {settings.ml_cv_folds}).")

    m = METRICS[task]
    scoring = {"primary": m["primary"]["key"], "secondary": m["secondary"]["key"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # e.g. unseen categories in a test fold: encoded as 'other'
        model_cv = cross_validate(_pipeline(prep, task, settings), X, y, cv=cv, scoring=scoring, error_score="raise")
        dummy_cv = cross_validate(dummy, X, y, cv=cv, scoring=scoring)

    sign = -1 if m["secondary"]["key"].startswith("neg_") else 1
    scores = model_cv["test_primary"]
    score = float(np.mean(scores))
    guess = float(np.mean(dummy_cv["test_primary"]))
    return {
        **base, "status": "ok", "rows": len(X), "folds": folds, "repeats": repeats, "notes": notes,
        "score": round(score, 4),
        "score_std": round(float(np.std(scores)), 4),
        # Standard error over ONE round of folds, not all repeats: repeated rounds reuse the same rows,
        # so their scores are strongly correlated. Dividing by all 15 understated the noise in testing
        # (a pure-noise target was reported as 'signal' at 150 rows).
        "score_se": round(float(np.std(scores, ddof=1) / np.sqrt(folds)), 4),
        "fold_scores": [round(float(v), 4) for v in model_cv["test_primary"]],
        "secondary": round(sign * float(np.mean(model_cv["test_secondary"])), 4),
        "guess": round(guess, 4),
        "guess_secondary": round(sign * float(np.mean(dummy_cv["test_secondary"])), 4),
        "lift": round(score - guess, 4),
    }


def _verdict(result: dict) -> list[str]:
    m = METRICS[result["task"]]
    label, target = m["primary"]["label"], result["target"]
    raw, cleaned = result["raw"], result["cleaned"]
    trusted = cleaned if cleaned["status"] == "ok" else raw
    out = []
    if trusted["lift"] < max(m["weak_lift"], 2 * trusted["score_se"]):
        out.append(f"The model is barely better than guessing ({label} {f3(trusted['score'])} vs "
                   f"{f3(trusted['guess'])}): the features carry little information about '{target}'. "
                   "A good ML Readiness score means the data is healthy, not that it can predict this target.")
    else:
        out.append(f"The model beats guessing by +{f3(trusted['lift'])} {label} "
                   f"({f3(trusted['score'])} vs {f3(trusted['guess'])}): the features do carry signal about '{target}'.")

    change, noise = result["change"], result["noise"]
    if change is None:
        return out + _suspect_verdict(result, label)
    if not cleaned["steps"] and not any(e["reason"].startswith("dropped by cleaning") for e in cleaned["excluded"]):
        out.append("The report found nothing to clean, so both runs use the same data.")
        return out + _suspect_verdict(result, label)
    inflating = []
    if any("duplicate row" in s for s in cleaned["steps"]):
        inflating.append("duplicate rows (the same row lands in both training and test folds)")
    if change > noise:
        out.append(f"Cleaning raised {label} by +{f3(change)} (noise ±{f3(noise)}): "
                   "the fixes in this report make a measurable difference.")
    elif change < -noise and inflating:
        out.append(f"Cleaning lowered {label} by {f3(abs(change))} (noise ±{f3(noise)}). That is expected here: "
                   f"{' and '.join(inflating)} made the raw score look better than it really is. "
                   "The cleaned score is the honest estimate.")
    elif change < -noise:
        out.append(f"Cleaning lowered {label} by {f3(abs(change))} (noise ±{f3(noise)}): some of the removed rows "
                   "or columns carried signal. Review the dropped columns before applying every fix.")
    else:
        out.append(f"The difference ({'+' if change >= 0 else '-'}{f3(abs(change))}) is within the run-to-run noise (±{f3(noise)}): cleaning has "
                   "no measurable effect on this model, which copes with these defects. The fixes still matter "
                   "for anything that reads the values directly (averages, dashboards, simpler models).")
    if inflating and change >= -noise:
        out.append("Note: the raw score may still be inflated by " + " and ".join(inflating) + ".")
    return out + _suspect_verdict(result, label)


def _suspect_verdict(result: dict, label: str) -> list[str]:
    w = result.get("without_suspects")
    if not w or w["status"] != "ok" or result["cleaned"]["status"] != "ok":
        return []
    cols = ", ".join(f"'{c}'" for c in w["columns"])
    drop = result["cleaned"]["score"] - w["score"]
    noise = max(MEANINGFUL_CHANGE, 2 * float(np.hypot(result["cleaned"]["score_se"], w["score_se"])))
    if drop > noise:
        return [f"Most of the model's skill comes from {cols}: without it, {label} falls from "
                f"{f3(result['cleaned']['score'])} to {f3(w['score'])}. The report flagged it as possible target "
                f"leakage. If {cols} is known BEFORE '{result['target']}' is decided, this is genuine signal; if it "
                "is recorded after, or derived from, the outcome, expect the lower score in real use."]
    return [f"Leaving out {cols} (flagged as possible leakage) changes {label} by less than the noise "
            f"({f3(result['cleaned']['score'])} vs {f3(w['score'])}): the model does not depend on it."]


def _importance(data, analysis: dict, target: str, task: str, suspects: tuple, settings: Settings) -> dict | None:
    """Phase 8: which columns the cleaned-data model relies on, joined with their data problems."""
    from datatrust.ml.importance import compute_importance, explain
    prep = prepare(data, analysis, target, clean=True, settings=settings)
    if len(prep.numeric) + len(prep.categorical) == 0 or len(prep.X) < settings.ml_min_rows:
        return None
    m = METRICS[task]
    try:
        imp = compute_importance(prep, task, m["primary"]["key"], lambda: _pipeline(prep, task, settings), settings)
    except ValueError as exc:            # e.g. a class too small to split: the baseline already says so
        logger.warning("Feature importance skipped: %s", exc)
        return None
    return explain(imp, analysis, m["primary"]["label"], list(suspects))


def run_baseline(df: pd.DataFrame, analysis: dict, target: str | None,
                 settings: Settings = DEFAULT_SETTINGS) -> dict:
    """Train and score the baseline model on raw and cleaned data. Never raises for data problems."""
    if not target:
        return {"status": "skipped", "reason": "A baseline model needs a target column."}
    task = task_of(analysis)
    if task is None:
        return {"status": "skipped", "reason": f"No target summary for '{target}'."}
    try:
        import sklearn
    except ImportError:
        return {"status": "skipped", "reason": "scikit-learn is not installed: pip install -r requirements.txt"}

    data, sampled = _sample(df, target, task, settings)
    raw = _evaluate(prepare(data, analysis, target, clean=False, settings=settings), task, settings)
    cleaned = _evaluate(prepare(data, analysis, target, clean=True, settings=settings), task, settings)
    suspects = tuple(c for c in _deduction_columns(analysis, "target_leakage") if c in data and c != target)
    without = None
    if suspects:
        without = _evaluate(prepare(data, analysis, target, clean=True, settings=settings, drop_columns=suspects),
                            task, settings)
        without["columns"] = list(suspects)
    m = METRICS[task]
    result = {
        "status": "ok" if "ok" in (raw["status"], cleaned["status"]) else "skipped",
        "target": target, "task": task, "model": MODEL_NAME, "sklearn_version": sklearn.__version__,
        "metric": m["primary"], "secondary_metric": m["secondary"],
        "sampled_rows": sampled, "rows_in_file": len(df),
        "raw": raw, "cleaned": cleaned, "without_suspects": without,
        "change": (round(cleaned["score"] - raw["score"], 4)
                   if raw["status"] == cleaned["status"] == "ok" else None),
        "ml_readiness": analysis["scores"]["ml_readiness"]["score"],
    }
    result["noise"] = (round(max(MEANINGFUL_CHANGE, 2 * float(np.hypot(raw["score_se"], cleaned["score_se"]))), 4)
                       if result["change"] is not None else None)
    if result["status"] == "skipped":
        result["reason"] = cleaned.get("reason") or raw.get("reason")
        return result
    result["verdict"] = _verdict(result)
    result["importance"] = _importance(data, analysis, target, task, suspects, settings)
    logger.info("Baseline model for '%s': raw %s, cleaned %s (%s)", target,
                raw.get("score"), cleaned.get("score"), m["primary"]["label"])
    return result
