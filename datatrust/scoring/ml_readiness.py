"""
datatrust/scoring/ml_readiness.py

ML READINESS SCORE: "Could I train a reliable model on this data today?"

  Data trust      (40%)  the Data Trust Score itself: bad data -> bad model
  Sample size     (15%)  enough rows overall, and enough rows per feature
  Feature quality (20%)  useless, redundant or hard-to-use columns
  Target          (25%)  only when a target is given: class balance, missing labels, leakage

If no target is given, the target dimension is skipped, the other weights are
rescaled to sum to 1, and the result says the assessment is partial.

CAP: ML Readiness can never exceed the Data Trust Score. A model cannot be more
trustworthy than the data it learns from. (Without this rule, the dirty sample file
scored Trust 79.8 (C) but ML 84.1 (B), because size and target were fine.)
While the cap applies, each trust point fixed raises ML Readiness one-for-one, so
the trust deductions are shown at their full impact in Top actions, and other
deductions show 0 impact (fixing them can't move the score until trust improves).

Rules of thumb used (and why):
  Sample size  - fewer than ~10 rows per feature makes overfitting likely;
                 under ~100 rows any model is fragile. The weaker of the two wins.
  Redundancy   - charged per column you could DROP, not per pair: 4 columns that are
                 all ~0.95 correlated form 6 pairs, but only 3 columns are redundant.
  Leakage      - a feature that predicts the target almost perfectly (>= 0.95) usually
                 contains the answer (e.g. 'refund_issued' when predicting 'complaint').
                 The model looks brilliant in testing and fails in production.
"""
from __future__ import annotations

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.profiling import DatasetProfile, SemanticType
from datatrust.quality import IssueType, QualityReport
from datatrust.scoring.models import (
    Deduction,
    DimensionScore,
    ScoreResult,
    combine,
    finalize_dimension,
    grade_for,
)
from datatrust.stats import StatisticsReport

_BALANCE_POINTS = {"balanced": 0.0, "moderate": 10.0, "high": 35.0, "severe": 65.0}
_SINGLE_CLASS_CAP = 40.0


def score_ml_readiness(trust: ScoreResult, quality: QualityReport, stats: StatisticsReport,
                       profile: DatasetProfile, target: str | None = None,
                       settings: Settings = DEFAULT_SETTINGS) -> ScoreResult:
    raw_weights = dict(settings.ml_weights)
    if target is None:
        raw_weights.pop("target")
    total = sum(raw_weights.values())
    w = {name: weight / total for name, weight in raw_weights.items()}

    notes: list[str] = []
    caps: list[str] = []
    features = [c for c in profile.columns
                if c.semantic_type not in (SemanticType.IDENTIFIER, SemanticType.EMPTY) and c.name != target]

    dimensions = [
        _data_trust(trust, w["data_trust"]),
        finalize_dimension("sample_size", w["sample_size"], _sample_size(profile.n_rows, len(features))),
        finalize_dimension("feature_quality", w["feature_quality"],
                           _feature_quality(profile, quality, stats, target, notes)),
    ]
    if target is not None:
        target_deductions, single_class = _target(target, quality, stats, settings)
        dimensions.append(finalize_dimension("target", w["target"], target_deductions))
    else:
        single_class = False
        notes.append("No target given: target balance and leakage were not assessed (partial score). "
                     "Use --target COLUMN for a complete assessment.")

    score = combine(dimensions)
    if score > trust.score:
        caps.append(f"Capped at the Data Trust Score ({trust.score}): a model cannot be more "
                    f"trustworthy than its data (uncapped: {score}).")
        score = trust.score
        for d in dimensions[0].deductions:  # data_trust: now every trust point counts fully
            d.impact = d.trust_impact
        for dim in dimensions[1:]:          # the rest can't move the score until trust improves
            for d in dim.deductions:
                d.impact = 0.0
        if any(dim.deductions for dim in dimensions[1:]):
            notes.append("Sample size, feature and target improvements are listed under their dimensions; "
                         "they will raise this score only after data-trust issues are fixed.")
    if single_class and score > _SINGLE_CLASS_CAP:
        score = _SINGLE_CLASS_CAP
        caps.append(f"Capped at {_SINGLE_CLASS_CAP:g}: the target has only one class, so no classifier can learn.")

    return ScoreResult(name="ML Readiness Score", score=score, grade=grade_for(score),
                       dimensions=dimensions, caps=caps, notes=notes)


# ------------------------------------------------------------------ dimensions
def _data_trust(trust: ScoreResult, weight: float) -> DimensionScore:
    """Re-uses the trust score; its deductions are carried over so they appear in Top actions."""
    carried = []
    for d in trust.all_deductions():
        carried.append(Deduction(
            dimension="data_trust", points=d.points, reason=d.reason, recommendation=d.recommendation,
            column=d.column, issue_type=d.issue_type, impact=round(d.impact * weight, 2),
            trust_impact=d.impact,
        ))
    return DimensionScore(name="data_trust", weight=weight, score=trust.score, deductions=carried,
                          note=f"Equals the Data Trust Score ({trust.score}); deductions shown are the trust "
                               f"deductions, weighted by their trust dimension.")


def _sample_size(n_rows: int, n_features: int) -> list[Deduction]:
    by_rows = _band(n_rows, ((1000, 100), (500, 90), (200, 75), (100, 60), (50, 40)), floor=20)
    ratio = n_rows / n_features if n_features else float(n_rows)
    by_ratio = _band(ratio, ((50, 100), (20, 85), (10, 65), (5, 40)), floor=20)
    score = min(by_rows, by_ratio)
    if score >= 100:
        return []
    if by_ratio <= by_rows:
        reason = (f"Only {ratio:.1f} rows per feature ({n_rows} rows, {n_features} features); "
                  "models may overfit below ~10-20 rows per feature.")
        rec = "Collect more rows, or reduce features (drop redundant/uninformative columns)."
    elif score >= 75:
        reason = f"{n_rows} rows is adequate but on the small side; evaluation results will vary between splits."
        rec = "Use cross-validation rather than a single train/test split; more data would help."
    else:
        reason = f"Only {n_rows} rows in total; small datasets give unstable models and unreliable evaluation."
        rec = "Collect more data, and use cross-validation instead of a single train/test split."
    return [Deduction(dimension="sample_size", points=100 - score, reason=reason, recommendation=rec)]


def _feature_quality(profile: DatasetProfile, quality: QualityReport, stats: StatisticsReport,
                     target: str | None, notes: list[str]) -> list[Deduction]:
    out: list[Deduction] = []
    for issue in quality.issues:
        if issue.column == target:
            continue
        if issue.issue_type == IssueType.CONSTANT_COLUMN:
            out.append(Deduction("feature_quality", 15, f"'{issue.column}' is constant: no information.",
                                 "Drop it before training.", issue.column, issue.issue_type.value))
        elif issue.issue_type == IssueType.NEAR_CONSTANT_COLUMN:
            out.append(Deduction("feature_quality", 5, f"'{issue.column}' is near-constant.",
                                 issue.recommendation, issue.column, issue.issue_type.value))
        elif issue.issue_type == IssueType.OUTLIERS:
            out.append(Deduction("feature_quality", 2, f"Outliers in '{issue.column}'.",
                                 "Check them; consider capping or a log transform.",
                                 issue.column, issue.issue_type.value))

    for col in _redundant_columns(stats):
        out.append(Deduction("feature_quality", 5, f"'{col['drop']}' is redundant with '{col['keep']}' "
                             f"({col['method']} {col['value']}).",
                             f"Keep one of '{col['keep']}' / '{col['drop']}' (usually the more interpretable).",
                             col["drop"], "redundant_feature"))

    for c in profile.columns:
        if c.semantic_type == SemanticType.TEXT and c.name != target:
            out.append(Deduction("feature_quality", 5, f"'{c.name}' is free text ({c.unique_count} distinct values).",
                                 "Extract structured parts (e.g. brand from a name) or drop it; "
                                 "raw text can't be used by most models.", c.name, "free_text"))
        if c.semantic_type == SemanticType.IDENTIFIER and c.name != target:
            notes.append(f"'{c.name}' is an identifier: exclude it from model features (no points deducted).")
    return out


def _redundant_columns(stats: StatisticsReport) -> list[dict]:
    """Greedy: walk redundant pairs from strongest down; each pair whose columns are both
    still kept marks its second column as droppable. Columns, not pairs, are charged."""
    dropped: set[str] = set()
    result = []
    for a in sorted((x for x in stats.associations if x.redundant), key=lambda x: -abs(x.value)):
        if a.column_a in dropped or a.column_b in dropped:
            continue
        dropped.add(a.column_b)
        result.append({"keep": a.column_a, "drop": a.column_b, "method": a.method, "value": a.value})
    return result


def _target(target: str, quality: QualityReport, stats: StatisticsReport,
            settings: Settings) -> tuple[list[Deduction], bool]:
    out: list[Deduction] = []
    points = dict(settings.severity_points)
    t = stats.target
    single_class = False

    if t is not None and t.task == "classification":
        if t.n_classes is not None and t.n_classes < 2:
            single_class = True
            out.append(Deduction("target", 100, f"Target '{target}' has a single class.",
                                 "A classifier needs at least two classes; check the data extract.",
                                 target, "single_class"))
        elif _BALANCE_POINTS.get(t.balance, 0):
            out.append(Deduction("target", _BALANCE_POINTS[t.balance],
                                 f"Class imbalance in '{target}': minority class '{t.minority_class}' is "
                                 f"{t.minority_pct}% (ratio {t.imbalance_ratio}, {t.balance}).",
                                 "Use stratified splits and class weights; judge models by F1/recall, not accuracy.",
                                 target, "class_imbalance"))

    for issue in quality.issues:
        if issue.column == target and issue.issue_type in (IssueType.MISSING, IssueType.DISGUISED_MISSING):
            out.append(Deduction("target", points[issue.severity.value],
                                 f"{issue.affected_rows} row(s) have no usable target value.",
                                 "Rows without a target can't be used for training; drop them, don't impute the target.",
                                 target, issue.issue_type.value))

    for a in stats.associations:
        if a.involves_target and abs(a.value) >= settings.leakage_threshold:
            feature = a.column_b if a.column_a == target else a.column_a
            out.append(Deduction("target", 40,
                                 f"Possible target leakage: '{feature}' predicts '{target}' almost perfectly "
                                 f"({a.method} {a.value}).",
                                 f"Confirm '{feature}' is known BEFORE the outcome; if not, remove it.",
                                 feature, "target_leakage"))
    return out, single_class


def _band(value: float, bands: tuple[tuple[float, float], ...], floor: float) -> float:
    for threshold, score in bands:
        if value >= threshold:
            return float(score)
    return float(floor)
