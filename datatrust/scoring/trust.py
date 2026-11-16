"""
datatrust/scoring/trust.py

DATA TRUST SCORE: "Can I trust what this data says?"

Four standard data-quality dimensions (as used in data-governance frameworks):

  Completeness (30%)  missing values, placeholders like 'Unknown', empty columns
  Uniqueness   (20%)  duplicate rows, conflicting duplicate IDs, duplicates hidden by IDs
  Validity     (30%)  text in number columns, impossible dates, impossible values
  Consistency  (20%)  the same label spelled differently, numbers written as words

How a dimension is scored:
  Start at 100. Each Phase 2 issue in that dimension deducts points by severity
  (LOW 4, MEDIUM 10, HIGH 20, CRITICAL 35). Floor at 0. Because Phase 2 already
  sets severity from the share of rows affected (with floors for real errors),
  rare problems cost little and widespread problems cost a lot.

Final score = weighted average of the four dimensions.

Deliberately NOT part of trust:
  - Outliers: extreme is not the same as wrong (a very rich customer is real).
  - Constant / near-constant columns: the values are correct, just uninformative.
  Both affect ML readiness instead.

Two explainability rules:
  1. Each problem is charged once. Repeated IDs caused purely by exact duplicate
     rows cost nothing extra; the duplicate rows already paid.
  2. A CRITICAL issue caps the score at 69 (grade D). One unusable column must not
     be averaged away by three perfect dimensions.
"""
from __future__ import annotations

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.quality import IssueType, QualityReport, Severity
from datatrust.scoring.models import Deduction, ScoreResult, combine, finalize_dimension, grade_for

DIMENSION_OF: dict[IssueType, str] = {
    IssueType.MISSING: "completeness",
    IssueType.DISGUISED_MISSING: "completeness",
    IssueType.EMPTY_COLUMN: "completeness",
    IssueType.DUPLICATE_ROWS: "uniqueness",
    IssueType.DUPLICATE_IDS: "uniqueness",
    IssueType.DUPLICATES_HIDDEN_BY_ID: "uniqueness",
    IssueType.TYPE_MISMATCH: "validity",
    IssueType.INVALID_DATE: "validity",
    IssueType.OUT_OF_RANGE: "validity",
    IssueType.INCONSISTENT_FORMAT: "consistency",
    IssueType.NUMBERS_AS_WORDS: "consistency",
}
# A crashed check means we can't vouch for that dimension.
_CHECK_DIMENSION = {"missing": "completeness", "duplicates": "uniqueness",
                    "validity": "validity", "consistency": "consistency"}
_CRITICAL_CAP = 69.0


def score_trust(quality: QualityReport, settings: Settings = DEFAULT_SETTINGS) -> ScoreResult:
    points = dict(settings.severity_points)
    weights = dict(settings.trust_weights)
    by_dim: dict[str, list[Deduction]] = {name: [] for name in weights}
    notes: list[str] = []

    for issue in quality.issues:
        if issue.issue_type == IssueType.CHECK_FAILED:
            dim = _CHECK_DIMENSION.get(issue.check)
            if dim:
                by_dim[dim].append(Deduction(
                    dimension=dim, points=points["high"], reason=issue.message,
                    recommendation=issue.recommendation, issue_type=issue.issue_type.value))
            continue

        dim = DIMENSION_OF.get(issue.issue_type)
        if dim is None:
            continue  # outliers / low-information columns are scored in ML readiness

        if issue.issue_type == IssueType.DUPLICATE_IDS and issue.details.get("conflicting_records", 0) == 0:
            notes.append(f"Repeated IDs in '{issue.column}' come only from exact duplicate rows; "
                         "charged once under duplicate rows.")
            continue

        by_dim[dim].append(Deduction(
            dimension=dim,
            points=points[issue.severity.value],
            reason=f"[{issue.severity.value.upper()}] {issue.message}",
            recommendation=issue.recommendation,
            column=issue.column,
            issue_type=issue.issue_type.value,
        ))

    dimensions = [finalize_dimension(name, weights[name], by_dim[name]) for name in weights]
    score = combine(dimensions)

    caps: list[str] = []
    critical = [i for i in quality.issues if i.severity == Severity.CRITICAL]
    if critical and score > _CRITICAL_CAP:
        score = _CRITICAL_CAP
        caps.append(f"Capped at {_CRITICAL_CAP:g}: {len(critical)} CRITICAL issue(s) make part of the "
                    f"data unusable, e.g. {critical[0].message}")

    return ScoreResult(name="Data Trust Score", score=score, grade=grade_for(score),
                       dimensions=dimensions, caps=caps, notes=notes)
