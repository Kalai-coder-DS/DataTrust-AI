"""
datatrust/quality/duplicates.py

Duplicate detection at two levels:

  1. Exact duplicate rows - every column identical. Usually caused by a double
     import or a bad join. They overweight some records and leak between
     train/test splits in ML.

  2. Duplicate identifiers - the same customer_id appears more than once.
     If those rows are exact duplicates, it's the same problem as (1).
     If the rows DIFFER, it is worse: two conflicting versions of one entity,
     and we cannot tell which is correct. Reported as 'conflicting records'.

  3. Duplicates hidden by an ID - rows identical in every column EXCEPT an
     auto-generated id (1, 2, 3...). Check (1) misses them because the ids differ.
     Classic cause: a surrogate key was added after the data was duplicated.
     Found in a real insurance.csv: 2 people entered twice with different ids.

     Guard against false alarms: if the non-ID columns are all low-cardinality
     (e.g. sex, smoker, region), two different people can match by pure chance.
     So this check only runs when at least one remaining column is mostly unique
     (like a precise 'charges' amount), which makes a chance match very unlikely.
"""
from __future__ import annotations

import pandas as pd

from datatrust.profiling import DatasetProfile, SemanticType
from datatrust.quality.base import QualityCheck
from datatrust.quality.models import IssueType, QualityIssue, Severity, pct, severity_from_pct, to_python

# A column this unique (e.g. a precise money amount) makes accidental full-row matches unlikely.
_MIN_DISTINCTIVE_UNIQUE_PCT = 50.0


class DuplicateCheck(QualityCheck):
    name = "duplicates"

    def run(self, df: pd.DataFrame, profile: DatasetProfile) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        n = len(df)
        thresholds = self.settings.severity_thresholds

        # keep="first": the first occurrence is the original, later ones are the duplicates.
        dup_rows = df.duplicated(keep="first")
        n_dup = int(dup_rows.sum())
        if n_dup:
            p = pct(n_dup, n)
            issues.append(QualityIssue(
                check=self.name,
                issue_type=IssueType.DUPLICATE_ROWS,
                severity=severity_from_pct(p, thresholds, floor=Severity.MEDIUM),
                column=None,
                affected_rows=n_dup,
                affected_pct=p,
                message=f"{n_dup} exact duplicate row(s) ({p}%).",
                recommendation="Remove duplicates with df.drop_duplicates() after confirming "
                               "they are not legitimate repeated events.",
                examples=[int(i) for i in df.index[dup_rows][: self.settings.max_examples]],
                details={"examples_are": "row indices of the repeated copies"},
            ))

        hidden = self._duplicates_hidden_by_id(df, profile, dup_rows)
        if hidden:
            issues.append(hidden)

        for col in profile.columns:
            if col.semantic_type != SemanticType.IDENTIFIER:
                continue
            ids = df[col.name]
            repeated = ids.notna() & ids.duplicated(keep="first")
            n_repeated = int(repeated.sum())
            if not n_repeated:
                continue

            # How many repeated IDs are NOT explained by an exact duplicate row?
            conflicting = int((repeated & ~dup_rows).sum())
            p = pct(n_repeated, n)
            sev = Severity.HIGH if conflicting else severity_from_pct(p, thresholds, floor=Severity.MEDIUM)
            if conflicting:
                msg = (f"{n_repeated} repeated value(s) in identifier '{col.name}'; "
                       f"{conflicting} belong to rows with DIFFERENT data (conflicting records).")
                rec = "Investigate conflicting records with the data owner; decide which version is correct."
            else:
                msg = f"{n_repeated} repeated value(s) in identifier '{col.name}', all from exact duplicate rows."
                rec = "Removing the exact duplicate rows will also fix these repeated identifiers."
            issues.append(QualityIssue(
                check=self.name,
                issue_type=IssueType.DUPLICATE_IDS,
                severity=sev,
                column=col.name,
                affected_rows=n_repeated,
                affected_pct=p,
                message=msg,
                recommendation=rec,
                examples=[to_python(v) for v in ids[repeated].unique()[: self.settings.max_examples]],
                details={"conflicting_records": conflicting},
            ))
        return issues

    def _duplicates_hidden_by_id(self, df: pd.DataFrame, profile: DatasetProfile,
                                 exact_dup: pd.Series) -> QualityIssue | None:
        id_cols = [c.name for c in profile.columns if c.semantic_type == SemanticType.IDENTIFIER]
        rest = [c for c in profile.columns
                if c.name not in id_cols and c.semantic_type != SemanticType.EMPTY]
        if not id_cols or len(rest) < 2:
            return None
        if not any(c.unique_pct >= _MIN_DISTINCTIVE_UNIQUE_PCT for c in rest):
            return None  # chance matches too likely to call them duplicates

        same_content = df[[c.name for c in rest]].duplicated(keep="first")
        hidden = same_content & ~exact_dup  # exact duplicates are already reported
        count = int(hidden.sum())
        if not count:
            return None

        p = pct(count, len(df))
        return QualityIssue(
            check=self.name,
            issue_type=IssueType.DUPLICATES_HIDDEN_BY_ID,
            severity=severity_from_pct(p, self.settings.severity_thresholds, floor=Severity.MEDIUM),
            column=None,
            affected_rows=count,
            affected_pct=p,
            message=(f"{count} row(s) are identical to another row except for "
                     f"{', '.join(repr(c) for c in id_cols)} (duplicates hidden by an ID column)."),
            recommendation="These are probably the same record entered twice. Confirm, then drop with "
                           f"df.drop_duplicates(subset=[all columns except {', '.join(id_cols)}]).",
            examples=[int(i) for i in df.index[hidden][: self.settings.max_examples]],
            details={"ignored_id_columns": id_cols,
                     "examples_are": "row indices of the repeated copies"},
        )
