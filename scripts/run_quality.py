"""
scripts/run_quality.py

Run the quality analysis on a real CSV and export the results for further work.

    python scripts/run_quality.py path/to/your_data.csv
    python scripts/run_quality.py ../../insurance.csv

Outputs (in reports/):
    <name>_issues.csv    - one row per issue; open in Excel / Power BI
    <name>_quality.json  - full machine-readable report

The code also shows how to fetch results in your own Python / notebooks, using the
same three steps every caller (CLI, notebook, Django later) follows:
    1. CSVLoader       -> load the file safely
    2. DataProfiler    -> describe the data      (Phase 1)
    3. QualityAnalyzer -> find the problems      (Phase 2)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Let this script import the `datatrust` package when run from the scripts/ folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from datatrust.ingestion import CSVLoader  # noqa: E402
from datatrust.profiling import DataProfiler  # noqa: E402
from datatrust.quality import IssueType, QualityAnalyzer, Severity  # noqa: E402

from datatrust.exceptions import DataTrustError  # noqa: E402

if len(sys.argv) != 2:
    print("Usage: python scripts/run_quality.py path/to/your_data.csv")
    sys.exit(2)
csv_path = sys.argv[1]

# ---------------------------------------------------------------- 1. RUN
try:
    result = CSVLoader().load(csv_path)      # LoadResult: .df, .encoding, .warnings ...
except DataTrustError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)
df = result.df
profile = DataProfiler().profile(df)         # DatasetProfile (Phase 1)
report = QualityAnalyzer().analyze(df, profile)  # QualityReport (Phase 2)

# ---------------------------------------------------------------- 2. FETCH
print(f"\nFile: {result.file_name}  ({len(df)} rows x {df.shape[1]} columns)")
print(f"Total issues: {len(report.issues)}")
print("By severity :", report.severity_counts())

# a) Loop over issues, most severe first
print("\nAll issues:")
for issue in report.sorted_issues():
    print(f"  [{issue.severity.value:8}] {issue.column or '(whole dataset)':16} {issue.message}")

# b) Only the serious ones (MEDIUM or worse)
serious = [i for i in report.issues if i.severity.rank >= Severity.MEDIUM.rank]
print(f"\nSerious issues (medium+): {len(serious)}")

# c) Issues for one column (here: the column with the most issues)
worst = pd.Series([i.column for i in report.issues if i.column]).value_counts()
column = worst.index[0] if not worst.empty else df.columns[0]
print(f"\nIssues in column '{column}':")
for issue in report.filter(column=column):
    print(f"  - {issue.issue_type.value}: {issue.affected_rows} rows, e.g. {issue.examples[:3]}")

# d) One type of issue
for issue in report.filter(IssueType.OUTLIERS):
    print(f"\nOutliers in '{issue.column}': {issue.affected_rows} rows, method = {issue.details['method']}")

# e) As a pandas DataFrame - easiest to sort, filter or export to Excel / Power BI
issues_df = pd.DataFrame([i.to_dict() for i in report.sorted_issues()])
if not issues_df.empty:
    print("\nIssues as a DataFrame:")
    print(issues_df[["severity", "issue_type", "column", "affected_rows", "affected_pct"]].to_string(index=False))

# ---------------------------------------------------------------- 3. SAVE
out_dir = Path("reports")
out_dir.mkdir(exist_ok=True)
stem = Path(csv_path).stem
issues_df.to_csv(out_dir / f"{stem}_issues.csv", index=False, encoding="utf-8-sig")  # BOM: Excel shows text correctly
(out_dir / f"{stem}_quality.json").write_text(json.dumps(report.to_dict(), indent=2, default=str))
print(f"\nSaved: reports/{stem}_issues.csv and reports/{stem}_quality.json")
