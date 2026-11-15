"""
main.py

Command-line entry point (Phase 1: profiling, Phase 2: quality issues).

    python main.py data/sample/customers_dirty.csv
    python main.py data/sample/customers_dirty.csv --json reports/profile.json

The CLI is a thin layer: it wires CSVLoader -> DataProfiler -> QualityAnalyzer and formats output.
All logic lives in the `datatrust` package, so the Django app (Phase 6) will call
exactly the same classes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datatrust.exceptions import DataTrustError
from datatrust.ingestion import CSVLoader
from datatrust.logger import get_logger
from datatrust.profiling import DataProfiler
from datatrust.quality import QualityAnalyzer

logger = get_logger("cli")


def format_report(load_summary: dict, profile: dict) -> str:
    lines = [
        "=" * 78,
        f" DataTrust AI - Dataset Profile: {load_summary['file_name']}",
        "=" * 78,
        f" Rows: {profile['n_rows']:,}   Columns: {profile['n_columns']}   "
        f"Memory: {profile['memory_bytes'] / 1024:,.1f} KB",
        f" Encoding: {load_summary['encoding']}   Delimiter: {load_summary['delimiter']!r}",
        f" Missing cells: {profile['total_missing_cells']:,} ({profile['missing_cells_pct']}%)",
        " Column types: " + ", ".join(
            f"{k}={v}" for k, v in profile["semantic_type_counts"].items() if v
        ),
    ]
    if load_summary["warnings"]:
        lines.append("-" * 78)
        lines.append(" Load warnings:")
        lines += [f"  ! {w}" for w in load_summary["warnings"]]

    lines.append("-" * 78)
    lines.append(f" {'Column':<18}{'pandas dtype':<14}{'Semantic':<13}{'Null %':>8}{'Unique':>8}  Notes")
    lines.append("-" * 78)
    for c in profile["columns"]:
        notes = []
        if c["stored_as_text"]:
            notes.append(f"stored as text, {c['parse_success_pct']}% parse")
        lines.append(
            f" {c['name'][:17]:<18}{c['pandas_dtype'][:13]:<14}{c['semantic_type']:<13}"
            f"{c['null_pct']:>8}{c['unique_count']:>8}  {'; '.join(notes)}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


_SEVERITY_ICON = {"critical": "[!!!]", "high": "[!! ]", "medium": "[!  ]", "low": "[.  ]"}


def format_quality(quality: dict) -> str:
    counts = quality["severity_counts"]
    lines = [
        f" DATA QUALITY ISSUES: {quality['total_issues']} found  "
        f"(critical={counts['critical']}, high={counts['high']}, "
        f"medium={counts['medium']}, low={counts['low']})",
        "=" * 78,
    ]
    if not quality["issues"]:
        lines.append(" No issues detected.")
    for i, issue in enumerate(quality["issues"], start=1):
        lines.append(f" {i:>2}. {_SEVERITY_ICON[issue['severity']]} {issue['severity'].upper():<8} {issue['message']}")
        if issue["examples"]:
            examples = ", ".join(str(e) for e in issue["examples"][:3])
            lines.append(f"       e.g. {examples}")
        lines.append(f"       -> {issue['recommendation']}")
    lines.append("=" * 78)
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DataTrust AI - profile a CSV dataset")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--json", dest="json_out", help="Optional path to save the profile as JSON")
    args = parser.parse_args(argv)

    try:
        result = CSVLoader().load(args.csv_path)
        profile = DataProfiler().profile(result.df)
        quality = QualityAnalyzer().analyze(result.df, profile)
    except DataTrustError as exc:
        logger.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    load_summary, profile_dict, quality_dict = result.summary(), profile.to_dict(), quality.to_dict()
    print(format_report(load_summary, profile_dict))
    print(format_quality(quality_dict))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"file": load_summary, "profile": profile_dict, "quality": quality_dict}, indent=2, default=str))
        print(f"Profile saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
