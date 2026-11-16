"""
main.py

Command-line entry point (Phase 1: profiling, Phase 2: quality issues, Phase 3: statistics,
Phase 4: Data Trust and ML Readiness scores).

    python main.py data/sample/customers_dirty.csv
    python main.py data/sample/customers_dirty.csv --json reports/profile.json
    python main.py data/sample/customers_dirty.csv --target churned

The CLI is a thin layer: it wires CSVLoader -> DataProfiler -> QualityAnalyzer -> StatisticsAnalyzer
-> score_dataset and formats output.
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
from datatrust.scoring import GRADE_MEANING, score_dataset
from datatrust.stats import StatisticsAnalyzer

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


def _num(v) -> str:
    """Compact number formatting for the table (1,234.5 / 0.0123 / 1.2e+07)."""
    if v is None:
        return "-"
    a = abs(v)
    if a >= 1e7:
        return f"{v:.2e}"
    if a >= 100:
        return f"{v:,.0f}"
    return f"{v:.2f}"


_MAX_ASSOCIATIONS_SHOWN = 10


def format_statistics(stats: dict) -> str:
    lines = [" STATISTICAL SUMMARY", "=" * 78]
    if stats["numeric"]:
        lines.append(f" {'Numeric column':<16}{'Mean':>11}{'Median':>11}{'Std':>11}{'Skew':>7}  Shape")
        lines.append("-" * 78)
        for s in stats["numeric"]:
            lines.append(f" {s['column'][:15]:<16}{_num(s['mean']):>11}{_num(s['median']):>11}"
                         f"{_num(s['std']):>11}{s['skewness']:>7.2f}  {s['shape']}")
            if s["excluded"]:
                lines.append(f" {'':16}({s['excluded']} invalid value(s) excluded from these statistics)")
        lines.append("-" * 78)
    if stats["categorical"]:
        lines.append(f" {'Label column':<16}{'Categories':>11}  {'Most common':<26}{'Rare':>5}")
        lines.append("-" * 78)
        for s in stats["categorical"]:
            top = f"{s['top_value']} ({s['top_pct']}%)"
            lines.append(f" {s['column'][:15]:<16}{s['n_categories']:>11}  {top[:25]:<26}{s['rare_categories']:>5}")
        lines.append("-" * 78)
    for s in stats["datetime"]:
        lines.append(f" Dates '{s['column']}': {s['min'][:10]} to {s['max'][:10]} ({s['span_days']} days)")
    if stats["associations"]:
        shown = stats["associations"][:_MAX_ASSOCIATIONS_SHOWN]
        lines.append(f" Notable relationships between columns ({len(stats['associations'])} found):")
        for a in shown:
            extra = f" (Pearson {a['details']['pearson']})" if a["details"].get("pearson") is not None else ""
            if a["involves_target"]:
                flag = "  <- signal for the target"
            elif a["redundant"]:
                flag = "  <- REDUNDANT features"
            else:
                flag = ""
            lines.append(f"  - {a['column_a']} ~ {a['column_b']}: {a['strength']} "
                         f"({a['method']} = {a['value']}){extra}{flag}")
        hidden = len(stats["associations"]) - len(shown)
        if hidden:
            lines.append(f"  ... and {hidden} more (use --json to see all)")
    else:
        lines.append(" No strong relationships between columns.")
    t = stats["target"]
    if t:
        lines.append("-" * 78)
        if t["task"] == "regression":
            lines.append(f" Target '{t['column']}': regression (predicting a number); see its numeric summary above.")
        else:
            lines.append(f" Target '{t['column']}': classification, {t['n_classes']} classes {t['class_counts']}")
            lines.append(f"   Minority class '{t['minority_class']}' = {t['minority_pct']}%, "
                         f"imbalance ratio {t['imbalance_ratio']} -> {t['balance'].upper()}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "#" * filled + "." * (width - filled)


def format_scores(scores: dict) -> str:
    lines = [" SCORES", "=" * 78]
    for key in ("trust", "ml_readiness"):
        r = scores[key]
        lines.append(f" {r['name']:<20} {r['score']:>5} / 100   grade {r['grade']} ({GRADE_MEANING[r['grade']]})")
        for d in r["dimensions"]:
            label = d["name"].replace("_", " ")
            lines.append(f"   {label:<17} [{_bar(d['score'])}] {d['score']:>5}  (weight {d['weight']:.0%})")
        for cap in r["caps"]:
            lines.append(f"   ! {cap}")
        lines.append("")
    actions = scores["ml_readiness"]["top_actions"]
    if actions:
        lines.append(" TOP ACTIONS (biggest score gain first):")
        for i, a in enumerate(actions, start=1):
            lines.append(f"  {i}. +{a['impact']:.1f} pts  {a['reason']}")
            if a["recommendation"]:
                lines.append(f"              -> {a['recommendation']}")
    notes = scores["trust"]["notes"] + scores["ml_readiness"]["notes"]
    if notes:
        lines.append(" Notes:")
        lines += [f"  - {n}" for n in notes]
    lines.append("=" * 78)
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DataTrust AI - profile a CSV dataset")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--json", dest="json_out", help="Optional path to save the full analysis as JSON")
    parser.add_argument("--target", help="Column a model should predict (e.g. churned); adds class-balance analysis")
    args = parser.parse_args(argv)

    try:
        result = CSVLoader().load(args.csv_path)
        profile = DataProfiler().profile(result.df)
        quality = QualityAnalyzer().analyze(result.df, profile)
        stats = StatisticsAnalyzer().analyze(result.df, profile, target=args.target)
        scores = score_dataset(quality, stats, profile, target=args.target)
    except (DataTrustError, ValueError) as exc:
        logger.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    load_summary, profile_dict, quality_dict = result.summary(), profile.to_dict(), quality.to_dict()
    stats_dict, scores_dict = stats.to_dict(), scores.to_dict()
    print(format_report(load_summary, profile_dict))
    print(format_quality(quality_dict))
    print(format_statistics(stats_dict))
    print(format_scores(scores_dict))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"file": load_summary, "profile": profile_dict,
                                    "quality": quality_dict, "statistics": stats_dict,
                                    "scores": scores_dict}, indent=2, default=str))
        print(f"Profile saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
