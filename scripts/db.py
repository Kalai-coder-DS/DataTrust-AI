"""
scripts/db.py

Work with stored analyses (the database set in .env: MySQL, or SQLite by default).

    python scripts/db.py init                      create the database and tables
    python scripts/db.py datasets                  every dataset with its latest scores
    python scripts/db.py history insurance.csv     every run of one dataset, with score changes
    python scripts/db.py issues 3                  the issues found in run #3
    python scripts/db.py report 3 --html reports/run3.html   re-create the HTML report of run #3

Store analyses with:  python main.py path/to/file.csv --save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datatrust.reporting import render_html  # noqa: E402
from datatrust.storage import StorageError, open_repository  # noqa: E402


def _arrow(change) -> str:
    if change is None:
        return "  (first run)"
    if change > 0:
        return f"  ▲ +{change}"
    if change < 0:
        return f"  ▼ {change}"
    return "  = no change"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DataTrust AI - stored analyses")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create the database and tables")
    sub.add_parser("datasets", help="list datasets with their latest scores")
    p_hist = sub.add_parser("history", help="all runs of one dataset")
    p_hist.add_argument("name", help="dataset file name, e.g. insurance.csv")
    p_iss = sub.add_parser("issues", help="issues found in one run")
    p_iss.add_argument("run_id", type=int)
    p_rep = sub.add_parser("report", help="re-create the HTML report of a stored run")
    p_rep.add_argument("run_id", type=int)
    p_rep.add_argument("--html", required=True, help="output path, e.g. reports/run3.html")
    args = parser.parse_args(argv)

    try:
        repo = open_repository()
    except StorageError as exc:
        print(f"ERROR (database): {exc}", file=sys.stderr)
        return 1

    try:
        if args.command == "init":
            print(f"Ready: {repo.db.config.describe()} (tables: datasets, analysis_runs, issues)")

        elif args.command == "datasets":
            rows = repo.list_datasets()
            if not rows:
                print("No analyses stored yet. Run: python main.py your_file.csv --save")
            for r in rows:
                print(f" {r['name']:<32} runs={r['runs']:<3} last={r['last_analyzed']}  "
                      f"trust={r['trust_score']} ({r['trust_grade']})  ml={r['ml_score']} ({r['ml_grade']})")

        elif args.command == "history":
            rows = repo.history(args.name)
            if not rows:
                print(f"No runs stored for '{args.name}'.")
            for r in rows:
                changed = "" if r["data_changed"] is None else ("  data changed" if r["data_changed"] else "  same data")
                print(f" run #{r['run_id']:<4} {r['analyzed_at']}  trust={r['trust_score']:>5} ({r['trust_grade']})"
                      f"  ml={r['ml_score']:>5} ({r['ml_grade']})  issues={r['total_issues']:<3}"
                      f"{_arrow(r['trust_change'])}{changed}")

        elif args.command == "issues":
            for i in repo.issues(args.run_id):
                print(f" [{i['severity'].upper():<8}] {i['column_name'] or '(dataset)':<18} {i['message']}")

        elif args.command == "report":
            out = Path(args.html)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render_html(repo.get_analysis(args.run_id)), encoding="utf-8")
            print(f"HTML report for run #{args.run_id} saved to {out}")
    except StorageError as exc:
        print(f"ERROR (database): {exc}", file=sys.stderr)
        return 1
    finally:
        repo.db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
