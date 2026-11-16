"""
datatrust/storage/repository.py

The ONLY place that knows the table layout. Everything else calls these methods:

    repo = AnalysisRepository(db)
    repo.init_schema()
    run_id = repo.save(analysis)          # analysis = pipeline.analyze_file(...)
    repo.history("insurance.csv")         # every run of that dataset, oldest first
    repo.get_analysis(run_id)             # the full stored analysis (re-render as HTML)

This is the "repository pattern": swapping MySQL for another database, or adding a
cache, changes this file and database.py only - never the CLI, the report or Django.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

from datatrust.logger import get_logger
from datatrust.storage.database import Database, StorageError
from datatrust.storage.schema import schema_for

logger = get_logger(__name__)


class RunNotFound(StorageError):
    """No analysis run with the requested id (the database itself is fine)."""


class AnalysisRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------ schema
    def init_schema(self) -> None:
        with self.db.transaction():
            for statement in schema_for(self.db.backend):
                self.db.execute(statement)
        logger.info("Schema ready (%s)", self.db.backend)

    # -------------------------------------------------------------------- save
    def save(self, analysis: dict) -> int:
        """Store one analysis (run + its issues) in a single transaction. Returns the run id."""
        try:
            f, scores, meta = analysis["file"], analysis["scores"], analysis["meta"]
            trust, ml = scores["trust"], scores["ml_readiness"]
            dims = {d["name"]: d["score"] for d in trust["dimensions"]}
            issues = analysis["quality"]["issues"]
            sha = f["sha256"]
        except KeyError as exc:
            raise StorageError(f"Analysis is missing the field {exc}; was it produced by analyze_file?") from exc

        analyzed_at = _to_db_datetime(meta["analyzed_at"])
        with self.db.transaction():
            dataset_id = self._dataset_id(f["file_name"], analyzed_at)
            run_id = self.db.execute(
                """INSERT INTO analysis_runs
                   (dataset_id, analyzed_at, file_sha256, n_rows, n_columns, target,
                    trust_score, trust_grade, ml_score, ml_grade,
                    completeness, uniqueness, validity, consistency,
                    total_issues, tool_version, report_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (dataset_id, analyzed_at, sha, f["rows"], f["columns"], meta.get("target"),
                 trust["score"], trust["grade"], ml["score"], ml["grade"],
                 dims["completeness"], dims["uniqueness"], dims["validity"], dims["consistency"],
                 len(issues), meta["version"], json.dumps(analysis, default=str)),
            )
            self.db.executemany(
                """INSERT INTO issues (run_id, severity, issue_type, column_name, affected_rows,
                                       affected_pct, message, recommendation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [(run_id, i["severity"], i["issue_type"], i["column"], i["affected_rows"],
                  i["affected_pct"], i["message"], i["recommendation"]) for i in issues],
            )
        logger.info("Saved run #%s for %s (%d issues)", run_id, f["file_name"], len(issues))
        return int(run_id)

    def _dataset_id(self, name: str, seen_at: str) -> int:
        rows = self.db.query("SELECT id FROM datasets WHERE name = ?", (name,))
        if rows:
            return int(rows[0]["id"])
        return int(self.db.execute("INSERT INTO datasets (name, first_seen) VALUES (?, ?)", (name, seen_at)))

    # ------------------------------------------------------------------- read
    def list_datasets(self) -> list[dict]:
        """Every dataset with its number of runs and its latest scores."""
        rows = self.db.query(
            """SELECT d.name, COUNT(r.id) AS runs, MAX(r.analyzed_at) AS last_analyzed
               FROM datasets d JOIN analysis_runs r ON r.dataset_id = d.id
               GROUP BY d.id, d.name
               ORDER BY last_analyzed DESC""")
        for row in rows:
            latest = self.history(row["name"])[-1]
            row.update(trust_score=latest["trust_score"], trust_grade=latest["trust_grade"],
                       ml_score=latest["ml_score"], ml_grade=latest["ml_grade"])
        return [_clean(r) for r in rows]

    def history(self, name: str) -> list[dict]:
        """All runs of one dataset, oldest first, with the change in trust score since the previous run."""
        rows = [_clean(r) for r in self.db.query(
            """SELECT r.id AS run_id, r.analyzed_at, r.file_sha256, r.n_rows, r.target,
                      r.trust_score, r.trust_grade, r.ml_score, r.ml_grade,
                      r.completeness, r.uniqueness, r.validity, r.consistency, r.total_issues
               FROM analysis_runs r JOIN datasets d ON d.id = r.dataset_id
               WHERE d.name = ?
               ORDER BY r.analyzed_at, r.id""", (name,))]
        previous = None
        for row in rows:
            row["trust_change"] = None if previous is None else round(row["trust_score"] - previous["trust_score"], 1)
            row["data_changed"] = None if previous is None else row["file_sha256"] != previous["file_sha256"]
            previous = row
        return rows

    def get_analysis(self, run_id: int) -> dict:
        rows = self.db.query("SELECT report_json FROM analysis_runs WHERE id = ?", (run_id,))
        if not rows:
            raise RunNotFound(f"No analysis run with id {run_id}.")
        return json.loads(rows[0]["report_json"])

    def issues(self, run_id: int) -> list[dict]:
        return [_clean(r) for r in self.db.query(
            """SELECT severity, issue_type, column_name, affected_rows, affected_pct, message
               FROM issues WHERE run_id = ? ORDER BY id""", (run_id,))]


# ---------------------------------------------------------------- helpers
def _to_db_datetime(iso: str) -> str:
    """'2026-11-16T15:46:51+00:00' -> '2026-11-16 15:46:51' (UTC), valid for MySQL DATETIME and SQLite."""
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _clean(row: dict) -> dict:
    """MySQL returns Decimal and datetime objects; make every row plain JSON-friendly values."""
    out = {}
    for k, v in row.items():
        if isinstance(v, Decimal):
            v = float(v)
        elif isinstance(v, datetime):
            v = v.strftime("%Y-%m-%d %H:%M:%S")
        out[k] = v
    return out
