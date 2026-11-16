"""
datatrust/storage/schema.py

Three tables, normalised so each fact is stored once:

  datasets       one row per dataset (identified by file name)
  analysis_runs  one row per analysis: when, file fingerprint, scores, grades, full JSON
  issues         one row per data-quality issue found in a run

    datasets 1 ──< analysis_runs 1 ──< issues

Why keep both summary columns AND the full JSON in analysis_runs?
  - The columns (trust_score, validity, ...) make history queries fast and simple SQL.
  - The JSON keeps everything else, so any past report can be re-rendered exactly.

file_sha256 is a fingerprint of the file's bytes: two runs with the same fingerprint
analysed identical data; a different fingerprint means the file changed.

The MySQL and SQLite versions differ only where the two databases genuinely differ
(AUTO_INCREMENT vs AUTOINCREMENT, ENGINE, text types). sql/schema_mysql.sql holds the
MySQL version for running in MySQL Workbench.
"""

MYSQL_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS datasets (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        name        VARCHAR(255) NOT NULL,
        first_seen  DATETIME     NOT NULL,
        UNIQUE KEY uq_datasets_name (name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS analysis_runs (
        id             INT AUTO_INCREMENT PRIMARY KEY,
        dataset_id     INT          NOT NULL,
        analyzed_at    DATETIME     NOT NULL,
        file_sha256    CHAR(64)     NOT NULL,
        n_rows         INT          NOT NULL,
        n_columns      INT          NOT NULL,
        target         VARCHAR(255) NULL,
        trust_score    DECIMAL(5,1) NOT NULL,
        trust_grade    CHAR(1)      NOT NULL,
        ml_score       DECIMAL(5,1) NOT NULL,
        ml_grade       CHAR(1)      NOT NULL,
        completeness   DECIMAL(5,1) NOT NULL,
        uniqueness     DECIMAL(5,1) NOT NULL,
        validity       DECIMAL(5,1) NOT NULL,
        consistency    DECIMAL(5,1) NOT NULL,
        total_issues   INT          NOT NULL,
        tool_version   VARCHAR(20)  NOT NULL,
        report_json    LONGTEXT     NOT NULL,
        KEY ix_runs_dataset_time (dataset_id, analyzed_at),
        CONSTRAINT fk_runs_dataset FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS issues (
        id             INT AUTO_INCREMENT PRIMARY KEY,
        run_id         INT          NOT NULL,
        severity       VARCHAR(10)  NOT NULL,
        issue_type     VARCHAR(40)  NOT NULL,
        column_name    VARCHAR(255) NULL,
        affected_rows  INT          NOT NULL,
        affected_pct   DECIMAL(6,2) NOT NULL,
        message        TEXT         NOT NULL,
        recommendation TEXT         NOT NULL,
        KEY ix_issues_run (run_id),
        KEY ix_issues_type (issue_type),
        CONSTRAINT fk_issues_run FOREIGN KEY (run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]

SQLITE_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS datasets (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,
        first_seen  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS analysis_runs (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id     INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
        analyzed_at    TEXT    NOT NULL,
        file_sha256    TEXT    NOT NULL,
        n_rows         INTEGER NOT NULL,
        n_columns      INTEGER NOT NULL,
        target         TEXT,
        trust_score    REAL    NOT NULL,
        trust_grade    TEXT    NOT NULL,
        ml_score       REAL    NOT NULL,
        ml_grade       TEXT    NOT NULL,
        completeness   REAL    NOT NULL,
        uniqueness     REAL    NOT NULL,
        validity       REAL    NOT NULL,
        consistency    REAL    NOT NULL,
        total_issues   INTEGER NOT NULL,
        tool_version   TEXT    NOT NULL,
        report_json    TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_runs_dataset_time ON analysis_runs (dataset_id, analyzed_at)",
    """
    CREATE TABLE IF NOT EXISTS issues (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id         INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
        severity       TEXT    NOT NULL,
        issue_type     TEXT    NOT NULL,
        column_name    TEXT,
        affected_rows  INTEGER NOT NULL,
        affected_pct   REAL    NOT NULL,
        message        TEXT    NOT NULL,
        recommendation TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_issues_run ON issues (run_id)",
    "CREATE INDEX IF NOT EXISTS ix_issues_type ON issues (issue_type)",
]


def schema_for(backend: str) -> list[str]:
    return MYSQL_SCHEMA if backend == "mysql" else SQLITE_SCHEMA
