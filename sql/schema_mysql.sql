-- sql/schema_mysql.sql
-- DataTrust AI storage schema for MySQL 8.
-- You don't need to run this by hand: `python scripts/db.py init` creates everything.
-- It is here so you can read the design, or create the tables yourself in MySQL Workbench.
--
--   datasets 1 ──< analysis_runs 1 ──< issues
--                                1 ──o run_owners   (public demo only)

CREATE DATABASE IF NOT EXISTS datatrust CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE datatrust;

CREATE TABLE IF NOT EXISTS datasets (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    first_seen  DATETIME     NOT NULL,
    UNIQUE KEY uq_datasets_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Deployment (public demo): which visitor uploaded a run. No row = a public run
-- (CLI runs, local web uploads, the demo's sample data). A separate table, so existing
-- databases need no ALTER TABLE.
CREATE TABLE IF NOT EXISTS run_owners (
    run_id   INT         NOT NULL PRIMARY KEY,
    owner    VARCHAR(64) NOT NULL,
    KEY ix_run_owners_owner (owner),
    CONSTRAINT fk_owner_run FOREIGN KEY (run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
