# Changelog

What changed in each phase, so you can see at a glance which files are new or modified.
Each phase is also a Git tag (`git checkout phase-2` to see the project as it was).

## Phase 6 — Django web app

Decision: **Option B** — Django calls the existing `datatrust/storage` repository; no Django ORM
models, no migrations, no second definition of the tables.

**New files**
- `datatrust/web/services.py` — framework-free page logic: upload checks, analyse + save,
  data for every page, one DB connection per request
- `datatrust/web/charts.py` — score-history chart (inline SVG, no JavaScript)
- `webapp/manage.py`, `webapp/datatrust_site/` (settings, urls, wsgi) — Django project
- `webapp/dashboard/` — views, form, URLs, CSP middleware, 7 templates, Django tests
- `tests/test_web_services.py` — 30 tests: uploads, names, size limits, pages, chart
- `tests/test_webapp.py` — template checks against real page data (no Django needed) +
  runs `manage.py check` and the Django page tests when Django is installed

**Modified files**
- `datatrust/storage/repository.py` — `RunNotFound` (subclass of `StorageError`): "no such run"
  (404) vs "database down" (503)
- `datatrust/reporting/html_report.py` — public names for grade/severity labels, shared with the web pages
- `requirements.txt` — adds `Django>=5.2.8`
- `README.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`

**Tests:** 235 → 271 without Django (+2 that run when Django is installed, covering 25 Django page tests)

## Phase 5b — Store analyses in MySQL

**New files**
- `datatrust/storage/__init__.py` — `open_repository()`: read `.env`, connect, create tables
- `datatrust/storage/database.py` — MySQL (PyMySQL) / SQLite connection, `.env` loading, transactions
- `datatrust/storage/schema.py` — tables `datasets`, `analysis_runs`, `issues`
- `datatrust/storage/repository.py` — save / history / list / re-load analyses (repository pattern)
- `scripts/db.py` — `init`, `datasets`, `history`, `issues`, `report` commands
- `sql/schema_mysql.sql` — the MySQL schema, readable / runnable in Workbench
- `sql/example_queries.sql` — 6 example queries (window functions, LAG, GROUP BY/HAVING)
- `.env.example` — template for your local `.env` (database settings)
- `tests/test_storage.py` — 23 tests on SQLite (round-trip, history, rollback, SQL injection, CLI)

**Modified files**
- `datatrust/pipeline.py` — adds the file's SHA-256 fingerprint (detects changed data between runs)
- `main.py` — new `--save` option
- `requirements.txt` — adds `PyMySQL`
- `.gitignore` — ignores local `*.db` files
- `README.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`

**Tests:** 212 → 235

## Phase 5a — HTML report

**New files**
- `datatrust/pipeline.py` — `analyze_file(path, target)`: runs every phase, returns one dict
- `datatrust/reporting/__init__.py`, `datatrust/reporting/html_report.py` — self-contained HTML report
- `tests/test_reporting.py` — 10 tests (pipeline, report sections, offline, XSS escaping, CLI)

**Modified files**
- `main.py` — uses `analyze_file`; new `--html PATH` option; JSON export now includes `meta`
- `README.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`

**Tests:** 202 → 212

## Phase 4 — Data Trust Score and ML Readiness Score

**New files**
- `datatrust/scoring/__init__.py` — package entry point (`score_dataset`)
- `datatrust/scoring/models.py` — `Deduction`, `DimensionScore`, `ScoreResult`, grades A–F
- `datatrust/scoring/trust.py` — Data Trust Score: completeness, uniqueness, validity, consistency
- `datatrust/scoring/ml_readiness.py` — ML Readiness Score: data trust, sample size, feature quality, target
- `tests/test_scoring.py` — 36 unit tests (traceability, caps, leakage, imbalance, redundancy)
- `tests/test_scoring_integration.py` — 4 tests, incl. dirty-file scores derived by hand

**Modified files**
- `datatrust/config.py` — Phase 4 settings (severity points, dimension weights, leakage threshold)
- `main.py` — new SCORES section and TOP ACTIONS; scores added to `--json`
- `README.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`

**Not changed:** ingestion, profiling, quality, stats.

**Tests:** 162 → 202

## Phase 3 fix — redundancy for label and mixed pairs

Found by running on the real CarPrice data: `fueltype ~ fuelsystem` (Cramér's V 0.985) and
`compressionratio ~ fueltype` (eta 0.984) were not marked redundant, because redundancy was
only checked for number–number pairs.

**Modified files**
- `datatrust/stats/associations.py` — redundancy (>= 0.9) now applies to Cramér's V and eta too
- `datatrust/stats/models.py` — comment updated
- `tests/test_stats.py` — 2 new tests (CarPrice-style redundant pairs; moderate pair not redundant)
- `CHANGELOG.md`, `docs/ARCHITECTURE.md`

**Tests:** 160 → 162

## Phase 3 — Statistical analysis

**New files**
- `datatrust/stats/__init__.py` — package entry point (`StatisticsAnalyzer`)
- `datatrust/stats/models.py` — result containers: numeric / categorical / datetime summaries, associations, target summary
- `datatrust/stats/descriptive.py` — per-column statistics on valid values only; distribution shape in plain language
- `datatrust/stats/associations.py` — Spearman + Pearson, Cramér's V (bias-corrected), correlation ratio (eta)
- `datatrust/stats/target.py` — classification vs regression target; class balance
- `datatrust/stats/analyzer.py` — runs all of the above, returns one `StatisticsReport`
- `tests/test_stats.py` — 39 unit tests with hand-checkable answers
- `tests/test_stats_integration.py` — 8 tests on the sample datasets
- `CHANGELOG.md` — this file

**Modified files**
- `datatrust/config.py` — new Phase 3 settings (association thresholds, rare-category %, max classes)
- `main.py` — new STATISTICAL SUMMARY section; new `--target COLUMN` option; statistics added to `--json`
- `tests/test_cli.py` — tests for the statistics section and `--target`
- `README.md`, `docs/ARCHITECTURE.md` — Phase 3 status, checklist and design notes

**Not changed:** ingestion, profiling, quality checks, sample data generator.

**Tests:** 111 → 160

## Phase 2 — Data quality detection

New `datatrust/quality/` package (missing, duplicates incl. hidden-by-ID, validity, consistency,
outliers with log-scale for skewed data, low-information columns), `scripts/run_quality.py`,
improved sample generator with exact `dirty_manifest.json`. Tests: 56 → 111.

## Phase 1 — Foundation

Config, logging, custom exceptions, robust CSV loader, profiler with semantic-type inference,
CLI, sample data generator. 56 tests.
