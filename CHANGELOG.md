# Changelog

What changed in each phase, so you can see at a glance which files are new or modified.
Each phase is also a Git tag (`git checkout phase-2` to see the project as it was).

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
