# Changelog

What changed in each phase, so you can see at a glance which files are new or modified.
Each phase is also a Git tag (`git checkout phase-2` to see the project as it was).

## Phase 9 fix 2 — no false alarms on "no critical issues"; show rejected text

Found on real runs: qwen2.5:7b and llama3.2:3b were both rejected for mentioning "critical issues" when the
sample has none. Two models making the same "mistake" pointed at the check: a correct sentence such as
"there are no critical issues" was rejected because the check ignored the "no".

- `datatrust/genai/guard.py` — "no / none / not / without / free of ... critical issues" in the same sentence is accepted
- `datatrust/genai/summary.py` — keeps the rejected AI text (`rejected_text`) so the user can see what was written
- `datatrust/genai/providers.py` — a timeout says "no answer within N seconds; raise DATATRUST_AI_TIMEOUT"
  instead of "cannot reach" (the server was reached; qwen2.5:7b was slow on a laptop CPU)
- `main.py`, `datatrust/reporting/html_report.py`, `_summary.html`, `present.py` — show the rejected text
- `tests/test_genai.py` — 7 tests

**Tests:** 351 → 358

## Phase 9 fix — guard checks what numbers mean

First real run with Ollama (llama3.2:3b, Windows) wrote "seven problems" (12 exist), "beats guessing by a
balanced accuracy of 0.693" (the lift is 0.193) and "the most critical problems" (0 critical). Each number also
appeared elsewhere in the facts, so the number check passed them.

- `datatrust/genai/guard.py` — new checks: "<N> issues/problems" (digits or words) must be a real issue count;
  "beats guessing by X" must be the real lift; "critical"/"high" issues only when there are some
- `datatrust/genai/facts.py` — adds `grade_means`, `issues_listed_below`, `model.beats_guessing_by`
- `datatrust/genai/providers.py` — prompt rules for grades, counts and the model comparison
- `tests/test_genai.py` — 8 tests, including the exact text Ollama produced (now rejected)

**Tests:** 343 → 351

## Phase 9 — Plain-language summaries (template, Ollama, Claude API)

Three providers behind one interface. The template is the default and always works offline; Ollama (local, free)
and the Claude API are optional and need no new Python packages (HTTP via the standard library).

**New files** (`datatrust/genai/`)
- `facts.py` — the verified facts every provider receives, copied from the analysis; nothing recalculated
- `guard.py` — rejects AI text that is empty, too long, or contains a number not in the facts
- `providers.py` — `TemplateProvider`, `OllamaProvider`, `ClaudeProvider`; shared prompt; injectable HTTP transport
- `summary.py` — `summarize()`: provider choice (`auto`/`template`/`ollama`/`claude`), configuration from the
  environment, automatic fallback to the template with the reason recorded
- `present.py` — display values shared by report, CLI and web page
- `webapp/dashboard/templates/dashboard/_summary.html` — run page section
- `tests/test_genai.py` — 27 tests; network blocked for the whole file

**Modified files**
- `datatrust/pipeline.py` — `analyze_file(..., summary=None)`; result under `"summary"` (nothing added by default)
- `datatrust/reporting/html_report.py` — Summary section under the scores
- `main.py` — `--summary [PROVIDER]`
- `datatrust/web/services.py`, `webapp/dashboard/forms.py`, `views.py`, `upload.html`, `run.html`, `base.html`, `tests.py`
- `tests/test_web_services.py`, `tests/test_webapp.py` — summary option; template check covers `_summary.html`
- `.env.example`, `README.md`, `docs/ARCHITECTURE.md`

**Tests:** 315 → 343 (with Django installed; Django page tests inside them: 28 → 32); every one runs offline

## Phase 8 — What the model relies on (feature importance + data problems)

Decision: **permutation importance** (scikit-learn, already required) instead of SHAP. It gives one number per
original column, the level at which the report lists problems, in the model's own metric, on held-out rows.
SHAP explains single predictions, which this tool does not need, and could not be installed or tested here.

**New files**
- `datatrust/ml/importance.py` — cross-validated permutation importance; joins each column with its Phase 2
  value problems; verdict: what the model relies on, which fixes come first, which can wait
- `webapp/dashboard/templates/dashboard/_importance.html` — run page section
- `tests/test_importance.py` — 12 tests with known causes

**Modified files**
- `datatrust/ml/baseline.py` — runs importance on the cleaned data; result under `ml.importance`
- `datatrust/ml/present.py` — `importance_view` shared by report, CLI and web page
- `datatrust/reporting/html_report.py`, `main.py`, `datatrust/web/services.py`, `webapp/dashboard/` — new section
- `datatrust/config.py` — `ml_importance_repeats`

**Found while testing (kept as tests)**
1. Checked against the generator: the sample's churn depends only on `tenure_months`, and that is the only
   column found to be relied on, for every seed tried.
2. With a "2 standard errors" rule, `plan` (drawn at random) and a pure-noise column were called "relied on":
   on a few hundred rows a random column correlates a little with the target by chance, in every fold alike.
   Now a column must be 3 standard errors above zero AND at least 5% of the top column.

**Tests:** 303 → 315 (with Django installed)

## Phase 7 — Baseline model (does cleaning change what a model learns?)

**New files**
- `datatrust/ml/features.py` — model input from the table: *raw* (as uploaded) or *cleaned* (the report's fixes)
- `datatrust/ml/baseline.py` — gradient boosting vs guessing, repeated cross-validation, noise-aware verdict;
  a third run without columns flagged as possible leakage
- `datatrust/ml/present.py` — display values shared by the HTML report, the CLI and the web page
- `webapp/dashboard/templates/dashboard/_baseline.html` — run page section
- `tests/test_ml.py` — 22 tests with known answers (duplicates, leakage, regression signal, pure noise, seeds)

**Modified files**
- `datatrust/pipeline.py` — `analyze_file(..., baseline=False, name=None)`; result under `"ml"`
- `datatrust/config.py` — `ml_*` settings (sample size, folds, repeats, seed)
- `datatrust/reporting/html_report.py` — "Baseline model" section
- `datatrust/web/services.py` — `baseline=` option; uploads analysed under their real name
  (fixes the log line "Analysis complete for upload.csv"); error messages show the uploaded name
- `webapp/dashboard/` — upload form checkbox, run page section, 3 Django tests
- `main.py` — `--baseline`
- `requirements.txt` — adds `scikit-learn`
- `tests/test_web_services.py` — the 6 Windows file-name regression tests from Phase 6 fix + baseline option

**Found while testing (kept as tests)**
1. On the 515-row sample, the raw-vs-cleaned gap moved between 0.001 and 0.034 just by changing the random
   seed. A fixed threshold would report noise as an effect, so changes are judged against ~2 standard errors.
2. A pure-noise target scored 0.558 on 150 rows and was first called "signal": repeated folds are correlated,
   so the standard error is now taken over one round of folds.
3. `price = 3x + noise` flags `x` as possible leakage. Dropping it automatically took R² from 0.97 to 0, so
   cleaning keeps such columns and a third run measures the model without them; a person decides.

4. On Windows with pandas 3.1.1, merged label spellings kept pandas' `pd.NA` for missing values and
   scikit-learn failed with "boolean value of NA is ambiguous" (pandas 3.0 did not show it). Text columns
   and the target are now built as plain Python values; a test checks no `pd.NA` reaches scikit-learn.

**Tests:** 273 → 303 (with Django installed; the Django page tests inside them: 25 → 28)

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
