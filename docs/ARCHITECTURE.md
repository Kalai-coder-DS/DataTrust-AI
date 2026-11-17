# DataTrust AI — Architecture & Development Plan

*Intelligent Data Quality, ML Readiness & Responsible AI Platform*

## 1. Architecture

DataTrust AI is a **layered pipeline**. Each layer has one job, takes the previous
layer's output, and knows nothing about the layers above it. The analysis engine is
a plain Python package (`datatrust/`), so the CLI, the test suite, and later the
Django web app all call the **same** code.

```
                 ┌──────────────────────────────────────────────┐
  Interfaces     │  CLI (main.py)   │   Django web app (Phase 6) │
                 └───────────────┬──────────────────────────────┘
                                 │ calls
 ┌───────────────────────────────▼─────────────────────────────────────┐
 │                      datatrust  (core engine)                        │
 │                                                                      │
 │  ingestion ──► profiling ──► quality ──► statistics ──► scoring      │
 │  (load CSV)    (types,       (missing,    (summaries,    (Trust &    │
 │                 shape)        dupes,       distribu-      ML Readi-  │
 │                               invalid,     tions,         ness       │
 │                               outliers)    correlation)   scores)    │
 │                                                  │                   │
 │                                                  ▼                   │
 │                           reporting (JSON / HTML report)             │
 │                                                                      │
 │  ml (Ph 7) ── explain / SHAP (Ph 8) ── genai explanations (Ph 9)     │
 │                                                                      │
 │  cross-cutting: config · logger · exceptions                         │
 └───────────────────────────────┬─────────────────────────────────────┘
                                 │
                 ┌───────────────▼──────────────┐
  Persistence    │  MySQL (Phase 5): datasets,   │
                 │  analysis runs, scores, issues│
                 └──────────────────────────────┘
```

### Design principles (interview talking points)

| Principle | How it shows up |
|---|---|
| Separation of concerns | Loader never cleans data; profiler never judges quality; scoring never parses files. |
| Dependency injection | Every class takes a `Settings` object → tests inject custom thresholds without touching globals. |
| Fail loudly, fail clearly | Custom exception hierarchy (`DataTrustError` → subclasses) — expected user errors vs. real bugs. |
| Observe, don't mutate | A quality tool must see the raw data. No silent fixes at load time. |
| Ground-truth testing | The dirty dataset is generated with a manifest of *exactly* which defects were injected, so detectors are tested against known answers. |
| Framework-independent core | Django is a delivery mechanism added in Phase 6, not the foundation. |

## 2. Target folder structure

Items marked *(Ph N)* are created in that phase — not before.

```
datatrust-ai/
├── main.py                       # CLI entry point
├── requirements.txt
├── pytest.ini
├── README.md
├── docs/ARCHITECTURE.md
├── datatrust/                    # core engine (pure Python)
│   ├── __init__.py
│   ├── config.py                 # all thresholds & settings
│   ├── logger.py                 # console + rotating file logs
│   ├── exceptions.py             # custom error hierarchy
│   ├── ingestion/loader.py       # Ph 1: safe CSV loading
│   ├── profiling/profiler.py     # Ph 1: structure & semantic types
│   ├── quality/                  # Ph 2: missing, duplicates, invalid, consistency, outliers
│   ├── stats/                    # Ph 3: summaries, distributions, associations, target balance
│   ├── scoring/                  # Ph 4: Data Trust Score, ML Readiness Score (traceable deductions)
│   ├── pipeline.py               # Ph 5a: analyze_file() - one entry point for every phase
│   ├── reporting/                # Ph 5a: self-contained HTML report
│   ├── storage/                  # Ph 5b: MySQL/SQLite repository layer
│   ├── ml/                       # Ph 7: baseline model, raw vs cleaned (features, baseline, present)
│   ├── explain/                  # Ph 8: SHAP
│   └── genai/                    # Ph 9: plain-language summaries (facts, guard, providers, summary)
│   ├── web/                      # Ph 6: framework-free page logic for Django (services, charts)
├── webapp/                       # Ph 6: Django project (manage.py, datatrust_site/, dashboard/)
├── scripts/generate_sample_data.py
├── data/sample/                  # reproducible test datasets + manifest
├── tests/                        # pytest suite, mirrors datatrust/
└── logs/
```

## 3. Phase-wise plan

Every phase ends with: all tests passing, a CLI/demo run on the sample data, and your approval.

| Phase | Deliverable | New tech |
|---|---|---|
| **1** | Foundation: config, logging, errors, robust CSV loader, data profiler with semantic-type inference, sample data generator, tests | pandas, numpy, pytest |
| **2** | Quality detectors: missing (incl. disguised like "Unknown", "N/A"), exact & key duplicates, invalid values (type, range, impossible dates, inconsistent categories), outliers (IQR + robust z-score), constant columns | — |
| **3** | Statistical analysis: descriptive stats, skewness/kurtosis, distributions, correlations (Pearson/Spearman, Cramér's V), class balance | `statistics`, numpy |
| **4** | Scoring engine: transparent, weighted **Data Trust Score** (completeness, uniqueness, validity, consistency) and **ML Readiness Score** (size, feature quality, leakage hints, imbalance, cardinality) — every point deduction traceable to an issue | — |
| **5** | Reports + persistence: JSON & HTML report with charts and recommendations; MySQL schema + repository layer to store runs and history | matplotlib, MySQL connector |
| **6** | Django web app: upload, async-safe analysis, dashboard, report pages, run history | Django |
| **7** | ML baseline: auto-train simple model on the target column to validate readiness empirically (before vs after cleaning) | scikit-learn |
| **8** | Explainability: SHAP feature importance, data-quality impact on model | shap |
| **9** | GenAI: plain-language explanations of issues & fixes (with a no-API fallback) | LLM API |
| **10** | Hardening: Docker, CI (GitHub Actions), performance tuning, final docs | Docker |

## 4. Phase 1 — what was built

| File | Purpose |
|---|---|
| `datatrust/config.py` | Frozen `Settings` dataclass; env-var overrides; every threshold in one place |
| `datatrust/logger.py` | `get_logger(__name__)`; console + `logs/datatrust.log` (rotating 5×2 MB) |
| `datatrust/exceptions.py` | `DataTrustError` → `FileValidationError`, `DataLoadError`, `EmptyDatasetError` |
| `datatrust/ingestion/loader.py` | Validates file → detects encoding (BOM/UTF-8/cp1252/latin-1) → sniffs delimiter → parses with C engine, falls back to tolerant parser that counts malformed rows → flags duplicate/blank/padded headers |
| `datatrust/profiling/profiler.py` | Dataset & column profiles; infers **semantic type** (empty, boolean, numeric, datetime, categorical, identifier, text) and flags numbers/dates stored as text |
| `main.py` | CLI: prints profile, optional `--json` export |
| `scripts/generate_sample_data.py` | Seeded clean (500 rows) & dirty (515 rows) datasets + `dirty_manifest.json` ground truth |
| `tests/` | 56 tests: loader edge cases, type inference, manifest-checked realistic data, CLI exit codes |

### Semantic type inference rules (priority order)

1. All null → `empty`
2. Native bool → `boolean`; native datetime → `datetime`
3. Native numeric → `boolean` if only {0,1}; `identifier` if ID-named or a 1-step sequence of ≥10 unique ints; else `numeric`
4. Text with ≤2 case-insensitive values in {yes,no,true,false,y,n,t,f,0,1} → `boolean`
5. ≥90% of text parses as numbers → `numeric` (stored_as_text)
6. ≥90% parses as dates → `datetime` (stored_as_text)
7. All unique (or ≥95% and ID-named), short, no spaces → `identifier`
8. ≤50 distinct values or ≤5% distinct ratio → `categorical`
9. Otherwise → `text`

## 5. Phase 2 — Data quality detection

Every check implements `QualityCheck.run(df, profile) -> list[QualityIssue]`, so the
analyzer runs any list of checks without knowing what they do. One crashing check is
recorded as a `check_failed` issue and the others still run.

| File | Purpose |
|---|---|
| `quality/models.py` | `Severity`, `IssueType`, `QualityIssue`, `QualityReport`; severity bands by % of rows affected |
| `quality/base.py` | Abstract `QualityCheck` contract |
| `quality/rules.py` | Shared definitions: disguised-missing tokens, safe numeric/date coercion, name-based range rules |
| `quality/missing.py` | Empty columns, true missing, disguised missing (`Unknown`, `?`, `-999`) |
| `quality/duplicates.py` | Exact duplicate rows; repeated IDs, split into "from exact duplicates" vs **conflicting records**; rows duplicated except for an auto-generated id (**duplicates hidden by an ID**) |
| `quality/validity.py` | Text in numeric columns, impossible dates, out-of-range values (age 0–120, money ≥ 0) |
| `quality/consistency.py` | Same label spelled differently (`Chennai`/`chennai`); numbers written as words |
| `quality/outliers.py` | IQR **and** modified z-score must agree; log scale for right-skewed positive data |
| `quality/columns.py` | Constant and near-constant (≥ 99% one value) columns |
| `quality/analyzer.py` | Runs all checks, isolates failures, returns one `QualityReport` |
| `scripts/run_quality.py` | Runs the analysis on any real CSV; exports `reports/<name>_issues.csv` (Excel/Power BI) and `<name>_quality.json` |

### Key design decisions (interview talking points)

1. **Each cell is reported once.** `Unknown` counts as disguised-missing, not also as a
   type error; `age = 210` counts as out-of-range, not also as an outlier.
2. **Severity floors.** Errors (invalid values, disguised missing, duplicates) are never
   below MEDIUM, however rare: one impossible age is still wrong. Outliers start at LOW,
   because extreme ≠ wrong.
3. **Consensus outliers.** IQR alone over-flags skewed data; modified z-score uses
   median/MAD so outliers can't hide each other (masking). Requiring both cuts false alarms.
4. **Log scale for skewed data.** On insurance-style `charges`, raw-scale detection
   flagged 89 values; log-scale flagged 1. The planted outliers are still all found.
5. **Surrogate keys can hide duplicates.** Testing on a real `insurance.csv` showed 2 people
   entered twice with different `id` values, invisible to a plain `duplicated()`. The check now
   also compares rows with ID columns excluded, but only when a highly unique column (e.g. a
   precise amount) makes accidental matches unlikely.
6. **Ground-truth tests.** The generator picks non-overlapping rows for every defect, so
   `dirty_manifest.json` is exact and integration tests assert exact counts.

### Known limitations (planned)

- Range rules are name-based; Phase 6 will let users define rules per column in the UI.
- Typos in free text (e.g. `toyouta` vs `toyota` in CarPrice `CarName`) need fuzzy
  matching; candidate for a later phase.
- Skewness uses the standard moment estimate, which outliers can inflate; Phase 3's
  statistics module will provide robust alternatives.

## 6. Phase 3 — Statistical analysis

Package `datatrust/stats/` (named `stats`, not `statistics`, so it can never be confused
with Python's built-in `statistics` module).

| File | Purpose |
|---|---|
| `stats/models.py` | `NumericSummary`, `CategoricalSummary`, `DatetimeSummary`, `Association`, `TargetSummary`, `StatisticsReport` |
| `stats/descriptive.py` | Mean, median, std, quartiles, CV, skewness (3 ways), kurtosis, shape label; category counts, rare categories, entropy; date range |
| `stats/associations.py` | Spearman (primary) + Pearson; bias-corrected Cramér's V; correlation ratio (eta) |
| `stats/target.py` | User-chosen target: classification (class balance) or regression |
| `stats/analyzer.py` | Runs everything; marks feature–target pairs as signal, not redundancy |

### Key design decisions (interview talking points)

1. **Statistics on valid values only.** Placeholders, unparseable text and impossible values
   are excluded and counted (`excluded`). One `age = 210` must not distort the mean.
2. **Mean vs median shows outlier influence.** Dirty `monthly_spend`: mean 7,003 vs median
   2,181, caused by 7 values. The clean file's mean and median are close.
3. **Spearman is primary, Pearson shown alongside.** In the dirty file, income vs spend has
   Pearson 0.02 but Spearman 0.62; in the clean file both are ~0.65. Ranks resist outliers.
   A big gap between the two is itself a diagnostic.
4. **"Is the skew caused by outliers?" is tested directly**: skewness is recomputed inside
   the IQR fences. First version used quartile (Bowley) skewness, but on 205 rows it was too
   noisy and wrongly labelled a genuinely skewed price column. Caught by testing.
5. **Redundancy ≠ predictive signal.** Two *features* associated ≥ 0.9 are redundant
   (multicollinearity). A feature strongly related to the *target* is useful signal. First
   version flagged both as redundant; fixed after testing with a target.
   Redundancy applies to every pair type: on the real CarPrice data, `fueltype ~ fuelsystem`
   (Cramér's V 0.985) and `compressionratio ~ fueltype` (eta 0.984) were initially missed
   because only number–number pairs were checked.
6. **Bias-corrected Cramér's V.** The uncorrected version reports spurious association when
   there are many categories and few rows.
7. **The target is chosen by the user**, never guessed: a wrong guess would silently produce a
   misleading ML Readiness Score in Phase 4.

### Known limitations (planned)

- Full pipeline takes ~9 s on 206k rows (pairwise associations, repeated parsing). Fine for
  now; optimisation is part of Phase 10 hardening.
- Associations are pairwise only; Phase 4 adds leakage hints (near-perfect target association).

## 7. Phase 4 — Data Trust Score and ML Readiness Score

Package `datatrust/scoring/`. Core rule: **every point lost is a `Deduction` that names its
cause**, and deductions add up exactly to the score (tested).

### Data Trust Score — "can I trust what this data says?"

| Dimension | Weight | Phase 2 issues charged here |
|---|---|---|
| Completeness | 30% | missing values, disguised missing, empty columns |
| Uniqueness | 20% | duplicate rows, conflicting duplicate IDs, duplicates hidden by an ID |
| Validity | 30% | type mismatch, invalid dates, out-of-range values |
| Consistency | 20% | inconsistent label spelling, numbers written as words |

- Each dimension: `100 - Σ points`, floored at 0. Points per issue by severity:
  **LOW 4, MEDIUM 10, HIGH 20, CRITICAL 35**.
- Score = weighted average. Grade: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, F < 60.
- Outliers and constant columns are **not** trust problems (values are correct); they count in ML readiness.

Worked example (dirty sample file):
`0.3×76 + 0.2×90 + 0.3×70 + 0.2×90 = 79.8 → C`

### ML Readiness Score — "could I train a reliable model on this today?"

| Dimension | Weight | What it measures |
|---|---|---|
| Data trust | 40% | the Data Trust Score |
| Sample size | 15% | min(score by total rows, score by rows per feature) |
| Feature quality | 20% | constant (−15) / near-constant (−5) columns, redundant columns (−5 per droppable column), free text (−5), outliers (−2) |
| Target | 25% | class imbalance (moderate −10, high −35, severe −65), missing target values, **leakage** (feature–target association ≥ 0.95: −40) |

Without `--target`, the target dimension is skipped, weights are rescaled, and the result
says the assessment is partial.

### Key design decisions (interview talking points)

1. **Explainable over clever.** Fixed, documented points per severity rather than a learned
   or opaque formula: anyone can recompute the score by hand from the issue list.
2. **Each problem charged once.** Repeated IDs caused purely by duplicate rows cost nothing
   extra.
3. **Caps for things averages hide.**
   - A CRITICAL issue caps trust at 69 (D): an unusable column must not be averaged away.
   - ML Readiness ≤ Data Trust: *a model can't be more trustworthy than its data*. Without
     this, the dirty file scored Trust C but ML B. While capped, Top Actions show only trust
     fixes (other fixes can't move the score until trust improves), with their full impact.
   - A single-class target caps ML readiness at 40 (F).
4. **Redundancy charged per droppable column, not per pair.** Three near-identical columns
   form three pairs but only two columns are redundant (greedy selection, strongest first).
5. **Top actions ranked by impact** = points × dimension weight: the exact score gain from
   fixing that one problem.
6. **Leakage hint.** A feature that predicts the target almost perfectly usually contains
   the answer; it inflates test scores and fails in production.

### Known limitations (planned)

- Weights and points are sensible defaults, not learned; Phase 7 (baseline models) will
  check empirically whether higher ML readiness really means better models.
- Leakage detection is association-based; time-based leakage needs domain knowledge.

## 8. Phase 5a — HTML report

| File | Purpose |
|---|---|
| `pipeline.py` | `analyze_file(path, target)` runs load → profile → quality → stats → scores and returns one JSON-serialisable dict |
| `reporting/html_report.py` | `render_html(analysis)` → one self-contained HTML page |

### Key design decisions (interview talking points)

1. **One pipeline function.** The CLI, the HTML report, MySQL storage (5b) and Django (6)
   all call `analyze_file`, so they can never run the phases differently.
2. **Render from the dict, not live objects.** A stored analysis can be re-rendered later
   without the original CSV, which is what the web app's history page will do.
3. **Self-contained page.** Inline CSS and SVG, no JavaScript, no external files: opens
   offline, can be emailed, prints cleanly. Tested.
4. **Escape all user data (XSS).** Column names and values come from untrusted files; every
   one passes through `html.escape`. Tested with a column named `<script>…</script>`.
5. **Readable charts.** Scores as stat tiles; dimensions as single-hue bars; numeric columns
   as box plots built from the Phase 3 summary (no raw data needed); severity as icon +
   label + colour, never colour alone; light and dark mode.

## 9. Phase 5b — Storing analyses in MySQL

```
datasets 1 ──< analysis_runs 1 ──< issues
```

| Table | One row per | Key columns |
|---|---|---|
| `datasets` | dataset (file name) | `name` (unique), `first_seen` |
| `analysis_runs` | analysis | `analyzed_at`, `file_sha256`, scores, grades, 4 trust dimensions, `report_json` |
| `issues` | quality issue in a run | `severity`, `issue_type`, `column_name`, `affected_rows` |

| File | Purpose |
|---|---|
| `storage/database.py` | Connection (PyMySQL or sqlite3), `.env` loading, `?`→`%s` placeholders, transactions, friendly connection errors |
| `storage/schema.py` | DDL for MySQL and SQLite |
| `storage/repository.py` | `save`, `history`, `list_datasets`, `get_analysis`, `issues` |
| `scripts/db.py` | Command-line access to stored runs |
| `sql/*.sql` | Schema and example analytical queries for MySQL Workbench |

### Key design decisions (interview talking points)

1. **Repository pattern.** Only `repository.py` knows the tables. The CLI, the report and
   (later) Django call its methods, so the storage can change without touching them.
2. **Summary columns + full JSON.** Scores live in typed columns for fast, simple SQL
   (trends, rankings); the full analysis is kept as JSON so any past report can be
   re-rendered exactly (tested: the re-rendered HTML is identical).
3. **File fingerprint (SHA-256).** Tells "the data changed" apart from "the same file analysed
   again", so a score change can be attributed to the data.
4. **Parameterised queries only.** No value is ever pasted into SQL text (SQL injection);
   tested with a file named `x'; DROP TABLE issues; --.csv`. The one exception, the database
   name in `CREATE DATABASE`, is validated to letters, digits and underscores.
5. **Transactions.** A run and its issues are saved all-or-nothing; tested by forcing a
   failure halfway and checking nothing was left behind.
6. **Secrets outside the code.** Credentials come from `.env` (git-ignored) or real
   environment variables (which win); the password is never printed, logged or in `repr`.
7. **Tests on SQLite, production on MySQL.** Same repository code, different driver: tests
   run anywhere in seconds with no server. MySQL is verified with the README checklist.

### Known limitation found while testing: banded scoring has cliffs

Removing the 15 duplicate rows from the dirty file raised trust by only **+0.2**: the
duplicates fix gained +2.0, but with 500 instead of 515 rows, city's missing share became
exactly 5.0% and crossed the LOW→MEDIUM band (−1.8). Severity bands make scores easy to
explain but cause jumps near thresholds. A candidate improvement is proportional
deductions (points scaled by % of rows affected), traded against simplicity.

## 10. Phase 6 — Django web app

```
browser ──► Django view (webapp/dashboard/views.py)      parse request, render template
                 │
                 ▼
            datatrust.web (services.py, charts.py)          all page logic, no Django imports
                 │                         │
                 ▼                         ▼
            pipeline.analyze_file     storage.AnalysisRepository ──► MySQL
```

| Page | URL | Data from |
|---|---|---|
| Dashboard | `/` | `dashboard_page` → `repo.list_datasets()` |
| Analyse a CSV | `/upload/` | `analyze_upload` → `analyze_file` + `repo.save` |
| Dataset history | `/datasets/<name>/` | `dataset_page` → `repo.history()` + `trend_svg` |
| Run | `/runs/<id>/` | `run_page` → `repo.get_analysis()` |
| Full report / JSON | `/runs/<id>/report/`, `/runs/<id>/json/` | `stored_analysis` → `render_html` / JSON |

### Key design decisions (interview talking points)

1. **Option B: no Django ORM models.** The tables already have one owner, `repository.py`. Mirroring them
   as `managed=False` models would mean two definitions to keep in sync and two ways to write data.
   Cost: no Django admin. `DATABASES = {}`; tests use `SimpleTestCase`.
2. **Thin views, framework-free services.** Each view opens a repository, calls one function in
   `datatrust.web`, renders a template. The logic is tested with plain pytest; Django tests check the wiring.
3. **Templates checked against real data.** Django renders a misspelt variable as an empty string, so a
   page can return 200 and still be wrong. `tests/test_webapp.py` resolves every variable, loop, include
   and `{% url %}` in the templates against page data from a real analysis (and is itself tested on typos).
4. **Errors by cause.** Bad upload → form again with the message (400); unknown run or dataset → 404;
   database unreachable → 503 page with the password-free `StorageError` message, never a stack trace.
   `RunNotFound` (a `StorageError`) separates the first two database cases.
5. **Upload safety.** Only the base name is kept (Windows browsers send full paths), control characters are
   removed, the size limit is enforced while streaming (the declared size can lie), the file goes to a private
   temp folder and is deleted after analysis. Its original name becomes the dataset name, so web and CLI
   runs of the same file share one history.
6. **Defence in depth against XSS.** Django autoescaping for every value, plus a strict
   Content-Security-Policy (no scripts at all), `X-Frame-Options: DENY`, `nosniff`, CSRF on the upload form.
   Tested with a file named `<img src=x onerror=alert(1)>.csv` containing a `<script>` column.
7. **One connection per request.** PyMySQL connections are not thread-safe; each request opens and closes its
   own. The schema is created once per process (first request), not on every request.

### Known limitations (planned)

- Analysis runs inside the request: fine for the sample files (< 1 s), ~9 s for 206k rows. A background
  task queue is a Phase 10 hardening item.
- No login: anyone who can reach the server can see every dataset. Fine for `runserver` on your own
  machine; needed before any shared deployment.
- Per-column validation rules in the UI (noted in Phase 2) are not built yet.

## 11. Phase 7 — Baseline model

The scores say whether data is *healthy*; Phase 7 checks what that means for an actual model.

| Run | Data | Why |
|---|---|---|
| Guessing | ignores features (most frequent class / mean) | the floor every model must beat |
| Raw | as uploaded; only conversions a model needs (text numbers → numbers, dates → days) | what a hurried analyst gets |
| Cleaned | the report's own fixes: duplicates (also hidden by an ID) removed, placeholders and impossible values → missing, label spellings merged, constant and redundant columns dropped | what following the report gets |
| Without suspects | cleaned, minus columns flagged as possible target leakage | how much the model depends on them |

| File | Purpose |
|---|---|
| `ml/features.py` | `prepare(df, analysis, target, clean)`: X, y, steps applied, columns not used and why |
| `ml/baseline.py` | `run_baseline(df, analysis, target)`: cross-validated scores and verdict, never raises for data problems |
| `ml/present.py` | display values shared by the report, CLI and web page |

### Key design decisions (interview talking points)

1. **A measuring instrument, not a model to deploy.** One fixed gradient-boosting model (100 trees, not tuned),
   same seed every run: differences come from the data, not from tuning luck.
2. **Fixes come from the analysis, not re-detection.** The cleaned run applies exactly what the report told the
   user to fix, and lists each step with counts.
3. **Fair metrics.** Balanced accuracy and macro F1 for classes (imbalance-proof; guessing = 1/k); R² and MAE for
   numbers (guessing the mean = 0).
4. **Noise-aware verdicts.** 5-fold CV repeated 3× on files ≤ 5,000 rows; a change or a claim of signal must exceed
   ~2 standard errors (over one round of folds, since repeats are correlated). Tested across seeds.
5. **Dirty data can score *better*.** Duplicate rows land in both training and test folds; a leaking column gives
   the answer away. Tested: a pure-noise target with each row repeated 4× scores 0.99 raw, 0.56 de-duplicated.
6. **A person confirms leakage.** Strong genuine predictors look like leakage statistically (`price = 3x` → `x` is
   flagged). Cleaning keeps them; a third run shows the score without them and the verdict explains both readings.
7. **Readiness ≠ signal.** Healthy data can still be useless for a target; the verdict says so when the model
   barely beats guessing.
8. **Fast enough for a web request.** Files over 20,000 rows are sampled (class-stratified); 60,000 rows ≈ 6 s.
   Stored inside the analysis JSON, so history and re-rendered reports keep it; no schema change.

### Known limitations (planned)

- One model family. A linear model next to the boosting model would show which defects hurt *simpler* models
  (the verdict mentions this; Phase 8 can add it alongside SHAP).
- Outliers are not changed by cleaning (extreme ≠ wrong); their effect on the model is not measured separately.
- Score history (dataset page, `sql/`) does not show the model score yet: it lives in the stored JSON only.
- Time-based leakage and train/test drift need dates and domain knowledge; not attempted.

## 12. Phase 8 — What the model relies on

`ml/importance.py`, run as part of the Phase 7 baseline (same model, same cleaned data):

1. For each of 5 folds: train on 4, then shuffle one column at a time in the 5th and record how much the
   score drops (5 shuffles per column). Average over folds; the spread is the noise.
2. Join every column with the Phase 2 issues on its values (missing, placeholders, impossible values,
   spelling variants, outliers, bad dates, text in numbers).
3. Verdict: which columns the model relies on; **fix first** = relied on AND has a problem;
   **can wait** = has a problem but no measurable effect on this model.

### Key design decisions (interview talking points)

1. **Permutation importance, not SHAP.** Per original column (not per one-hot piece), in the model's metric,
   on held-out rows, no new dependency. SHAP's strength, explaining single predictions, is not needed here.
2. **Verified against ground truth.** The sample generator makes churn depend on tenure only; the tool finds
   exactly that, for every seed tested.
3. **Strict "relied on" rule.** 3 standard errors *and* ≥ 5% of the top column. A 2-SE rule called randomly
   generated columns important, because chance correlation in a small sample is shared by every fold.
4. **Turns the report into a priority list.** Data problems matter most where the model looks.

### Known limitations

- Correlated columns share importance (shuffle one, the model uses the other). Cleaning drops redundant
  columns first, which removes most of this; the report states it.
- "Can wait" is about *this* model only: a simpler model, or a dashboard reading the raw values, can still be
  hurt by those problems.
- Importance is computed on the cleaned data, so it describes the model you get after following the report.

## 13. Phase 9 — Plain-language summaries

```
analysis (Phases 1-8) ──► facts.py ──► provider ──► guard.py ──► summary
                          verified      template      numbers      text + who wrote it
                          numbers       ollama        must be in   + why it fell back
                          only          claude        the facts
                                          │ any failure
                                          └──────────► template
```

### Key design decisions (interview talking points)

1. **The AI explains; the engine decides.** Providers receive only facts copied from the analysis. The
   report always shows the engine's scores; the summary is a separate, labelled section.
2. **Grounding is enforced, not requested.** The prompt tells the model to use only the facts, and
   `guard.py` then rejects any text with a number that is not in them (an invented or recalculated score,
   percentage or count). Rejected text falls back to the template, and the reason is shown.
3. **Graceful degradation.** `auto` = Claude if a key is set, else a running Ollama with the model downloaded
   (2-second check), else the template. The tool never fails because an AI is missing or down.
4. **Privacy option.** Ollama runs the model on the user's computer: no data leaves it, no cost, no key.
5. **No new dependencies.** Both APIs are plain HTTP + JSON through `urllib`; the transport is injected, so the
   27 tests run with a fake and the network blocked.
6. **Secrets.** Keys only from the environment / `.env` (git-ignored); masked in errors; never stored in results.

### Known limitations

- The guard checks numbers, not wording: an AI can still phrase something vaguely or emphasise the wrong
  point. Labelling the source ("Claude API", "Local AI") lets the reader judge.
- Small local models on a CPU are slow (30-90 s) and write less well than the Claude API; a web upload waits
  that long. Background jobs are a Phase 10 item.
- The summary is written once, at analysis time, and stored; it does not change if the report is re-rendered.
