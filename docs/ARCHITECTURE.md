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
│   ├── statistics/               # Ph 3: summaries, distributions, correlations
│   ├── scoring/                  # Ph 4: Data Trust Score, ML Readiness Score
│   ├── reporting/                # Ph 5: JSON + HTML reports
│   ├── storage/                  # Ph 5: MySQL repository layer
│   ├── ml/                       # Ph 7: scikit-learn baseline models
│   ├── explain/                  # Ph 8: SHAP
│   └── genai/                    # Ph 9: plain-language explanations
├── webapp/                       # Ph 6: Django project
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
