# DataTrust AI

**Intelligent Data Quality, ML Readiness & Responsible AI Platform**

Upload a CSV and find out whether it can be trusted — and whether it's ready for Machine Learning.

> Status: **Phase 4 — Data Trust Score & ML Readiness Score** (Phases 1–3 ✅ verified).
> What changed in each phase: [CHANGELOG.md](CHANGELOG.md)
> See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture and the full phase plan.

## Setup (Windows)

```powershell
cd datatrust-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux: `source .venv/bin/activate` instead of the second line.

## Run

```bash
python scripts/generate_sample_data.py              # (re)create sample datasets
python main.py data/sample/customers_dirty.csv      # profile a file
python main.py data/sample/customers_dirty.csv --json reports/profile.json
python main.py data/sample/customers_dirty.csv --target churned   # + class balance for a target
python scripts/run_quality.py path/to/your_data.csv  # export issues to reports/ (Excel-ready CSV + JSON)
python -m pytest                                     # run all tests
```

## Phase 4 verification checklist

- [ ] `python -m pytest` → **202 passed**
- [ ] `python main.py data/sample/customers_clean.csv --target churned` → **Trust 100 (A)**, **ML Readiness 95.6 (A)**
- [ ] `python main.py data/sample/customers_dirty.csv --target churned` →
  - **Trust 79.8 (C)**: completeness 76, uniqueness 90, validity 70, consistency 90
  - **ML Readiness 79.8 (C)**, capped at the trust score (uncapped 84.1)
  - TOP ACTIONS list the trust fixes, each worth +3.0 points
- [ ] Your data: `python main.py ../../insurance.csv --target charges` and
      `python main.py ../../CarPrice_Assignment.csv --target price`

On Windows, if `python` points to another installation, use `.venv\Scripts\python.exe` instead of `python`.

## Configuration

Override via environment variables (no code changes):

| Variable | Default | Meaning |
|---|---|---|
| `DATATRUST_MAX_FILE_MB` | 200 | Upload size limit |
| `DATATRUST_CAT_MAX_UNIQUE` | 50 | Max distinct values for a categorical column |
| `DATATRUST_CAT_MAX_RATIO` | 0.05 | …or max distinct/rows ratio |
| `DATATRUST_LOG_LEVEL` | INFO | DEBUG / INFO / WARNING / ERROR |

## Note on pandas versions

Tested with pandas 3.0 (Linux) and Python 3.14 on Windows. On pandas 2.x, text columns show dtype `object` instead of `str` in the report — behaviour is otherwise identical, because the code uses `pandas.api.types` checks rather than dtype names.
