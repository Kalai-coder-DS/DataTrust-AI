# DataTrust AI

**Intelligent Data Quality, ML Readiness & Responsible AI Platform**

Upload a CSV and find out whether it can be trusted — and whether it's ready for Machine Learning.

> Status: **Phase 5a — HTML report** (Phases 1–4 ✅ verified).
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
python main.py data/sample/customers_dirty.csv --target churned --html reports/dirty.html  # shareable report
python scripts/run_quality.py path/to/your_data.csv  # export issues to reports/ (Excel-ready CSV + JSON)
python -m pytest                                     # run all tests
```

## Phase 5a verification checklist

- [ ] `python -m pytest` → **212 passed**
- [ ] `python main.py data/sample/customers_dirty.csv --target churned --html reports/dirty.html`
      → open `reports/dirty.html` in your browser: two score tiles (79.8, grade C), top actions,
      issues with severity badges, charts
- [ ] Your data: `python main.py ../../CarPrice_Assignment.csv --target price --html reports/carprice.html`

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
