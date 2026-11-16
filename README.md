# DataTrust AI

**Intelligent Data Quality, ML Readiness & Responsible AI Platform**

Upload a CSV and find out whether it can be trusted — and whether it's ready for Machine Learning.

> Status: **Phase 6 — Django web app** (Phases 1–5b ✅ verified).
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
python main.py data/sample/customers_dirty.csv --target churned --save   # store in MySQL (see setup below)
python scripts/db.py history customers_dirty.csv                        # score history of a dataset
python scripts/run_quality.py path/to/your_data.csv  # export issues to reports/ (Excel-ready CSV + JSON)
python -m pytest                                     # run all tests
```

## Web app (Phase 6)

```bash
pip install -r requirements.txt                      # adds Django
python webapp/manage.py runserver                    # open http://127.0.0.1:8000/
python webapp/manage.py test dashboard               # the web app's page tests
```

Pages: **Dashboard** (every dataset, latest scores) · **Analyse a CSV** (upload + optional target) ·
**Dataset history** (score trend chart + every run) · **Run** (scores, top actions, issues) ·
**Full report** (the Phase 5a HTML report) · **JSON download**.
The web app uses the same database as `--save` (your MySQL via `.env`), through
`datatrust.storage`; Django itself has no models and no database of its own.

## Connect to your MySQL (one-time setup)

1. `pip install -r requirements.txt` (adds **PyMySQL**)
2. Copy `.env.example` to `.env` and put your MySQL password in `.env`
   (`.env` is in `.gitignore`: it never goes to GitHub)
3. `python scripts/db.py init` → creates database `datatrust` and its 3 tables

## Phase 6 verification checklist

- [ ] `pip install -r requirements.txt` → Django installs
- [ ] `python -m pytest` → **all passed, 0 skipped** (with Django installed this also runs
      `manage.py check` and every Django page test)
- [ ] `python webapp/manage.py runserver`, open http://127.0.0.1:8000/ → dashboard lists the datasets already in MySQL
- [ ] **Analyse a CSV** → upload `insurance.csv` with target `charges` → run page shows the same scores as the CLI
- [ ] Upload it again → dataset page shows 2 runs, `no change`, `same data`, and the chart
- [ ] Upload with target `nope` → the form comes back with "not found. Available: ..."
- [ ] **Open full report** and **Download analysis (JSON)** on a run page
- [ ] Stop MySQL80, reload the dashboard → "Database unavailable" page (not a crash); start it again
- [ ] In MySQL Workbench: `SELECT * FROM analysis_runs ORDER BY id DESC LIMIT 5;` shows the web uploads

On Windows, if `python` points to another installation, use `.venv\Scripts\python.exe` instead of `python`.

## Configuration

Override via environment variables (no code changes):

| Variable | Default | Meaning |
|---|---|---|
| `DATATRUST_MAX_FILE_MB` | 200 | Upload size limit |
| `DATATRUST_CAT_MAX_UNIQUE` | 50 | Max distinct values for a categorical column |
| `DATATRUST_CAT_MAX_RATIO` | 0.05 | …or max distinct/rows ratio |
| `DATATRUST_LOG_LEVEL` | INFO | DEBUG / INFO / WARNING / ERROR |
| `DATATRUST_DEBUG` | 1 | Web app: `0` for production-style settings |
| `DATATRUST_SECRET_KEY` | (dev key) | Web app: required when `DATATRUST_DEBUG=0` |
| `DATATRUST_ALLOWED_HOSTS` | localhost,127.0.0.1 | Web app: host names it answers to |

## Note on pandas versions

Tested with pandas 3.0 (Linux) and Python 3.14 on Windows. On pandas 2.x, text columns show dtype `object` instead of `str` in the report — behaviour is otherwise identical, because the code uses `pandas.api.types` checks rather than dtype names.
