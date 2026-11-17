# DataTrust AI

**Intelligent Data Quality, ML Readiness & Responsible AI Platform**

Upload a CSV and find out whether it can be trusted — and whether it's ready for Machine Learning.

> Status: **Phase 9 — Plain-language summaries** (Phases 1–8 ✅ verified).
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
python main.py data/sample/customers_dirty.csv --target churned --baseline   # Phase 7: train a model, raw vs cleaned
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
Tick **Also train a baseline model** on the upload form (needs a target) to add the Phase 7 section.
The web app uses the same database as `--save` (your MySQL via `.env`), through
`datatrust.storage`; Django itself has no models and no database of its own.

## Connect to your MySQL (one-time setup)

1. `pip install -r requirements.txt` (adds **PyMySQL**)
2. Copy `.env.example` to `.env` and put your MySQL password in `.env`
   (`.env` is in `.gitignore`: it never goes to GitHub)
3. `python scripts/db.py init` → creates database `datatrust` and its 3 tables

## Plain-language summaries (Phase 9)

```bash
python main.py data/sample/customers_dirty.csv --target churned --baseline --summary            # best available
python main.py data/sample/customers_dirty.csv --target churned --summary template               # offline, instant
```

| Provider | Needs | Notes |
|---|---|---|
| `template` | nothing | default fallback; offline, instant, same text every time |
| `ollama` | [Ollama](https://ollama.com) installed + `ollama pull llama3.2:3b` | free, local, data never leaves the computer; 30-90 s on a laptop without a GPU |
| `claude` | `ANTHROPIC_API_KEY` in `.env` | paid per use; fastest and best writing |
| `auto` | — | Claude if a key is set, else Ollama if running, else template |

Every provider gets the same verified facts. AI text that contains a number not in the analysis is rejected, and any
failure falls back to the template with the reason shown. Scores always come from the engine, never from the AI.

## Phase 9 verification checklist

- [ ] `python -m pytest` → **all passed, 0 skipped**, with no internet needed
- [ ] `python main.py data/sample/customers_dirty.csv --target churned --baseline --summary template` →
      a **PLAIN-LANGUAGE SUMMARY** at the end
- [ ] `--summary claude` without a key → template summary plus "ANTHROPIC_API_KEY is not set"
- [ ] Web app: the upload form has **Plain-language summary**; the run page shows a **Summary** under the scores
- [ ] Optional: install Ollama, `ollama pull llama3.2:3b`, then `--summary ollama` → "Local AI (Ollama), model llama3.2:3b"

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
| `DATATRUST_ML_MAX_ROWS` | 20000 | Baseline model: rows used (a class-stratified sample above this) |
| `DATATRUST_SUMMARY_PROVIDER` | auto | Summary: auto / template / ollama / claude |
| `ANTHROPIC_API_KEY` | (none) | Enables the Claude provider. Keep it in `.env` only |
| `DATATRUST_CLAUDE_MODEL` | claude-haiku-5-5 | Claude model |
| `DATATRUST_OLLAMA_URL` | http://localhost:11434 | Ollama server |
| `DATATRUST_OLLAMA_MODEL` | llama3.2:3b | Local model (about 2 GB) |
| `DATATRUST_AI_TIMEOUT` | 120 | Seconds to wait for an AI answer |

## Note on pandas versions

Tested with pandas 3.0 (Linux) and Python 3.14 on Windows. On pandas 2.x, text columns show dtype `object` instead of `str` in the report — behaviour is otherwise identical, because the code uses `pandas.api.types` checks rather than dtype names.
