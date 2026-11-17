"""
datatrust/web/services.py

What each web page needs, as plain functions. A Django view does only three things:

    with repository() as repo:                      # 1. open the database (datatrust.storage)
        page = dataset_page(repo, name)             # 2. call ONE function here
    return render(request, "dataset.html", page)   # 3. render a template

Errors the views translate into HTTP responses:
    UploadRejected  -> the upload form again, with the message      (400)
    NotFound        -> page not found                               (404)
    StorageError    -> "database unavailable" page                  (503)
"""
from __future__ import annotations

import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.exceptions import DataTrustError
from datatrust.logger import get_logger
from datatrust.ml.present import baseline_view, importance_view
from datatrust.pipeline import analyze_file
from datatrust.reporting.html_report import GRADE_CLASS, GRADE_MEANING, SEVERITY_DISPLAY
from datatrust.storage import AnalysisRepository, Database, DBConfig, RunNotFound, load_env_file, open_repository
from datatrust.web.charts import trend_svg

logger = get_logger(__name__)

SEVERITY_ORDER = ("critical", "high", "medium", "low")
MAX_NAME_LENGTH = 255          # datasets.name is VARCHAR(255)
TEMP_FILE_NAME = "upload.csv"  # safe name for the uploaded file on disk


class UploadRejected(DataTrustError):
    """The uploaded file (or the target column) can't be analysed. The message is safe to show."""


class NotFound(DataTrustError):
    """The requested dataset or run does not exist."""


# ------------------------------------------------------------------ database
_initialised: set[tuple] = set()
_init_lock = threading.Lock()


def _config_key(c: DBConfig) -> tuple:
    return (c.backend, str(c.sqlite_path), c.host, c.port, c.database)


@contextmanager
def repository(environ: dict | None = None) -> Iterator[AnalysisRepository]:
    """
    One database connection per request, closed afterwards (connections are not shared
    between threads). The first request in a process also creates the database and tables
    (open_repository); later requests only connect.
    """
    load_env_file(environ=environ)
    config = DBConfig.from_env(environ)
    key = _config_key(config)
    repo: AnalysisRepository | None = None
    if key not in _initialised:
        with _init_lock:                      # held only while creating the schema
            if key not in _initialised:
                repo = open_repository(environ)
                _initialised.add(key)
    if repo is None:
        repo = AnalysisRepository(Database(config).connect())
    try:
        yield repo
    finally:
        repo.db.close()


def reset_repository_state() -> None:
    """Forget which databases were initialised (tests switch databases between cases)."""
    with _init_lock:
        _initialised.clear()


# -------------------------------------------------------------------- upload
def clean_file_name(name: str) -> str:
    """
    The uploaded file's name becomes the dataset name (that is how history groups runs).
    Keep only the base name (browsers on Windows may send 'C:\\Users\\..\\data.csv'),
    drop control characters, and fit the database column.
    """
    base = (name or "").replace("\\", "/").split("/")[-1]
    base = "".join(ch for ch in base if ch.isprintable()).strip()
    if base in ("", ".", ".."):
        raise UploadRejected("The uploaded file has no usable name.")
    if len(base) > MAX_NAME_LENGTH:
        stem, dot, ext = base.rpartition(".")
        base = (stem[: MAX_NAME_LENGTH - len(ext) - 1] + dot + ext) if dot else base[:MAX_NAME_LENGTH]
    return base


def validate_upload(name: str, size: int | None, settings: Settings = DEFAULT_SETTINGS) -> str:
    """Cheap checks before reading the file: extension and declared size. Returns the cleaned name."""
    cleaned = clean_file_name(name)
    suffix = Path(cleaned).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise UploadRejected(f"'{cleaned}' is not a CSV file. Allowed: {', '.join(settings.allowed_extensions)}.")
    if size is not None and size == 0:
        raise UploadRejected(f"'{cleaned}' is empty.")
    if size is not None and size > settings.max_file_size_bytes:
        raise UploadRejected(f"'{cleaned}' is {size / 1024 / 1024:.1f} MB; the limit is {settings.max_file_size_mb} MB.")
    return cleaned


def analyze_upload(
    chunks: Iterable[bytes],
    file_name: str,
    repo: AnalysisRepository,
    target: str | None = None,
    size: int | None = None,
    settings: Settings = DEFAULT_SETTINGS,
    baseline: bool = False,
) -> int:
    """
    Save the uploaded bytes to a private temporary folder, run the full pipeline (the same
    analyze_file the CLI uses) under the ORIGINAL file name, store the result, delete the file.
    Returns the new run id.

    The file on disk is always called 'upload.csv', never the uploaded name: a name that is
    legal on the uploader's computer can be illegal on the server (Windows forbids ? < > : * | "
    and names like CON.csv). Found by running the Django tests on Windows.

    baseline=True also trains the Phase 7 baseline model (only when a target is given).
    """
    name = validate_upload(file_name, size, settings)
    target = (target or "").strip() or None
    workdir = Path(tempfile.mkdtemp(prefix="datatrust-upload-"))
    try:
        path = workdir / TEMP_FILE_NAME
        written = 0
        with path.open("wb") as fh:
            for chunk in chunks:
                written += len(chunk)
                if written > settings.max_file_size_bytes:   # don't trust the declared size
                    raise UploadRejected(f"'{name}' is larger than the {settings.max_file_size_mb} MB limit.")
                fh.write(chunk)
        if written == 0:
            raise UploadRejected(f"'{name}' is empty.")
        try:
            analysis = analyze_file(path, target=target, baseline=baseline and target is not None, name=name)
        except ValueError as exc:            # invalid target column
            raise UploadRejected(str(exc)) from exc
        except UploadRejected:
            raise
        except DataTrustError as exc:        # unreadable / empty / malformed file
            raise UploadRejected(str(exc).replace(TEMP_FILE_NAME, name)) from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    run_id = repo.save(analysis)             # StorageError is NOT an upload problem: let it through
    logger.info("Web upload %s analysed and saved as run #%s", name, run_id)
    return run_id


# --------------------------------------------------------------------- pages
def _grade(grade: str) -> dict:
    return {"letter": grade, "css": GRADE_CLASS.get(grade, ""), "meaning": GRADE_MEANING.get(grade, "")}


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def dashboard_page(repo: AnalysisRepository) -> dict:
    datasets = repo.list_datasets()
    for d in datasets:
        d["trust"] = _grade(d["trust_grade"])
        d["ml"] = _grade(d["ml_grade"])
    return {
        "datasets": datasets,
        "total_datasets": len(datasets),
        "total_runs": sum(d["runs"] for d in datasets),
    }


def dataset_page(repo: AnalysisRepository, name: str) -> dict:
    history = repo.history(name)
    if not history:
        raise NotFound(f"No dataset called '{name}' has been analysed yet.")
    runs = []
    for row in reversed(history):    # newest first in the table
        change = row["trust_change"]
        runs.append({
            **row,
            "trust": _grade(row["trust_grade"]),
            "ml": _grade(row["ml_grade"]),
            "trust_change_label": "first run" if change is None else (f"{change:+.1f}" if change else "no change"),
            "data_label": "-" if row["data_changed"] is None else ("new data" if row["data_changed"] else "same data"),
        })
    return {
        "name": name,
        "runs": runs,
        "latest": runs[0],
        "first_run": history[0],
        "chart": trend_svg(history, name),
    }


def stored_analysis(repo: AnalysisRepository, run_id: int) -> dict:
    """The full stored analysis (for the HTML report and the JSON download)."""
    try:
        return repo.get_analysis(run_id)
    except RunNotFound as exc:
        raise NotFound(str(exc)) from exc


def _score_card(score: dict) -> dict:
    return {
        "name": score["name"],
        "score": score["score"],
        "grade": _grade(score["grade"]),
        "caps": score.get("caps", []),
        "notes": score.get("notes", []),
        "dimensions": [
            {"label": _label(d["name"]), "score": d["score"], "weight_pct": round(d["weight"] * 100),
             "width": max(0.0, min(100.0, float(d["score"]))), "note": d.get("note", "")}
            for d in score["dimensions"]
        ],
    }


def run_page(repo: AnalysisRepository, run_id: int) -> dict:
    a = stored_analysis(repo, run_id)
    q = a["quality"]
    issues = []
    for i in sorted(q["issues"], key=lambda i: SEVERITY_ORDER.index(i["severity"])):
        icon, label, css = SEVERITY_DISPLAY[i["severity"]]
        issues.append({**i, "sev_icon": icon, "sev_label": label, "sev_css": css,
                       "examples_text": ", ".join(str(x) for x in i.get("examples", [])[:3])})
    counts = q["severity_counts"]
    return {
        "run_id": run_id,
        "file": a["file"],
        "meta": a["meta"],
        "dataset_name": a["file"]["file_name"],
        "trust": _score_card(a["scores"]["trust"]),
        "ml": _score_card(a["scores"]["ml_readiness"]),
        "top_actions": a["scores"]["ml_readiness"]["top_actions"][:5],
        "issues": issues,
        "severity_summary": [
            {"key": s, "icon": SEVERITY_DISPLAY[s][0], "label": SEVERITY_DISPLAY[s][1],
             "css": SEVERITY_DISPLAY[s][2], "count": counts.get(s, 0)}
            for s in SEVERITY_ORDER
        ],
        "total_issues": q["total_issues"],
        "baseline": baseline_view(a.get("ml")),
        "importance": importance_view(a.get("ml")),
    }
