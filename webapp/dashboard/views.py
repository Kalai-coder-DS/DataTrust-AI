"""
Views: thin on purpose. Each one opens the database through datatrust.storage, calls ONE
function from datatrust.web, and renders a template. No SQL and no analysis logic here.

    UploadRejected -> upload form again with the message   (400)
    NotFound       -> error page                           (404)
    StorageError   -> "database unavailable" page          (503)
"""
from __future__ import annotations

import functools
import json

from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_http_methods

from datatrust.config import DEFAULT_SETTINGS
from datatrust.logger import get_logger
from datatrust.reporting import render_html
from datatrust.storage import StorageError
from datatrust.web import (
    NotFound,
    UploadRejected,
    analyze_upload,
    dashboard_page,
    dataset_page,
    repository,
    run_page,
    stored_analysis,
)
from dashboard.forms import UploadForm

logger = get_logger(__name__)


def _error(request, status: int, title: str, message: str) -> HttpResponse:
    return render(request, "dashboard/error.html",
                  {"status": status, "title": title, "message": message}, status=status)


def handles_errors(view):
    """Turn NotFound / StorageError into friendly 404 / 503 pages."""
    @functools.wraps(view)
    def wrapper(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except NotFound as exc:
            return _error(request, 404, "Not found", str(exc))
        except StorageError as exc:
            # StorageError messages never contain the password (see datatrust.storage.database).
            logger.error("Database unavailable: %s", exc)
            return _error(request, 503, "Database unavailable",
                          f"{exc} Check the database settings in .env, then try again.")
    return wrapper


@require_GET
@handles_errors
def home(request):
    with repository() as repo:
        page = dashboard_page(repo)
    return render(request, "dashboard/home.html", page)


@require_http_methods(["GET", "POST"])
@handles_errors
def upload(request):
    limit_mb = DEFAULT_SETTINGS.max_file_size_mb
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data["file"]
            try:
                with repository() as repo:
                    run_id = analyze_upload(f.chunks(), f.name, repo,
                                            target=form.cleaned_data["target"], size=f.size,
                                            baseline=form.cleaned_data["baseline"])
            except UploadRejected as exc:
                form.add_error(None, str(exc))
            else:
                return redirect("dashboard:run", run_id=run_id)
        return render(request, "dashboard/upload.html", {"form": form, "limit_mb": limit_mb}, status=400)
    return render(request, "dashboard/upload.html", {"form": UploadForm(), "limit_mb": limit_mb})


@require_GET
@handles_errors
def dataset(request, name: str):
    with repository() as repo:
        page = dataset_page(repo, name)
    return render(request, "dashboard/dataset.html", page)


@require_GET
@handles_errors
def run(request, run_id: int):
    with repository() as repo:
        page = run_page(repo, run_id)
    return render(request, "dashboard/run.html", page)


def _download(response: HttpResponse, filename: str) -> HttpResponse:
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@require_GET
@handles_errors
def report(request, run_id: int):
    """The full self-contained HTML report (Phase 5a), re-rendered from the stored analysis."""
    with repository() as repo:
        analysis = stored_analysis(repo, run_id)
    response = HttpResponse(render_html(analysis), content_type="text/html; charset=utf-8")
    if request.GET.get("download"):
        _download(response, f"datatrust-run-{run_id}.html")
    return response


@require_GET
@handles_errors
def analysis_json(request, run_id: int):
    with repository() as repo:
        analysis = stored_analysis(repo, run_id)
    body = json.dumps(analysis, indent=2, ensure_ascii=False, default=str)
    return _download(HttpResponse(body, content_type="application/json; charset=utf-8"),
                     f"datatrust-run-{run_id}.json")


def page_not_found(request, exception=None):
    return _error(request, 404, "Not found", "There is no page at this address.")


def server_error(request):
    return _error(request, 500, "Something went wrong",
                  "An unexpected error occurred. The details are in logs/datatrust.log.")
