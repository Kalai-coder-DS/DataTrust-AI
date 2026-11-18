"""
Views: thin on purpose. Each one opens the database through datatrust.storage, calls ONE
function from datatrust.web, and renders a template. No SQL and no analysis logic here.

    UploadRejected -> upload form again with the message   (400; 429 for demo rate limits)
    NotFound       -> error page                           (404)
    StorageError   -> "database unavailable" page          (503)

Public demo (deployment): request.datatrust_owner (set by VisitorMiddleware) is passed as
visible_to / owner, so a visitor sees public runs and their own uploads only. Locally it is
None and everything behaves as before.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import re

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_http_methods

from datatrust.config import DEFAULT_SETTINGS
from datatrust.logger import get_logger
from datatrust.reporting import render_html
from datatrust.storage import StorageError
from datatrust.web.auth import COOKIE_NAME, MAX_AGE, LoginThrottle, check_password, make_token, safe_next
from datatrust.web.demo import UploadRateLimiter, check_public_upload, client_address
from datatrust.web.deploy import DemoLimits
from datatrust.web.jobs import JobManager
from datatrust.web.services import start_upload_job
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

# Phase 10: one job manager per server process (see datatrust/web/jobs.py for its limits).
jobs = JobManager(workers=int(os.environ.get("DATATRUST_WEB_WORKERS", "2") or 2))
JOB_ID = re.compile(r"^[0-9a-f]{32}$")


throttle = LoginThrottle()
upload_limiter = UploadRateLimiter()      # public demo: uploads per hour (visitor, address, server)

DEMO_SUMMARY = "template"   # no Ollama or Claude key on the demo server: don't make visitors wait for them


def _owner(request) -> str | None:
    """The visitor's owner tag on the public demo; None locally (everything visible)."""
    return getattr(request, "datatrust_owner", None)


def _demo(request) -> bool:
    return bool(getattr(request, "datatrust_demo", False))


class DemoLimitReached(UploadRejected):
    """Too many uploads or a full queue on the public demo (HTTP 429)."""


def _demo_gate(request, limits: DemoLimits) -> None:
    """Public demo checks before a file is even saved. Raises DemoLimitReached."""
    owner = _owner(request)
    if jobs.pending(owner) >= 1:
        raise DemoLimitReached("You already have an analysis running. Wait for it to finish, then try again.")
    if jobs.pending() >= limits.max_pending_jobs:
        raise DemoLimitReached("The demo server is busy with other visitors' analyses. Try again in a minute.")
    wait = upload_limiter.check_and_record([f"visitor:{owner}", f"ip:{client_address(request.META)}"],
                                           limits.uploads_per_hour, limits.uploads_per_hour_total)
    if wait:
        raise DemoLimitReached(f"Upload limit reached on the public demo. Try again in about "
                               f"{max(1, round(wait / 60))} minute(s).")


def background_enabled() -> bool:
    """DATATRUST_WEB_BACKGROUND=0 analyses inside the request, as before Phase 10."""
    return os.environ.get("DATATRUST_WEB_BACKGROUND", "1").strip() != "0"


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
        page = dashboard_page(repo, visible_to=_owner(request))
    return render(request, "dashboard/home.html", page)


@require_http_methods(["GET", "POST"])
@handles_errors
def upload(request):
    limit_mb = DEFAULT_SETTINGS.max_file_size_mb
    demo = _demo(request)
    limits = DemoLimits.from_env() if demo else None
    context = {"limit_mb": limit_mb, "demo_limits": dataclasses.asdict(limits) if limits else None}
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        status = 400
        if form.is_valid():
            f = form.cleaned_data["file"]
            summary = form.cleaned_data["summary"]
            if demo and summary:
                summary = DEMO_SUMMARY
            options = {"target": form.cleaned_data["target"], "size": f.size,
                       "baseline": form.cleaned_data["baseline"], "summary": summary}
            if demo:
                options["owner"] = _owner(request)
                options["check"] = lambda path, name: check_public_upload(path, name, limits)
            try:
                if demo:
                    _demo_gate(request, limits)
                    with repository() as repo:      # keep the demo database small
                        repo.prune_owned_runs(limits.retention_hours, limits.max_stored_runs)
                if background_enabled():
                    job_id = start_upload_job(jobs, f.chunks(), f.name, **options)
                    return redirect("dashboard:job", job_id=job_id)
                with repository() as repo:
                    run_id = analyze_upload(f.chunks(), f.name, repo, **options)
            except UploadRejected as exc:
                form.add_error(None, str(exc))
                if isinstance(exc, DemoLimitReached):
                    status = 429
            else:
                return redirect("dashboard:run", run_id=run_id)
        return render(request, "dashboard/upload.html", {"form": form, **context}, status=status)
    return render(request, "dashboard/upload.html", {"form": UploadForm(), **context})


@require_http_methods(["GET", "POST"])
@handles_errors
def login(request):
    """Phase 10: single-user login (password from DATATRUST_WEB_PASSWORD). See datatrust/web/auth.py."""
    from dashboard.middleware import web_password
    password = web_password()
    next_url = safe_next(request.POST.get("next") or request.GET.get("next"))
    if not password:
        return redirect(next_url)                         # login is switched off
    context = {"next": next_url, "error": None}
    status = 200
    if request.method == "POST":
        client = request.META.get("REMOTE_ADDR", "unknown")
        wait = throttle.locked_for(client)
        if wait:
            context["error"] = f"Too many wrong passwords. Try again in {wait} seconds."
            status = 429
        elif check_password(request.POST.get("password"), password):
            throttle.succeeded(client)
            response = redirect(next_url)
            response.set_cookie(COOKIE_NAME, make_token(settings.SECRET_KEY, password), max_age=MAX_AGE,
                                httponly=True, samesite="Lax", secure=not settings.DEBUG)
            logger.info("Web login from %s", client)
            return response
        else:
            throttle.failed(client)
            logger.warning("Wrong web password from %s", client)
            context["error"] = "Wrong password."
            status = 401
    return render(request, "dashboard/login.html", context, status=status)


@require_http_methods(["POST"])
@handles_errors
def logout(request):
    response = redirect("dashboard:login")
    response.delete_cookie(COOKIE_NAME, samesite="Lax")
    return response


@require_GET
@handles_errors
def job(request, job_id: str):
    """Phase 10: 'Analysing...' page. Refreshes itself until the run page is ready."""
    found = jobs.get(job_id) if JOB_ID.match(job_id) else None
    if found is not None and found.get("owner") != _owner(request):
        found = None                       # public demo: another visitor's job looks like no job
    if found is None:
        raise NotFound("This analysis is not known to the server. It may have finished long ago, "
                       "or the server was restarted while it ran; check the dashboard.")
    if found["state"] == "done":
        return redirect("dashboard:run", run_id=found["run_id"])
    status = 200 if found["state"] in ("queued", "running") else {"user": 400, "database": 503}.get(
        found["error_kind"], 500)
    return render(request, "dashboard/job.html", {"job": found, "refresh_seconds": 3}, status=status)


@require_GET
@handles_errors
def dataset(request, name: str):
    with repository() as repo:
        page = dataset_page(repo, name, visible_to=_owner(request))
    return render(request, "dashboard/dataset.html", page)


@require_GET
@handles_errors
def run(request, run_id: int):
    with repository() as repo:
        page = run_page(repo, run_id, visible_to=_owner(request))
    return render(request, "dashboard/run.html", page)


def _download(response: HttpResponse, filename: str) -> HttpResponse:
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@require_GET
@handles_errors
def report(request, run_id: int):
    """The full self-contained HTML report (Phase 5a), re-rendered from the stored analysis."""
    with repository() as repo:
        analysis = stored_analysis(repo, run_id, visible_to=_owner(request))
    response = HttpResponse(render_html(analysis), content_type="text/html; charset=utf-8")
    if request.GET.get("download"):
        _download(response, f"datatrust-run-{run_id}.html")
    return response


@require_GET
@handles_errors
def analysis_json(request, run_id: int):
    with repository() as repo:
        analysis = stored_analysis(repo, run_id, visible_to=_owner(request))
    body = json.dumps(analysis, indent=2, ensure_ascii=False, default=str)
    return _download(HttpResponse(body, content_type="application/json; charset=utf-8"),
                     f"datatrust-run-{run_id}.json")


def page_not_found(request, exception=None):
    return _error(request, 404, "Not found", "There is no page at this address.")


def server_error(request):
    return _error(request, 500, "Something went wrong",
                  "An unexpected error occurred. The details are in logs/datatrust.log.")
