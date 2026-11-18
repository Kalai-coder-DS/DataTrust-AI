"""
Middleware (each one is a thin wrapper; the logic lives in datatrust/web/):

  UploadSizeLimitMiddleware        (deployment) refuse a request body bigger than the upload limit
                                   BEFORE Django reads it (Django would otherwise write the whole
                                   body to a temporary file first, whatever its size).
  VisitorMiddleware                (deployment) public demo: a signed visitor cookie per browser,
                                   so each visitor sees only public runs and their own uploads.
  LoginRequiredMiddleware          (Phase 10) the optional single-user login.
  ContentSecurityPolicyMiddleware  CSP header on every response (defence in depth against XSS).
                                   The pages use no JavaScript and no external files, so the
                                   policy can be strict: only inline styles (needed for the bar
                                   widths) and data: images are allowed.
"""
import os
from urllib.parse import urlencode

from django.conf import settings
from django.shortcuts import redirect, render
from django.urls import reverse

from datatrust.config import DEFAULT_SETTINGS
from datatrust.web.auth import COOKIE_NAME, check_token
from datatrust.web.demo import VISITOR_COOKIE, VISITOR_MAX_AGE, new_visitor_cookie, owner_tag, read_visitor_cookie

# Room for the multipart wrapping and the form's other fields around the file itself.
BODY_OVERHEAD_BYTES = 256 * 1024


class UploadSizeLimitMiddleware:
    """413 for a request whose declared size is over the upload limit; the body is never read."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            length = int(request.META.get("CONTENT_LENGTH") or 0)
        except ValueError:
            length = 0
        limit = DEFAULT_SETTINGS.max_file_size_bytes + BODY_OVERHEAD_BYTES
        if length > limit:
            return render(request, "dashboard/error.html", {
                "status": 413, "title": "File too large",
                "message": f"Uploads are limited to {DEFAULT_SETTINGS.max_file_size_mb} MB. "
                           "Choose a smaller CSV file."}, status=413)
        return self.get_response(request)


class VisitorMiddleware:
    """
    Public demo only (settings.DATATRUST_PUBLIC_DEMO). Sets:
        request.datatrust_demo   True / False
        request.datatrust_owner  the visitor's owner tag, or None (local app: everything is shared)
    A browser without a valid visitor cookie gets a new one with this response.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.datatrust_demo = bool(getattr(settings, "DATATRUST_PUBLIC_DEMO", False))
        request.datatrust_owner = None
        new_cookie = None
        if request.datatrust_demo:
            visitor_id = read_visitor_cookie(request.COOKIES.get(VISITOR_COOKIE), settings.SECRET_KEY)
            if visitor_id is None:
                visitor_id, new_cookie = new_visitor_cookie(settings.SECRET_KEY)
            request.datatrust_owner = owner_tag(visitor_id, settings.SECRET_KEY)
        response = self.get_response(request)
        if new_cookie:
            response.set_cookie(VISITOR_COOKIE, new_cookie, max_age=VISITOR_MAX_AGE, httponly=True,
                                samesite="Lax", secure=not settings.DEBUG)
        return response


def web_password() -> str:
    """Read at request time, so a changed .env/environment applies without code changes (and tests can set it)."""
    return os.environ.get("DATATRUST_WEB_PASSWORD", "").strip()


class LoginRequiredMiddleware:
    """
    Phase 10: every page except the login page needs a valid login cookie, when a password is set.
    The logic lives in datatrust/web/auth.py; this only connects it to Django.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        password = web_password()
        request.datatrust_login_enabled = bool(password)
        if password and request.path != reverse("dashboard:login"):
            if not check_token(request.COOKIES.get(COOKIE_NAME), settings.SECRET_KEY, password):
                target = request.get_full_path() if request.method == "GET" else "/"
                return redirect(f"{reverse('dashboard:login')}?{urlencode({'next': target})}")
        return self.get_response(request)


CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; "
    "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
)


class ContentSecurityPolicyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if "Content-Security-Policy" not in response:
            response["Content-Security-Policy"] = CSP
        return response
