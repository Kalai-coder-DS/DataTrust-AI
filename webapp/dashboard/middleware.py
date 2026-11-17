"""
Content-Security-Policy header on every response (defence in depth against XSS).

The pages use no JavaScript and no external files, so the policy can be strict: only
inline styles (needed for the bar widths) and data: images are allowed. Even if some
uploaded text slipped through escaping, a <script> in it could not run.
"""
import os
from urllib.parse import urlencode

from django.conf import settings
from django.shortcuts import redirect
from django.urls import reverse

from datatrust.web.auth import COOKIE_NAME, check_token


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
