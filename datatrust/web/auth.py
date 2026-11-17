"""
datatrust/web/auth.py  (Phase 10)

Single-user login for the web app, WITHOUT a database (decision "Option B": no Django ORM, so
Django's user tables are not available).

  - The password is DATATRUST_WEB_PASSWORD (environment or .env, which git ignores).
    Not set -> no login (local development). With DATATRUST_DEBUG=0 the app refuses to start
    without one (see webapp/datatrust_site/settings.py).
  - After a correct password the browser gets a signed cookie: "<expiry>.<HMAC-SHA256>".
    The HMAC key is the Django SECRET_KEY, and the signed message includes a fingerprint of the
    password, so changing the password logs everyone out. Nothing secret is stored in the cookie.
  - Comparisons use hmac.compare_digest (constant time: no timing leaks).
  - 5 wrong passwords from one address lock it out for 60 seconds (slows down guessing).
  - "next" after login only accepts paths on this site (no open redirect to another site).

Standard library only; the Django middleware and views in webapp/dashboard/ are thin wrappers.
"""
from __future__ import annotations

import hashlib
import hmac
import threading
import time

COOKIE_NAME = "datatrust_auth"
MAX_AGE = 12 * 3600            # seconds a login lasts
MAX_FAILURES = 5
LOCKOUT_SECONDS = 60


def _fingerprint(password: str) -> str:
    return hashlib.sha256(("datatrust-password:" + password).encode("utf-8")).hexdigest()


def _signature(secret: str, password: str, expiry: int) -> str:
    message = f"{expiry}|{_fingerprint(password)}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def make_token(secret: str, password: str, now: float | None = None, max_age: int = MAX_AGE) -> str:
    expiry = int((time.time() if now is None else now) + max_age)
    return f"{expiry}.{_signature(secret, password, expiry)}"


def check_token(token: str | None, secret: str, password: str, now: float | None = None) -> bool:
    if not token or "." not in token:
        return False
    expiry_text, signature = token.split(".", 1)
    if not expiry_text.isdigit():
        return False
    expiry = int(expiry_text)
    if expiry < (time.time() if now is None else now):
        return False
    return hmac.compare_digest(signature, _signature(secret, password, expiry))


def check_password(given: str | None, expected: str) -> bool:
    return hmac.compare_digest((given or "").encode("utf-8"), expected.encode("utf-8"))


def safe_next(url: str | None, default: str = "/") -> str:
    """Only a path on this site: '/runs/3/' yes; 'https://evil.example', '//evil', '/\\evil' no."""
    if not url or not url.startswith("/") or url.startswith("//") or url.startswith("/\\"):
        return default
    if any(ch in url for ch in "\r\n\t") or "\\" in url:
        return default
    return url


class LoginThrottle:
    """Counts wrong passwords per client address; locks it out for a while after too many."""

    def __init__(self, max_failures: int = MAX_FAILURES, lockout_seconds: int = LOCKOUT_SECONDS):
        self.max_failures, self.lockout_seconds = max_failures, lockout_seconds
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, client: str, now: float) -> list[float]:
        times = [t for t in self._failures.get(client, []) if now - t < self.lockout_seconds]
        self._failures[client] = times
        return times

    def locked_for(self, client: str, now: float | None = None) -> int:
        """Seconds until this client may try again (0 = allowed)."""
        now = time.time() if now is None else now
        with self._lock:
            times = self._recent(client, now)
            if len(times) < self.max_failures:
                return 0
            return max(1, int(self.lockout_seconds - (now - times[-self.max_failures])) + 1)

    def failed(self, client: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._recent(client, now).append(now)

    def succeeded(self, client: str) -> None:
        with self._lock:
            self._failures.pop(client, None)
