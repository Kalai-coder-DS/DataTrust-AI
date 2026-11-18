"""
datatrust/web/deploy.py  (Deployment)

Every setting that differs between "my laptop" and "the public server", computed from
environment variables WITHOUT Django, so pytest can check each rule (Django's settings.py just
copies the result).

Three ways the app runs:

    local (default)   DATATRUST_DEBUG unset/1   -> exactly as before this file existed:
                      DEBUG on, localhost only, no HTTPS rules, the dev key is allowed.
    private server    DATATRUST_DEBUG=0          -> secret key AND a web password required.
    public demo       DATATRUST_DEBUG=0 + DATATRUST_PUBLIC_DEMO=1
                      -> no password (anyone can try it), but every upload is limited
                         (DemoLimits) and visible only to the visitor who sent it.

Secrets are only ever read from the environment (Render's dashboard, or .env locally, which
git ignores); nothing secret is written in render.yaml or anywhere else in the repository.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

DEV_SECRET_KEY = "dev-only-insecure-key-do-not-use-in-production"
MIN_SECRET_LENGTH = 32
HSTS_SECONDS = 30 * 24 * 3600     # browsers remember "HTTPS only" for 30 days


class DeploymentConfigError(Exception):
    """The environment variables describe an unsafe or impossible setup. The message says what to set."""


def _flag(env: dict, name: str, default: str = "0") -> bool:
    return env.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _int(env: dict, name: str, default: int, minimum: int = 0) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise DeploymentConfigError(f"{name} must be a whole number, not '{raw}'.") from exc
    if value < minimum:
        raise DeploymentConfigError(f"{name} must be at least {minimum}.")
    return value


def django_settings(environ: dict | None = None) -> dict:
    """
    The deployment-dependent Django settings, as a dict of setting name -> value.
    Raises DeploymentConfigError (settings.py turns it into ImproperlyConfigured).
    """
    env = os.environ if environ is None else environ
    debug = env.get("DATATRUST_DEBUG", "1").strip() == "1"
    demo = _flag(env, "DATATRUST_PUBLIC_DEMO")

    if demo and debug:
        raise DeploymentConfigError("DATATRUST_PUBLIC_DEMO=1 needs DATATRUST_DEBUG=0: never show debug "
                                    "pages (with settings and tracebacks) to the public.")

    secret = env.get("DATATRUST_SECRET_KEY", "").strip()
    if not secret:
        if not debug:
            raise DeploymentConfigError("Set DATATRUST_SECRET_KEY when DATATRUST_DEBUG=0.")
        secret = DEV_SECRET_KEY
    elif not debug and (secret == DEV_SECRET_KEY or len(secret) < MIN_SECRET_LENGTH):
        raise DeploymentConfigError(f"DATATRUST_SECRET_KEY is too weak: use a random value of at least "
                                    f"{MIN_SECRET_LENGTH} characters (Render's generateValue does this).")

    password = env.get("DATATRUST_WEB_PASSWORD", "").strip()
    if not debug and not demo and not password:
        raise DeploymentConfigError("Set DATATRUST_WEB_PASSWORD when DATATRUST_DEBUG=0 (the web app needs a login), "
                                    "or DATATRUST_PUBLIC_DEMO=1 for an open, limited public demo.")

    hosts = [h.strip() for h in env.get("DATATRUST_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()]
    render_host = env.get("RENDER_EXTERNAL_HOSTNAME", "").strip()   # set by Render automatically
    if render_host and render_host not in hosts:
        hosts.append(render_host)

    https = not debug and _flag(env, "DATATRUST_HTTPS", "1")
    settings = {
        "DEBUG": debug,
        "SECRET_KEY": secret,
        "ALLOWED_HOSTS": hosts,
        "DATATRUST_PUBLIC_DEMO": demo,
        "CSRF_COOKIE_SECURE": not debug,                # as before: production cookies need HTTPS
        "CSRF_TRUSTED_ORIGINS": [f"https://{h}" for h in hosts if https and h not in ("localhost", "127.0.0.1")],
        "SECURE_SSL_REDIRECT": https,
        "SECURE_HSTS_SECONDS": HSTS_SECONDS if https else 0,
        "SECURE_HSTS_INCLUDE_SUBDOMAINS": False,        # onrender.com subdomains belong to other people
        "SECURE_HSTS_PRELOAD": False,
        # Render ends HTTPS at its proxy and forwards plain HTTP with this header. Trust it ONLY
        # behind such a proxy (https on); otherwise a client could claim "https" itself.
        "SECURE_PROXY_SSL_HEADER": ("HTTP_X_FORWARDED_PROTO", "https") if https else None,
        # `manage.py check --deploy` would warn about the two HSTS options above. Both are off on
        # purpose: other people's sites live on onrender.com subdomains, and preloading is a
        # permanent browser-list decision for a domain you own.
        "SILENCED_SYSTEM_CHECKS": ["security.W005", "security.W021"] if https else [],
    }
    return settings


@dataclass(frozen=True)
class DemoLimits:
    """
    Limits for uploads on the public demo (read at request time, so tests can change them).
    The file-size limit itself is DATATRUST_MAX_FILE_MB (datatrust.config), used everywhere.
    """
    max_rows: int = 20_000              # DATATRUST_DEMO_MAX_ROWS
    max_columns: int = 50               # DATATRUST_DEMO_MAX_COLUMNS
    uploads_per_hour: int = 10          # DATATRUST_DEMO_UPLOADS_PER_HOUR      (per visitor and per address)
    uploads_per_hour_total: int = 60    # DATATRUST_DEMO_UPLOADS_PER_HOUR_TOTAL (whole server)
    max_pending_jobs: int = 3           # DATATRUST_DEMO_MAX_PENDING (queued + running, whole server)
    retention_hours: int = 24           # DATATRUST_DEMO_RETENTION_HOURS (visitor uploads deleted after)
    max_stored_runs: int = 300          # DATATRUST_DEMO_MAX_STORED_RUNS (oldest visitor runs deleted beyond)

    @classmethod
    def from_env(cls, environ: dict | None = None) -> "DemoLimits":
        env = os.environ if environ is None else environ
        d = cls()
        return cls(
            max_rows=_int(env, "DATATRUST_DEMO_MAX_ROWS", d.max_rows, 1),
            max_columns=_int(env, "DATATRUST_DEMO_MAX_COLUMNS", d.max_columns, 1),
            uploads_per_hour=_int(env, "DATATRUST_DEMO_UPLOADS_PER_HOUR", d.uploads_per_hour, 1),
            uploads_per_hour_total=_int(env, "DATATRUST_DEMO_UPLOADS_PER_HOUR_TOTAL", d.uploads_per_hour_total, 1),
            max_pending_jobs=_int(env, "DATATRUST_DEMO_MAX_PENDING", d.max_pending_jobs, 1),
            retention_hours=_int(env, "DATATRUST_DEMO_RETENTION_HOURS", d.retention_hours, 1),
            max_stored_runs=_int(env, "DATATRUST_DEMO_MAX_STORED_RUNS", d.max_stored_runs, 1),
        )
