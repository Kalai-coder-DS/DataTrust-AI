"""
Django settings for the DataTrust AI web app (Phase 6).

Deliberately small:
  - No Django ORM models and no Django database (DATABASES = {}). Every read and write goes
    through datatrust.storage (repository pattern), which talks to MySQL itself. Decision
    "Option B": one data-access layer for the CLI and the web app, no duplicated table
    definitions, no migrations to keep in sync with sql/schema_mysql.sql.
  - No admin, auth or sessions apps (they need ORM tables). CSRF protection still works:
    Django keeps the CSRF secret in a cookie.
  - CSS is inline in the base template, so no static-files setup is needed.

Settings come from environment variables (or the project's .env file):
    DATATRUST_DEBUG          1 (default, local development) or 0
    DATATRUST_SECRET_KEY     required when DATATRUST_DEBUG=0
    DATATRUST_ALLOWED_HOSTS  comma-separated, default "localhost,127.0.0.1"
    DATATRUST_PUBLIC_DEMO    1 = open public demo (no login, limited uploads), see docs/DEPLOYMENT.md
    DATATRUST_DB, DATATRUST_MYSQL_*   database settings, see .env.example

Deployment: DEBUG, the secret key, allowed hosts and the HTTPS rules are computed by
datatrust/web/deploy.py (plain Python, tested with pytest); this file only copies them.
Locally (no DATATRUST_DEBUG set) everything is exactly as it was before deployment.
"""
import sys
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent          # webapp/
PROJECT_ROOT = BASE_DIR.parent                             # repository root (holds datatrust/)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datatrust.storage import load_env_file  # noqa: E402  (needs PROJECT_ROOT on sys.path)

load_env_file()   # values in the real environment win over .env

from datatrust.web.deploy import DeploymentConfigError, django_settings  # noqa: E402

# Phase 10 rule, kept: with DATATRUST_DEBUG=0 a web password is required, unless this is the
# open public demo (DATATRUST_PUBLIC_DEMO=1), which has upload limits and private uploads instead.
try:
    _deploy = django_settings()
except DeploymentConfigError as exc:
    raise ImproperlyConfigured(str(exc)) from exc

DEBUG = _deploy["DEBUG"]
SECRET_KEY = _deploy["SECRET_KEY"]
ALLOWED_HOSTS = _deploy["ALLOWED_HOSTS"]
DATATRUST_PUBLIC_DEMO = _deploy["DATATRUST_PUBLIC_DEMO"]

# HTTPS (production only; all off locally). Render ends HTTPS at its proxy, which sends
# X-Forwarded-Proto: https; plain-http visits are redirected and browsers told to stay on https.
CSRF_TRUSTED_ORIGINS = _deploy["CSRF_TRUSTED_ORIGINS"]
SECURE_SSL_REDIRECT = _deploy["SECURE_SSL_REDIRECT"]
SECURE_HSTS_SECONDS = _deploy["SECURE_HSTS_SECONDS"]
SECURE_HSTS_INCLUDE_SUBDOMAINS = _deploy["SECURE_HSTS_INCLUDE_SUBDOMAINS"]
SECURE_HSTS_PRELOAD = _deploy["SECURE_HSTS_PRELOAD"]
SECURE_PROXY_SSL_HEADER = _deploy["SECURE_PROXY_SSL_HEADER"]
SILENCED_SYSTEM_CHECKS = _deploy["SILENCED_SYSTEM_CHECKS"]

INSTALLED_APPS = [
    "django.contrib.humanize",   # number formatting in templates (no database tables)
    "dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "dashboard.middleware.UploadSizeLimitMiddleware",     # refuse oversized bodies before parsing
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "dashboard.middleware.VisitorMiddleware",             # public demo: private uploads per browser
    "dashboard.middleware.LoginRequiredMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "dashboard.middleware.ContentSecurityPolicyMiddleware",
]

ROOT_URLCONF = "datatrust_site.urls"
WSGI_APPLICATION = "datatrust_site.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

# Option B: no Django-managed database. See the module docstring.
DATABASES = {}

# Uploads: files over 2.5 MB are streamed to a temporary file by Django, not held in memory.
# The real size limit (DATATRUST_MAX_FILE_MB) is enforced in datatrust.web while streaming.
FILE_UPLOAD_MAX_MEMORY_SIZE = 2_621_440

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"          # analysis times are stored in UTC
USE_I18N = False
USE_TZ = True

# Security headers (Django defaults made explicit).
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SECURE = _deploy["CSRF_COOKIE_SECURE"]

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {"django": {"handlers": ["console"], "level": "WARNING"}},
}
