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
    DATATRUST_DB, DATATRUST_MYSQL_*   database settings, see .env.example
"""
import os
import sys
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent          # webapp/
PROJECT_ROOT = BASE_DIR.parent                             # repository root (holds datatrust/)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datatrust.storage import load_env_file  # noqa: E402  (needs PROJECT_ROOT on sys.path)

load_env_file()   # values in the real environment win over .env

DEBUG = os.environ.get("DATATRUST_DEBUG", "1").strip() == "1"

SECRET_KEY = os.environ.get("DATATRUST_SECRET_KEY", "")
if not SECRET_KEY:
    if not DEBUG:
        raise ImproperlyConfigured("Set DATATRUST_SECRET_KEY when DATATRUST_DEBUG=0.")
    SECRET_KEY = "dev-only-insecure-key-do-not-use-in-production"

# Phase 10: login. Without a password the app is open (fine on your own computer, with the
# default ALLOWED_HOSTS = localhost only). In production mode a password is required.
if not DEBUG and not os.environ.get("DATATRUST_WEB_PASSWORD", "").strip():
    raise ImproperlyConfigured("Set DATATRUST_WEB_PASSWORD when DATATRUST_DEBUG=0 (the web app needs a login).")

ALLOWED_HOSTS = [h.strip() for h in os.environ.get("DATATRUST_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()]

INSTALLED_APPS = [
    "django.contrib.humanize",   # number formatting in templates (no database tables)
    "dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
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
if not DEBUG:
    CSRF_COOKIE_SECURE = True

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {"django": {"handlers": ["console"], "level": "WARNING"}},
}
