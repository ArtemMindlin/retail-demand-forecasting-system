"""Django settings for the retail forecasting dashboard.

Deliberately database-free: the system's state lives in on-disk artifacts under
``reports/`` (Parquet/CSV written by the pipeline), not in a relational store.
Sessions therefore use the signed-cookie backend, and no app in
``INSTALLED_APPS`` defines models. There is nothing to migrate.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root: src/retail_forecasting/api/settings.py -> up 4.
BASE_DIR = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parent


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"false", "0", "no"}


# ── Core ──────────────────────────────────────────────────────────────────────
# Sessions are signed with SECRET_KEY; rotating it logs everyone out, which is
# an acceptable trade for not persisting a session store.
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-in-prod")
DEBUG = _env_flag("DJANGO_DEBUG", False)
ALLOWED_HOSTS = [h for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",") if h]

# TLS is terminated by the reverse proxy, so Django must trust its forwarded
# scheme header — otherwise request.is_secure() is always False behind Caddy and
# secure cookies are never honoured.
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

ROOT_URLCONF = "retail_forecasting.api.urls"
WSGI_APPLICATION = "retail_forecasting.api.wsgi.application"
ASGI_APPLICATION = "retail_forecasting.api.asgi.application"

INSTALLED_APPS = [
    "django.contrib.sessions",
    "django.contrib.staticfiles",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "retail_forecasting.api.middleware.LoginRequiredMiddleware",
]

# No relational store. Any accidental ORM use fails loudly instead of silently
# creating a SQLite file.
DATABASES: dict[str, dict[str, str]] = {}

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [APP_DIR / "templates"],
        "APP_DIRS": False,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "retail_forecasting.api.context.navigation",
                "retail_forecasting.api.context.whatif",
            ],
            "builtins": ["retail_forecasting.api.templatetags.dashboard"],
        },
    },
]

# ── Sessions & auth ───────────────────────────────────────────────────────────
SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"
SESSION_COOKIE_NAME = "rf_session"
SESSION_COOKIE_AGE = 60 * 60 * 24 * 7  # 7 days
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
# Safari drops Secure cookies over http://localhost (Chrome/Firefox allow them),
# so local dev sets COOKIE_SECURE=false. Production keeps the secure default.
SESSION_COOKIE_SECURE = _env_flag("COOKIE_SECURE", True)
CSRF_COOKIE_SECURE = SESSION_COOKIE_SECURE
CSRF_TRUSTED_ORIGINS = [
    o for o in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",") if o
]

# Single operator account, supplied by environment — same contract as before.
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "ArtemMindlin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "")

# Paths reachable without a session.
PUBLIC_URL_NAMES = {"login", "health"}

# ── Static files ──────────────────────────────────────────────────────────────
STATIC_URL = "/static/"
STATICFILES_DIRS = [APP_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        if DEBUG
        else "django.contrib.staticfiles.storage.ManifestStaticFilesStorage"
    },
}

# ── Pipeline artifacts ────────────────────────────────────────────────────────
# CWD-relative by default, matching how the pipeline and the Makefile targets
# already resolve these paths.
REPORTS_DIR = Path(os.environ.get("RETAIL_REPORTS_DIR", "reports"))
CONFIG_PATH = Path(os.environ.get("RETAIL_CONFIG_PATH", "configs/experiment.yaml"))

# Rate limit for the pipeline-trigger endpoint.
RUN_RATE_LIMIT_MAX = 3
RUN_RATE_LIMIT_WINDOW = 600  # seconds

# ── I18N ──────────────────────────────────────────────────────────────────────
LANGUAGE_CODE = "es-es"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
