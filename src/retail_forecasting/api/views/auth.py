"""Login and logout for the single operator account.

Credentials come from the environment, as before. Two things changed in the
port: login is now a real form POST (so the browser's password manager works and
there is CSRF protection), and an unset ``AUTH_PASSWORD`` refuses every login
instead of accepting an empty one.
"""

from __future__ import annotations

import hmac

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods


def _credentials_valid(username: str, password: str) -> bool:
    """Constant-time credential check.

    An empty configured password means the deployment never set ``AUTH_PASSWORD``.
    The previous implementation let an empty submitted password through in that
    case; here it is treated as "login disabled".
    """
    expected_password = settings.AUTH_PASSWORD
    if not expected_password:
        return False
    return hmac.compare_digest(username, settings.AUTH_USERNAME) and hmac.compare_digest(
        password, expected_password
    )


def _safe_next(request: HttpRequest) -> str:
    """Return the ``?next=`` target when it is a local path, else the dashboard."""
    target = str(request.POST.get("next") or request.GET.get("next") or "")
    if target.startswith("/") and not target.startswith("//"):
        return target
    return str(reverse("dashboard"))


@require_http_methods(["GET", "POST"])
def login(request: HttpRequest) -> HttpResponse:
    """Render the login screen and authenticate the operator."""
    if request.session.get("authenticated"):
        return redirect(_safe_next(request))

    if request.method == "GET":
        return render(request, "login.html", {"next": request.GET.get("next", "")})

    username = request.POST.get("username", "")
    password = request.POST.get("password", "")

    if not _credentials_valid(username, password):
        return render(
            request,
            "login.html",
            {
                "error": "Credenciales incorrectas",
                "next": request.POST.get("next", ""),
                "username": username,
            },
            status=401,
        )

    request.session.cycle_key()
    request.session["authenticated"] = True
    request.session.set_expiry(settings.SESSION_COOKIE_AGE)
    return redirect(_safe_next(request))


@require_http_methods(["POST"])
def logout(request: HttpRequest) -> HttpResponse:
    """Drop the session and return to the login screen."""
    request.session.flush()
    return redirect(reverse("login"))
