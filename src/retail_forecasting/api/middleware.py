"""Session gate for the dashboard.

Replaces the hand-rolled FastAPI ``auth_middleware``. The rule is unchanged:
everything requires a session except a small public allowlist. What differs is
the failure mode — browser navigations get redirected to the login page instead
of a bare 401, while JSON and HTMX callers still get a machine-readable refusal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.urls import reverse


class LoginRequiredMiddleware:
    """Require an authenticated session for every view outside the allowlist."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        return self.get_response(request)

    def process_view(
        self,
        request: HttpRequest,
        view_func: Callable[..., HttpResponse],
        view_args: tuple[Any, ...],
        view_kwargs: dict[str, Any],
    ) -> HttpResponse | None:
        if getattr(view_func, "login_exempt", False):
            return None

        match = request.resolver_match
        url_name = match.url_name if match else None
        if url_name in settings.PUBLIC_URL_NAMES:
            return None

        if request.session.get("authenticated"):
            return None

        return self._deny(request)

    @staticmethod
    def _deny(request: HttpRequest) -> HttpResponse:
        if request.path.startswith("/api/"):
            return JsonResponse({"detail": "Not authenticated"}, status=401)

        if request.headers.get("HX-Request"):
            # Tell htmx to hard-redirect rather than swapping a login page into
            # some inner fragment target.
            response = HttpResponse(status=401)
            response["HX-Redirect"] = reverse("login")
            return response

        login_url = reverse("login")
        if request.path != "/":
            return redirect(f"{login_url}?next={request.get_full_path()}")
        return redirect(login_url)
