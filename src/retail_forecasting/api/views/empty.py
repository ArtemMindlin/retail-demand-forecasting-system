"""The one empty state, built in one place.

Invariant 32: a missing artifact renders an explicit empty state that names the command
which produces it. Four templates used to say that four times; now the shape is a
function signature, so a new caller cannot omit the command by accident.
"""

from __future__ import annotations

from typing import Any

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


def empty_state(
    request: HttpRequest,
    *,
    label: str,
    title: str,
    detail: str,
    icon: str = "boxes",
    hint: str | None = None,
    show_run_button: bool = False,
    page_title: str | None = None,
    status: int = 200,
) -> HttpResponse:
    """Render the shared empty state. ``hint`` is where the command goes."""
    context: dict[str, Any] = {
        "label": label,
        "title": title,
        "detail": detail,
        "icon": icon,
        "hint": hint,
        "show_run_button": show_run_button,
        "page_title": page_title or title,
    }
    return render(request, "views/empty_state.html", context, status=status)
