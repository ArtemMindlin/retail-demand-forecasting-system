"""Process-wide :class:`ArtifactStore`, configured from Django settings.

The store itself knows nothing about Django; this module is the single seam
where the two meet, so tests can build their own store against a tmp_path
without touching settings.
"""

from __future__ import annotations

from django.conf import settings

from retail_forecasting.api.services.runs import ArtifactStore

_store: ArtifactStore | None = None


def get_store() -> ArtifactStore:
    """Return the shared store, building it on first use."""
    global _store
    if _store is None:
        _store = ArtifactStore(settings.REPORTS_DIR, settings.CONFIG_PATH)
    return _store


def reset_store() -> None:
    """Drop the shared store. Used by tests and after a settings override."""
    global _store
    _store = None
