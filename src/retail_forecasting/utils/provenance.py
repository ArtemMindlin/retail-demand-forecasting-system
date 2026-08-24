from __future__ import annotations

import subprocess
from datetime import UTC, datetime

_frozen_commit: tuple[str | None] | None = None


def freeze_git_commit() -> str | None:
    """Read HEAD once, at the start of a run, and pin it for the rest of the process.

    Every artifact a run writes is stamped when it is written, which for a long run is hours
    after the code started executing. Committing in the meantime made the stamp name a commit
    that did not exist at launch: a 4h39m imputation search reported the EDA commit landed
    while it ran, not the search code that produced its number.
    """
    global _frozen_commit
    _frozen_commit = (_read_git_commit(),)
    return _frozen_commit[0]


def get_git_commit() -> str | None:
    """The frozen commit when a run pinned one, otherwise HEAD as it stands right now.

    The fallback keeps importers that never call `freeze_git_commit` -- the dashboard, the
    test suite -- working unchanged.
    """
    if _frozen_commit is not None:
        return _frozen_commit[0]
    return _read_git_commit()


def _read_git_commit() -> str | None:
    head = _git("rev-parse", "--short", "HEAD")
    if head is None:
        return None
    status = _git("status", "--porcelain")
    return f"{head}-dirty" if status else head


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()
