from __future__ import annotations

import subprocess
from datetime import UTC, datetime


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()
