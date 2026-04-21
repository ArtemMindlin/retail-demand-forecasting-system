from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_GENERATED_PLACEHOLDERS = {
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
    "reports/.gitkeep",
}
GENERATED_PREFIXES = (
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "reports/",
    "tmp/",
)


def test_generated_artifacts_are_not_tracked_as_source_files() -> None:
    tracked_files = _tracked_files()
    violations = [
        path
        for path in tracked_files
        if (REPO_ROOT / path).exists()
        and _is_generated_artifact(path)
        and path not in ALLOWED_GENERATED_PLACEHOLDERS
    ]

    assert violations == []


def test_project_gitignore_covers_local_generated_artifacts() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    required_patterns = [
        ".DS_Store",
        "data/raw/*",
        "data/interim/*",
        "data/processed/*",
        "reports/*",
        "tmp/*",
        "output/*",
    ]

    for pattern in required_patterns:
        assert pattern in gitignore


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def _is_generated_artifact(path: str) -> bool:
    return path.endswith(".DS_Store") or path.startswith(GENERATED_PREFIXES)
