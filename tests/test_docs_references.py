from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = [
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "README.md",
    *sorted((REPO_ROOT / "docs").rglob("*.md")),
]
LOCAL_PATH_PREFIXES = (
    "AGENTS.md",
    "README.md",
    "configs/",
    "docs/",
    "src/",
    "tests/",
)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


def test_documented_local_references_exist() -> None:
    violations = []

    for doc_path in DOC_PATHS:
        text = doc_path.read_text(encoding="utf-8")
        for reference in _local_references(text):
            target = REPO_ROOT / reference
            if not target.exists():
                violations.append(
                    f"{doc_path.relative_to(REPO_ROOT)} references missing path: {reference}"
                )

    assert violations == []


def _local_references(text: str) -> set[str]:
    references = set()

    for match in MARKDOWN_LINK_RE.finditer(text):
        reference = _normalize_reference(match.group(1))
        if _is_local_project_reference(reference):
            references.add(reference)

    text_without_code_blocks = FENCED_CODE_BLOCK_RE.sub("", text)
    for match in INLINE_CODE_RE.finditer(text_without_code_blocks):
        for token in match.group(1).split():
            reference = _normalize_reference(token)
            if _is_local_project_reference(reference):
                references.add(reference)

    return references


def _normalize_reference(reference: str) -> str:
    normalized = reference.strip().split("#", maxsplit=1)[0]
    return normalized.strip(".,:;()[]{}'\"")


def _is_local_project_reference(reference: str) -> bool:
    if not reference:
        return False
    if "<" in reference or ">" in reference:
        return False
    if "YYYY" in reference or "short-kebab-case-title" in reference:
        return False
    if "://" in reference or reference.startswith("/"):
        return False
    return reference.startswith(LOCAL_PATH_PREFIXES)
