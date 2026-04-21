from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "retail_forecasting"
RAW_COLUMN_NAMES = {"dt", "sale_amount", "stock_hour6_22_cnt"}
RAW_COLUMN_OWNER = PACKAGE_ROOT / "data" / "fresh_retailnet.py"


def test_raw_dataset_column_names_stay_inside_data_loader() -> None:
    violations = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path == RAW_COLUMN_OWNER:
            continue

        for line_number, raw_column in _raw_column_literals(path):
            violations.append(
                f"{path.relative_to(PACKAGE_ROOT)}:{line_number} uses raw column "
                f"`{raw_column}` outside data/fresh_retailnet.py"
            )

    assert violations == []


def _raw_column_literals(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in RAW_COLUMN_NAMES:
                matches.append((node.lineno, node.value))

    return matches
