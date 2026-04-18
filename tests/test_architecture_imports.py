from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "retail_forecasting"
FIRST_PARTY_PREFIX = "retail_forecasting"

FORBIDDEN_LAYER_IMPORTS = {
    "config": {
        "data",
        "drift",
        "evaluation",
        "features",
        "forecasting",
        "inventory",
        "models",
        "visualization",
    },
    "data": {
        "drift",
        "evaluation",
        "features",
        "forecasting",
        "inventory",
        "models",
        "visualization",
    },
    "drift": {
        "data",
        "evaluation",
        "features",
        "forecasting",
        "inventory",
        "models",
        "visualization",
    },
    "evaluation": {
        "data",
        "drift",
        "features",
        "forecasting",
        "inventory",
        "models",
    },
    "features": {
        "data",
        "drift",
        "evaluation",
        "forecasting",
        "inventory",
        "models",
        "visualization",
    },
    "inventory": {
        "data",
        "drift",
        "evaluation",
        "features",
        "forecasting",
        "models",
        "visualization",
    },
    "models": {
        "data",
        "drift",
        "evaluation",
        "features",
        "forecasting",
        "inventory",
        "visualization",
    },
    "run": {
        "data",
        "drift",
        "evaluation",
        "features",
        "inventory",
        "models",
        "visualization",
    },
    "utils": {
        "data",
        "drift",
        "evaluation",
        "features",
        "forecasting",
        "inventory",
        "models",
        "visualization",
    },
    "visualization": {
        "data",
        "drift",
        "evaluation",
        "features",
        "forecasting",
        "inventory",
        "models",
    },
}


def test_layer_imports_do_not_cross_forbidden_boundaries() -> None:
    violations = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source_layer = _source_layer(path)
        forbidden_layers = FORBIDDEN_LAYER_IMPORTS.get(source_layer, set())
        imported_layers = _first_party_imported_layers(path)
        blocked_layers = sorted(imported_layers & forbidden_layers)

        if blocked_layers:
            violations.append(
                f"{path.relative_to(PACKAGE_ROOT)} imports forbidden layer(s): "
                f"{', '.join(blocked_layers)}"
            )

    assert violations == []


def _source_layer(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT)
    if len(relative.parts) == 1:
        return relative.stem
    return relative.parts[0]


def _first_party_imported_layers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_layers: set[str] = set()

    for node in ast.walk(tree):
        module_name = _imported_module_name(node)
        layer = _first_party_layer(module_name)
        if layer is not None:
            imported_layers.add(layer)

    return imported_layers


def _imported_module_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.ImportFrom):
        return node.module

    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == FIRST_PARTY_PREFIX or alias.name.startswith(
                f"{FIRST_PARTY_PREFIX}.",
            ):
                return alias.name

    return None


def _first_party_layer(module_name: str | None) -> str | None:
    if module_name is None:
        return None

    if module_name == FIRST_PARTY_PREFIX:
        return None

    prefix = f"{FIRST_PARTY_PREFIX}."
    if not module_name.startswith(prefix):
        return None

    remainder = module_name.removeprefix(prefix)
    return remainder.split(".", maxsplit=1)[0]
