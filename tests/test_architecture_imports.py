from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "retail_forecasting"
FIRST_PARTY_PREFIX = "retail_forecasting"

ALLOWED_LAYER_IMPORTS = {
    "__init__": set(),
    "api": {"config", "forecasting"},
    "config": {"contracts"},
    "contracts": set(),
    "data": {"config", "contracts", "utils"},
    "drift": {"contracts"},
    "eda": {"config", "contracts", "data", "eda", "utils"},
    "evaluation": {"config", "contracts", "utils", "visualization"},
    "features": {"config", "contracts"},
    "forecasting": {
        "config",
        "contracts",
        "data",
        "drift",
        "evaluation",
        "features",
        "forecasting",
        "inventory",
        "models",
        "utils",
    },
    "inventory": {"config"},
    "models": {"config", "contracts", "utils"},
    "run": {"config", "contracts", "forecasting", "simulation"},
    "simulation": {
        "config",
        "data",
        "drift",
        "features",
        "forecasting",
        "inventory",
        "models",
    },
    "utils": set(),
    "visualization": {"config", "evaluation", "inventory", "utils"},
}


def test_layer_imports_do_not_cross_forbidden_boundaries() -> None:
    violations = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source_layer = _source_layer(path)

        if source_layer not in ALLOWED_LAYER_IMPORTS:
            violations.append(
                f"{path.relative_to(PACKAGE_ROOT)}: layer `{source_layer}` is not "
                f"registered in ALLOWED_LAYER_IMPORTS — add it."
            )
            continue

        allowed_layers = ALLOWED_LAYER_IMPORTS[source_layer] | {source_layer}
        imported_layers = _first_party_imported_layers(path)
        blocked_layers = sorted(imported_layers - allowed_layers)

        if blocked_layers:
            violations.append(
                f"{path.relative_to(PACKAGE_ROOT)} imports forbidden layer(s): "
                f"{', '.join(blocked_layers)}. Allowed for `{source_layer}`: "
                f"{', '.join(sorted(allowed_layers)) or 'none'}"
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
        for module_name in _imported_module_names(node):
            layer = _first_party_layer(module_name)
            if layer is not None:
                imported_layers.add(layer)

    return imported_layers


def _imported_module_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.ImportFrom):
        return {node.module} if node.module else set()

    if isinstance(node, ast.Import):
        return {
            alias.name
            for alias in node.names
            if alias.name == FIRST_PARTY_PREFIX or alias.name.startswith(f"{FIRST_PARTY_PREFIX}.")
        }

    return set()


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
