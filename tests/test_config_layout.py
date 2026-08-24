"""The configs/ tree is one folder per run mode, and each file declares only what it uses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from retail_forecasting.config import load_config
from retail_forecasting.contracts.contracts_config import MODE_SECTIONS, RunMode

CONFIGS = Path("configs")

# Fields that identify the dataset and the model store. Splitting one config per mode
# duplicates them, so they are the ones that can silently drift apart.
# `dataset.processed_panel_dir` is deliberately absent: the OPS plane streams its own
# dedicated split from `data/processed/ops_sim/`, built by scripts/build_ops_sim_split.py.
SHARED_FIELDS = (
    ("dataset", "hf_dataset_id"),
    ("dataset", "splits"),
    ("dataset", "local_cache_dir"),
    ("models", "models_dir"),
    ("models", "imputation_params_filename"),
)


def _config_files() -> list[Path]:
    return sorted(CONFIGS.glob("*/*.yaml"))


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_every_config_lives_in_a_run_mode_folder() -> None:
    modes = set(MODE_SECTIONS)
    folders = {path.parent.name for path in _config_files()}
    assert folders <= modes, f"carpetas sin run mode: {sorted(folders - modes)}"
    assert not list(CONFIGS.glob("*.yaml")), "los configs viven en carpetas por modo, no en la raíz"


@pytest.mark.parametrize("path", _config_files(), ids=lambda p: str(p))
def test_config_declares_only_sections_its_mode_reads(path: Path) -> None:
    mode: RunMode = path.parent.name  # type: ignore[assignment]
    declared = set(_load(path))
    extra = declared - MODE_SECTIONS[mode]
    assert not extra, (
        f"{path} declara secciones que el modo '{mode}' nunca lee: {sorted(extra)}. "
        "Un mando que no hace nada es peor que uno ausente."
    )


@pytest.mark.parametrize("path", _config_files(), ids=lambda p: str(p))
def test_config_run_mode_matches_its_folder(path: Path) -> None:
    declared = _load(path).get("project", {}).get("run_mode")
    assert declared == path.parent.name, (
        f"{path} declara run_mode='{declared}' pero vive en '{path.parent.name}/'"
    )


def test_shared_dataset_and_model_fields_do_not_drift() -> None:
    """One config per mode duplicates these; nothing else stops them diverging.

    Compared as EFFECTIVE values, so a config that omits the field counts as declaring the
    default. Comparing only what is written down missed the dangerous case: one config
    declaring a value while the others silently take the default is divergence, and for
    `imputation_params_filename` it sends the readers to a file the writer never wrote.
    """
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for path in _config_files():
        payload = _load(path)
        settings = load_config(path)
        for section, field in SHARED_FIELDS:
            if section not in MODE_SECTIONS[payload["project"]["run_mode"]]:
                continue
            effective = getattr(getattr(settings, section), field)
            seen.setdefault((section, field), {})[str(path)] = repr(effective)

    divergentes = {key: values for key, values in seen.items() if len(set(values.values())) > 1}
    assert not divergentes, f"valores efectivos divergentes entre configs: {divergentes}"
