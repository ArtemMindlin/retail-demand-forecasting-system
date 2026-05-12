from __future__ import annotations

import sys
from pathlib import Path

import pytest

from retail_forecasting.run import main


def test_main_reports_configuration_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invalid_config_path = tmp_path / "invalid_config.yaml"
    invalid_config_path.write_text(
        """
models:
  seasonal_period: -1
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retail-forecast",
            "--config",
            str(invalid_config_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert str(exc_info.value) == (
        "Invalid configuration:\n- models.seasonal_period must be greater than 0."
    )
