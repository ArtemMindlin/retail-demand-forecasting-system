"""The console helpers that give every run mode the same shape."""

from __future__ import annotations

import logging

import pytest

from retail_forecasting.utils.logging import _HEADER_EVERY_ROWS, Table, fields, thousands

COLUMNS = {"trial": 7, "estado": 8, "MAE": 6}


@pytest.fixture
def captured(request: pytest.FixtureRequest) -> tuple[logging.Logger, list[str]]:
    """A package logger and the lines it emits.

    Not `caplog`: the package root sets ``propagate = False`` on purpose, so records never
    reach the root handler pytest installs, and a test built on it would pass empty.
    """
    lines: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    logger = logging.getLogger(f"retail_forecasting.tests.{request.node.name}")
    handler = Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    request.addfinalizer(lambda: logger.removeHandler(handler))
    return logger, lines


def test_rows_line_up_under_their_headings(captured: tuple[logging.Logger, list[str]]) -> None:
    """The heading is what lets a row drop its labels, so the two must share their columns.

    No `header()` call: the first row opens the table. That step was separate and every caller
    made it immediately, so it was only a step to forget.
    """
    logger, lines = captured
    table = Table(logger, COLUMNS)

    table.row({"trial": "1/300", "estado": "completo", "MAE": "0.5157"})

    heading, divider, row = lines[1], lines[2], lines[3]
    assert len(heading) == len(row) == len(divider)
    assert heading.index("MAE") + len("MAE") == row.index("0.5157") + len("0.5157")


def test_a_row_stays_one_line_a_narrow_pane_can_hold() -> None:
    """Wrapping is the whole reason this is a table: a wrapped row runs into the next one."""
    columns = {
        "trial": 7,
        "estado": 8,
        "MAE": 6,
        "mejor": 6,
        "podados": 7,
        "s/trial": 7,
        "pasado": 6,
        "queda": 6,
    }
    width = 2 + sum(columns.values()) + 2 * (len(columns) - 1)

    assert width <= 80


def test_the_heading_comes_back_so_a_long_run_reads_as_blocks(
    captured: tuple[logging.Logger, list[str]],
) -> None:
    """Three hundred unbroken rows are unreadable, and the column names scroll out of reach."""
    logger, lines = captured
    table = Table(logger, COLUMNS)

    for trial in range(_HEADER_EVERY_ROWS + 1):
        table.row({"trial": f"{trial}/300", "estado": "completo", "MAE": "0.5157"})

    assert sum(1 for line in lines if set(line.strip()) == {"-"}) == 2


def test_a_missing_value_leaves_its_column_blank_instead_of_shifting_the_rest(
    captured: tuple[logging.Logger, list[str]],
) -> None:
    """A row that drops a field must not silently slide the next field into its column."""
    logger, lines = captured
    table = Table(logger, COLUMNS)

    table.row({"trial": "1/300", "MAE": "0.5157"})

    assert lines[1].index("MAE") + len("MAE") == lines[3].index("0.5157") + len("0.5157")


def test_setup_lines_share_one_column_whoever_writes_them(
    captured: tuple[logging.Logger, list[str]],
) -> None:
    """Hand-spaced setup lines drifted a character out of true; the width lives in one place."""
    logger, lines = captured

    fields(logger, {"panel": "500 series"})
    fields(logger, {"referencia": "defaults sin sintonizar"})

    assert lines[0].index("500") == lines[1].index("defaults")


def test_thousands_groups_digits_the_spanish_way() -> None:
    """`1,800` reads as one point eight to a Spanish reader."""
    assert thousands(1800) == "1.800"
    assert thousands(45000) == "45.000"
