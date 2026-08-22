"""Console logging for the run modes, with one style for all of them.

The package configures a logger root of its OWN (`retail_forecasting`) with
``propagate = False``, rather than touching the process-wide root logger. That is the
pattern Optuna uses, and it is the one that fits a codebase that is a library and an
application at the same time: importing anything from `retail_forecasting` must not
reconfigure logging for Django, for the test suite or for a notebook, yet
`python -m retail_forecasting.run` has to print something readable without ceremony.

Library code only ever calls ``get_logger(__name__)`` and emits. ``configure()`` is called
from the CLI entry point and nowhere else.

The three helpers below exist so the shape of a run is the same whichever mode produced
it: a rule with the title, aligned ``key   value`` lines for the setup, and a progress line
whose columns line up from one call to the next.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping

_LIBRARY_ROOT = "retail_forecasting"
_RULE_WIDTH = 74
_FIELD_WIDTH = 16

_handler: logging.Handler | None = None


class _PlainFormatter(logging.Formatter):
    """Message-only for INFO, level-prefixed above it.

    An experiment log is read as a narrative, so the timestamp and logger name that a
    server log needs are noise here. Warnings and errors do get marked, because those are
    the lines someone scrolls back to find.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno <= logging.INFO:
            return message
        return f"{record.levelname.lower()}: {message}"


def configure(level: int = logging.INFO) -> None:
    """Attach the console handler to the package logger. Called from the CLI only."""
    global _handler

    root = logging.getLogger(_LIBRARY_ROOT)
    if _handler is None:
        _handler = logging.StreamHandler(stream=sys.stderr)
        _handler.setFormatter(_PlainFormatter())
        root.addHandler(_handler)
        # Never hand these records to the process-wide root: whoever imports us owns that.
        root.propagate = False
    root.setLevel(level)


def set_verbosity(level: int) -> None:
    """Raise or lower the package's log level, mirroring `optuna.logging.set_verbosity`."""
    logging.getLogger(_LIBRARY_ROOT).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """The logger a module should emit through."""
    return logging.getLogger(name)


def thousands(value: float) -> str:
    """Group digits the Spanish way, so ``1,800`` does not read as one point eight."""
    return f"{value:,.0f}".replace(",", ".")


def rule(logger: logging.Logger, title: str) -> None:
    """Open a section: a horizontal rule with the title above it."""
    logger.info("")
    logger.info(title.upper())
    logger.info("=" * _RULE_WIDTH)


def fields(logger: logging.Logger, values: Mapping[str, object]) -> None:
    """The setup of a run, one aligned line per entry, so runs can be diffed by eye."""
    for key, value in values.items():
        logger.info("  %-*s %s", _FIELD_WIDTH, key, value)


def progress(logger: logging.Logger, values: Mapping[str, object]) -> None:
    """One progress line, columns aligned across calls.

    Deliberately a new line rather than a carriage return over the previous one: with
    `logging` every call is its own record, and a long run is usually read afterwards in a
    redirected file, where overwriting leaves the carriage returns behind.
    """
    logger.info("  " + "   ".join(f"{key} {value}" for key, value in values.items()))
