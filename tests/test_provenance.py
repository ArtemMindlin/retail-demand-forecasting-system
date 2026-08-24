from __future__ import annotations

import pytest

from retail_forecasting.utils import provenance


@pytest.fixture(autouse=True)
def _unfrozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record the unfrozen state so a test that freezes cannot leak into the next one."""
    monkeypatch.setattr(provenance, "_frozen_commit", None)


def test_an_unfrozen_process_reads_head_every_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback importers rely on: no freeze, no pinning."""
    seen = iter(["aaaaaaa", "bbbbbbb"])
    monkeypatch.setattr(provenance, "_read_git_commit", lambda: next(seen))

    assert provenance.get_git_commit() == "aaaaaaa"
    assert provenance.get_git_commit() == "bbbbbbb"


def test_freezing_pins_the_commit_against_a_later_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: a commit landing mid-run must not rewrite what the run reports."""
    monkeypatch.setattr(provenance, "_read_git_commit", lambda: "aaaaaaa")
    assert provenance.freeze_git_commit() == "aaaaaaa"

    monkeypatch.setattr(provenance, "_read_git_commit", lambda: "bbbbbbb")
    assert provenance.get_git_commit() == "aaaaaaa"


def test_freezing_outside_a_checkout_stays_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frozen-to-None must not read as never-frozen, or the pin silently lifts."""
    monkeypatch.setattr(provenance, "_read_git_commit", lambda: None)
    assert provenance.freeze_git_commit() is None

    monkeypatch.setattr(provenance, "_read_git_commit", lambda: "bbbbbbb")
    assert provenance.get_git_commit() is None


def test_a_clean_worktree_stamps_the_bare_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provenance,
        "_git",
        lambda *args: "aaaaaaa" if args[0] == "rev-parse" else None,
    )
    assert provenance._read_git_commit() == "aaaaaaa"


def test_a_modified_worktree_is_marked_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare hash over uncommitted edits describes code that never ran."""
    monkeypatch.setattr(
        provenance,
        "_git",
        lambda *args: "aaaaaaa" if args[0] == "rev-parse" else " M src/thing.py",
    )
    assert provenance._read_git_commit() == "aaaaaaa-dirty"


def test_no_head_reports_nothing_rather_than_a_dirty_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "_git", lambda *args: None)
    assert provenance._read_git_commit() is None
