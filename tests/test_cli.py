from pathlib import Path

import pytest

from cheta import __version__
from cheta.cli import main


def test_cli_help_runs() -> None:
    assert main([]) == 0


def test_version_is_set() -> None:
    assert __version__


def test_enroll_cancelled_without_consent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    assert main(["--data-dir", str(tmp_path), "enroll", "--name", "Dave"]) == 1
    assert not list(tmp_path.glob("*.db"))


def test_list_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--data-dir", str(tmp_path), "list"]) == 0
    assert "No one enrolled" in capsys.readouterr().out
