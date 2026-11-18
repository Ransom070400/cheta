from cheta import __version__
from cheta.cli import main


def test_cli_runs(capsys: object) -> None:
    assert main([]) == 0


def test_version_is_set() -> None:
    assert __version__
