"""Command line interface: ``cheta enroll|list|forget|demo`` (implemented in later milestones)."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(prog="cheta", description=__doc__)
    parser.add_subparsers(dest="command")
    parser.parse_args(argv)
    parser.print_help()
    return 0
