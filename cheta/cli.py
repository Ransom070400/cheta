"""Command line interface: ``cheta enroll|list|forget|demo``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from cheta.identity import Identity


def _identity(args: argparse.Namespace) -> Identity:
    return Identity(app=args.app, camera=args.camera, data_dir=args.data_dir)


def cmd_enroll(args: argparse.Namespace) -> int:
    """Enroll a person from the camera."""
    if not args.yes:
        answer = input(
            f"Store a face template for {args.name!r} on this device? "
            "Raw images are never saved. [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled: consent not given.")
            return 1
    idt = _identity(args)
    try:
        person = idt.enroll(args.name, consent=True, on_prompt=print)
    finally:
        idt.close()
    print(f"Enrolled {person.name} ({person.id}) with 5 encrypted face samples.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List enrolled people."""
    idt = _identity(args)
    try:
        people = idt.store.list_people()
    finally:
        idt.close()
    for p in people:
        print(f"{p.id}  {p.name}")
    if not people:
        print("No one enrolled.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="cheta", description=__doc__)
    parser.add_argument("--app", default="cheta", help="app namespace (default: cheta)")
    parser.add_argument("--data-dir", default="~/.cheta")
    parser.add_argument("--camera", choices=["webcam", "reachy"], default="webcam")
    sub = parser.add_subparsers(dest="command")

    enroll = sub.add_parser("enroll", help="enroll a person from the camera")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--yes", action="store_true", help="confirm consent without prompting")
    enroll.set_defaults(func=cmd_enroll)

    lst = sub.add_parser("list", help="list enrolled people")
    lst.set_defaults(func=cmd_list)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))
