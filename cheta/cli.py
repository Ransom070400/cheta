"""Command line interface: ``cheta enroll|list|forget|demo``."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from cheta.face.matcher import DEFAULT_THRESHOLD
from cheta.identity import DEFAULT_VOICE_THRESHOLD, Identity


def _identity(args: argparse.Namespace) -> Identity:
    return Identity(
        app=args.app,
        camera=args.camera,
        data_dir=args.data_dir,
        threshold=getattr(args, "threshold", DEFAULT_THRESHOLD),
        liveness=not getattr(args, "no_liveness", False),
        listen_voice=getattr(args, "voice", False),
        on_prompt=print,
    )


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
        person = idt.enroll(args.name, consent=True, on_prompt=print, voice=not args.no_voice)
    finally:
        idt.close()
    what = "face samples" if args.no_voice else "face and voice samples"
    print(f"Enrolled {person.name} ({person.id}) with encrypted {what}.")
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


def cmd_demo(args: argparse.Namespace) -> int:
    """Live recognition: print who is in view and arrival/left events until Ctrl-C."""
    idt = _identity(args)
    idt.on_person_arrived(lambda p: print(f"[arrived] {p.name} ({p.confidence:.2f})"))
    idt.on_person_left(lambda p: print(f"[left]    {p.name}"))
    idt.start()
    print("Watching the camera. Press Ctrl-C to stop.")
    last = ""
    try:
        while True:
            who = idt.who_is_here()
            line = f"{who.name} ({who.confidence:.2f})" if who else "nobody known"
            if line != last:
                print(f"here: {line}")
                last = line
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        idt.close()
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    """Hard-delete a person: face samples and memories."""
    idt = _identity(args)
    try:
        matches = idt.store.find_by_name(args.name) if args.name else []
        if args.id:
            person = idt.store.get_person(args.id)
            matches = [person] if person else []
        if not matches:
            print("No such person.")
            return 1
        if len(matches) > 1:
            print("Several people match; pass --id:")
            for p in matches:
                print(f"  {p.id}  {p.name}")
            return 1
        person = matches[0]
        if not args.yes:
            answer = input(f"Permanently delete {person.name} and all their data? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Cancelled.")
                return 1
        idt.forget(person)
    finally:
        idt.close()
    print(f"Forgot {person.name}.")
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    """Record a few seconds from the microphone and say who is speaking."""
    idt = Identity(
        app=args.app,
        camera=args.camera,
        data_dir=args.data_dir,
        voice_threshold=args.threshold,
    )
    try:
        print(f"Speak for {args.seconds:.0f} seconds...")
        who = idt.identify_voice(seconds=args.seconds)
    finally:
        idt.close()
    print(f"speaker: {who.name} ({who.confidence:.2f})" if who else "speaker: unknown")
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
    enroll.add_argument("--no-voice", action="store_true", help="enroll by face only")
    enroll.add_argument("--yes", action="store_true", help="confirm consent without prompting")
    enroll.set_defaults(func=cmd_enroll)

    demo = sub.add_parser("demo", help="live recognition from the camera")
    demo.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    demo.add_argument("--voice", action="store_true", help="also listen to the mic and fuse voice")
    demo.add_argument(
        "--no-liveness", action="store_true", help="skip the head-turn check (photos will pass!)"
    )
    demo.set_defaults(func=cmd_demo)

    forget = sub.add_parser("forget", help="delete a person and all their data")
    who = forget.add_mutually_exclusive_group(required=True)
    who.add_argument("--name")
    who.add_argument("--id")
    forget.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    forget.set_defaults(func=cmd_forget)

    listen = sub.add_parser("listen", help="identify the speaker from the microphone")
    listen.add_argument("--seconds", type=float, default=4.0)
    listen.add_argument("--threshold", type=float, default=DEFAULT_VOICE_THRESHOLD)
    listen.set_defaults(func=cmd_listen)

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
