"""Command line interface: ``cheta enroll|list|forget|demo``."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from cheta.calibrate import load_voice_dir, recommend_threshold, score_voices
from cheta.face.matcher import DEFAULT_THRESHOLD
from cheta.identity import DEFAULT_VOICE_THRESHOLD, Identity
from cheta.person import Person
from cheta.voice import EcapaEmbedder, SileroVAD


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


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Recommend a voice threshold from recordings in a folder (nothing is enrolled or stored)."""
    try:
        enrol, probes = load_voice_dir(Path(args.dir))
        genuine, impostor = score_voices(enrol, probes, SileroVAD(), EcapaEmbedder())
        rec = recommend_threshold(genuine, impostor, max_far=args.max_far)
    except ValueError as exc:
        print(f"Cannot calibrate: {exc}")
        return 1
    print(f"{len(enrol)} people, {rec.n_genuine} genuine and {rec.n_impostor} impostor scores.")
    print(f"genuine  : median {sorted(genuine)[len(genuine) // 2]:.2f}, lowest {min(genuine):.2f}")
    print(
        f"impostor : median {sorted(impostor)[len(impostor) // 2]:.2f}, highest {max(impostor):.2f}"
    )
    print(
        f"Recommended voice_threshold = {rec.threshold:.3f}  "
        f"(false accepts {rec.false_accept_rate:.1%}, false rejects {rec.false_reject_rate:.1%})"
    )
    if rec.n_impostor < 100:
        print("Note: few impostor scores; the false-accept estimate is coarse. Use more clips.")
    return 0


def _one_person(idt: Identity, name: str) -> Person | None:
    matches = idt.store.find_by_name(name)
    if not matches:
        print(f"No one called {name!r} is enrolled.")
        return None
    if len(matches) > 1:
        print(f"Several people are called {name!r}; use `cheta list` to tell them apart.")
        return None
    return matches[0]


def cmd_remember(args: argparse.Namespace) -> int:
    """Store a fact about an enrolled person."""
    idt = _identity(args)
    try:
        person = _one_person(idt, args.name)
        if person is None:
            return 1
        idt.remember(person, args.text, expires_in_days=args.expires_days)
    finally:
        idt.close()
    print(f"Remembered for {person.name}: {args.text}")
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    """Look up what is remembered about a person."""
    idt = _identity(args)
    try:
        person = _one_person(idt, args.name)
        if person is None:
            return 1
        memories = idt.recall(person, args.query, k=args.k)
    finally:
        idt.close()
    for m in memories:
        print(f"{m.score:.2f}  {m.text}")
    if not memories:
        print("Nothing remembered.")
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

    calibrate = sub.add_parser(
        "calibrate", help="recommend a voice threshold from recordings (<name>_<n>.wav)"
    )
    calibrate.add_argument("dir", help="folder of <name>_1.wav (enrol), <name>_2.wav ... (tests)")
    calibrate.add_argument("--max-far", type=float, default=0.01, help="false-accept target")
    calibrate.set_defaults(func=cmd_calibrate)

    remember = sub.add_parser("remember", help="store a fact about an enrolled person")
    remember.add_argument("--name", required=True)
    remember.add_argument("text")
    remember.add_argument("--expires-days", type=int, default=None)
    remember.set_defaults(func=cmd_remember)

    recall = sub.add_parser("recall", help="look up facts about an enrolled person")
    recall.add_argument("--name", required=True)
    recall.add_argument("query")
    recall.add_argument("-k", type=int, default=5)
    recall.set_defaults(func=cmd_recall)

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
