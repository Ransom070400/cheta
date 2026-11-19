"""Greeter demo: when someone you have enrolled walks up, the robot turns toward them and
says "Welcome back <name>" plus one thing it remembers about them.

    # one-time setup (use --app greeter for everything)
    cheta --app greeter enroll --name Dave
    cheta --app greeter remember --name Dave "Dave is training for a marathon"

    python examples/greeter.py                  # laptop webcam, mic and speakers
    python examples/greeter.py --camera reachy  # Reachy Mini (pip install "cheta[robot]")

The same script runs in both modes; only the camera, the head and the speaker change.
Everything runs on the device. Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from cheta import Identity, Person
from cheta.speech import Speaker

COOLDOWN_S = 30.0  # do not greet the same person again within this time
DEFAULT_TOPIC = "what do they like or care about"


class Head(Protocol):
    """Anything that can look toward a pixel in the camera image."""

    def look_at(self, u: int, v: int) -> None: ...


class PrintHead:
    """Stand-in for the robot's head when running on a laptop."""

    def look_at(self, u: int, v: int) -> None:
        print(f"[head] turning toward pixel ({u}, {v})")


class Greeter:
    """Greets people as they arrive: look at them, then speak."""

    def __init__(
        self,
        idt: Identity,
        head: Head,
        speaker: Speaker,
        topic: str = DEFAULT_TOPIC,
        cooldown_s: float = COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.idt = idt
        self.head = head
        self.speaker = speaker
        self.topic = topic
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._last_greeted: dict[str, float] = {}
        self._busy = threading.Lock()
        idt.on_person_arrived(self.on_arrived)

    def line_for(self, person: Person) -> str:
        """What to say: a welcome plus the single most relevant remembered fact, if any."""
        facts = self.idt.recall(person, self.topic, k=1)
        line = f"Welcome back {person.name}."
        return f"{line} I remember: {facts[0].text}" if facts else line

    def greet(self, person: Person) -> None:
        """Turn toward the person, then speak. Blocks until finished."""
        if not self._busy.acquire(blocking=False):
            return  # already mid-greeting
        try:
            position = self.idt.face_position()
            if position is not None:
                self.head.look_at(*position)
            line = self.line_for(person)
            print(f"[greet] {line}")
            self.speaker.say(line)
        finally:
            self._busy.release()

    def on_arrived(self, person: Person) -> None:
        """Arrival callback: greet on a worker thread so recognition keeps running."""
        now = self._clock()
        last = self._last_greeted.get(person.id)
        if last is not None and now - last < self.cooldown_s:
            return
        self._last_greeted[person.id] = now
        threading.Thread(target=self.greet, args=(person,), daemon=True).start()


def build(args: argparse.Namespace) -> tuple[Identity, Head, Speaker, Any]:
    """Create the identity, head and speaker for the chosen camera. Returns the robot (or None)."""
    if args.camera == "reachy":
        from cheta.robot import ReachyHead, connect_robot
        from cheta.speech import robot_speaker

        mini = connect_robot()
        mini.wake_up()
        idt = Identity(
            app=args.app, camera="reachy", data_dir=args.data_dir, robot=mini, on_prompt=print
        )
        return idt, ReachyHead(mini), robot_speaker(mini), mini
    idt = Identity(app=args.app, camera="webcam", data_dir=args.data_dir, on_prompt=print)
    return idt, PrintHead(), Speaker(), None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--camera", choices=["webcam", "reachy"], default="webcam")
    parser.add_argument("--app", default="greeter")
    parser.add_argument("--data-dir", default="~/.cheta")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="what to recall about people")
    args = parser.parse_args(argv)

    idt, head, speaker, robot = build(args)
    Greeter(idt, head, speaker, topic=args.topic)
    idt.start()
    print("Watching for people you have enrolled. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        idt.close()
        if robot is not None:
            robot.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
