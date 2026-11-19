"""M7: on arrival the robot turns toward the person and says Welcome back <name> + one fact."""

import importlib.util
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import cheta.identity as identity_mod
from cheta import Identity
from cheta.face.embedder import FaceObservation
from cheta.robot import ReachyHead
from cheta.storage import Store
from tests.conftest import TURNING, FakeSource, FakeTextEmbedder, rand_emb
from tests.test_robot import FakeMini

spec = importlib.util.spec_from_file_location(
    "greeter", Path(__file__).parent.parent / "examples" / "greeter.py"
)
assert spec and spec.loader
greeter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(greeter)


class TurningFace:
    """Dave's face at pixel (400, 200), turning his head so liveness passes."""

    def __init__(self) -> None:
        self.i = 0

    def observe(self, frame: Any) -> FaceObservation:
        yaw = TURNING[self.i % len(TURNING)]
        self.i += 1
        return FaceObservation(rand_emb(1), yaw, center=(400.0, 200.0))


class FakeSpeaker:
    def __init__(self, delay: float = 0.0) -> None:
        self.said: list[str] = []
        self.done = threading.Event()
        self.delay = delay

    def say(self, text: str) -> None:
        time.sleep(self.delay)
        self.said.append(text)
        self.done.set()


class RecordingHead:
    def __init__(self) -> None:
        self.looked: list[tuple[int, int]] = []

    def look_at(self, u: int, v: int) -> None:
        self.looked.append((u, v))


@pytest.fixture
def idt(store: Store, monkeypatch: pytest.MonkeyPatch) -> Identity:
    monkeypatch.setattr(identity_mod, "LOOP_INTERVAL_S", 0.005)
    dave = store.add_person("Dave", consent=True)
    store.add_face_embeddings(dave.id, [rand_emb(1)])
    ident = Identity(
        "greeter",
        store=store,
        face_embedder=TurningFace(),
        frame_source=FakeSource(),
        text_embedder=FakeTextEmbedder(),
    )
    ident.remember(dave, "Dave is training for a marathon")
    ident.remember(dave, "Dave's dog is called Biscuit")
    return ident


def test_robot_turns_toward_the_person_and_greets_with_a_fact(
    idt: Identity, monkeypatch: pytest.MonkeyPatch
) -> None:
    mini = FakeMini()
    speaker = FakeSpeaker()
    greeter.Greeter(idt, ReachyHead(mini), speaker, topic="marathon training")  # type: ignore[arg-type]
    idt.start()
    try:
        assert speaker.done.wait(5)
    finally:
        idt.stop()
    assert mini.looked == [(400, 200, 0.8)]  # the head turned toward the face
    assert speaker.said == ["Welcome back Dave. I remember: Dave is training for a marathon"]


def test_same_greeter_works_on_a_laptop_with_a_print_head(
    idt: Identity, capsys: pytest.CaptureFixture[str]
) -> None:
    speaker = FakeSpeaker()
    greeter.Greeter(idt, greeter.PrintHead(), speaker, topic="what is the dog called")
    idt.start()
    try:
        assert speaker.done.wait(5)
    finally:
        idt.stop()
    assert "[head] turning toward pixel (400, 200)" in capsys.readouterr().out
    assert speaker.said[0].startswith("Welcome back Dave. I remember: Dave's dog is called")


def test_person_without_memories_just_gets_a_welcome(store: Store) -> None:
    dave = store.add_person("Dave", consent=True)
    ident = Identity("g", store=store, text_embedder=FakeTextEmbedder(), liveness=False)
    g = greeter.Greeter(ident, RecordingHead(), FakeSpeaker())  # type: ignore[arg-type]
    assert g.line_for(dave) == "Welcome back Dave."


def test_does_not_regreet_within_the_cooldown(idt: Identity) -> None:
    now = [0.0]
    speaker = FakeSpeaker()
    g = greeter.Greeter(idt, RecordingHead(), speaker, cooldown_s=30, clock=lambda: now[0])
    dave = idt.store.list_people()[0]
    g.on_arrived(dave)
    assert speaker.done.wait(2)
    speaker.done.clear()
    now[0] = 10.0
    g.on_arrived(dave)
    assert not speaker.done.wait(0.3)  # too soon
    now[0] = 31.0
    g.on_arrived(dave)
    assert speaker.done.wait(2)
    assert len(speaker.said) == 2


def test_speaking_does_not_block_recognition(
    idt: Identity, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow greeting must not freeze the camera loop (that would fake a 'left' event)."""
    left = threading.Event()
    idt.on_person_left(lambda p: left.set())
    speaker = FakeSpeaker(delay=0.5)
    greeter.Greeter(idt, RecordingHead(), speaker)  # type: ignore[arg-type]
    idt.start()
    try:
        assert speaker.done.wait(5)
        assert not left.is_set()
        assert idt.who_is_here() is not None  # still tracking while/after speaking
    finally:
        idt.stop()


def test_unknown_camera_args_are_rejected() -> None:
    with pytest.raises(SystemExit):
        greeter.main(["--camera", "toaster"])
