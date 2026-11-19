import threading
import time

import pytest

from cheta import Identity
from cheta.fusion import SINGLE_MODALITY_FACTOR
from cheta.person import Person
from cheta.storage import Store
from tests.conftest import FakeSource, LiveEmbedder, rand_emb


def setup(store: Store) -> tuple[Identity, LiveEmbedder, Person]:
    p = store.add_person("Dave", consent=True)
    store.add_face_embeddings(p.id, [rand_emb(1)])
    emb = LiveEmbedder()
    idt = Identity(
        "test", store=store, face_embedder=emb, frame_source=FakeSource(), liveness=False
    )
    return idt, emb, p


def test_who_is_here_known(store: Store) -> None:
    idt, emb, p = setup(store)
    emb.current = rand_emb(1)
    who = idt.who_is_here()
    assert who is not None and who.id == p.id and who.name == "Dave"
    assert who.confidence == pytest.approx(SINGLE_MODALITY_FACTOR)  # perfect, but face only


def test_stranger_and_no_face_return_none(store: Store) -> None:
    idt, emb, _ = setup(store)
    emb.current = rand_emb(42)
    assert idt.who_is_here() is None
    emb.current = None
    assert idt.who_is_here() is None


def test_threshold_configurable(store: Store) -> None:
    idt, emb, _ = setup(store)
    idt.threshold = 1.01
    emb.current = rand_emb(1)
    assert idt.who_is_here() is None


def test_arrived_and_left_events(store: Store, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cheta.identity as mod

    monkeypatch.setattr(mod, "LOOP_INTERVAL_S", 0.01)
    idt, emb, _ = setup(store)
    idt._absence_seconds = 0.1
    arrived, left = threading.Event(), threading.Event()
    idt.on_person_arrived(lambda p: arrived.set())
    idt.on_person_left(lambda p: left.set())
    emb.current = rand_emb(1)
    idt.start()
    try:
        assert arrived.wait(3)
        who = idt.who_is_here()
        assert who is not None and who.name == "Dave"
        emb.current = None
        assert left.wait(3)
        assert idt.who_is_here() is None
    finally:
        idt.stop()


def test_callback_errors_do_not_kill_loop(store: Store, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cheta.identity as mod

    monkeypatch.setattr(mod, "LOOP_INTERVAL_S", 0.01)
    idt, emb, _ = setup(store)
    ok = threading.Event()

    def boom(p: Person) -> None:
        raise RuntimeError("x")

    idt.on_person_arrived(boom)
    idt.on_person_arrived(lambda p: ok.set())
    emb.current = rand_emb(1)
    idt.start()
    try:
        assert ok.wait(3)
    finally:
        idt.stop()


def test_face_position_tracks_the_recognised_face(store: Store) -> None:
    from cheta.face.embedder import FaceObservation

    class Positioned(LiveEmbedder):
        def observe(self, frame):  # type: ignore[no-untyped-def]
            return FaceObservation(self.current, 0.0, center=(320.4, 180.9))

    idt, emb, _ = setup(store)
    idt._embedder = Positioned()
    idt._embedder.current = rand_emb(1)  # type: ignore[attr-defined]
    assert idt.face_position() is None  # nothing seen yet
    assert idt.who_is_here() is not None
    assert idt.face_position() == (320, 180)
    idt._embedder.current = rand_emb(42)  # type: ignore[attr-defined]  # a stranger
    idt._face_center = None
    idt.who_is_here()
    assert idt.face_position() is None  # strangers are not tracked


def test_face_position_goes_stale(store: Store) -> None:
    idt, _, _ = setup(store)
    idt._face_center = ((10.0, 20.0), time.monotonic() - 5)
    assert idt.face_position() is None
    assert idt.face_position(max_age_s=10) == (10, 20)
