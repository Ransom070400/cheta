import threading
import time
from itertools import cycle

import numpy as np
import pytest

import cheta.identity as identity_mod
from cheta import Identity
from cheta.face.embedder import head_yaw
from cheta.fusion import SINGLE_MODALITY_FACTOR, Evidence
from cheta.identity import EnrollmentError, LivenessError
from cheta.liveness import HeadTurnLiveness
from cheta.person import Person
from cheta.storage import Store
from tests.conftest import (
    PHOTO,
    TURNING,
    CycleEmbedder,
    FakeAudio,
    FakeSource,
    FakeVAD,
    FakeVoiceEmbedder,
    rand_emb,
)


def feed(live: HeadTurnLiveness, yaws: list[float], dt: float = 0.1) -> bool:
    result = False
    for i, y in enumerate(yaws):
        result = live.update(y, i * dt)
    return result


# -- the check itself --------------------------------------------------------------------------


def test_head_turn_passes() -> None:
    turn = [0.0, -0.1, -0.3, -0.35, -0.35, -0.3, -0.1, 0.1, 0.3, 0.35, 0.35, 0.3, 0.1, 0.0]
    assert feed(HeadTurnLiveness(), turn)


def test_photo_jitter_fails() -> None:
    assert not feed(HeadTurnLiveness(), list(PHOTO) * 20)


def test_flat_photo_tilted_by_hand_fails() -> None:
    # Spread seen when simulating a photo tilted +-45 degrees, rolled and shifted: 0.19 total.
    wobble = [-0.14, -0.1, -0.05, 0.0, 0.03, 0.045, 0.0, -0.05, -0.1, -0.14] * 3
    assert not feed(HeadTurnLiveness(), wobble)


def test_single_noisy_frame_cannot_pass() -> None:
    assert not feed(HeadTurnLiveness(), [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert not feed(HeadTurnLiveness(), [0.0, 0.9, -0.9, 0.0, 0.9, -0.9, 0.0, 0.0, 0.0])


def test_turn_must_happen_within_window() -> None:
    live = HeadTurnLiveness(window_s=1.0)
    slow = [i * 0.05 for i in range(10)]  # drifts 0.45 total, but over 4.5 s at dt=0.5
    assert not feed(live, slow, dt=0.5)


def test_passed_latches_and_reset_clears() -> None:
    live = HeadTurnLiveness()
    assert feed(live, list(TURNING) * 2)
    assert live.update(0.0, 99.0)
    live.reset()
    assert not live.passed


def test_head_yaw_from_landmarks() -> None:
    def face(nose_x: float) -> np.ndarray:
        row = np.zeros(15, dtype=np.float32)
        row[4:14] = [40, 50, 80, 50, nose_x, 70, 45, 90, 75, 90]  # eyes, nose, mouth corners
        return row

    assert head_yaw(face(60)) == pytest.approx(0.0)  # frontal: nose centred
    assert head_yaw(face(70)) > 0.2 and head_yaw(face(50)) < -0.2  # turned either way


# -- enrollment --------------------------------------------------------------------------------


def make(store: Store, yaws: tuple[float, ...], emb=None, **kw):  # type: ignore[no-untyped-def]
    return Identity(
        "test",
        store=store,
        face_embedder=CycleEmbedder(emb if emb is not None else rand_emb(1), yaws),
        frame_source=FakeSource(),
        **kw,
    )


def test_enroll_requires_head_turn(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity_mod, "ENROLL_TIMEOUT_S", 0.3)
    prompts: list[str] = []
    idt = make(store, PHOTO)
    with pytest.raises(LivenessError):
        idt.enroll(
            "Dave", consent=True, voice=False, on_prompt=prompts.append, _sleep=lambda s: None
        )
    assert store.list_people() == []  # a photo enrolls nothing
    assert any("turn your head" in p for p in prompts)
    assert issubclass(LivenessError, EnrollmentError)


def test_enroll_succeeds_with_head_turn(store: Store) -> None:
    idt = make(store, TURNING)
    person = idt.enroll("Dave", consent=True, voice=False, _sleep=lambda s: None)
    assert len(store.face_embeddings(person.id)) == 5


def test_enroll_ignores_turn_by_a_different_person(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Swap:
        """5 samples of Dave holding still, then someone else turns their head."""

        def __init__(self) -> None:
            self.n = 0
            self.turn = cycle(TURNING)

        def observe(self, frame):  # type: ignore[no-untyped-def]
            from cheta.face.embedder import FaceObservation

            self.n += 1
            if self.n <= 5:
                return FaceObservation(rand_emb(1), 0.0)
            return FaceObservation(rand_emb(99), next(self.turn))

    monkeypatch.setattr(identity_mod, "ENROLL_TIMEOUT_S", 0.3)
    idt = Identity("test", store=store, face_embedder=Swap(), frame_source=FakeSource())
    with pytest.raises(LivenessError):
        idt.enroll("Dave", consent=True, voice=False, _sleep=lambda s: None)
    assert store.list_people() == []


# -- recognition -------------------------------------------------------------------------------


def enrolled(store: Store) -> Person:
    p = store.add_person("Dave", consent=True)
    store.add_face_embeddings(p.id, [rand_emb(1)])
    return p


def test_photo_is_not_recognised_one_shot(store: Store) -> None:
    enrolled(store)
    idt = make(store, PHOTO)
    idt._live = HeadTurnLiveness(window_s=0.3)
    assert idt.who_is_here() is None


def test_live_person_is_recognised_one_shot(store: Store) -> None:
    p = enrolled(store)
    who = make(store, TURNING).who_is_here()
    assert who is not None and who.id == p.id


def test_photo_never_arrives_in_background_loop(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(identity_mod, "LOOP_INTERVAL_S", 0.005)
    enrolled(store)
    idt = make(store, PHOTO)
    arrived = threading.Event()
    idt.on_person_arrived(lambda p: arrived.set())
    idt.start()
    try:
        assert not arrived.wait(0.6)
        assert idt.who_is_here() is None
    finally:
        idt.stop()


def test_live_person_arrives_in_background_loop(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(identity_mod, "LOOP_INTERVAL_S", 0.005)
    enrolled(store)
    idt = make(store, TURNING)
    arrived = threading.Event()
    idt.on_person_arrived(lambda p: arrived.set())
    idt.start()
    try:
        assert arrived.wait(3)
    finally:
        idt.stop()


def test_liveness_can_be_disabled(store: Store) -> None:
    p = enrolled(store)
    who = make(store, PHOTO, liveness=False).who_is_here()
    assert who is not None and who.id == p.id


def test_verification_expires_after_absence(store: Store) -> None:
    p = enrolled(store)
    idt = make(store, TURNING, absence_seconds=1.0)
    obs = idt._observe()
    assert idt._decide(obs, 0.0) is None
    for i in range(1, 30):  # the person turns their head
        idt._decide(idt._observe(), i * 0.03)
    assert idt._verified_id == p.id
    idt._decide(None, 5.0)  # gone for longer than absence_seconds
    assert idt._verified_id is None  # a returning photo would have to pass liveness again


def test_prompts_user_to_turn(store: Store) -> None:
    enrolled(store)
    prompts: list[str] = []
    idt = make(store, PHOTO, on_prompt=prompts.append)
    idt._decide(idt._observe(), 0.0)
    assert any("turn your head" in p for p in prompts)


# -- fusion inside Identity --------------------------------------------------------------------


def with_voice(store: Store, person_voice: Person | None = None) -> tuple[Identity, Person]:
    p = enrolled(store)
    idt = make(
        store,
        TURNING,
        absence_seconds=1000.0,  # keep liveness verified across the long gaps these tests use
        voice_embedder=FakeVoiceEmbedder(),
        vad=FakeVAD(),
    )
    owner = person_voice or p
    store.add_voice_embeddings(
        owner.id, [FakeVoiceEmbedder().embed(np.full(16_000, 0.07, np.float32))]
    )
    return idt, p


def warm_up(idt: Identity) -> None:
    for i in range(30):
        idt._decide(idt._observe(), i * 0.03)


def test_face_and_voice_agree_gives_higher_confidence(store: Store) -> None:
    idt, p = with_voice(store)
    warm_up(idt)
    face_only = idt._decide(idt._observe(), 1.0)
    idt._voice_evidence = (Evidence(p.id, 1.0, idt.voice_threshold), 1.0)
    both = idt._decide(idt._observe(), 1.1)
    assert face_only is not None and both is not None
    assert face_only.confidence == pytest.approx(SINGLE_MODALITY_FACTOR)
    assert both.confidence == pytest.approx(1.0) and both.confidence > face_only.confidence


def test_face_and_voice_conflict_gives_nobody(store: Store) -> None:
    idt, p = with_voice(store)
    other = store.add_person("Ann", consent=True)
    warm_up(idt)
    idt._voice_evidence = (Evidence(other.id, 0.9, idt.voice_threshold), 1.0)
    assert idt._decide(idt._observe(), 1.1) is None


def test_stale_voice_is_ignored(store: Store) -> None:
    idt, p = with_voice(store)
    warm_up(idt)
    idt._voice_evidence = (Evidence(p.id, 1.0, idt.voice_threshold), 0.0)
    who = idt._decide(idt._observe(), 100.0)  # voice evidence is 100 s old
    assert who is not None and who.confidence == pytest.approx(SINGLE_MODALITY_FACTOR)


def test_voice_alone_cannot_start_a_presence(store: Store) -> None:
    p = enrolled(store)
    idt = make(store, PHOTO, voice_embedder=FakeVoiceEmbedder(), vad=FakeVAD())  # face never live
    idt._voice_evidence = (Evidence(p.id, 1.0, idt.voice_threshold), 1.0)
    assert idt._decide(idt._observe(), 1.1) is None  # a recording of their voice is not enough


def test_voice_keeps_a_present_person_present(store: Store) -> None:
    idt, p = with_voice(store)
    warm_up(idt)
    idt._tracker.update(Person(p.id, "Dave", 0.8), 1.0)
    idt._tracker.update(Person(p.id, "Dave", 0.8), 1.1)
    assert idt._tracker.current is not None
    idt._voice_evidence = (Evidence(p.id, 0.9, idt.voice_threshold), 2.0)
    who = idt._decide(None, 2.0)  # face turned away, but they are talking
    assert who is not None and who.id == p.id and who.confidence < 0.8 + 1e-9


def test_listener_thread_feeds_fusion(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity_mod, "LOOP_INTERVAL_S", 0.005)
    p = enrolled(store)
    store.add_voice_embeddings(p.id, [FakeVoiceEmbedder().embed(np.full(16_000, 0.07, np.float32))])
    idt = Identity(
        "test",
        store=store,
        face_embedder=CycleEmbedder(rand_emb(1), TURNING),
        frame_source=FakeSource(),
        audio_source=FakeAudio(np.full(16_000 * 5, 0.07, dtype=np.float32)),
        vad=FakeVAD(),
        voice_embedder=FakeVoiceEmbedder(),
        listen_voice=True,
    )
    idt.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            who = idt.who_is_here()
            if who is not None and who.confidence > 0.99:
                break
            time.sleep(0.02)
        assert who is not None and who.confidence > 0.99  # face + voice together
    finally:
        idt.stop()
