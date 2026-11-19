import pytest

import cheta.identity as identity_mod
from cheta import Identity
from cheta.identity import EnrollmentError
from tests.conftest import FakeEmbedder, FakeSource, rand_emb


def make(store, seq):  # type: ignore[no-untyped-def]
    return Identity("test", store=store, face_embedder=FakeEmbedder(seq), frame_source=FakeSource())


def test_refuses_without_consent(store) -> None:  # type: ignore[no-untyped-def]
    idt = make(store, [rand_emb(i) for i in range(5)])
    with pytest.raises(PermissionError):
        idt.enroll("Dave", consent=False)
    assert store.list_people() == []


def test_enroll_saves_five_samples_skipping_bad_frames(store) -> None:  # type: ignore[no-untyped-def]
    seq = [None, rand_emb(0), rand_emb(1), None, rand_emb(2), rand_emb(3), rand_emb(4)]
    idt = make(store, seq)
    prompts: list[str] = []
    person = idt.enroll(
        "Dave", consent=True, on_prompt=prompts.append, voice=False, _sleep=lambda _s: None
    )
    assert person.name == "Dave"
    assert len(store.face_embeddings(person.id)) == 5
    assert prompts[-1] == "Captured sample 5/5"


def test_enroll_times_out_without_storing(store, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(identity_mod, "ENROLL_TIMEOUT_S", 0.0)
    idt = make(store, [rand_emb(0)])
    with pytest.raises(EnrollmentError):
        idt.enroll("Dave", consent=True, _sleep=lambda _s: None)
    assert store.list_people() == []


def test_empty_name_rejected(store) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        make(store, []).enroll("  ", consent=True)
