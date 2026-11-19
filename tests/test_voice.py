import numpy as np
import pytest

from cheta import Identity
from cheta.identity import EnrollmentError
from cheta.storage import Store
from cheta.voice import SAMPLE_RATE, split_windows
from tests.conftest import (
    FakeAudio,
    FakeEmbedder,
    FakeSource,
    FakeVAD,
    FakeVoiceEmbedder,
    rand_emb,
)


def tone(value: float, seconds: float) -> np.ndarray:
    return np.full(int(SAMPLE_RATE * seconds), value, dtype=np.float32)


def make(store: Store, audio: FakeAudio) -> Identity:
    return Identity(
        "test",
        store=store,
        face_embedder=FakeEmbedder([rand_emb(i) for i in range(5)]),
        frame_source=FakeSource(),
        audio_source=audio,
        vad=FakeVAD(),
        voice_embedder=FakeVoiceEmbedder(),
    )


def test_split_windows() -> None:
    assert len(split_windows(tone(1, 2.0))) == 1
    assert len(split_windows(tone(1, 3.0))) == 1
    assert len(split_windows(tone(1, 5.0))) == 2  # 3 s windows, 1.5 s stride


def test_enroll_stores_voice_samples(store: Store) -> None:
    audio = FakeAudio(tone(0.07, 5.0))
    idt = make(store, audio)
    person = idt.enroll("Dave", consent=True, _sleep=lambda _s: None)
    assert audio.requested == [5.0]
    voices = store.voice_embeddings(person.id)
    assert len(voices) == 3  # two overlapping windows + the whole utterance
    assert idt.export(person)["voice_samples"] == 3


def test_enroll_retries_then_fails_when_silent(store: Store) -> None:
    audio = FakeAudio(tone(0.0, 5.0))
    idt = make(store, audio)
    with pytest.raises(EnrollmentError):
        idt.enroll("Dave", consent=True, _sleep=lambda _s: None)
    assert len(audio.requested) == 3
    assert store.list_people() == []  # nothing stored, face samples included


def test_identify_enrolled_speaker_from_three_seconds(store: Store) -> None:
    idt = make(store, FakeAudio(tone(0.07, 5.0)))
    person = idt.enroll("Dave", consent=True, _sleep=lambda _s: None)
    who = idt.identify_voice(tone(0.07, 3.0))
    assert who is not None and who.id == person.id and who.confidence > 0.99


def test_rejects_other_speaker(store: Store) -> None:
    idt = make(store, FakeAudio(tone(0.07, 5.0)))
    idt.enroll("Dave", consent=True, _sleep=lambda _s: None)
    assert idt.identify_voice(tone(0.42, 3.0)) is None


def test_too_little_speech_returns_none(store: Store) -> None:
    idt = make(store, FakeAudio(tone(0.07, 5.0)))
    idt.enroll("Dave", consent=True, _sleep=lambda _s: None)
    assert idt.identify_voice(tone(0.07, 1.0)) is None


def test_identify_records_from_mic_when_no_audio_given(store: Store) -> None:
    audio = FakeAudio(tone(0.07, 5.0))
    idt = make(store, audio)
    idt.enroll("Dave", consent=True, _sleep=lambda _s: None)
    assert idt.identify_voice() is not None
    assert audio.requested[-1] == 4.0


def test_forget_removes_voice_rows(store: Store) -> None:
    idt = make(store, FakeAudio(tone(0.07, 5.0)))
    person = idt.enroll("Dave", consent=True, _sleep=lambda _s: None)
    idt.forget(person)
    assert store._db.execute("SELECT COUNT(*) FROM voice_embedding").fetchone()[0] == 0
    assert idt.identify_voice(tone(0.07, 3.0)) is None
