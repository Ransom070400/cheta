from pathlib import Path

import numpy as np
import pytest
from cryptography.fernet import Fernet, InvalidToken

from cheta.storage import Store
from tests.conftest import rand_emb


def test_consent_required(store: Store) -> None:
    with pytest.raises(PermissionError):
        store.add_person("Dave", consent=False)
    assert store.list_people() == []


def test_embeddings_roundtrip(store: Store) -> None:
    p = store.add_person("Dave", consent=True)
    embs = [rand_emb(i) for i in range(5)]
    store.add_face_embeddings(p.id, embs)
    got = store.face_embeddings(p.id)
    assert len(got) == 5 and all(np.array_equal(a, b) for a, b in zip(got, embs, strict=True))


def test_db_unreadable_without_key(tmp_path: Path, key: bytes) -> None:
    s = Store(tmp_path, "test", key=key)
    p = s.add_person("Dave", consent=True)
    e = rand_emb(7)
    s.add_face_embeddings(p.id, [e])
    s.close()

    # Raw bytes on disk never contain the plaintext embedding.
    assert e.tobytes() not in s.path.read_bytes()

    # Opening with the wrong key cannot decrypt.
    other = Store(tmp_path, "test", key=Fernet.generate_key())
    with pytest.raises(InvalidToken):
        other.face_embeddings(p.id)


def test_delete_leaves_no_rows(store: Store) -> None:
    p = store.add_person("Dave", consent=True)
    store.add_face_embeddings(p.id, [rand_emb(1)])
    store.delete_person(p.id)
    assert store.list_people() == [] and store.all_face_embeddings() == {}
    assert store._db.execute("SELECT COUNT(*) FROM face_embedding").fetchone()[0] == 0


def test_db_file_is_0600(store: Store) -> None:
    assert store.path.stat().st_mode & 0o777 == 0o600


def _person(store: Store) -> str:
    return store.add_person("Dave", consent=True).id


def test_memory_embedding_is_encrypted_at_rest(store: Store) -> None:
    pid = _person(store)
    emb = rand_emb(1, 384)
    store.add_memory(pid, "a fact", emb)
    raw = store._db.execute("SELECT embedding FROM memory").fetchone()[0]
    assert raw != emb.tobytes() and emb.tobytes() not in raw
    assert raw.startswith(b"gAAAA")  # a Fernet token
    [hit] = store.search_memories(pid, emb, k=1)
    assert hit.text == "a fact" and hit.score == pytest.approx(1.0, abs=1e-5)


def test_search_ranks_by_similarity(store: Store) -> None:
    pid = _person(store)
    near, mid, far = rand_emb(1, 384), rand_emb(2, 384), rand_emb(3, 384)
    query = near + 0.1 * mid
    for text, emb in (("far", far), ("near", near), ("mid", mid)):
        store.add_memory(pid, text, emb)
    assert [m.text for m in store.search_memories(pid, query, k=3)][0] == "near"
    scores = [m.score for m in store.search_memories(pid, query, k=3)]
    assert scores == sorted(scores, reverse=True)
    assert len(store.search_memories(pid, query, k=2)) == 2


def test_search_ties_prefer_the_newest(store: Store) -> None:
    pid = _person(store)
    emb = rand_emb(1, 384)
    store.add_memory(pid, "old", emb)
    store.add_memory(pid, "new", emb)
    assert [m.text for m in store.search_memories(pid, emb, k=2)] == ["new", "old"]


def test_search_is_scoped_to_one_person(store: Store) -> None:
    a, b = _person(store), store.add_person("Eve", consent=True).id
    emb = rand_emb(1, 384)
    store.add_memory(a, "dave fact", emb)
    store.add_memory(b, "eve fact", emb)
    assert [m.text for m in store.search_memories(a, emb, k=5)] == ["dave fact"]


def test_search_skips_expired_and_handles_empty_or_zero_k(store: Store) -> None:
    pid = _person(store)
    emb = rand_emb(1, 384)
    assert store.search_memories(pid, emb, k=3) == []
    store.add_memory(pid, "gone", emb, expires_at=10.0)
    store.add_memory(pid, "kept", emb)
    assert [m.text for m in store.search_memories(pid, emb, k=5, now=100.0)] == ["kept"]
    assert store.search_memories(pid, emb, k=0) == []


def test_legacy_unencrypted_memory_embedding_still_searchable(store: Store) -> None:
    pid = _person(store)
    emb = rand_emb(1, 384)
    store._db.execute(
        "INSERT INTO memory (person_id, text, embedding, created_at) VALUES (?, ?, ?, 1.0)",
        (pid, "legacy", emb.tobytes()),
    )
    assert [m.text for m in store.search_memories(pid, emb, k=1)] == ["legacy"]


def test_memory_embedding_wrong_key_fails(tmp_path: Path, key: bytes) -> None:
    s1 = Store(tmp_path, "app", key=key)
    pid = s1.add_person("Dave", consent=True).id
    s1.add_memory(pid, "x", rand_emb(1, 384))
    s1.close()
    s2 = Store(tmp_path, "app", key=Fernet.generate_key())
    with pytest.raises(InvalidToken):
        s2.search_memories(pid, rand_emb(1, 384), k=1)
