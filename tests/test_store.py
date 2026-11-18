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
