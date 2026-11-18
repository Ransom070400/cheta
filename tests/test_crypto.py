from pathlib import Path

import numpy as np
import pytest
from cryptography.fernet import Fernet, InvalidToken

from cheta.storage.crypto import decrypt_embedding, encrypt_embedding, load_or_create_key
from tests.conftest import rand_emb


def test_roundtrip(key: bytes) -> None:
    e = rand_emb(1)
    assert np.array_equal(decrypt_embedding(key, encrypt_embedding(key, e)), e)


def test_wrong_key_fails(key: bytes) -> None:
    token = encrypt_embedding(key, rand_emb(1))
    with pytest.raises(InvalidToken):
        decrypt_embedding(Fernet.generate_key(), token)


def test_key_file_is_0600_and_stable(tmp_path: Path) -> None:
    k1 = load_or_create_key(tmp_path, "app", use_keyring=False)
    k2 = load_or_create_key(tmp_path, "app", use_keyring=False)
    assert k1 == k2
    assert (tmp_path / "app.key").stat().st_mode & 0o777 == 0o600
