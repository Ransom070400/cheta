import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from cryptography.fernet import Fernet, InvalidToken

from cheta.storage import crypto
from cheta.storage.crypto import (
    decrypt_embedding,
    decrypt_text,
    encrypt_embedding,
    encrypt_text,
    load_or_create_key,
)
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


def test_text_round_trip_and_is_not_plaintext() -> None:
    key = Fernet.generate_key()
    token = encrypt_text(key, "Dave's dog is called Biscuit ☺")
    assert b"Biscuit" not in token
    assert decrypt_text(key, token) == "Dave's dog is called Biscuit ☺"


def test_text_wrong_key_fails() -> None:
    token = encrypt_text(Fernet.generate_key(), "secret")
    with pytest.raises(InvalidToken):
        decrypt_text(Fernet.generate_key(), token)


class HangingKeyring:
    """Stands in for the ``keyring`` module with calls that block until released."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def get_password(self, service: str, username: str) -> str | None:
        self.release.wait(30)
        return None

    def set_password(self, service: str, username: str, password: str) -> None:
        self.release.wait(30)


@pytest.fixture
def hung(monkeypatch: pytest.MonkeyPatch) -> Iterator[HangingKeyring]:
    fake = HangingKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fake)
    monkeypatch.setattr(crypto, "KEYRING_TIMEOUT_S", 0.2)
    yield fake
    fake.release.set()  # let the stuck daemon threads finish


def test_hung_keyring_on_fresh_install_falls_back_to_key_file(
    tmp_path: Path, hung: HangingKeyring
) -> None:
    started = time.monotonic()
    key = load_or_create_key(tmp_path, "app")
    assert time.monotonic() - started < 5  # did not wait for the 30 s block
    assert (tmp_path / "app.key").read_bytes() == key
    assert load_or_create_key(tmp_path, "app") == key  # stable on the next run


def test_hung_keyring_with_existing_database_refuses_to_invent_a_key(
    tmp_path: Path, hung: HangingKeyring
) -> None:
    (tmp_path / "app.db").write_bytes(b"existing data")
    with pytest.raises(RuntimeError, match="did not respond"):
        load_or_create_key(tmp_path, "app")
    assert not (tmp_path / "app.key").exists()


def test_hung_keyring_with_existing_key_file_uses_it(tmp_path: Path, hung: HangingKeyring) -> None:
    (tmp_path / "app.db").write_bytes(b"existing data")
    (tmp_path / "app.key").write_bytes(b"stored-key\n")
    assert load_or_create_key(tmp_path, "app") == b"stored-key"


def test_keyring_that_raises_still_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        def get_password(self, service: str, username: str) -> str | None:
            raise RuntimeError("no backend")

        def set_password(self, service: str, username: str, password: str) -> None:
            raise RuntimeError("no backend")

    monkeypatch.setitem(sys.modules, "keyring", Broken())
    key = load_or_create_key(tmp_path, "app")
    assert (tmp_path / "app.key").read_bytes() == key


def test_working_keyring_stores_the_key_without_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Memory:
        def __init__(self) -> None:
            self.data: dict[str, str] = {}

        def get_password(self, service: str, username: str) -> str | None:
            return self.data.get(username)

        def set_password(self, service: str, username: str, password: str) -> None:
            self.data[username] = password

    monkeypatch.setitem(sys.modules, "keyring", Memory())
    key = load_or_create_key(tmp_path, "app")
    assert not (tmp_path / "app.key").exists()
    assert load_or_create_key(tmp_path, "app") == key
