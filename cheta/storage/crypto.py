"""Encryption key management and Fernet helpers for biometric embeddings.

The key lives in the OS keyring when one is available, otherwise in a ``0600`` key file inside
the data directory. Embeddings and memory text are stored only as Fernet tokens.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
from cryptography.fernet import Fernet
from numpy.typing import NDArray

T = TypeVar("T")

KEYRING_SERVICE = "cheta"
KEYRING_TIMEOUT_S = 5.0  # a locked or prompt-blocked keychain can hang instead of raising


class KeyringTimeout(Exception):
    """The OS keyring did not answer in time."""


def _with_timeout(fn: Callable[[], T], timeout: float) -> T:
    """Run ``fn`` in a daemon thread; raise ``KeyringTimeout`` if it has not finished in time.

    A blocked call cannot be killed, so it is left behind on its daemon thread.
    """
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # re-raised in the caller
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True, name="cheta-keyring")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise KeyringTimeout(f"keyring did not answer within {timeout:g}s")
    if "error" in box:
        raise box["error"]
    return cast(T, box["value"])


def _keyring_get(username: str) -> str | None:
    """Return the stored key, or None if there is none or no usable backend.

    Raises ``KeyringTimeout`` if the keyring hangs.
    """

    def get() -> str | None:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, username)

    try:
        return _with_timeout(get, KEYRING_TIMEOUT_S)
    except KeyringTimeout:
        raise
    except Exception:  # no usable backend (headless Linux, CI, ...)
        return None


def _keyring_set(username: str, value: str) -> bool:
    def put() -> bool:
        import keyring

        keyring.set_password(KEYRING_SERVICE, username, value)
        return keyring.get_password(KEYRING_SERVICE, username) == value

    try:
        return _with_timeout(put, KEYRING_TIMEOUT_S)
    except Exception:  # includes KeyringTimeout: fall back to the key file
        return False


def load_or_create_key(data_dir: Path, app: str, use_keyring: bool = True) -> bytes:
    """Return the Fernet key for ``app``, creating and persisting it on first use.

    If the keyring hangs, the key file is used. A *new* key is never invented for a database that
    already exists, since the real key may be sitting in the unresponsive keyring.
    """
    username = f"{app}:{data_dir.resolve()}"
    key_file = data_dir / f"{app}.key"
    keyring_hung = False

    if use_keyring:
        try:
            existing = _keyring_get(username)
        except KeyringTimeout:
            existing, keyring_hung = None, True
        if existing:
            return existing.encode()

    if key_file.exists():
        return key_file.read_bytes().strip()

    if keyring_hung and (data_dir / f"{app}.db").exists():
        raise RuntimeError(
            f"The OS keyring did not respond and there is no key file ({key_file}) for the "
            "existing database. Unlock or fix the keyring and try again; cheta will not create "
            "a new key, because that would make the stored data unreadable."
        )

    key = Fernet.generate_key()
    if use_keyring and not keyring_hung and _keyring_set(username, key.decode()):
        return key

    data_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)
    return key


def encrypt_embedding(key: bytes, embedding: NDArray[np.float32]) -> bytes:
    """Encrypt a float32 embedding vector into a Fernet token."""
    return Fernet(key).encrypt(np.asarray(embedding, dtype=np.float32).tobytes())


def decrypt_embedding(key: bytes, token: bytes) -> NDArray[np.float32]:
    """Decrypt a Fernet token back into a float32 embedding vector.

    Raises ``cryptography.fernet.InvalidToken`` if the key is wrong or the data was tampered with.
    """
    arr: NDArray[np.float32] = np.frombuffer(Fernet(key).decrypt(token), dtype=np.float32).copy()
    return arr


def encrypt_text(key: bytes, text: str) -> bytes:
    """Encrypt a UTF-8 string into a Fernet token."""
    return Fernet(key).encrypt(text.encode("utf-8"))


def decrypt_text(key: bytes, token: bytes) -> str:
    """Decrypt a Fernet token back into a string.

    Raises ``cryptography.fernet.InvalidToken`` if the key is wrong or the data was tampered with.
    """
    return Fernet(key).decrypt(token).decode("utf-8")
