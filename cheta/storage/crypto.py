"""Encryption key management and Fernet helpers for biometric embeddings.

The key lives in the OS keyring when one is available, otherwise in a ``0600`` key file inside
the data directory. Embeddings are stored only as Fernet tokens.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet
from numpy.typing import NDArray

KEYRING_SERVICE = "cheta"


def _keyring_get(username: str) -> str | None:
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, username)
    except Exception:  # no usable backend (headless Linux, CI, ...)
        return None


def _keyring_set(username: str, value: str) -> bool:
    try:
        import keyring

        keyring.set_password(KEYRING_SERVICE, username, value)
        return keyring.get_password(KEYRING_SERVICE, username) == value
    except Exception:
        return False


def load_or_create_key(data_dir: Path, app: str, use_keyring: bool = True) -> bytes:
    """Return the Fernet key for ``app``, creating and persisting it on first use."""
    username = f"{app}:{data_dir.resolve()}"
    key_file = data_dir / f"{app}.key"

    if use_keyring:
        existing = _keyring_get(username)
        if existing:
            return existing.encode()

    if key_file.exists():
        return key_file.read_bytes().strip()

    key = Fernet.generate_key()
    if use_keyring and _keyring_set(username, key.decode()):
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
    return np.frombuffer(Fernet(key).decrypt(token), dtype=np.float32).copy()
