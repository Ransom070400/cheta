from pathlib import Path

import numpy as np
import pytest
from cryptography.fernet import Fernet
from numpy.typing import NDArray

from cheta.face.embedder import Frame
from cheta.storage import Store


class FakeEmbedder:
    """Returns a scripted sequence of embeddings (None = no usable face)."""

    def __init__(self, seq: list[NDArray[np.float32] | None]) -> None:
        self.seq = list(seq)

    def embed(self, frame: Frame) -> NDArray[np.float32] | None:
        return self.seq.pop(0) if self.seq else None


class FakeSource:
    def __init__(self) -> None:
        self.closed = False

    def read(self) -> Frame | None:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def rand_emb(seed: int, dim: int = 128) -> NDArray[np.float32]:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def key() -> bytes:
    return Fernet.generate_key()


@pytest.fixture
def store(tmp_path: Path, key: bytes) -> Store:
    return Store(tmp_path, "test", key=key)


class LiveEmbedder:
    """Embedder whose output the test can change (who is in front of the camera)."""

    def __init__(self) -> None:
        self.current: NDArray[np.float32] | None = None

    def embed(self, frame: Frame) -> NDArray[np.float32] | None:
        return self.current


class FakeTextEmbedder:
    """Deterministic bag-of-words embedder (hashed tokens) so CI needs no model download."""

    DIM = 384

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]

    def _embed(self, text: str) -> NDArray[np.float32]:
        import zlib

        v = np.zeros(self.DIM, dtype=np.float32)
        for tok in self._tokens(text):
            v[zlib.crc32(tok.encode()) % self.DIM] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def embed_document(self, text: str) -> NDArray[np.float32]:
        return self._embed(text)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._embed(text)
