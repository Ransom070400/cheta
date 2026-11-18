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
    """Embedder whose output can be changed from the test (simulates who is in front of the camera)."""

    def __init__(self) -> None:
        self.current: NDArray[np.float32] | None = None

    def embed(self, frame: Frame) -> NDArray[np.float32] | None:
        return self.current
