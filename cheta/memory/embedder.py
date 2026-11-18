"""Text embeddings for memory (BAAI/bge-small-en-v1.5 via sentence-transformers)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cheta.models import download_model

DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class TextEmbedder(Protocol):
    """Embeds text into L2-normalised vectors of size ``DIM``."""

    def embed_document(self, text: str) -> NDArray[np.float32]: ...

    def embed_query(self, text: str) -> NDArray[np.float32]: ...


class BgeEmbedder:
    """BGE small English embedder; the model is loaded lazily on first use."""

    def __init__(self) -> None:
        self._model: object | None = None

    def _encode(self, text: str) -> NDArray[np.float32]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(str(download_model("bge")), device="cpu")
        vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
        return np.asarray(vec, dtype=np.float32)

    def embed_document(self, text: str) -> NDArray[np.float32]:
        return self._encode(text)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._encode(QUERY_PREFIX + text)
