"""Text embeddings for memory: BAAI/bge-small-en-v1.5 as ONNX (no torch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cheta.models import download_model

DIM = 384
MAX_TOKENS = 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class TextEmbedder(Protocol):
    """Embeds text into L2-normalised vectors of size ``DIM``."""

    def embed_document(self, text: str) -> NDArray[np.float32]: ...

    def embed_query(self, text: str) -> NDArray[np.float32]: ...


class BgeEmbedder:
    """BGE small English embedder on onnxruntime; the model is loaded lazily on first use.

    Matches sentence-transformers' output for this model: the [CLS] token's final hidden state,
    L2-normalised.
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = Path(download_model("bge"))  # snapshot dir holding onnx/model.onnx + tokenizer.json
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(root / "onnx" / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        tok.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer = tok

    def _encode(self, text: str) -> NDArray[np.float32]:
        if self._session is None:
            self._load()
        enc = self._tokenizer.encode(text)
        ids = np.asarray([enc.ids], dtype=np.int64)
        feeds = {
            "input_ids": ids,
            "attention_mask": np.asarray([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.zeros_like(ids),
        }
        hidden = self._session.run(None, feeds)[0]
        vec = np.asarray(hidden[0, 0], dtype=np.float32)  # [CLS]
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm else vec

    def embed_document(self, text: str) -> NDArray[np.float32]:
        return self._encode(text)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._encode(QUERY_PREFIX + text)
