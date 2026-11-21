"""Speaker embeddings: speechbrain's ECAPA-TDNN (spkrec-ecapa-voxceleb) exported to ONNX.

The ONNX graph includes the filterbank front end, so it takes raw 16 kHz audio and needs no torch
or speechbrain at runtime. ``scripts/export_ecapa_onnx.py`` produces it from the original weights;
its output matches speechbrain's ``encode_batch`` to better than 0.99999 cosine similarity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cheta.models import download_model
from cheta.voice.audio import Audio


class VoiceEmbedder(Protocol):
    """Turns speech audio into an L2-normalised speaker embedding."""

    def embed(self, audio: Audio) -> NDArray[np.float32]: ...


class EcapaEmbedder:
    """ECAPA-TDNN speaker encoder on onnxruntime; the model is loaded lazily on first use."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._path = str(model_path) if model_path else None
        self._session: Any = None

    def _load(self) -> Any:
        if self._session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                self._path or str(download_model("ecapa")),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def embed(self, audio: Audio) -> NDArray[np.float32]:
        wav = np.asarray(audio, dtype=np.float32)[None, :]
        vec = np.asarray(self._load().run(None, {"wav": wav})[0][0], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm else vec
