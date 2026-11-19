"""Speaker embeddings with speechbrain's ECAPA-TDNN (spkrec-ecapa-voxceleb)."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cheta.models import download_model
from cheta.voice.audio import Audio


class VoiceEmbedder(Protocol):
    """Turns speech audio into an L2-normalised speaker embedding."""

    def embed(self, audio: Audio) -> NDArray[np.float32]: ...


class EcapaEmbedder:
    """ECAPA-TDNN speaker encoder; the model is loaded lazily from the local cache."""

    def __init__(self) -> None:
        self._model: Any = None

    def embed(self, audio: Audio) -> NDArray[np.float32]:
        import torch

        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier

            src = str(download_model("ecapa"))
            self._model = EncoderClassifier.from_hparams(
                source=src, savedir=src, run_opts={"device": "cpu"}
            )
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))[None, :]
        with torch.no_grad():
            emb = self._model.encode_batch(wav)
        vec: NDArray[np.float32] = emb.squeeze().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec
