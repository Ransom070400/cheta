"""Voice activity detection with silero-vad (ONNX)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from cheta.models import download_model
from cheta.voice.audio import SAMPLE_RATE, Audio

CHUNK = 512  # samples per VAD step at 16 kHz
CONTEXT = 64  # silero-vad v5 prepends the last 64 samples of the previous chunk
SPEECH_PROB = 0.5


class VoiceActivityDetector(Protocol):
    """Extracts only the speech from a recording."""

    def speech_only(self, audio: Audio) -> Audio: ...


class SileroVAD:
    """silero-vad via onnxruntime. The model is loaded lazily."""

    def __init__(self, model_path: str | Path | None = None, threshold: float = SPEECH_PROB):
        self._path = str(model_path) if model_path else None
        self.threshold = threshold
        self._session: Any = None

    def _load(self) -> Any:
        if self._session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                self._path or str(download_model("silero_vad")),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def probabilities(self, audio: Audio) -> list[float]:
        """Per-chunk speech probability."""
        session = self._load()
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT), dtype=np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        probs: list[float] = []
        for start in range(0, len(audio) - CHUNK + 1, CHUNK):
            chunk = audio[start : start + CHUNK][None, :]
            x = np.concatenate([context, chunk], axis=1)
            out, state = session.run(None, {"input": x, "state": state, "sr": sr})
            context = x[:, -CONTEXT:]
            probs.append(float(out[0][0]))
        return probs

    def speech_only(self, audio: Audio) -> Audio:
        probs = self.probabilities(audio)
        kept = [
            audio[i * CHUNK : (i + 1) * CHUNK] for i, p in enumerate(probs) if p >= self.threshold
        ]
        return np.concatenate(kept) if kept else np.zeros(0, dtype=np.float32)
