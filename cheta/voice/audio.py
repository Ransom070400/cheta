"""Microphone capture. Audio is held in memory only; it is never written to disk."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000

Audio = NDArray[np.float32]


class AudioSource(Protocol):
    """Records mono float32 audio at 16 kHz."""

    def record(self, seconds: float) -> Audio: ...


class MicSource:
    """Laptop (or robot) microphone via sounddevice."""

    def __init__(self, device: int | str | None = None) -> None:
        self._device = device

    def record(self, seconds: float) -> Audio:
        import sounddevice as sd

        frames = int(seconds * SAMPLE_RATE)
        data = sd.rec(
            frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=self._device
        )
        sd.wait()
        return np.asarray(data[:, 0], dtype=np.float32)
