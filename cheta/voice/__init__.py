"""Voice activity detection and speaker embedding."""

from __future__ import annotations

from cheta.voice.audio import SAMPLE_RATE, Audio, AudioSource, MicSource
from cheta.voice.embedder import EcapaEmbedder, VoiceEmbedder
from cheta.voice.vad import SileroVAD, VoiceActivityDetector

WINDOW_S = 3.0
STRIDE_S = 1.5


def split_windows(
    speech: Audio, window_s: float = WINDOW_S, stride_s: float = STRIDE_S
) -> list[Audio]:
    """Split speech into overlapping windows (one window if it is shorter than ``window_s``)."""
    win, stride = int(window_s * SAMPLE_RATE), int(stride_s * SAMPLE_RATE)
    if len(speech) <= win:
        return [speech]
    return [speech[s : s + win] for s in range(0, len(speech) - win + 1, stride)]


__all__ = [
    "SAMPLE_RATE",
    "Audio",
    "AudioSource",
    "EcapaEmbedder",
    "MicSource",
    "SileroVAD",
    "VoiceActivityDetector",
    "VoiceEmbedder",
    "split_windows",
]
