"""Adapters for the Reachy Mini SDK (optional: ``pip install "cheta[robot]"``).

The SDK is imported lazily, so the rest of cheta works without it. Each adapter takes an
already-connected ``ReachyMini`` so apps keep control of the connection and the motors.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cheta.face.embedder import Frame
from cheta.voice.audio import SAMPLE_RATE, Audio

INSTALL_HINT = 'The Reachy Mini SDK is not installed. Run: pip install "cheta[robot]"'


def connect_robot(**kwargs: Any) -> Any:
    """Connect to a Reachy Mini (extra keyword arguments go to ``ReachyMini``)."""
    try:
        from reachy_mini import ReachyMini
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return ReachyMini(**kwargs)


def to_mono_16k(samples: NDArray[np.float32], rate: int) -> Audio:
    """Downmix to mono and resample to cheta's 16 kHz."""
    mono = samples if samples.ndim == 1 else samples.mean(axis=1)
    mono = np.asarray(mono, dtype=np.float32)
    if rate == SAMPLE_RATE:
        return mono
    if rate % SAMPLE_RATE == 0:  # integer ratio: block average doubles as a low-pass filter
        step = rate // SAMPLE_RATE
        usable = len(mono) - len(mono) % step
        return np.asarray(mono[:usable].reshape(-1, step).mean(axis=1), dtype=np.float32)
    n_out = int(len(mono) * SAMPLE_RATE / rate)
    x_new = np.linspace(0, len(mono) - 1, n_out)
    return np.asarray(np.interp(x_new, np.arange(len(mono)), mono), dtype=np.float32)


def resample(samples: NDArray[np.float32], from_rate: int, to_rate: int) -> NDArray[np.float32]:
    """Linear-interpolation resampling of mono audio (used for speech playback)."""
    if from_rate == to_rate or len(samples) == 0:
        return samples
    n_out = int(len(samples) * to_rate / from_rate)
    x_new = np.linspace(0, len(samples) - 1, n_out)
    return np.asarray(np.interp(x_new, np.arange(len(samples)), samples), dtype=np.float32)


class ReachyFrameSource:
    """Camera frames from the robot (BGR), as a cheta ``FrameSource``."""

    def __init__(self, mini: Any) -> None:
        self._mini = mini

    def read(self) -> Frame | None:
        frame: Frame | None = self._mini.media.get_frame()
        return frame

    def close(self) -> None:
        """Nothing to release: the app owns the robot connection."""


class ReachyAudioSource:
    """Microphone audio from the robot, as a cheta ``AudioSource`` (16 kHz mono)."""

    def __init__(self, mini: Any, poll_s: float = 0.02) -> None:
        self._mini = mini
        self._poll_s = poll_s

    def record(self, seconds: float) -> Audio:
        media = self._mini.media
        rate = int(media.get_input_audio_samplerate())
        wanted = int(seconds * rate)
        chunks: list[NDArray[np.float32]] = []
        have = 0
        deadline = time.monotonic() + seconds + 2.0
        media.start_recording()
        try:
            while have < wanted and time.monotonic() < deadline:
                chunk = media.get_audio_sample()
                if chunk is None or len(chunk) == 0:
                    time.sleep(self._poll_s)
                    continue
                chunks.append(chunk)
                have += len(chunk)
        finally:
            media.stop_recording()
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return to_mono_16k(np.concatenate(chunks)[:wanted], rate)


class ReachyHead:
    """Turn the robot's head toward a pixel in its camera image."""

    def __init__(self, mini: Any) -> None:
        self._mini = mini

    def look_at(self, u: int, v: int, duration: float = 0.8) -> None:
        """Look at pixel ``(u, v)``; coordinates are clamped inside the image."""
        width, height = self._mini.media.camera.resolution
        self._mini.look_at_image(
            min(max(int(u), 1), int(width) - 1),
            min(max(int(v), 1), int(height) - 1),
            duration=duration,
        )

    def wake_up(self) -> None:
        """Move to the neutral pose."""
        self._mini.wake_up()


class RobotSink:
    """Play synthesised speech through the robot's speaker."""

    def __init__(self, mini: Any) -> None:
        self._mini = mini

    def play(self, samples: NDArray[np.float32], rate: int) -> None:
        media = self._mini.media
        out_rate = int(media.get_output_audio_samplerate())
        data = resample(np.asarray(samples, dtype=np.float32), rate, out_rate)
        media.start_playing()
        try:
            media.push_audio_sample(data)
            time.sleep(len(data) / out_rate + 0.3)  # let the buffer drain before stopping
        finally:
            media.stop_playing()
