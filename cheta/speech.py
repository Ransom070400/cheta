"""Local text-to-speech. Nothing is sent off the device.

Synthesis uses the operating system's engine (``say`` on macOS, ``espeak-ng`` on Linux) writing
a temporary WAV that is read back and deleted; playback goes to the laptop speakers or the robot.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cheta.robot import RobotSink

Samples = NDArray[np.float32]


class TextToSpeech(Protocol):
    """Turns text into mono float32 samples and their sample rate."""

    def synthesize(self, text: str) -> tuple[Samples, int]: ...


class SpeechSink(Protocol):
    """Plays mono float32 samples."""

    def play(self, samples: Samples, rate: int) -> None: ...


def read_wav(path: Path) -> tuple[Samples, int]:
    """Read a 16-bit PCM WAV as mono float32."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM audio from the TTS engine")
        rate, channels = wav.getframerate(), wav.getnchannels()
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    samples = data.reshape(-1, channels).mean(axis=1) if channels > 1 else data
    return (samples / 32768.0).astype(np.float32), rate


class SystemTTS:
    """macOS ``say`` or Linux ``espeak-ng``."""

    def __init__(
        self,
        voice: str | None = None,
        run: Callable[..., Any] = subprocess.run,
        platform: str = sys.platform,
    ) -> None:
        self.voice = voice
        self._run = run
        self._platform = platform

    def _command(self, out: Path) -> list[str]:
        if self._platform == "darwin":
            cmd = ["say", "-o", str(out), "--data-format=LEI16@22050", "-f", "-"]
            return cmd + (["-v", self.voice] if self.voice else [])
        if self._platform.startswith("linux") and shutil.which("espeak-ng"):
            cmd = ["espeak-ng", "-w", str(out), "--stdin"]
            return cmd + (["-v", self.voice] if self.voice else [])
        raise RuntimeError(
            "no text-to-speech engine found: need `say` (macOS) or `espeak-ng` (Linux)"
        )

    def synthesize(self, text: str) -> tuple[Samples, int]:
        with tempfile.TemporaryDirectory(prefix="cheta-tts-") as tmp:
            out = Path(tmp) / "speech.wav"
            # The text goes in on stdin, so a name like "-rf" can never be read as an option.
            self._run(self._command(out), input=text.encode(), check=True, capture_output=True)
            return read_wav(out)


class SoundDeviceSink:
    """Play through the computer's default output device."""

    def play(self, samples: Samples, rate: int) -> None:
        import sounddevice as sd

        sd.play(samples, rate)
        sd.wait()


class Speaker:
    """Say things out loud."""

    def __init__(self, tts: TextToSpeech | None = None, sink: SpeechSink | None = None) -> None:
        self._tts = tts or SystemTTS()
        self._sink = sink or SoundDeviceSink()

    def say(self, text: str) -> None:
        """Speak ``text`` and wait until it has finished."""
        if not text.strip():
            return
        samples, rate = self._tts.synthesize(text)
        self._sink.play(samples, rate)


def robot_speaker(mini: Any, tts: TextToSpeech | None = None) -> Speaker:
    """A speaker that talks through the robot's own speaker."""
    return Speaker(tts=tts, sink=RobotSink(mini))
