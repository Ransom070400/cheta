import shutil
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cheta.speech import Speaker, SystemTTS, read_wav, robot_speaker
from tests.test_robot import FakeMini


def write_wav(path: Path, samples: np.ndarray, rate: int = 22_050, channels: int = 1) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def test_read_wav_roundtrip(tmp_path: Path) -> None:
    tone = np.sin(np.linspace(0, 50, 2205)).astype(np.float32) * 0.5
    write_wav(tmp_path / "a.wav", tone)
    samples, rate = read_wav(tmp_path / "a.wav")
    assert rate == 22_050 and len(samples) == 2205
    assert np.allclose(samples, tone, atol=1e-3)


def test_read_wav_downmixes_stereo(tmp_path: Path) -> None:
    both = np.stack([np.full(100, 0.5), np.full(100, -0.5)], axis=1).reshape(-1)
    write_wav(tmp_path / "s.wav", both, channels=2)
    samples, _ = read_wav(tmp_path / "s.wav")
    assert len(samples) == 100 and np.allclose(samples, 0.0, atol=1e-3)


class FakeRun:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, cmd: list[str], **kw: Any) -> None:
        self.calls.append((cmd, kw))
        write_wav(Path(cmd[cmd.index("-o") + 1]), np.zeros(441, dtype=np.float32))


def test_macos_say_gets_text_on_stdin_not_argv() -> None:
    run = FakeRun()
    samples, rate = SystemTTS(voice="Samantha", run=run, platform="darwin").synthesize("-rf evil")
    cmd, kw = run.calls[0]
    assert cmd[0] == "say" and "-rf evil" not in cmd  # never an argument
    assert kw["input"] == b"-rf evil" and "-f" in cmd and "-v" in cmd and "Samantha" in cmd
    assert rate == 22_050 and len(samples) == 441


def test_no_engine_gives_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="text-to-speech"):
        SystemTTS(platform="win32").synthesize("hi")


class FakeTTS:
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.last = text
        return np.ones(10, dtype=np.float32), 16_000


class FakeSink:
    def __init__(self) -> None:
        self.played: list[tuple[np.ndarray, int]] = []

    def play(self, samples: np.ndarray, rate: int) -> None:
        self.played.append((samples, rate))


def test_speaker_synthesises_then_plays() -> None:
    sink = FakeSink()
    Speaker(tts=FakeTTS(), sink=sink).say("Welcome back Dave")
    assert len(sink.played) == 1 and sink.played[0][1] == 16_000


def test_speaker_ignores_empty_text() -> None:
    sink = FakeSink()
    Speaker(tts=FakeTTS(), sink=sink).say("   ")
    assert sink.played == []


def test_robot_speaker_uses_the_robot_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cheta.robot.time.sleep", lambda _s: None)
    mini = FakeMini()
    robot_speaker(mini, tts=FakeTTS()).say("hello")
    assert mini.media.calls == ["start_playing", "stop_playing"]


@pytest.mark.skipif(
    not (sys.platform == "darwin" and shutil.which("say")), reason="needs macOS `say`"
)
def test_real_macos_say_produces_audio() -> None:
    samples, rate = SystemTTS().synthesize("Welcome back Dave")
    assert rate == 22_050 and len(samples) > rate // 2  # at least half a second of speech
    assert float(np.abs(samples).max()) > 0.05
