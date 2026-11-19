from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cheta.robot import (
    ReachyAudioSource,
    ReachyFrameSource,
    ReachyHead,
    RobotSink,
    connect_robot,
    resample,
    to_mono_16k,
)


class FakeMedia:
    """Mimics the parts of reachy_mini's MediaManager that cheta uses."""

    def __init__(self, rate: int = 16_000) -> None:
        self.rate = rate
        self.frame: Any = np.full((480, 640, 3), 7, dtype=np.uint8)
        self.camera = SimpleNamespace(resolution=(640, 480))
        self.calls: list[str] = []
        self.pushed: list[np.ndarray] = []
        self._chunks = 0

    def get_frame(self) -> Any:
        return self.frame

    def start_recording(self) -> None:
        self.calls.append("start_recording")

    def stop_recording(self) -> None:
        self.calls.append("stop_recording")

    def get_input_audio_samplerate(self) -> int:
        return self.rate

    def get_output_audio_samplerate(self) -> int:
        return self.rate

    def get_audio_sample(self) -> Any:
        self._chunks += 1
        if self._chunks == 1:
            return None  # nothing buffered yet
        n = self.rate // 10
        left = np.full(n, 0.5, dtype=np.float32)
        right = np.full(n, 0.1, dtype=np.float32)
        return np.stack([left, right], axis=1)  # the SDK returns stereo (n, 2)

    def start_playing(self) -> None:
        self.calls.append("start_playing")

    def push_audio_sample(self, data: np.ndarray) -> None:
        self.pushed.append(data)

    def stop_playing(self) -> None:
        self.calls.append("stop_playing")


class FakeMini:
    def __init__(self, rate: int = 16_000) -> None:
        self.media = FakeMedia(rate)
        self.looked: list[tuple[int, int, float]] = []
        self.woke = False

    def look_at_image(self, u: int, v: int, duration: float = 1.0) -> None:
        self.looked.append((u, v, duration))

    def wake_up(self) -> None:
        self.woke = True


def test_frames_come_from_the_robot_camera() -> None:
    src = ReachyFrameSource(FakeMini())
    frame = src.read()
    assert frame is not None and frame.shape == (480, 640, 3)
    src.close()


def test_no_frame_is_none() -> None:
    mini = FakeMini()
    mini.media.frame = None
    assert ReachyFrameSource(mini).read() is None


def test_record_returns_mono_16k_of_the_right_length() -> None:
    mini = FakeMini()
    audio = ReachyAudioSource(mini, poll_s=0.0).record(0.5)
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) == 8_000
    assert audio[0] == pytest.approx(0.3)  # mean of the two channels
    assert mini.media.calls == ["start_recording", "stop_recording"]


def test_record_downsamples_a_48k_microphone() -> None:
    audio = ReachyAudioSource(FakeMini(rate=48_000), poll_s=0.0).record(0.5)
    assert len(audio) == 8_000


def test_record_stops_even_if_reading_fails() -> None:
    mini = FakeMini()

    def boom() -> Any:
        raise RuntimeError("mic unplugged")

    mini.media.get_audio_sample = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        ReachyAudioSource(mini).record(0.5)
    assert mini.media.calls[-1] == "stop_recording"


def test_look_at_passes_pixels_and_clamps_to_the_image() -> None:
    mini = FakeMini()
    head = ReachyHead(mini)
    head.look_at(320, 240)
    head.look_at(0, 9999)  # the SDK asserts 0 < u < width, 0 < v < height
    assert mini.looked == [(320, 240, 0.8), (1, 479, 0.8)]
    head.wake_up()
    assert mini.woke


def test_robot_sink_plays_and_always_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cheta.robot.time.sleep", lambda _s: None)
    mini = FakeMini()
    RobotSink(mini).play(np.zeros(1600, dtype=np.float32), 16_000)
    assert mini.media.calls == ["start_playing", "stop_playing"]
    assert len(mini.media.pushed[0]) == 1600


def test_robot_sink_resamples_to_the_output_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cheta.robot.time.sleep", lambda _s: None)
    mini = FakeMini(rate=48_000)
    RobotSink(mini).play(np.zeros(22_050, dtype=np.float32), 22_050)
    assert len(mini.media.pushed[0]) == 48_000


def test_resampling_helpers() -> None:
    x = np.sin(np.linspace(0, 20, 4800)).astype(np.float32)
    assert len(to_mono_16k(x, 48_000)) == 1600
    assert len(to_mono_16k(x, 44_100)) == int(4800 * 16_000 / 44_100)
    assert resample(x, 16_000, 16_000) is x
    assert len(resample(x, 24_000, 48_000)) == 9600


def test_connect_robot_explains_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name == "reachy_mini":
            raise ImportError("no module")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"cheta\[robot\]"):
        connect_robot()
