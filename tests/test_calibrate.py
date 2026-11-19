import wave
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from cheta import cli
from cheta.calibrate import MIN_SPEECH_S, load_voice_dir, recommend_threshold, score_voices
from cheta.voice import SAMPLE_RATE
from tests.conftest import FakeVAD, rand_emb


def test_threshold_clears_every_impostor_when_no_false_accepts_allowed() -> None:
    rec = recommend_threshold([0.8, 0.9, 0.7], [0.2, 0.3, 0.6], max_far=0.0)
    assert rec.threshold == pytest.approx(0.601)  # just above the highest impostor
    assert rec.false_accept_rate == 0.0 and rec.false_reject_rate == 0.0
    assert (rec.n_genuine, rec.n_impostor) == (3, 3)


def test_threshold_is_lowest_that_meets_the_false_accept_target() -> None:
    # One impostor of three (the 0.6) may be accepted, so the threshold sits just above 0.3.
    rec = recommend_threshold([0.8, 0.9, 0.7], [0.2, 0.3, 0.6], max_far=0.34)
    assert rec.threshold == pytest.approx(0.301)
    assert rec.false_accept_rate == pytest.approx(1 / 3)
    assert rec.false_reject_rate == 0.0


def test_overlapping_scores_cost_false_rejects() -> None:
    rec = recommend_threshold([0.5, 0.9], [0.6], max_far=0.0)
    assert rec.false_accept_rate == 0.0 and rec.false_reject_rate == 0.5


def test_rounding_never_breaks_the_false_accept_target() -> None:
    rec = recommend_threshold([0.9], [0.68249], max_far=0.0)
    assert rec.threshold >= 0.683 and rec.false_accept_rate == 0.0


@pytest.mark.parametrize("gen,imp", [([], [0.1]), ([0.9], [])])
def test_needs_both_kinds_of_scores(gen: list[float], imp: list[float]) -> None:
    with pytest.raises(ValueError):
        recommend_threshold(gen, imp)


@pytest.mark.parametrize("bad", [-0.1, 1.0])
def test_max_far_validated(bad: float) -> None:
    with pytest.raises(ValueError):
        recommend_threshold([0.9], [0.1], max_far=bad)


class NoisyVoice:
    """Audio filled with ``person + 0.01*take`` embeds near that person's vector, with jitter."""

    def embed(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        m = float(audio.mean())
        person = int(round(m))
        take = int(round((m - person) * 100)) % 50
        v = rand_emb(person, 192) + 0.3 * rand_emb(1000 + take, 192)
        return (v / np.linalg.norm(v)).astype(np.float32)


def clip(person: int, take: int = 0, seconds: float = 4.0) -> NDArray[np.float32]:
    return np.full(int(seconds * SAMPLE_RATE), person + 0.01 * take, dtype=np.float32)


def test_scores_separate_genuine_from_impostor() -> None:
    enrol = {"ann": clip(1), "bob": clip(2), "cy": clip(3)}
    probes = {n: [clip(i, 1), clip(i, 2)] for n, i in (("ann", 1), ("bob", 2), ("cy", 3))}
    genuine, impostor = score_voices(enrol, probes, FakeVAD(), NoisyVoice())
    assert len(genuine) == len(impostor) > 0
    assert min(genuine) > 0.8 and max(impostor) < 0.4
    rec = recommend_threshold(genuine, impostor, max_far=0.0)
    assert max(impostor) < rec.threshold <= min(genuine)
    assert rec.false_accept_rate == 0.0 and rec.false_reject_rate == 0.0


def test_needs_two_people() -> None:
    with pytest.raises(ValueError, match="two people"):
        score_voices({"ann": clip(1)}, {"ann": [clip(1, 1)]}, FakeVAD(), NoisyVoice())


def test_probe_without_enrolment_is_rejected() -> None:
    with pytest.raises(ValueError, match="no enrolment"):
        score_voices({"ann": clip(1), "bob": clip(2)}, {"eve": [clip(5)]}, FakeVAD(), NoisyVoice())


def test_short_enrolment_is_rejected() -> None:
    short = clip(1, seconds=MIN_SPEECH_S / 2)
    with pytest.raises(ValueError, match="not enough speech"):
        score_voices({"ann": short, "bob": clip(2)}, {"bob": [clip(2, 1)]}, FakeVAD(), NoisyVoice())


def test_probes_with_too_little_speech_are_skipped_and_all_short_is_an_error() -> None:
    short = clip(2, 1, seconds=MIN_SPEECH_S / 2)
    enrol = {"ann": clip(1), "bob": clip(2)}
    genuine, _ = score_voices(enrol, {"bob": [short, clip(2, 2)]}, FakeVAD(), NoisyVoice())
    assert len(genuine) >= 1
    with pytest.raises(ValueError, match="no probe clip"):
        score_voices(enrol, {"bob": [short]}, FakeVAD(), NoisyVoice())


def write_wav(path: Path, value: float, seconds: float = 4.0, rate: int = SAMPLE_RATE) -> None:
    # 16-bit audio holds -1..1, so store value/4 (values here are 1..3) and scale back on read
    data = np.full(int(seconds * rate), int(value / 4 * 32767), dtype=np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


def make_dir(tmp_path: Path) -> Path:
    for name, i in (("ann", 1), ("bob", 2), ("cy", 3)):
        for take in (1, 2, 3):
            write_wav(tmp_path / f"{name}_{take}.wav", i + 0.01 * (take - 1))
    return tmp_path


def test_load_voice_dir_uses_lowest_number_to_enrol(tmp_path: Path) -> None:
    enrol, probes = load_voice_dir(make_dir(tmp_path))
    assert set(enrol) == set(probes) == {"ann", "bob", "cy"}
    assert all(len(p) == 2 for p in probes.values())
    assert float(enrol["bob"].mean()) * 4 == pytest.approx(2.0, abs=0.01)


def test_load_voice_dir_resamples_to_16k(tmp_path: Path) -> None:
    write_wav(tmp_path / "ann_1.wav", 1.0, seconds=2.0, rate=8000)
    write_wav(tmp_path / "ann_2.wav", 1.0, seconds=2.0, rate=8000)
    enrol, _ = load_voice_dir(tmp_path)
    assert len(enrol["ann"]) == 2 * SAMPLE_RATE


def test_load_voice_dir_names_may_contain_underscores(tmp_path: Path) -> None:
    write_wav(tmp_path / "mary_jane_1.wav", 1.0)
    write_wav(tmp_path / "mary_jane_2.wav", 1.0)
    enrol, _ = load_voice_dir(tmp_path)
    assert list(enrol) == ["mary_jane"]


def test_load_voice_dir_rejects_bad_names_and_lone_clips(tmp_path: Path) -> None:
    write_wav(tmp_path / "ann.wav", 1.0)
    with pytest.raises(ValueError, match="<name>_<number>"):
        load_voice_dir(tmp_path)
    (tmp_path / "ann.wav").unlink()
    write_wav(tmp_path / "ann_1.wav", 1.0)
    with pytest.raises(ValueError, match="at least two clips"):
        load_voice_dir(tmp_path)


class WavVoice(NoisyVoice):
    """NoisyVoice for audio that went through a WAV file (values were stored divided by 4)."""

    def embed(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        return super().embed(audio * 4)


def test_calibrate_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "SileroVAD", FakeVAD)
    monkeypatch.setattr(cli, "EcapaEmbedder", WavVoice)
    assert cli.main(["calibrate", str(make_dir(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "3 people" in out and "Recommended voice_threshold" in out
    assert "false accepts 0.0%" in out and "false rejects 0.0%" in out


def test_calibrate_command_reports_bad_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["calibrate", str(tmp_path)]) == 1  # no recordings at all
    assert "Cannot calibrate" in capsys.readouterr().out
