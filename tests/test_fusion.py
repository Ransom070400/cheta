import pytest

from cheta.fusion import SINGLE_MODALITY_FACTOR, Evidence, fuse


def ev(pid: str, score: float, threshold: float = 0.4) -> Evidence:
    return Evidence(pid, score, threshold)


def test_no_evidence() -> None:
    assert fuse(None, None) is None


def test_normalisation_bounds() -> None:
    assert ev("a", 0.4).normalised == 0.0
    assert ev("a", 1.0).normalised == 1.0
    assert ev("a", 0.1).normalised == 0.0  # below threshold clamps
    assert 0.0 < ev("a", 0.7).normalised < 1.0


def test_single_modality_has_lower_confidence() -> None:
    face_only = fuse(ev("a", 0.9), None)
    voice_only = fuse(None, ev("a", 0.9))
    assert face_only is not None and voice_only is not None
    assert face_only[0] == "a"
    assert face_only[1] == ev("a", 0.9).normalised * SINGLE_MODALITY_FACTOR
    assert voice_only[1] == face_only[1]


def test_agreement_beats_either_alone() -> None:
    both = fuse(ev("a", 0.7), ev("a", 0.7))
    single = fuse(ev("a", 0.7), None)
    assert both is not None and single is not None
    assert both[1] > single[1]
    assert both[1] <= 1.0


def test_conflict_returns_none() -> None:
    assert fuse(ev("a", 0.9), ev("b", 0.9)) is None


def test_perfect_both_is_one() -> None:
    assert fuse(ev("a", 1.0), ev("a", 1.0)) == ("a", 1.0)


def test_single_factor_is_configurable() -> None:
    perfect = ev("a", 1.0)
    assert fuse(perfect, None, single_factor=1.0) == ("a", 1.0)
    assert fuse(None, perfect, single_factor=0.5) == ("a", 0.5)
    both = fuse(perfect, perfect, single_factor=0.1)
    assert both == ("a", 1.0)  # the factor only affects a lone signal


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_single_factor_validated(bad: float) -> None:
    with pytest.raises(ValueError):
        fuse(ev("a", 0.9), None, single_factor=bad)
