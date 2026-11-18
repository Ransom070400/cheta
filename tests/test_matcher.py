from cheta.face.matcher import match
from tests.conftest import rand_emb


def test_best_person_wins() -> None:
    gallery = {"a": [rand_emb(1), rand_emb(2)], "b": [rand_emb(3)]}
    assert match(rand_emb(3), gallery) == ("b", 1.0) or match(rand_emb(3), gallery)[0] == "b"  # type: ignore[index]


def test_stranger_returns_none() -> None:
    gallery = {"a": [rand_emb(1)], "b": [rand_emb(2)]}
    assert match(rand_emb(99), gallery) is None


def test_threshold_is_configurable() -> None:
    gallery = {"a": [rand_emb(1)]}
    q = rand_emb(2)
    assert match(q, gallery, threshold=0.4) is None
    assert match(q, gallery, threshold=-1.0) is not None


def test_empty_gallery() -> None:
    assert match(rand_emb(1), {}) is None
