from pathlib import Path

import pytest

from cheta import models


def fake_downloader(**kwargs: object) -> str:
    """Mimic hf_hub_download's snapshot layout without touching the network."""
    path = Path(str(kwargs["cache_dir"])) / "snapshots" / "abc" / str(kwargs["filename"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return str(path)


def test_registry_has_expected_models() -> None:
    assert set(models.MODELS) == {"yunet", "sface", "silero_vad", "ecapa", "bge"}


def test_single_file_returns_file(tmp_path: Path) -> None:
    p = models.download_model("yunet", cache_dir=tmp_path, downloader=fake_downloader)
    assert p.name == "face_detection_yunet_2023mar.onnx"
    assert p.exists()


def test_nested_primary_returns_file(tmp_path: Path) -> None:
    p = models.download_model("silero_vad", cache_dir=tmp_path, downloader=fake_downloader)
    assert p.parts[-2:] == ("onnx", "model.onnx")


def test_multi_file_returns_snapshot_dir(tmp_path: Path) -> None:
    root = models.download_model("bge", cache_dir=tmp_path, downloader=fake_downloader)
    assert root.name == "abc"
    for f in models.MODELS["bge"].files:
        assert (root / f).exists()


def test_every_model_belongs_to_a_group() -> None:
    assert {m for names in models.GROUPS.values() for m in names} == set(models.MODELS)


def test_models_cli_rejects_unknown_name() -> None:
    with pytest.raises(SystemExit):
        models.main(["nope"])


def test_download_all(tmp_path: Path) -> None:
    out = models.download_all(cache_dir=tmp_path, downloader=fake_downloader)
    assert set(out) == set(models.MODELS)


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        models.download_model("nope")


@pytest.mark.parametrize("name", sorted(models.MODELS))
def test_every_model_is_pinned_to_a_full_commit_hash(name: str) -> None:
    rev = models.MODELS[name].revision
    assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_downloader_receives_the_pinned_revision(tmp_path: Path) -> None:
    seen: list[object] = []

    def spy(**kwargs: object) -> str:
        seen.append(kwargs["revision"])
        return fake_downloader(**kwargs)

    models.download_model("bge", cache_dir=tmp_path, downloader=spy)
    assert seen and set(seen) == {models.MODELS["bge"].revision}
