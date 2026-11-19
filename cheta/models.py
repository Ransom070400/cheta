"""Model registry and downloader.

All models come from the Hugging Face Hub and are cached locally. Downloading models is
the only network activity cheta performs; biometrics never leave the device.

Run ``python -m cheta.models`` to download every model.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class ModelSpec:
    """A model hosted on the Hugging Face Hub.

    ``revision`` is the exact commit of the repo that cheta was tested with; it is pinned so a
    later upstream change cannot silently alter embeddings (which would break stored people).
    ``files`` are the files to fetch. ``primary`` is the file returned as the model path for
    single-file models; for multi-file models the directory containing the files is returned.
    """

    name: str
    repo_id: str
    files: tuple[str, ...]
    purpose: str
    revision: str
    primary: str | None = None


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            name="yunet",
            repo_id="opencv/face_detection_yunet",
            files=("face_detection_yunet_2023mar.onnx",),
            purpose="face detection",
            revision="3cc26e7f1014a5ee5d74a42acee58bafc9d0a310",
            primary="face_detection_yunet_2023mar.onnx",
        ),
        ModelSpec(
            name="sface",
            repo_id="opencv/face_recognition_sface",
            files=("face_recognition_sface_2021dec.onnx",),
            purpose="face embedding",
            revision="3d7082438a6e4551e840c9b2bb60b71e8da4b524",
            primary="face_recognition_sface_2021dec.onnx",
        ),
        ModelSpec(
            name="silero_vad",
            repo_id="onnx-community/silero-vad",
            files=("onnx/model.onnx",),
            purpose="voice activity detection",
            revision="e71cae966052b992a7eca6b17738916ce0eca4ec",
            primary="onnx/model.onnx",
        ),
        ModelSpec(
            name="ecapa",
            repo_id="speechbrain/spkrec-ecapa-voxceleb",
            files=(
                "hyperparams.yaml",
                "embedding_model.ckpt",
                "mean_var_norm_emb.ckpt",
                "classifier.ckpt",
                "label_encoder.txt",
            ),
            purpose="speaker embedding",
            revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        ),
        ModelSpec(
            name="bge",
            repo_id="BAAI/bge-small-en-v1.5",
            files=(
                "config.json",
                "config_sentence_transformers.json",
                "model.safetensors",
                "modules.json",
                "sentence_bert_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
                "1_Pooling/config.json",
            ),
            purpose="text embedding for memory",
            revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        ),
    )
}

Downloader = Callable[..., str]


def download_model(
    name: str,
    cache_dir: str | Path | None = None,
    downloader: Downloader = hf_hub_download,
) -> Path:
    """Download one model and return its local path.

    Single-file models return the file path; multi-file models return their directory.
    """
    try:
        spec = MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}") from None

    paths = [
        Path(
            downloader(
                repo_id=spec.repo_id, filename=f, revision=spec.revision, cache_dir=cache_dir
            )
        )
        for f in spec.files
    ]
    if spec.primary is not None:
        return paths[spec.files.index(spec.primary)]
    # Multi-file: every file lives under the same snapshot directory.
    first, first_name = paths[0], spec.files[0]
    root = first
    for _ in Path(first_name).parts:
        root = root.parent
    return root


def download_all(
    names: Iterable[str] | None = None,
    cache_dir: str | Path | None = None,
    downloader: Downloader = hf_hub_download,
) -> dict[str, Path]:
    """Download the given models (default: all) and return name -> local path."""
    return {
        n: download_model(n, cache_dir=cache_dir, downloader=downloader)
        for n in (names if names is not None else MODELS)
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m cheta.models``."""
    parser = argparse.ArgumentParser(prog="python -m cheta.models", description=__doc__)
    parser.add_argument(
        "names", nargs="*", help=f"models to fetch (default: all of {sorted(MODELS)})"
    )
    parser.add_argument("--cache-dir", default=None, help="override the Hugging Face cache dir")
    args = parser.parse_args(argv)
    for name in args.names:
        if name not in MODELS:
            parser.error(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    for name in args.names or MODELS:
        print(f"downloading {name} ({MODELS[name].purpose}) from {MODELS[name].repo_id} ...")
        path = download_model(name, cache_dir=args.cache_dir)
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
