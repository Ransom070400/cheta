"""One-off: export speechbrain's ECAPA speaker encoder (features included) to a single ONNX file.

Needs torch + speechbrain, which cheta itself does not depend on:

    uv run --with torch --with speechbrain --with onnx python scripts/export_ecapa_onnx.py OUT.onnx

The graph takes mono 16 kHz float32 audio ``[1, samples]`` and returns the raw (not yet
L2-normalised) speaker embedding ``[1, 192]``: Fbank -> sentence mean-norm -> ECAPA-TDNN,
exactly what ``EncoderClassifier.encode_batch`` does.
"""

from __future__ import annotations

import sys

import torch
from speechbrain.inference.speaker import EncoderClassifier

from cheta.models import download_model


class Wrapper(torch.nn.Module):
    """Fbank + sentence mean-norm + ECAPA-TDNN, with the STFT written as a strided convolution.

    ONNX cannot export ``torch.stft`` with complex output, so the Hamming-windowed DFT is baked
    in as conv weights (same maths: n_fft=win=400, hop=160, centre padding with zeros).
    """

    N_FFT = 400
    HOP = 160

    def __init__(self, clf: EncoderClassifier) -> None:
        super().__init__()
        self.filterbank = clf.mods.compute_features.compute_fbanks
        self.model = clf.mods.embedding_model
        n = torch.arange(self.N_FFT, dtype=torch.float64)
        k = torch.arange(self.N_FFT // 2 + 1, dtype=torch.float64)[:, None]
        window = torch.hamming_window(self.N_FFT, dtype=torch.float64)
        angle = 2 * torch.pi * k * n / self.N_FFT
        basis = torch.cat([torch.cos(angle), -torch.sin(angle)]) * window  # (402, 400)
        self.register_buffer("basis", basis.float()[:, None, :])

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(wav[:, None, :], (self.N_FFT // 2, self.N_FFT // 2))
        spec = torch.nn.functional.conv1d(x, self.basis, stride=self.HOP)  # (1, 402, frames)
        re, im = spec[:, : self.N_FFT // 2 + 1], spec[:, self.N_FFT // 2 + 1 :]
        power = (re * re + im * im).transpose(1, 2)  # (1, frames, 201)
        feats = self.filterbank(power)
        feats = feats - feats.mean(dim=1, keepdim=True)  # InputNormalization(sentence, no std)
        return self.model(feats).squeeze(1)


def main(out: str) -> None:
    src = str(download_model("ecapa"))
    clf = EncoderClassifier.from_hparams(source=src, savedir=src, run_opts={"device": "cpu"})
    clf.eval()
    w = Wrapper(clf).eval()
    check = torch.randn(1, 40_000) * 0.1
    with torch.no_grad():
        ref = clf.encode_batch(check).squeeze()
        assert torch.nn.functional.cosine_similarity(w(check).squeeze(), ref, dim=0) > 0.9999
    wav = torch.randn(1, 48_000)
    torch.onnx.export(
        w,
        (wav,),
        out,
        input_names=["wav"],
        output_names=["embedding"],
        dynamic_axes={"wav": {1: "samples"}},
        opset_version=17,
        dynamo=False,
    )
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
