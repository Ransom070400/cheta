---
license: apache-2.0
library_name: onnx
base_model: speechbrain/spkrec-ecapa-voxceleb
tags: [speaker-recognition, speaker-verification, onnx, ecapa-tdnn]
---

# cheta-ecapa-onnx

[speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
exported to ONNX for [cheta](https://github.com/Ransom070400/cheta), so speaker embeddings run
without torch or speechbrain.

- **Input** `wav`: float32 `[1, samples]`, mono, 16 kHz (any length, at least ~0.5 s).
- **Output** `embedding`: float32 `[1, 192]`, not L2-normalised.
- The graph includes the filterbank front end (Hamming-windowed STFT as a convolution, mel
  filterbank, sentence mean-normalisation) and the ECAPA-TDNN encoder, i.e. what
  `EncoderClassifier.encode_batch` computes.
- Checked against speechbrain on clips from 0.5 s to 10 s: cosine similarity >= 0.9999999.

Weights are unchanged from the original model (Apache-2.0, trained on VoxCeleb). Re-create with
`scripts/export_ecapa_onnx.py` in the cheta repo.
