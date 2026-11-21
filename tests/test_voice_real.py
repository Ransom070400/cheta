"""Real ECAPA ONNX model from the Hub. Deselected by default (needs the model download).

Run with: uv run pytest -m network tests/test_voice_real.py
"""

import numpy as np
import pytest

from cheta.voice import EcapaEmbedder

pytestmark = pytest.mark.network


def test_real_ecapa_embeds_and_separates_signals() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(48_000) / 16_000
    voice_a = (0.3 * np.sin(2 * np.pi * 120 * t) + 0.02 * rng.standard_normal(48_000)).astype(
        np.float32
    )
    voice_b = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.02 * rng.standard_normal(48_000)).astype(
        np.float32
    )
    emb = EcapaEmbedder()
    a1, a2, b = emb.embed(voice_a), emb.embed(voice_a[:32_000]), emb.embed(voice_b)
    assert a1.shape == (192,) and abs(float(np.linalg.norm(a1)) - 1.0) < 1e-4
    assert float(a1 @ a2) > float(a1 @ b)
