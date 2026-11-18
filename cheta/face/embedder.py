"""Face detection (YuNet) and embedding (SFace) via OpenCV. Raw frames are never stored."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from cheta.models import download_model

Frame = NDArray[np.uint8]
MIN_DETECTION_SCORE = 0.85
MIN_FACE_SIZE = 80  # pixels, shortest side of the detected box


class FaceEmbedder(Protocol):
    """Turns a frame into an L2-normalised embedding of the single face in it."""

    def embed(self, frame: Frame) -> NDArray[np.float32] | None:
        """Return an embedding, or None when there is not exactly one usable face."""
        ...


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity of two vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


class SFaceEmbedder:
    """YuNet detector + SFace recognizer."""

    def __init__(self, yunet_path: str | Path | None = None, sface_path: str | Path | None = None):
        self._yunet_path = str(yunet_path or download_model("yunet"))
        self._sface_path = str(sface_path or download_model("sface"))
        self._detector = cv2.FaceDetectorYN.create(self._yunet_path, "", (320, 320))
        self._recognizer = cv2.FaceRecognizerSF.create(self._sface_path, "")

    def embed(self, frame: Frame) -> NDArray[np.float32] | None:
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is None:
            return None
        good = [
            f for f in faces if f[14] >= MIN_DETECTION_SCORE and min(f[2], f[3]) >= MIN_FACE_SIZE
        ]
        if len(good) != 1:
            return None
        aligned = self._recognizer.alignCrop(frame, good[0])
        feat = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm else None
