"""Face detection (YuNet) and embedding (SFace) via OpenCV. Raw frames are never stored."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from cheta.models import download_model

Frame = NDArray[np.uint8]
MIN_DETECTION_SCORE = 0.85
MIN_FACE_SIZE = 80  # pixels, shortest side of the detected box


@dataclass(frozen=True)
class FaceObservation:
    """What one frame shows: the face embedding and a head-turn (yaw) proxy.

    ``yaw`` is the nose's horizontal offset from the face centre in units of eye distance:
    about 0 facing the camera, growing in magnitude as the head turns.
    """

    embedding: NDArray[np.float32]
    yaw: float
    center: tuple[float, float] | None = None  # face centre in frame pixels (u right, v down)


class FaceEmbedder(Protocol):
    """Looks at one frame containing a single face."""

    def observe(self, frame: Frame) -> FaceObservation | None:
        """Return the observation, or None when there is not exactly one usable face."""
        ...


def head_yaw(face: NDArray[np.float32]) -> float:
    """Yaw proxy from a YuNet detection row (landmarks: eyes, nose, mouth corners).

    A real head turning moves the nose (which sits in front of the eye plane) sideways relative
    to the eyes and mouth. A flat photo, even tilted or moved, keeps this ratio almost constant.
    """
    r_eye, l_eye, nose, r_mouth, l_mouth = (face[4 + 2 * i : 6 + 2 * i] for i in range(5))
    eye_dist = float(abs(l_eye[0] - r_eye[0]))
    if eye_dist < 1e-6:
        return 0.0
    centre_x = (float(r_eye[0] + l_eye[0]) / 2 + float(r_mouth[0] + l_mouth[0]) / 2) / 2
    return (float(nose[0]) - centre_x) / eye_dist


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

    def observe(self, frame: Frame) -> FaceObservation | None:
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, detected = self._detector.detect(frame)
        if detected is None:
            return None
        faces: list[NDArray[np.float32]] = list(detected)
        good = [
            f for f in faces if f[14] >= MIN_DETECTION_SCORE and min(f[2], f[3]) >= MIN_FACE_SIZE
        ]
        if len(good) != 1:
            return None
        aligned = self._recognizer.alignCrop(frame, good[0])
        feat = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feat))
        if not norm:
            return None
        x, y, bw, bh = (float(v) for v in good[0][:4])
        return FaceObservation(
            embedding=feat / norm, yaw=head_yaw(good[0]), center=(x + bw / 2, y + bh / 2)
        )

    def embed(self, frame: Frame) -> NDArray[np.float32] | None:
        """Embedding only (no yaw)."""
        obs = self.observe(frame)
        return obs.embedding if obs else None
