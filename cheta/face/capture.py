"""Frame sources. Frames are held in memory only long enough to embed them."""

from __future__ import annotations

from typing import Protocol, cast

import cv2

from cheta.face.embedder import Frame


class FrameSource(Protocol):
    """Anything that yields BGR frames."""

    def read(self) -> Frame | None: ...

    def close(self) -> None: ...


class WebcamSource:
    """Laptop webcam via OpenCV."""

    def __init__(self, index: int = 0) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open webcam {index}")

    def read(self) -> Frame | None:
        ok, frame = self._cap.read()
        return cast(Frame, frame) if ok else None

    def close(self) -> None:
        self._cap.release()
