"""The public ``Identity`` entry point."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cheta.face.capture import FrameSource, WebcamSource
from cheta.face.embedder import FaceEmbedder, SFaceEmbedder
from cheta.face.matcher import DEFAULT_THRESHOLD, match
from cheta.person import Person
from cheta.presence import ABSENCE_SECONDS, ARRIVE_FRAMES, PresenceTracker
from cheta.storage import Store

FACE_SAMPLES = 5
SAMPLE_GAP_S = 0.6
ENROLL_TIMEOUT_S = 60.0
LOOP_INTERVAL_S = 0.1

log = logging.getLogger(__name__)
PersonCallback = Callable[[Person], None]


class EnrollmentError(RuntimeError):
    """Enrollment could not collect enough usable samples."""


class Identity:
    """Recognize people by face (and, in later milestones, voice) and remember facts."""

    def __init__(
        self,
        app: str,
        camera: Literal["webcam", "reachy"] = "webcam",
        data_dir: str | Path = "~/.cheta",
        *,
        threshold: float = DEFAULT_THRESHOLD,
        arrive_frames: int = ARRIVE_FRAMES,
        absence_seconds: float = ABSENCE_SECONDS,
        store: Store | None = None,
        face_embedder: FaceEmbedder | None = None,
        frame_source: FrameSource | None = None,
    ) -> None:
        if camera not in ("webcam", "reachy"):
            raise ValueError("camera must be 'webcam' or 'reachy'")
        self.app = app
        self.camera = camera
        self.threshold = threshold
        self.store = store or Store(data_dir, app)
        self._gallery: dict[str, list[NDArray[np.float32]]] | None = None
        self._arrived_cbs: list[PersonCallback] = []
        self._left_cbs: list[PersonCallback] = []
        self._arrive_frames = arrive_frames
        self._absence_seconds = absence_seconds
        self._tracker = PresenceTracker(arrive_frames, absence_seconds)
        self._latest: Person | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._embedder = face_embedder
        self._source = frame_source

    def _get_embedder(self) -> FaceEmbedder:
        if self._embedder is None:
            self._embedder = SFaceEmbedder()
        return self._embedder

    def _get_source(self) -> FrameSource:
        if self._source is None:
            if self.camera == "reachy":
                raise NotImplementedError("the Reachy camera arrives in M7")
            self._source = WebcamSource()
        return self._source

    def enroll(
        self,
        name: str,
        consent: bool,
        on_prompt: Callable[[str], None] | None = None,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> Person:
        """Capture 5 face samples and store them encrypted. Refuses without ``consent=True``."""
        if consent is not True:
            raise PermissionError("enroll() requires consent=True")
        if not name.strip():
            raise ValueError("name must not be empty")
        say = on_prompt or (lambda _m: None)

        embedder, source = self._get_embedder(), self._get_source()
        samples: list[NDArray[np.float32]] = []
        deadline = time.monotonic() + ENROLL_TIMEOUT_S
        say("Look at the camera.")
        while len(samples) < FACE_SAMPLES:
            if time.monotonic() > deadline:
                raise EnrollmentError(
                    f"only captured {len(samples)}/{FACE_SAMPLES} samples; "
                    "make sure exactly one face is clearly visible"
                )
            frame = source.read()
            emb = embedder.embed(frame) if frame is not None else None
            del frame  # never keep raw images
            if emb is None:
                continue
            samples.append(emb)
            say(f"Captured sample {len(samples)}/{FACE_SAMPLES}")
            _sleep(SAMPLE_GAP_S)

        person = self.store.add_person(name.strip(), consent=True)
        self.store.add_face_embeddings(person.id, samples)
        self._gallery = None
        return person

    def _recognize(self) -> Person | None:
        """Grab one frame and return the matching person, or None (no face, or a stranger)."""
        frame = self._get_source().read()
        if frame is None:
            return None
        emb = self._get_embedder().embed(frame)
        del frame  # never keep raw images
        if emb is None:
            return None
        if self._gallery is None:
            self._gallery = self.store.all_face_embeddings()
        hit = match(emb, self._gallery, self.threshold)
        if hit is None:
            return None
        person = self.store.get_person(hit[0])
        return Person(person.id, person.name, hit[1]) if person else None

    def who_is_here(self) -> Person | None:
        """Return the person in view (name + confidence), or None if nobody known is there.

        While ``start()`` is running this returns the person currently tracked; otherwise it
        looks at one fresh camera frame.
        """
        if self._thread is not None:
            with self._lock:
                return self._latest
        return self._recognize()

    def on_person_arrived(self, cb: PersonCallback) -> None:
        """Register a callback fired (on the background thread) when someone arrives."""
        self._arrived_cbs.append(cb)

    def on_person_left(self, cb: PersonCallback) -> None:
        """Register a callback fired (on the background thread) when someone leaves."""
        self._left_cbs.append(cb)

    def _fire(self, kind: str, person: Person) -> None:
        for cb in self._arrived_cbs if kind == "arrived" else self._left_cbs:
            try:
                cb(person)
            except Exception:
                log.exception("on_person_%s callback failed", kind)

    def _loop(self) -> None:
        while not self._stop.is_set():
            person = self._recognize()
            with self._lock:
                events = self._tracker.update(person, time.monotonic())
                self._latest = self._tracker.current
            for kind, who in events:
                self._fire(kind, who)
            self._stop.wait(LOOP_INTERVAL_S)

    def start(self) -> None:
        """Start background recognition and arrival/left events."""
        if self._thread is not None:
            return
        self._gallery = None
        self._tracker = PresenceTracker(self._arrive_frames, self._absence_seconds)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="cheta-recognize", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background recognition."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5)
        self._thread = None
        with self._lock:
            self._latest = None

    def close(self) -> None:
        """Release the camera and database."""
        self.stop()
        if self._source is not None:
            self._source.close()
        self.store.close()
