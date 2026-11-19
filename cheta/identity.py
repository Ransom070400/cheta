"""The public ``Identity`` entry point."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from cheta.face.capture import FrameSource, WebcamSource
from cheta.face.embedder import FaceEmbedder, FaceObservation, SFaceEmbedder, cosine_similarity
from cheta.face.matcher import DEFAULT_THRESHOLD, match
from cheta.fusion import Evidence, fuse
from cheta.liveness import CHALLENGE, HeadTurnLiveness
from cheta.memory import BgeEmbedder, Memory, TextEmbedder
from cheta.person import Person
from cheta.presence import ABSENCE_SECONDS, ARRIVE_FRAMES, PresenceTracker
from cheta.robot import ReachyAudioSource, ReachyFrameSource, connect_robot
from cheta.storage import Store
from cheta.voice import (
    SAMPLE_RATE,
    Audio,
    AudioSource,
    EcapaEmbedder,
    MicSource,
    SileroVAD,
    VoiceActivityDetector,
    VoiceEmbedder,
    split_windows,
)

FACE_SAMPLES = 5
SAMPLE_GAP_S = 0.6
ENROLL_TIMEOUT_S = 60.0
VOICE_RECORD_S = 5.0
VOICE_ATTEMPTS = 3
MIN_ENROLL_SPEECH_S = 3.0
MIN_IDENTIFY_SPEECH_S = 1.5
IDENTIFY_RECORD_S = 4.0
DEFAULT_VOICE_THRESHOLD = 0.5
LOOP_INTERVAL_S = 0.1
VOICE_CHUNK_S = 2.0  # background listener: record this much per step
VOICE_BUFFER_S = 4.0  # ... and identify from the most recent this many seconds
VOICE_TTL_S = 8.0  # voice evidence older than this is ignored
SNAPSHOT_NO_FACE_S = 1.0  # one-shot who_is_here gives up if no known face appears by then
PROMPT_EVERY_S = 10.0

log = logging.getLogger(__name__)
PersonCallback = Callable[[Person], None]


class EnrollmentError(RuntimeError):
    """Enrollment could not collect enough usable samples."""


class LivenessError(EnrollmentError):
    """The person did not turn their head, so liveness could not be confirmed."""


class Identity:
    """Recognize people by face and voice, check they are live, and remember facts."""

    def __init__(
        self,
        app: str,
        camera: Literal["webcam", "reachy"] = "webcam",
        data_dir: str | Path = "~/.cheta",
        *,
        threshold: float = DEFAULT_THRESHOLD,
        voice_threshold: float = DEFAULT_VOICE_THRESHOLD,
        arrive_frames: int = ARRIVE_FRAMES,
        absence_seconds: float = ABSENCE_SECONDS,
        liveness: bool = True,
        listen_voice: bool = False,
        on_prompt: Callable[[str], None] | None = None,
        robot: Any | None = None,
        store: Store | None = None,
        face_embedder: FaceEmbedder | None = None,
        frame_source: FrameSource | None = None,
        text_embedder: TextEmbedder | None = None,
        audio_source: AudioSource | None = None,
        voice_embedder: VoiceEmbedder | None = None,
        vad: VoiceActivityDetector | None = None,
    ) -> None:
        if camera not in ("webcam", "reachy"):
            raise ValueError("camera must be 'webcam' or 'reachy'")
        self.app = app
        self.camera = camera
        self.threshold = threshold
        self.voice_threshold = voice_threshold
        self.liveness = liveness
        self.listen_voice = listen_voice
        self._on_prompt = on_prompt
        self._audio = audio_source
        self._voice_embedder = voice_embedder
        self._vad = vad
        self._voice_gallery: dict[str, list[NDArray[np.float32]]] | None = None
        self.store = store or Store(data_dir, app)
        self._gallery: dict[str, list[NDArray[np.float32]]] | None = None
        self._arrived_cbs: list[PersonCallback] = []
        self._left_cbs: list[PersonCallback] = []
        self._arrive_frames = arrive_frames
        self._absence_seconds = absence_seconds
        self._tracker = PresenceTracker(arrive_frames, absence_seconds)
        self._latest: Person | None = None
        self._thread: threading.Thread | None = None
        self._voice_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._robot = robot
        self._embedder = face_embedder
        self._source = frame_source
        self._text_embedder = text_embedder
        self._live = HeadTurnLiveness()
        self._live_for: str | None = None
        self._verified_id: str | None = None
        self._verified_seen = 0.0
        self._last_prompt = float("-inf")
        self._voice_evidence: tuple[Evidence, float] | None = None
        self._face_center: tuple[tuple[float, float], float] | None = None

    # -- lazy components ---------------------------------------------------------------------

    def _get_embedder(self) -> FaceEmbedder:
        if self._embedder is None:
            self._embedder = SFaceEmbedder()
        return self._embedder

    def _get_source(self) -> FrameSource:
        if self._source is None:
            if self.camera == "reachy":
                self._source = ReachyFrameSource(self._get_robot())
            else:
                self._source = WebcamSource()
        return self._source

    def _get_robot(self) -> Any:
        """The connected ReachyMini; connects on first use if the app did not pass one."""
        if self._robot is None:
            self._robot = connect_robot()
        return self._robot

    def _get_audio(self) -> AudioSource:
        if self._audio is None:
            self._audio = (
                ReachyAudioSource(self._get_robot()) if self.camera == "reachy" else MicSource()
            )
        return self._audio

    def _get_vad(self) -> VoiceActivityDetector:
        if self._vad is None:
            self._vad = SileroVAD()
        return self._vad

    def _get_voice_embedder(self) -> VoiceEmbedder:
        if self._voice_embedder is None:
            self._voice_embedder = EcapaEmbedder()
        return self._voice_embedder

    def _say(self, message: str) -> None:
        if self._on_prompt is not None:
            self._on_prompt(message)

    def _observe(self) -> FaceObservation | None:
        """Read one frame and return what it shows. The frame is discarded immediately."""
        frame = self._get_source().read()
        if frame is None:
            return None
        obs = self._get_embedder().observe(frame)
        del frame  # never keep raw images
        return obs

    # -- enrollment --------------------------------------------------------------------------

    def enroll(
        self,
        name: str,
        consent: bool,
        on_prompt: Callable[[str], None] | None = None,
        *,
        voice: bool = True,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> Person:
        """Capture 5 face samples (plus a head turn) and ~5 s of voice, stored encrypted.

        Needs ``consent=True``. Pass ``voice=False`` to enroll by face only. With liveness on
        (the default) the person must turn their head, so a photo cannot be enrolled. Nothing
        is stored unless every step works.
        """
        if consent is not True:
            raise PermissionError("enroll() requires consent=True")
        if not name.strip():
            raise ValueError("name must not be empty")
        say = on_prompt or self._say

        samples = self._capture_face(say, _sleep)
        voice_samples = self._capture_voice(say) if voice else []

        person = self.store.add_person(name.strip(), consent=True)
        self.store.add_face_embeddings(person.id, samples)
        if voice_samples:
            self.store.add_voice_embeddings(person.id, voice_samples)
        self._gallery = None
        self._voice_gallery = None
        return person

    def _capture_face(
        self, say: Callable[[str], None], sleep: Callable[[float], None]
    ) -> list[NDArray[np.float32]]:
        samples: list[NDArray[np.float32]] = []
        live = HeadTurnLiveness()
        deadline = time.monotonic() + ENROLL_TIMEOUT_S
        challenged = False
        say("Look at the camera.")
        while len(samples) < FACE_SAMPLES or (self.liveness and not live.passed):
            if time.monotonic() > deadline:
                if len(samples) < FACE_SAMPLES:
                    raise EnrollmentError(
                        f"only captured {len(samples)}/{FACE_SAMPLES} samples; "
                        "make sure exactly one face is clearly visible"
                    )
                raise LivenessError("no head turn detected; a live person must turn their head")
            obs = self._observe()
            if obs is None:
                continue
            if len(samples) < FACE_SAMPLES:
                samples.append(obs.embedding)
                say(f"Captured sample {len(samples)}/{FACE_SAMPLES}")
                live.update(obs.yaw, time.monotonic())
                sleep(SAMPLE_GAP_S)
                continue
            if not challenged:
                say(CHALLENGE)
                challenged = True
            # Only count head turns by the same person who gave the samples.
            if cosine_similarity(obs.embedding, samples[0]) >= self.threshold:
                live.update(obs.yaw, time.monotonic())
        return samples

    def _capture_voice(self, say: Callable[[str], None]) -> list[NDArray[np.float32]]:
        """Record until there are enough seconds of speech; return voice embeddings."""
        audio, vad, embedder = self._get_audio(), self._get_vad(), self._get_voice_embedder()
        for _ in range(VOICE_ATTEMPTS):
            say(f"Say a few sentences out loud for {VOICE_RECORD_S:.0f} seconds.")
            recording = audio.record(VOICE_RECORD_S)
            speech = vad.speech_only(recording)
            del recording  # never keep raw audio
            if len(speech) >= MIN_ENROLL_SPEECH_S * SAMPLE_RATE:
                embeddings = [embedder.embed(w) for w in split_windows(speech)]
                if len(embeddings) > 1:  # also keep one embedding of the whole utterance
                    embeddings.append(embedder.embed(speech))
                say(f"Captured {len(embeddings)} voice samples")
                return embeddings
            say("I did not hear enough speech; let's try again.")
        raise EnrollmentError(
            f"need at least {MIN_ENROLL_SPEECH_S:.0f} s of speech; check the microphone"
        )

    # -- voice identification ----------------------------------------------------------------

    def _voice_evidence_for(self, speech: Audio) -> Evidence | None:
        if len(speech) < MIN_IDENTIFY_SPEECH_S * SAMPLE_RATE:
            return None
        if self._voice_gallery is None:
            self._voice_gallery = self.store.all_voice_embeddings()
        hit = match(
            self._get_voice_embedder().embed(speech), self._voice_gallery, self.voice_threshold
        )
        return Evidence(hit[0], hit[1], self.voice_threshold) if hit else None

    def identify_voice(
        self, audio: Audio | None = None, seconds: float = IDENTIFY_RECORD_S
    ) -> Person | None:
        """Identify the speaker from the microphone (or the given audio); None if unknown.

        Needs at least 1.5 s of detected speech. ``voice_threshold`` sets how strict the match is.
        """
        recording = audio if audio is not None else self._get_audio().record(seconds)
        speech = self._get_vad().speech_only(recording)
        del recording
        ev = self._voice_evidence_for(speech)
        fused = fuse(None, ev)
        return self._person(*fused) if fused else None

    def _person(self, person_id: str, confidence: float) -> Person | None:
        stored = self.store.get_person(person_id)
        return Person(stored.id, stored.name, confidence) if stored else None

    def _voice_loop(self) -> None:
        """Background listener: keeps the latest voice evidence fresh for fusion."""
        buffer = np.zeros(0, dtype=np.float32)
        while not self._stop.is_set():
            try:
                chunk = self._get_audio().record(VOICE_CHUNK_S)
                buffer = np.concatenate([buffer, chunk])[-int(VOICE_BUFFER_S * SAMPLE_RATE) :]
                ev = self._voice_evidence_for(self._get_vad().speech_only(buffer))
                if ev is not None:
                    with self._lock:
                        self._voice_evidence = (ev, time.monotonic())
            except Exception:
                log.exception("voice listener failed")
                self._stop.wait(1.0)

    def _fresh_voice(self, now: float) -> Evidence | None:
        with self._lock:
            held = self._voice_evidence
        return held[0] if held is not None and now - held[1] <= VOICE_TTL_S else None

    # -- face recognition, liveness and fusion -----------------------------------------------

    def _face_evidence(self, obs: FaceObservation | None) -> Evidence | None:
        if obs is None:
            return None
        if self._gallery is None:
            self._gallery = self.store.all_face_embeddings()
        hit = match(obs.embedding, self._gallery, self.threshold)
        return Evidence(hit[0], hit[1], self.threshold) if hit else None

    def _prompt_challenge(self, now: float) -> None:
        if now - self._last_prompt >= PROMPT_EVERY_S:
            self._last_prompt = now
            self._say(CHALLENGE)

    def _check_liveness(
        self, face: Evidence | None, obs: FaceObservation | None, now: float
    ) -> bool:
        """Update liveness state; return True if ``face`` may be trusted as a live person."""
        if face is None or obs is None:
            return False
        if not self.liveness or face.person_id == self._verified_id:
            if face.person_id == self._verified_id:
                self._verified_seen = now
            return True
        if face.person_id != self._live_for:
            self._live.reset()
            self._live_for = face.person_id
            self._last_prompt = float("-inf")
        self._prompt_challenge(now)
        if self._live.update(obs.yaw, now):
            self._verified_id, self._verified_seen = face.person_id, now
            return True
        return False

    def _expire_verification(self, now: float) -> None:
        if self._verified_id is not None and now - self._verified_seen > self._absence_seconds:
            self._verified_id = None
            self._live.reset()
            self._live_for = None

    def _decide(self, obs: FaceObservation | None, now: float) -> Person | None:
        """Face match + liveness + voice evidence -> the person (with fused confidence) or None."""
        self._expire_verification(now)
        face = self._face_evidence(obs)
        if not self._check_liveness(face, obs, now):
            face = None
        voice = self._fresh_voice(now)
        if face is None and voice is not None:
            # Voice alone cannot start a presence (it has no liveness); it only keeps an
            # already-verified person present while they are speaking off-camera.
            current = self._tracker.current
            if current is None or current.id != voice.person_id:
                voice = None
        fused = fuse(face, voice)
        if face is not None and obs is not None and obs.center is not None:
            with self._lock:
                self._face_center = (obs.center, time.monotonic())
        return self._person(*fused) if fused else None

    def _snapshot(self) -> Person | None:
        """One-shot recognition: look until a live, known face is confirmed or time runs out.

        Without liveness this inspects a single frame. With liveness it keeps watching: it gives up
        after ``SNAPSHOT_NO_FACE_S`` if no known face appears, or after the liveness window if a
        known face never turns its head (which is what a photo does).
        """
        start = time.monotonic()
        seen_known = False
        while True:
            now = time.monotonic()
            obs = self._observe()
            person = self._decide(obs, now)
            if person is not None:
                return person
            if not self.liveness:
                return None
            seen_known = seen_known or self._face_evidence(obs) is not None
            limit = self._live.window_s if seen_known else SNAPSHOT_NO_FACE_S
            if now - start > limit:
                return None

    def face_position(self, max_age_s: float = 1.0) -> tuple[int, int] | None:
        """Pixel ``(u, v)`` of the recognised, live face in the latest camera frame, if recent.

        Lets an app aim the robot's head at the person, e.g. from ``on_person_arrived``.
        """
        with self._lock:
            held = self._face_center
        if held is None or time.monotonic() - held[1] > max_age_s:
            return None
        return int(held[0][0]), int(held[0][1])

    def who_is_here(self) -> Person | None:
        """Return the person in view (name + confidence), or None if nobody known is there.

        While ``start()`` is running this returns the person currently tracked. Otherwise it
        watches the camera for a moment; with liveness on, the person must turn their head, so
        a call can take a few seconds.
        """
        if self._thread is not None:
            with self._lock:
                return self._latest
        return self._snapshot()

    # -- events ------------------------------------------------------------------------------

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
            now = time.monotonic()
            person = self._decide(self._observe(), now)
            with self._lock:
                events = self._tracker.update(person, now)
                self._latest = self._tracker.current
            for kind, who in events:
                self._fire(kind, who)
            self._stop.wait(LOOP_INTERVAL_S)

    def start(self) -> None:
        """Start background recognition and arrival/left events."""
        if self._thread is not None:
            return
        self._gallery = None
        self._voice_gallery = None
        self._tracker = PresenceTracker(self._arrive_frames, self._absence_seconds)
        self._live.reset()
        self._live_for = self._verified_id = None
        self._voice_evidence = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="cheta-recognize", daemon=True)
        self._thread.start()
        if self.listen_voice:
            self._voice_thread = threading.Thread(
                target=self._voice_loop, name="cheta-listen", daemon=True
            )
            self._voice_thread.start()

    def stop(self) -> None:
        """Stop background recognition."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5)
        if self._voice_thread is not None:
            self._voice_thread.join(timeout=VOICE_CHUNK_S + 3)
            self._voice_thread = None
        self._thread = None
        with self._lock:
            self._latest = None

    # -- memory ------------------------------------------------------------------------------

    def _get_text_embedder(self) -> TextEmbedder:
        if self._text_embedder is None:
            self._text_embedder = BgeEmbedder()
        return self._text_embedder

    def _require_person(self, person: Person) -> None:
        if self.store.get_person(person.id) is None:
            raise KeyError(f"unknown person {person.id!r}")

    def remember(self, person: Person, text: str, expires_in_days: int | None = None) -> None:
        """Store a fact about ``person``; it stops being recalled after ``expires_in_days``."""
        if not text.strip():
            raise ValueError("text must not be empty")
        if expires_in_days is not None and expires_in_days <= 0:
            raise ValueError("expires_in_days must be positive")
        self._require_person(person)
        expires_at = None if expires_in_days is None else time.time() + expires_in_days * 86400
        embedding = self._get_text_embedder().embed_document(text.strip())
        self.store.purge_expired()
        self.store.add_memory(person.id, text.strip(), embedding, expires_at)

    def recall(self, person: Person, query: str, k: int = 5) -> list[Memory]:
        """Return up to ``k`` unexpired memories of ``person``, most relevant first."""
        if k < 1:
            raise ValueError("k must be at least 1")
        self._require_person(person)
        query_embedding = self._get_text_embedder().embed_query(query)
        self.store.purge_expired()
        return self.store.search_memories(person.id, query_embedding, k)

    def forget(self, person: Person) -> None:
        """Hard-delete everything about ``person``: face and voice embeddings and memories."""
        self.store.delete_person(person.id)
        self._gallery = None
        self._voice_gallery = None
        with self._lock:
            if self._latest is not None and self._latest.id == person.id:
                self._latest = None
            if self._voice_evidence is not None and self._voice_evidence[0].person_id == person.id:
                self._voice_evidence = None
        if self._verified_id == person.id:
            self._verified_id = None

    def export(self, person: Person) -> dict[str, Any]:
        """Return everything stored about ``person`` as a JSON-serialisable dict.

        Biometric templates are reported as counts only; they are never exported.
        """
        stored = self.store.get_person(person.id)
        if stored is None:
            raise KeyError(f"unknown person {person.id!r}")
        return {
            "id": stored.id,
            "name": stored.name,
            "face_samples": len(self.store.face_embeddings(person.id)),
            "voice_samples": len(self.store.voice_embeddings(person.id)),
            "memories": [
                {"text": m.text, "created_at": m.created_at, "expires_at": m.expires_at}
                for m in self.store.list_memories(person.id)
            ],
        }

    def close(self) -> None:
        """Release the camera and database."""
        self.stop()
        if self._source is not None:
            self._source.close()
        self.store.close()
