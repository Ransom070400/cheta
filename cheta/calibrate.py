"""Pick a voice threshold from your own recordings.

Speaker-embedding scores depend on the microphone, the room and the people, so no single default
fits everyone. Record a few clips of each person, then measure how high *genuine* scores (a clip
against the same person) and *impostor* scores (a clip against everyone else) actually are.
``recommend_threshold`` picks the lowest threshold that keeps false accepts under your target.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cheta.voice import (
    SAMPLE_RATE,
    Audio,
    VoiceActivityDetector,
    VoiceEmbedder,
    split_windows,
)

MIN_SPEECH_S = 1.5


@dataclass(frozen=True)
class Recommendation:
    """A suggested threshold and what it would have done on the recorded data."""

    threshold: float
    false_accept_rate: float  # impostor probes scoring at or above the threshold
    false_reject_rate: float  # genuine probes scoring below it
    n_genuine: int
    n_impostor: int


def recommend_threshold(
    genuine: Sequence[float], impostor: Sequence[float], max_far: float = 0.01
) -> Recommendation:
    """Lowest threshold whose false-accept rate on ``impostor`` is at most ``max_far``.

    Lower thresholds reject fewer genuine users, so the lowest one that meets the false-accept
    target is the best. With few impostor scores the rate is coarse; collect more clips.
    """
    if not genuine or not impostor:
        raise ValueError("need at least one genuine and one impostor score")
    if not 0.0 <= max_far < 1.0:
        raise ValueError("max_far must be in [0, 1)")
    gen = np.sort(np.asarray(genuine, dtype=np.float64))
    imp = np.sort(np.asarray(impostor, dtype=np.float64))
    # Candidates: just above each impostor score, so "score >= threshold" excludes it.
    candidates = np.unique(imp + 1e-6)
    chosen = next(float(t) for t in candidates if float(np.mean(imp >= t)) <= max_far)
    chosen = math.ceil(chosen * 1000) / 1000  # round *up* so the false-accept target still holds
    return Recommendation(
        threshold=chosen,
        false_accept_rate=float(np.mean(imp >= chosen)),
        false_reject_rate=float(np.mean(gen < chosen)),
        n_genuine=len(gen),
        n_impostor=len(imp),
    )


def _n(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return x / (np.linalg.norm(x) or 1.0)


def score_voices(
    enrol: Mapping[str, Audio],
    probes: Mapping[str, Sequence[Audio]],
    vad: VoiceActivityDetector,
    embedder: VoiceEmbedder,
) -> tuple[list[float], list[float]]:
    """Return ``(genuine, impostor)`` scores the way ``Identity`` would produce them.

    Each person is enrolled from their ``enrol`` clip (windows plus the whole utterance, as in
    ``enroll()``). Every probe clip is split into windows; a window's score for a person is its
    best cosine similarity to that person's samples. The genuine score is the window against its
    own person, the impostor score is the *best* other person (the open-set worst case).
    """
    if len(enrol) < 2:
        raise ValueError("need at least two people to measure impostor scores")
    templates: dict[str, list[NDArray[np.float32]]] = {}
    for name, clip in enrol.items():
        speech = vad.speech_only(clip)
        if len(speech) < MIN_SPEECH_S * SAMPLE_RATE:
            raise ValueError(f"not enough speech in the enrolment clip for {name!r}")
        embs = [_n(embedder.embed(w)) for w in split_windows(speech)]
        if len(embs) > 1:
            embs.append(_n(embedder.embed(speech)))
        templates[name] = embs
    genuine: list[float] = []
    impostor: list[float] = []
    for name, clips in probes.items():
        if name not in templates:
            raise ValueError(f"probe clips for {name!r} have no enrolment clip")
        for clip in clips:
            speech = vad.speech_only(clip)
            if len(speech) < MIN_SPEECH_S * SAMPLE_RATE:
                continue
            for window in split_windows(speech):
                q = _n(embedder.embed(window))
                scores = {n: max(float(q @ t) for t in ts) for n, ts in templates.items()}
                genuine.append(scores[name])
                impostor.append(max(s for n, s in scores.items() if n != name))
    if not genuine:
        raise ValueError("no probe clip had enough speech")
    return genuine, impostor
