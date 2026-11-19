"""Combine face and voice evidence into one identity decision.

Each modality's cosine similarity is first normalised so its own threshold maps to 0 and a
perfect match to 1. Then:

* face and voice agree on the same person: confidence = 1 - (1 - face) * (1 - voice), so two
  independent signals are stronger than either alone;
* only one modality is available: its normalised score times ``single_factor`` (default
  ``SINGLE_MODALITY_FACTOR``, a lower confidence since there is nothing to corroborate it;
  pass 1.0 for no penalty);
* they name different people: no decision (None), because a conflict is safer than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

SINGLE_MODALITY_FACTOR = 0.8


@dataclass(frozen=True)
class Evidence:
    """One modality's best match: who, the raw cosine similarity, and that modality's threshold."""

    person_id: str
    score: float
    threshold: float

    @property
    def normalised(self) -> float:
        """Score rescaled so the threshold is 0 and a perfect match is 1."""
        if self.threshold >= 1.0:
            return 0.0
        return min(1.0, max(0.0, (self.score - self.threshold) / (1.0 - self.threshold)))


def fuse(
    face: Evidence | None,
    voice: Evidence | None,
    single_factor: float = SINGLE_MODALITY_FACTOR,
) -> tuple[str, float] | None:
    """Return ``(person_id, confidence)``, or None if there is no evidence or it conflicts."""
    if not 0.0 < single_factor <= 1.0:
        raise ValueError("single_factor must be in (0, 1]")
    if face is None and voice is None:
        return None
    if face is not None and voice is not None:
        if face.person_id != voice.person_id:
            return None
        confidence = 1.0 - (1.0 - face.normalised) * (1.0 - voice.normalised)
        return face.person_id, confidence
    only = face if face is not None else voice
    assert only is not None
    return only.person_id, only.normalised * single_factor
