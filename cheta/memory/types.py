"""Memory data type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Memory:
    """A remembered fact about a person. ``score`` is cosine similarity to the recall query."""

    id: int
    person_id: str
    text: str
    created_at: float
    expires_at: float | None = None
    score: float = 0.0
