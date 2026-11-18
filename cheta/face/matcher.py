"""Match a face embedding against enrolled galleries by cosine similarity."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DEFAULT_THRESHOLD = 0.4  # SFace's documented cosine threshold is 0.363; slightly stricter here


def match(
    query: NDArray[np.float32],
    gallery: dict[str, list[NDArray[np.float32]]],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str, float] | None:
    """Return ``(person_id, similarity)`` of the best match at or above ``threshold``, else None.

    A person's score is the highest cosine similarity across their stored embeddings.
    """
    q = query / (np.linalg.norm(query) or 1.0)
    best_id: str | None = None
    best = -1.0
    for person_id, embeddings in gallery.items():
        for emb in embeddings:
            score = float(np.dot(q, emb / (np.linalg.norm(emb) or 1.0)))
            if score > best:
                best_id, best = person_id, score
    if best_id is None or best < threshold:
        return None
    return best_id, best
