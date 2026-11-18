"""Public data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Person:
    """A known person. ``confidence`` is the recognition score in [0, 1]."""

    id: str
    name: str
    confidence: float = 1.0
