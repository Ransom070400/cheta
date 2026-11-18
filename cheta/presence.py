"""Debounced arrival/departure detection from a stream of per-frame recognitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from cheta.person import Person

ARRIVE_FRAMES = 2  # consecutive recognitions of the same person before "arrived"
ABSENCE_SECONDS = 3.0  # time without seeing the person before "left"


@dataclass
class PresenceTracker:
    """Turns noisy recognitions into arrived/left events. Time is passed in for testability."""

    arrive_frames: int = ARRIVE_FRAMES
    absence_seconds: float = ABSENCE_SECONDS
    current: Person | None = None
    _last_seen: float = 0.0
    _candidate: str | None = None
    _count: int = 0
    events: list[tuple[str, Person]] = field(default_factory=list)

    def update(self, person: Person | None, now: float) -> list[tuple[str, Person]]:
        """Feed one observation (None = nobody known). Returns new ``("arrived"|"left", person)``."""
        out: list[tuple[str, Person]] = []
        if person is not None:
            if self.current is not None and person.id == self.current.id:
                self._last_seen = now
                self.current = person
                self._candidate, self._count = None, 0
            else:
                if self._candidate == person.id:
                    self._count += 1
                else:
                    self._candidate, self._count = person.id, 1
                if self._count >= self.arrive_frames:
                    if self.current is not None:
                        out.append(("left", self.current))
                    out.append(("arrived", person))
                    self.current, self._last_seen = person, now
                    self._candidate, self._count = None, 0
        else:
            self._candidate, self._count = None, 0
            if self.current is not None and now - self._last_seen > self.absence_seconds:
                out.append(("left", self.current))
                self.current = None
        return out
