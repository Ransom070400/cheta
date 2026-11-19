"""Head-turn liveness: a live head turning moves the nose relative to the eyes; a photo does not.

This is a basic check against printed photos and photos shown on a phone screen. It does not
defend against video replays, masks or deepfakes.
"""

from __future__ import annotations

from collections import deque
from statistics import median

TURN_RANGE = 0.35  # required spread of the yaw proxy; a flat photo moved by hand stays < 0.2
WINDOW_S = 6.0  # the turn must happen within this many seconds
MIN_SAMPLES = 6
SMOOTH = 3  # median filter width, so a single noisy landmark frame cannot pass the check

CHALLENGE = "Please slowly turn your head to the left, then to the right."


class HeadTurnLiveness:
    """Feed one yaw value per frame; ``passed`` latches once a real head turn is seen."""

    def __init__(
        self,
        turn_range: float = TURN_RANGE,
        window_s: float = WINDOW_S,
        min_samples: int = MIN_SAMPLES,
    ) -> None:
        self.turn_range = turn_range
        self.window_s = window_s
        self.min_samples = min_samples
        self._samples: deque[tuple[float, float]] = deque()
        self.passed = False

    def update(self, yaw: float, now: float) -> bool:
        """Add an observation taken at time ``now`` (seconds); return whether liveness passed."""
        if self.passed:
            return True
        self._samples.append((now, yaw))
        while self._samples and now - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        if len(self._samples) >= max(self.min_samples, SMOOTH):
            ys = [y for _, y in self._samples]
            smoothed = [median(ys[i : i + SMOOTH]) for i in range(len(ys) - SMOOTH + 1)]
            if max(smoothed) - min(smoothed) >= self.turn_range:
                self.passed = True
        return self.passed

    def reset(self) -> None:
        """Forget everything and require a fresh turn."""
        self._samples.clear()
        self.passed = False
