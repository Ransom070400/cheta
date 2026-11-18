from cheta.person import Person
from cheta.presence import PresenceTracker

A, B = Person("a", "Ann", 0.9), Person("b", "Bob", 0.8)


def kinds(ev: list[tuple[str, Person]]) -> list[tuple[str, str]]:
    return [(k, p.name) for k, p in ev]


def test_arrival_is_debounced() -> None:
    t = PresenceTracker(arrive_frames=2)
    assert t.update(A, 0.0) == []
    assert kinds(t.update(A, 0.1)) == [("arrived", "Ann")]
    assert t.update(A, 0.2) == []  # no repeat


def test_single_blip_does_not_arrive() -> None:
    t = PresenceTracker(arrive_frames=2)
    t.update(A, 0.0)
    t.update(None, 0.1)
    assert t.update(A, 0.2) == []


def test_left_after_absence() -> None:
    t = PresenceTracker(arrive_frames=1, absence_seconds=3.0)
    t.update(A, 0.0)
    assert t.update(None, 2.0) == []  # still within grace period
    assert kinds(t.update(None, 3.5)) == [("left", "Ann")]
    assert t.current is None


def test_switch_person_emits_left_then_arrived() -> None:
    t = PresenceTracker(arrive_frames=1)
    t.update(A, 0.0)
    assert kinds(t.update(B, 0.1)) == [("left", "Ann"), ("arrived", "Bob")]
