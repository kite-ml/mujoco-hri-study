"""The ``EventLog`` interface — the one thing a host must implement.

A host (e.g. Kite → Postgres) subclasses ``EventLog`` and points it at its own
storage. The built-in ``JsonlEventLog`` / ``SqliteEventLog`` back the native runner
and tests. ``InMemoryEventLog`` is a trivial reference implementation.

Contract: ``emit`` persists one event; ``query`` returns events for a participant
(or all) in chronological order. Timestamps are the store's business — a store that
has a trustworthy server clock should stamp ``server_ts`` on emit if unset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from ..schemas.events import StudyEvent


class EventLog(ABC):
    @abstractmethod
    def emit(self, event: StudyEvent) -> None:
        """Persist a single event."""

    @abstractmethod
    def query(self, participant_id: Optional[str] = None) -> list[StudyEvent]:
        """Return events (for one participant, or all) in chronological order."""

    def emit_many(self, events: Iterable[StudyEvent]) -> None:
        for e in events:
            self.emit(e)

    def participants(self) -> list[str]:
        """Distinct participant ids seen, in first-appearance order."""
        seen: dict[str, None] = {}
        for e in self.query():
            seen.setdefault(e.participant_id, None)
        return list(seen)


class InMemoryEventLog(EventLog):
    """Reference implementation — keeps events in a list. Not durable."""

    def __init__(self) -> None:
        self._events: list[StudyEvent] = []

    def emit(self, event: StudyEvent) -> None:
        self._events.append(event)

    def query(self, participant_id: Optional[str] = None) -> list[StudyEvent]:
        evs = [e for e in self._events if participant_id in (None, e.participant_id)]
        return sorted(
            evs, key=lambda e: (e.ts() if e.ts() is not None else float("inf"))
        )
