"""Study event schema + an extensible registry of event types.

The event log *is* the analysis dataset and the source of truth for protocol
state, so every meaningful moment is an event. Event types are strings; the core
set below is what the engine and the ``measures`` module understand, and studies
may register their own without touching this package.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

#: Event types the engine + measures understand out of the box.
CORE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "session_start",
        "phase_enter",
        "phase_exit",
        "practice_start",
        "practice_end",
        "teleop_episode_start",
        "teleop_episode_end",
        "spec_edit",
        "spec_validated",
        "spec_marked_deployable",
        "generation_requested",
        "generation_completed",
        "train_started",
        "train_completed",
        "train_fallback_used",
        "policy_play_start",
        "policy_play_end",
        "survey_submitted",
        "eval_completed",
        "deploy_decision",
        "session_complete",
    }
)

_REGISTERED: set[str] = set(CORE_EVENT_TYPES)


def register_event_type(name: str) -> None:
    """Register a study-specific event type so tooling recognises it."""
    _REGISTERED.add(name)


def known_event_types() -> frozenset[str]:
    """All event types currently recognised (core + registered)."""
    return frozenset(_REGISTERED)


class StudyEvent(BaseModel):
    """One logged moment. ``server_ts`` is authoritative for timing; ``client_ts``
    is the browser/native clock and may be absent or skewed."""

    participant_id: str
    event_type: str
    phase: Optional[str] = None
    condition: Optional[str] = None
    task: Optional[str] = None
    variant: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    client_ts: Optional[float] = None
    server_ts: Optional[float] = None
    schema_version: int = SCHEMA_VERSION

    def ts(self) -> Optional[float]:
        """Best available timestamp: server clock preferred, client as fallback."""
        return self.server_ts if self.server_ts is not None else self.client_ts
