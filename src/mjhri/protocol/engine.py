"""Event-sourced protocol state machine.

The engine holds **no mutable state**: the current phase is *derived* from the event
log every time. That is what makes a session resume correctly after a browser
refresh, a crash, or a handoff between the web host and the native runner — the log
is the single source of truth, and two processes reading the same log agree on where
the participant is.

A study is a flat sequence of *steps* — the protocol's phases repeated once per
condition, in the participant's assigned condition order. Each step is opened by a
``phase_enter`` event and closed by a ``phase_exit``. Advancing is gated: a phase can
declare required events (e.g. ``spec_marked_deployable``) and required instruments
(surveys) that must appear before its ``phase_exit`` is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..schemas.events import StudyEvent
from ..schemas.protocol import PhaseSpec, StudyProtocol
from .counterbalance import Assignment


class GateError(RuntimeError):
    """Raised when advancing is attempted before the current phase's gates pass."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__("; ".join(missing) or "phase gate not satisfied")


@dataclass(frozen=True)
class ProtocolState:
    participant_id: str
    step_index: int
    condition_index: int
    phase_index: int
    condition_id: Optional[str]
    phase_name: Optional[str]
    entered: bool
    is_complete: bool
    total_steps: int


class ProtocolEngine:
    def __init__(self, protocol: StudyProtocol, assignment: Assignment):
        self.protocol = protocol
        self.assignment = assignment
        # Flattened step list, ordered by phase pass: every run completes its 'main'
        # phases first, then every run its 'deploy' phases (async-training two-visit
        # flow). With no 'deploy' phases this is a straight run-by-run sequence.
        main_phases = [p for p in protocol.phases if getattr(p, "group", "main") != "deploy"]
        deploy_phases = [p for p in protocol.phases if getattr(p, "group", "main") == "deploy"]
        self.steps: list[tuple[str, PhaseSpec]] = [
            (run_id, phase) for run_id in assignment.condition_order for phase in main_phases
        ] + [
            (run_id, phase) for run_id in assignment.condition_order for phase in deploy_phases
        ]

    # -- state derivation -------------------------------------------------

    @staticmethod
    def _ordered(events: list[StudyEvent]) -> list[StudyEvent]:
        """Chronological order, robust to unstamped events.

        A store that stamps ``server_ts`` (Postgres, JSONL, SQLite) gives a total
        order directly. A store that doesn't (the in-memory reference) leaves engine
        events unstamped — so an unstamped event inherits the timestamp of the last
        stamped event before it, preserving causal (insertion) order instead of being
        shoved to the end. Ties break on original index (stable)."""
        keyed = []
        last = float("-inf")
        for i, e in enumerate(events):
            ts = e.ts()
            if ts is not None:
                last = ts
            keyed.append(((ts if ts is not None else last), i, e))
        keyed.sort(key=lambda t: (t[0], t[1]))
        return [e for _, _, e in keyed]

    def state(self, events: list[StudyEvent]) -> ProtocolState:
        pid = self.assignment.participant_id
        total = len(self.steps)
        nphases = max(1, len(self.protocol.phases))
        n_exit = sum(1 for e in events if e.event_type == "phase_exit")
        n_enter = sum(1 for e in events if e.event_type == "phase_enter")
        if total == 0 or n_exit >= total:
            return ProtocolState(pid, total, len(self.assignment.condition_order),
                                 0, None, None, False, True, total)
        step_index = n_exit
        cond_id, phase = self.steps[step_index]
        return ProtocolState(
            participant_id=pid,
            step_index=step_index,
            condition_index=step_index // nphases,
            phase_index=step_index % nphases,
            condition_id=cond_id,
            phase_name=phase.name,
            entered=n_enter > n_exit,
            is_complete=False,
            total_steps=total,
        )

    def _current_phase_events(self, events: list[StudyEvent]) -> list[StudyEvent]:
        """Events logged since the current phase was entered (its occurrence)."""
        ordered = self._ordered(events)
        last_enter = max(
            (i for i, e in enumerate(ordered) if e.event_type == "phase_enter"),
            default=None,
        )
        return ordered[last_enter + 1:] if last_enter is not None else []

    # -- gates + advancement ----------------------------------------------

    def requirements(self, step_index: int) -> tuple[list[str], list[str]]:
        """Effective (events, instruments) gate for a step: the phase's requirements
        plus any the run spec adds for that phase (``extra_events`` / ``extra_instruments``
        keyed by phase name) — used to place, e.g., a NASA-TLX battery once per method
        rather than on every run. A run spec with ``skip_instruments: true`` waives ALL
        instrument gates for its steps (e.g. a demonstration/practice run)."""
        run_id, phase = self.steps[step_index]
        spec = self.assignment.task_by_condition.get(run_id, {})
        events = list(phase.requires_events) + list(spec.get("extra_events", {}).get(phase.name, []))
        if spec.get("skip_instruments"):
            return events, []
        insts = list(phase.requires_instruments) + list(spec.get("extra_instruments", {}).get(phase.name, []))
        return events, insts

    def can_advance(self, events: list[StudyEvent]) -> tuple[bool, list[str]]:
        state = self.state(events)
        if state.is_complete:
            return False, ["study already complete"]
        if not state.entered:
            return False, ["current phase not entered"]
        req_events, req_insts = self.requirements(state.step_index)
        phase_events = self._current_phase_events(events)
        have_types = {e.event_type for e in phase_events}
        submitted = {
            e.payload.get("instrument")
            for e in phase_events
            if e.event_type == "survey_submitted"
        }
        missing: list[str] = []
        for et in req_events:
            if et not in have_types:
                missing.append(f"requires event '{et}'")
        for iid in req_insts:
            if iid not in submitted:
                missing.append(f"requires instrument '{iid}'")
        return (not missing), missing

    def _event_for(self, step_index: int, event_type: str, **payload) -> StudyEvent:
        run_id, phase = self.steps[step_index]
        spec = self.assignment.task_by_condition.get(run_id, {})
        # A run spec may carry an explicit "condition" (the teaching method) distinct
        # from the run id; fall back to the run id when it doesn't (backward compat).
        return StudyEvent(
            participant_id=self.assignment.participant_id,
            event_type=event_type,
            phase=phase.name,
            condition=spec.get("condition", run_id),
            task=spec.get("task"),
            variant=spec.get("variant"),
            payload={**payload, "run": run_id},
        )

    def start(self, events: list[StudyEvent]) -> list[StudyEvent]:
        """Events to open the study (session_start + first phase_enter). Empty if it
        has already started — safe to call on every resume."""
        if not self.steps:
            return []
        if any(e.event_type in ("phase_enter", "session_start") for e in events):
            return []
        return [
            StudyEvent(
                participant_id=self.assignment.participant_id,
                event_type="session_start",
                payload={"assignment": self.assignment.as_dict()},
            ),
            self._event_for(0, "phase_enter"),
        ]

    def advance(self, events: list[StudyEvent]) -> list[StudyEvent]:
        """Events to close the current phase and open the next (or complete the
        study). Raises ``GateError`` if the current phase's gates are unmet."""
        ok, missing = self.can_advance(events)
        if not ok:
            raise GateError(missing)
        state = self.state(events)
        out = [self._event_for(state.step_index, "phase_exit")]
        nxt = state.step_index + 1
        if nxt < len(self.steps):
            out.append(self._event_for(nxt, "phase_enter"))
        else:
            out.append(
                StudyEvent(
                    participant_id=self.assignment.participant_id,
                    event_type="session_complete",
                )
            )
        return out
