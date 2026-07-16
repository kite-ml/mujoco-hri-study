"""Derived study measures — the dependent variables, computed from the event log.

Everything here is a pure function of a participant's events, so the analysis is
reproducible from the exported log alone (no hidden state, no separate metrics
pipeline). Study-specific event types are read defensively: a measure that has no
supporting events returns ``None`` rather than raising.
"""

from __future__ import annotations

from typing import Any, Optional

from ..schemas.events import StudyEvent


def _ordered(events: list[StudyEvent]) -> list[StudyEvent]:
    return sorted(events, key=lambda e: (e.ts() if e.ts() is not None else float("inf")))


def phase_durations(events: list[StudyEvent]) -> dict[tuple[Optional[str], Optional[str]], float]:
    """``{(condition, phase): seconds}`` from matched phase_enter/phase_exit."""
    enters: dict[tuple, float] = {}
    out: dict[tuple, float] = {}
    for e in _ordered(events):
        key = (e.condition, e.phase)
        if e.event_type == "phase_enter" and e.ts() is not None:
            enters[key] = e.ts()
        elif e.event_type == "phase_exit" and key in enters and e.ts() is not None:
            out[key] = e.ts() - enters[key]
    return out


def event_counts(events: list[StudyEvent], condition: Optional[str] = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        if condition is not None and e.condition != condition:
            continue
        counts[e.event_type] = counts.get(e.event_type, 0) + 1
    return counts


def time_to_deployable_spec(events: list[StudyEvent], condition: str) -> Optional[float]:
    """Seconds from entering the spec phase to marking the spec deployable, for one
    condition. ``None`` if the participant never reached a deployable spec."""
    ordered = _ordered([e for e in events if e.condition == condition])
    marked = next((e for e in ordered if e.event_type == "spec_marked_deployable"), None)
    if marked is None or marked.ts() is None:
        return None
    enter = next(
        (e for e in ordered if e.event_type == "phase_enter" and e.phase == marked.phase),
        None,
    )
    if enter is None or enter.ts() is None:
        return None
    return marked.ts() - enter.ts()


def deploy_decisions(events: list[StudyEvent]) -> dict[str, dict[str, Any]]:
    """``{condition: {"deploy": bool, "confidence": number, ...}}`` from the last
    deploy_decision event in each condition (its payload passed through)."""
    out: dict[str, dict[str, Any]] = {}
    for e in _ordered(events):
        if e.event_type == "deploy_decision" and e.condition is not None:
            out[e.condition] = dict(e.payload)
    return out


def success_rates(events: list[StudyEvent]) -> dict[str, float]:
    """``{condition: success_rate}`` from the last eval_completed per condition."""
    out: dict[str, float] = {}
    for e in _ordered(events):
        if e.event_type == "eval_completed" and e.condition is not None:
            rate = e.payload.get("success_rate")
            if isinstance(rate, (int, float)):
                out[e.condition] = float(rate)
    return out


def survey_responses(events: list[StudyEvent]) -> list[dict[str, Any]]:
    """One row per submitted survey: condition, phase, instrument, and responses."""
    rows: list[dict[str, Any]] = []
    for e in _ordered(events):
        if e.event_type == "survey_submitted":
            rows.append(
                {
                    "condition": e.condition,
                    "phase": e.phase,
                    "instrument": e.payload.get("instrument"),
                    "responses": e.payload.get("responses", {}),
                    "ts": e.ts(),
                }
            )
    return rows


def summarize(events: list[StudyEvent]) -> dict[str, Any]:
    """A per-participant summary keyed by condition, plus study-level fields.

    Returns ``{participant_id, complete, conditions: {cond: {...}}, surveys: [...]}``
    where each condition carries phase durations, event counts, time-to-deployable-spec,
    the deploy decision, and the objective success rate when present.
    """
    if not events:
        return {"participant_id": None, "complete": False, "conditions": {}, "surveys": []}
    pid = events[0].participant_id
    durations = phase_durations(events)
    ttds_by_cond = {}
    conditions = sorted({e.condition for e in events if e.condition is not None})
    deploys = deploy_decisions(events)
    rates = success_rates(events)
    per_cond: dict[str, Any] = {}
    for cond in conditions:
        cond_durations = {ph: sec for (c, ph), sec in durations.items() if c == cond}
        per_cond[cond] = {
            "phase_durations": cond_durations,
            "teach_time": cond_durations.get("teach"),
            "event_counts": event_counts(events, cond),
            "time_to_deployable_spec": time_to_deployable_spec(events, cond),
            "deploy": deploys.get(cond),
            "success_rate": rates.get(cond),
        }
    return {
        "participant_id": pid,
        "complete": any(e.event_type == "session_complete" for e in events),
        "conditions": per_cond,
        "surveys": survey_responses(events),
    }
