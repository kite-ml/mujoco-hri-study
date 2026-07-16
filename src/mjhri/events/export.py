"""Anonymized export — turn the event log into analysis-ready tables.

Two shapes: a long **events** table (one row per event, payload JSON in a column)
and a wide **summary** table (one row per participant×condition, the measures from
``summarize``). Both key only on the participant id — no account identity ever
appears — so the export is safe to share as study data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ..schemas.events import StudyEvent
from .measures import summarize

_EVENT_COLUMNS = [
    "participant_id", "server_ts", "client_ts", "condition", "task",
    "variant", "phase", "event_type", "payload",
]


def event_rows(events: Iterable[StudyEvent]) -> list[dict[str, Any]]:
    rows = []
    for e in events:
        rows.append(
            {
                "participant_id": e.participant_id,
                "server_ts": e.server_ts,
                "client_ts": e.client_ts,
                "condition": e.condition,
                "task": e.task,
                "variant": e.variant,
                "phase": e.phase,
                "event_type": e.event_type,
                "payload": json.dumps(e.payload, separators=(",", ":")),
            }
        )
    return rows


def write_events_csv(events: Iterable[StudyEvent], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_EVENT_COLUMNS)
        w.writeheader()
        w.writerows(event_rows(events))
    return path


def summary_rows(events_by_participant: dict[str, list[StudyEvent]]) -> list[dict[str, Any]]:
    """One flat row per participant×condition, suitable for stats software."""
    rows: list[dict[str, Any]] = []
    for pid, events in events_by_participant.items():
        summ = summarize(events)
        for cond, c in summ["conditions"].items():
            deploy = c.get("deploy") or {}
            rows.append(
                {
                    "participant_id": pid,
                    "condition": cond,
                    "complete": summ["complete"],
                    "teach_time_s": c.get("teach_time"),
                    "time_to_deployable_spec_s": c.get("time_to_deployable_spec"),
                    "success_rate": c.get("success_rate"),
                    "deploy": deploy.get("deploy"),
                    "deploy_confidence": deploy.get("confidence"),
                    "n_spec_edits": c.get("event_counts", {}).get("spec_edit"),
                    "n_teleop_episodes": c.get("event_counts", {}).get("teleop_episode_end"),
                }
            )
    return rows


def write_summary_csv(events_by_participant: dict[str, list[StudyEvent]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(events_by_participant)
    fields = list(rows[0].keys()) if rows else [
        "participant_id", "condition", "complete", "teach_time_s",
        "time_to_deployable_spec_s", "success_rate", "deploy", "deploy_confidence",
        "n_spec_edits", "n_teleop_episodes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path
