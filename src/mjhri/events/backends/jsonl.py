"""Append-only JSONL EventLog — one JSON object per line.

The simplest durable log: human-readable, greppable, and trivially recoverable.
Good default for the native runner on a researcher's laptop.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from ...schemas.events import StudyEvent
from ..log import EventLog


class JsonlEventLog(EventLog):
    def __init__(self, path: str | Path, *, stamp_server_ts: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stamp = stamp_server_ts
        self._lock = threading.Lock()

    def emit(self, event: StudyEvent) -> None:
        if self._stamp and event.server_ts is None:
            event = event.model_copy(update={"server_ts": time.time()})
        line = json.dumps(event.model_dump(), separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def query(self, participant_id: Optional[str] = None) -> list[StudyEvent]:
        if not self.path.exists():
            return []
        out: list[StudyEvent] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = StudyEvent.model_validate_json(line)
                if participant_id in (None, ev.participant_id):
                    out.append(ev)
        out.sort(key=lambda e: (e.ts() if e.ts() is not None else float("inf")))
        return out
