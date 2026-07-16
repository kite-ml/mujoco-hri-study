"""SQLite EventLog — durable, concurrent-reader-friendly, still a single file.

Good when a native-runner study accumulates many participants/events and you want
indexed queries without standing up a database.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from ...schemas.events import StudyEvent
from ..log import EventLog

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id TEXT NOT NULL,
    event_type     TEXT NOT NULL,
    phase          TEXT,
    condition      TEXT,
    task           TEXT,
    variant        TEXT,
    payload        TEXT NOT NULL DEFAULT '{}',
    client_ts      REAL,
    server_ts      REAL,
    schema_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_events_participant ON events(participant_id);
"""


class SqliteEventLog(EventLog):
    def __init__(self, path: str | Path, *, stamp_server_ts: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stamp = stamp_server_ts
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def emit(self, event: StudyEvent) -> None:
        server_ts = event.server_ts
        if self._stamp and server_ts is None:
            server_ts = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (participant_id, event_type, phase, condition, task, "
                "variant, payload, client_ts, server_ts, schema_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    event.participant_id, event.event_type, event.phase, event.condition,
                    event.task, event.variant, json.dumps(event.payload),
                    event.client_ts, server_ts, event.schema_version,
                ),
            )
            self._conn.commit()

    def query(self, participant_id: Optional[str] = None) -> list[StudyEvent]:
        sql = (
            "SELECT participant_id, event_type, phase, condition, task, variant, "
            "payload, client_ts, server_ts, schema_version FROM events"
        )
        args: tuple = ()
        if participant_id is not None:
            sql += " WHERE participant_id = ?"
            args = (participant_id,)
        # Order by best-available timestamp, then insertion id for stability.
        sql += " ORDER BY COALESCE(server_ts, client_ts, 1e18), id"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out: list[StudyEvent] = []
        for r in rows:
            out.append(
                StudyEvent(
                    participant_id=r[0], event_type=r[1], phase=r[2], condition=r[3],
                    task=r[4], variant=r[5], payload=json.loads(r[6]),
                    client_ts=r[7], server_ts=r[8], schema_version=r[9],
                )
            )
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()
