"""Events: the ``EventLog`` interface hosts implement, plus built-in local backends,
derived study measures, and anonymized export."""

from .backends.jsonl import JsonlEventLog
from .backends.sqlite import SqliteEventLog
from .log import EventLog, InMemoryEventLog
from .measures import summarize

__all__ = [
    "EventLog",
    "InMemoryEventLog",
    "JsonlEventLog",
    "SqliteEventLog",
    "summarize",
]
