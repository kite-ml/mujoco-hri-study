"""Protocol: deterministic counterbalancing + an event-sourced phase state machine."""

from .counterbalance import Assignment, assign, williams_square
from .engine import ProtocolEngine, ProtocolState

__all__ = [
    "Assignment",
    "ProtocolEngine",
    "ProtocolState",
    "assign",
    "williams_square",
]
