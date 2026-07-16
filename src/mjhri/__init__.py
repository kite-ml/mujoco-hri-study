"""mjhri: experiment engine for HRI studies in MuJoCo.

Importing ``mjhri`` pulls in the schema + protocol + events + instruments layers,
which depend only on pydantic (no mujoco). The scoring/teleop layers under
``mjhri.tasks`` / ``mjhri.teleop`` import mujoco lazily — import them directly when
you need them.
"""

__version__ = "0.1.0"

from .events import EventLog, InMemoryEventLog, JsonlEventLog, SqliteEventLog
from .events.measures import summarize
from .instruments import get_instrument, load_instrument
from .protocol import Assignment, ProtocolEngine, ProtocolState, assign, williams_square
from .schemas.events import CORE_EVENT_TYPES, StudyEvent, register_event_type
from .schemas.instruments import Instrument, InstrumentItem
from .schemas.protocol import Condition, CounterbalanceSpec, PhaseSpec, StudyProtocol
from .schemas.tasks import (
    RegionCriterion,
    ResetRandomization,
    StackCriterion,
    TaskSpec,
)

__all__ = [
    "CORE_EVENT_TYPES",
    "Assignment",
    "Condition",
    "CounterbalanceSpec",
    "EventLog",
    "InMemoryEventLog",
    "Instrument",
    "InstrumentItem",
    "JsonlEventLog",
    "PhaseSpec",
    "ProtocolEngine",
    "ProtocolState",
    "RegionCriterion",
    "ResetRandomization",
    "SqliteEventLog",
    "StackCriterion",
    "StudyEvent",
    "StudyProtocol",
    "TaskSpec",
    "assign",
    "get_instrument",
    "load_instrument",
    "register_event_type",
    "summarize",
    "williams_square",
]
