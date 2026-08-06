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
from .robots import SO_ARM100, RobotProfile, available_robots, get_robot, register_robot
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
    "RobotProfile",
    "SO_ARM100",
    "SqliteEventLog",
    "StackCriterion",
    "StudyEvent",
    "StudyProtocol",
    "TaskSpec",
    "assign",
    "available_robots",
    "get_instrument",
    "get_robot",
    "load_instrument",
    "register_event_type",
    "register_robot",
    "summarize",
    "williams_square",
]
