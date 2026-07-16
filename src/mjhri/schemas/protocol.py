"""Study protocol schema — the experiment design expressed as data.

A new study is (mostly) a new ``StudyProtocol`` + task specs + scene XMLs; no code
change. The engine (``mjhri.protocol.engine``) walks a participant through the
``phases`` for each assigned condition, enforcing the gates declared here.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

PhaseKind = Literal[
    "instruction", "practice", "teach", "train", "review", "survey", "decision"
]


class PhaseSpec(BaseModel):
    """One phase in the per-run flow, with the gates that block advancing.

    ``group`` orders the study into passes: every run completes its non-``deploy``
    phases first, then every run its ``deploy`` phases. That models "teach all runs
    now, come back and deploy all runs later" for long asynchronous training — the
    default single group reproduces a straight run-by-run flow.
    """

    name: str
    label: str = ""
    kind: PhaseKind = "teach"
    group: str = Field("main", description="phase pass; 'deploy' runs after all 'main' phases")
    requires_events: list[str] = Field(
        default_factory=list,
        description="event_types that must be logged in this phase before advancing",
    )
    requires_instruments: list[str] = Field(
        default_factory=list,
        description="instrument ids that must be submitted before advancing",
    )
    optional: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Condition(BaseModel):
    """A teaching modality (study arm)."""

    id: str = Field(description="short key, e.g. 'A'")
    label: str = Field(description="participant-facing name, e.g. 'Direct teleoperation'")
    modality: str = Field(description="machine key, e.g. 'teleop' | 'reward_spec' | 'description'")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CounterbalanceSpec(BaseModel):
    """How condition order and task assignment vary across participants."""

    method: Literal["williams", "latin_square", "fixed"] = "williams"
    balance_conditions: bool = True
    balance_tasks: bool = True


class StudyProtocol(BaseModel):
    """The whole design: arms, the phase sequence each arm runs, tasks, instruments."""

    id: str
    name: str
    conditions: list[Condition]
    phases: list[PhaseSpec] = Field(description="phase sequence run once per condition")
    tasks: list[str] = Field(default_factory=list, description="task ids, e.g. ['place','sort','stack']")
    variants: list[str] = Field(default_factory=lambda: ["medium", "hard"])
    instruments: list[str] = Field(default_factory=list, description="instrument ids used anywhere")
    counterbalance: CounterbalanceSpec = Field(default_factory=CounterbalanceSpec)
    schema_version: int = 1

    def condition_ids(self) -> list[str]:
        return [c.id for c in self.conditions]

    def phase(self, name: str) -> Optional[PhaseSpec]:
        return next((p for p in self.phases if p.name == name), None)
