"""``Scorer`` — evaluate a ``TaskSpec``'s success criteria against a MuJoCo state.

Used two ways: on a live ``MjData`` during/after a rollout, or on a restored qpos
snapshot in batch eval. Ground truth for "did the policy complete the task" is
exactly ``Scorer.from_spec(spec).score(model, data).success``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..schemas.tasks import RegionCriterion, StackCriterion, TaskSpec
from . import criteria as C


@dataclass
class CriterionResult:
    type: str
    success: bool
    detail: dict[str, Any] = field(default_factory=dict)
    label: str = ""


@dataclass
class Outcome:
    success: bool
    criteria: list[CriterionResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "criteria": [
                {"type": c.type, "success": c.success, "label": c.label, "detail": c.detail}
                for c in self.criteria
            ],
        }


class Scorer:
    def __init__(self, spec: TaskSpec):
        self.spec = spec

    @classmethod
    def from_spec(cls, spec: TaskSpec) -> "Scorer":
        return cls(spec)

    def score(self, model, data) -> Outcome:
        """Score against an already-forwarded ``MjData`` (call ``mj_forward``/``mj_step``
        first). Success = every criterion satisfied (empty criteria ⇒ vacuously true,
        but flagged in detail)."""
        results: list[CriterionResult] = []
        for crit in self.spec.success_criteria:
            if isinstance(crit, RegionCriterion):
                ok, detail = C.check_in_region(
                    model, data, obj=crit.object, region=crit.region, margin=crit.margin,
                    require_resting=crit.require_resting, rest_speed=crit.rest_speed,
                )
                results.append(CriterionResult("in_region", ok, detail, f"{crit.object}→{crit.region}"))
            elif isinstance(crit, StackCriterion):
                ok, detail = C.check_stack(
                    model, data, objects=list(crit.objects), xy_tol=crit.xy_tol,
                    require_contact=crit.require_contact, upright_tol=crit.upright_tol,
                    min_layers=crit.min_layers,
                )
                results.append(CriterionResult("stack", ok, detail, "+".join(crit.objects)))
            else:  # pragma: no cover - discriminated union guards this
                raise TypeError(f"unknown criterion type: {type(crit)!r}")
        success = bool(results) and all(r.success for r in results)
        return Outcome(success=success, criteria=results)
