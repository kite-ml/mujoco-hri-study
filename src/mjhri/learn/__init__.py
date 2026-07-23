"""Reusable learning building blocks: turn teaching into a policy on any MuJoCo scene.

Sim- and robot-agnostic, numpy + mujoco only (no torch, no cloud). The pieces compose
into the three teaching pathways an HRI study might compare:

* imitation  — ``plan_from_demo`` → ``PickPlaceController`` (one-shot), or
               ``NearestStatePolicy`` (memory-based BC over several demos)
* reward     — ``PickPlaceController`` params tuned by ``cem_optimize`` to a reward
* generation — a host-supplied plan (e.g. from language) → ``PickPlaceController`` →
               generated demos → ``NearestStatePolicy``

Evaluate anything with ``score_policy`` (success over N randomized rollouts, scored by
the task's :class:`~mjhri.tasks.scorer.Scorer`).
"""

from .controller import PickPlaceController, PickPlaceParams
from .imitation import NearestStatePolicy
from .optimize import cem_optimize
from .plan import plan_from_demo, plan_from_taskspec
from .rollout import rollout_once, score_policy

__all__ = [
    "PickPlaceController",
    "PickPlaceParams",
    "NearestStatePolicy",
    "cem_optimize",
    "plan_from_demo",
    "plan_from_taskspec",
    "rollout_once",
    "score_policy",
]
