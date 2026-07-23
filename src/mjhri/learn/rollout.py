"""Headless rollout + scoring of a controller/policy on a MuJoCo scene.

Runs a driving policy (anything exposing ``initial_ctrl()``, ``act(qpos, t)`` and an
optional ``carry_update(model, data, t)``) in plain CPU MuJoCo and scores the final
state with the task's :class:`~mjhri.tasks.scorer.Scorer`. Used to evaluate a trained
policy (success over N randomized rollouts) and as the reward signal for CEM tuning.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..tasks.scorer import Scorer


class _View:
    """Minimal view a policy's ``act`` expects: ``.qpos`` and ``.sim_time``."""
    __slots__ = ("qpos", "sim_time")

    def __init__(self, qpos, t):
        self.qpos, self.sim_time = qpos, t


def rollout_once(model: Any, policy: Any, spec: Any, *, grasp: Any = None,
                 control_hz: float = 30.0, settle_steps: int = 200, seed: Optional[int] = None) -> bool:
    """Run ``policy`` to completion on a fresh ``MjData`` and return whether ``spec`` is
    met. Grasping is the shared ``grasp`` mechanic (an :class:`~mjhri.learn.grasp.AutoGrasp`)
    when given — applied uniformly to every policy — else the policy's own ``carry_update``.
    Applies the task's reset randomization when ``seed`` is given."""
    import mujoco

    data = mujoco.MjData(model)
    if seed is not None and getattr(spec, "reset_randomization", None) is not None:
        try:
            from ..tasks.randomize import apply_randomization
            apply_randomization(model, data, spec.reset_randomization, rng=np.random.default_rng(seed))
        except Exception:
            pass
    mujoco.mj_forward(model, data)
    if grasp is not None:
        grasp.reset()

    dt = float(model.opt.timestep)
    ci = 1.0 / control_hz
    last = -1.0
    t = 0.0
    ctrl = np.asarray(policy.initial_ctrl()).reshape(-1)
    carry = getattr(policy, "carry_update", None) if grasp is None else None
    end = float(getattr(policy, "duration", 6.0)) + 1.0
    while t < end:
        if t - last >= ci:
            ctrl = np.asarray(policy.act(data.qpos.copy(), t)).reshape(-1)
            last = t
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        if grasp is not None:
            grasp.update(model, data, t)
        elif carry is not None:
            carry(model, data, t)
        t += dt
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        if grasp is not None:
            grasp.update(model, data, end)
    return bool(Scorer.from_spec(spec).score(model, data).success)


def score_policy(model: Any, policy_factory: Any, spec: Any, *, grasp_factory: Any = None,
                 n_rollouts: int = 20, control_hz: float = 30.0) -> dict:
    """Success rate of a policy over ``n_rollouts`` randomized resets. ``policy_factory``
    (and optional ``grasp_factory``) are called per rollout for fresh per-reset state."""
    successes = 0
    for i in range(n_rollouts):
        policy = policy_factory() if callable(policy_factory) else policy_factory
        grasp = grasp_factory() if callable(grasp_factory) else grasp_factory
        if rollout_once(model, policy, spec, grasp=grasp, control_hz=control_hz, seed=i):
            successes += 1
    rate = successes / n_rollouts if n_rollouts else 0.0
    return {"successes": successes, "n": n_rollouts, "rate": round(rate, 3)}
