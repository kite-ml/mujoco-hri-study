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


def _criteria_objects_targets(model, data, spec):
    """(object_body_id, target_world_xyz) pairs implied by the task's success criteria —
    a region's site centre, or (for a stack) each block's tower slot above the base.
    Used by ``rollout_metrics`` to measure continuous placement error generically."""
    from ..schemas.tasks import RegionCriterion, StackCriterion
    from ..tasks import criteria as C

    pairs = []
    for crit in getattr(spec, "success_criteria", []):
        if isinstance(crit, RegionCriterion):
            try:
                pairs.append((C.body_id(model, crit.object),
                              np.asarray(data.site_xpos[C.site_id(model, crit.region)], float)))
            except KeyError:
                continue
        elif isinstance(crit, StackCriterion):
            ids = [C.body_id(model, o) for o in crit.objects]
            base = C.centroid(data, min(ids, key=lambda b: C.centroid(data, b)[2]))
            for k, b in enumerate(sorted(ids, key=lambda b: C.centroid(data, b)[2])):
                pairs.append((b, np.array([base[0], base[1], base[2] + k * 0.03])))
    return pairs


def rollout_metrics(model: Any, policy: Any, spec: Any, *, grasp: Any = None,
                    ee_body: Optional[str] = None, ee_offset=(0.0, 0.0, 0.0),
                    control_hz: float = 30.0, settle_steps: int = 200,
                    seed: Optional[int] = None) -> dict:
    """Like :func:`rollout_once`, but returns CONTINUOUS per-episode terms — the raw
    material for a weighted reward objective — not just success:

    * ``success``     final Scorer success (bool)
    * ``reach``       [0,1], 1 = the gripper got right to an object (needs ``ee_body``)
    * ``grasped``     [0,1], fraction of target objects lifted >2 cm at some point
    * ``place_error`` mean final object→target distance (m), lower is better
    * ``time``        [0,1], fraction of the episode before first success (1 = never)

    These quantities exist inside the scorer/criteria layer but are discarded by
    :func:`rollout_once`/:func:`score_policy`; here they are surfaced so a host can build
    a reward that reflects e.g. "prioritise precise placement" vs "be fast".
    """
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

    ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body) if ee_body else -1
    off = np.asarray(ee_offset, float)

    def grasp_point():
        rot = np.zeros(3)
        mujoco.mju_rotVecQuat(rot, off, data.xquat[ee_bid])
        return np.asarray(data.xpos[ee_bid], float) + rot

    pairs = _criteria_objects_targets(model, data, spec)
    obj_ids = list({b for b, _ in pairs})
    start_z = {b: float(data.xpos[b][2]) for b in obj_ids}
    max_lift = {b: 0.0 for b in obj_ids}
    init_reach = min((float(np.linalg.norm(grasp_point() - data.xpos[b])) for b in obj_ids),
                     default=1.0) if ee_bid >= 0 else 1.0
    min_reach = init_reach

    dt = float(model.opt.timestep)
    ci = 1.0 / control_hz
    last = -1.0
    t = 0.0
    ctrl = np.asarray(policy.initial_ctrl()).reshape(-1)
    carry = getattr(policy, "carry_update", None) if grasp is None else None
    end = float(getattr(policy, "duration", 6.0)) + 1.0
    scorer = Scorer.from_spec(spec)
    first_success_t = None

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
        if ee_bid >= 0 and obj_ids:
            min_reach = min(min_reach, min(float(np.linalg.norm(grasp_point() - data.xpos[b])) for b in obj_ids))
        for b in obj_ids:
            max_lift[b] = max(max_lift[b], float(data.xpos[b][2]) - start_z[b])
        if first_success_t is None and scorer.score(model, data).success:
            first_success_t = t
        t += dt
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        if grasp is not None:
            grasp.update(model, data, end)

    place_err = float(np.mean([np.linalg.norm((data.xpos[b] - tgt)[:2]) for b, tgt in pairs])) if pairs else 1.0
    reach = 1.0 - min(min_reach / init_reach, 1.0) if (ee_bid >= 0 and init_reach > 1e-6) else 0.0
    grasped = float(np.mean([1.0 if max_lift[b] > 0.02 else 0.0 for b in obj_ids])) if obj_ids else 0.0
    time_frac = (first_success_t / end) if first_success_t is not None else 1.0
    return {
        "success": bool(scorer.score(model, data).success),
        "reach": round(float(reach), 4),
        "grasped": round(float(grasped), 4),
        "place_error": round(place_err, 4),
        "time": round(float(time_frac), 4),
    }


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
