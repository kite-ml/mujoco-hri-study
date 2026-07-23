"""Build a pick-and-place *plan* (which object → where) from a task or a demo.

A plan is a list of ``(grasp_pos, place_pos, body_name)`` in world coordinates, the
input to :class:`mjhri.learn.controller.PickPlaceController`. Two sources:

* ``plan_from_taskspec`` — the ground-truth plan for a :class:`~mjhri.schemas.tasks.TaskSpec`
  (each in_region criterion → put its object at its region; stack → build the tower).
  Used by reward-conditioned tuning and by language-generated demonstrations.
* ``plan_from_demo`` — infer the plan a human demonstrated (which object was picked up,
  where it was set down) from a recorded trajectory, so a single demo can be re-executed
  robustly even when the objects reset to slightly different positions.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

Pick = Tuple[np.ndarray, np.ndarray, str]
_BLOCK = 0.03


def plan_from_taskspec(model: Any, data: Any, spec: Any) -> List[Pick]:
    """Ground-truth plan: put each object in its target region / build the tower.
    ``data`` must be forwarded (``mj_forward``) so site/body world positions are valid."""
    import mujoco

    picks: List[Pick] = []
    per_region: dict[str, int] = {}
    stack_base = None
    for crit in spec.success_criteria:
        ctype = str(getattr(crit, "type", "") or crit.__class__.__name__).lower()
        objs = getattr(crit, "objects", None)
        if objs and "stack" in ctype:
            for k, ob in enumerate(objs):
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ob)
                if bid < 0:
                    continue
                grasp = np.asarray(data.xpos[bid], np.float64).copy()
                grasp[2] = _BLOCK / 2
                if stack_base is None:
                    stack_base = grasp.copy()
                place = stack_base.copy()
                place[2] = _BLOCK / 2 + k * _BLOCK
                picks.append((grasp, place, ob))
            continue
        ob, region = getattr(crit, "object", None), getattr(crit, "region", None)
        if not ob or not region:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ob)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, region)
        if bid < 0 or sid < 0:
            continue
        grasp = np.asarray(data.xpos[bid], np.float64).copy()
        grasp[2] = max(grasp[2], _BLOCK / 2)
        n = per_region.get(region, 0)
        per_region[region] = n + 1
        col, row = n % 2, n // 2
        place = np.asarray(data.site_xpos[sid], np.float64).copy()
        place = place + np.array([(col - 0.5) * _BLOCK * 1.05, (row - 0.5) * _BLOCK * 1.05, 0.0])
        place[2] = _BLOCK / 2 + 0.01
        picks.append((grasp, place, ob))
    return picks


def plan_from_demo(
    model: Any,
    ee_traj: Sequence[Sequence[float]],
    grip_traj: Sequence[float],
    object_bodies: Sequence[str],
    grip_closed_below: float = 0.5,
) -> List[Pick]:
    """Infer the demonstrated plan from a recorded EE path + gripper signal.

    On each gripper close→…→open cycle: the object grasped is the one nearest the EE at
    the close; the place target is the EE position at the open. Objects' current world
    positions are read fresh (so re-execution retargets to where they actually are).
    Returns picks in demonstrated order. Robust to a single demonstration.
    """
    import mujoco

    ee = np.asarray(ee_traj, np.float64)
    grip = np.asarray(grip_traj, np.float64)
    if ee.ndim != 2 or ee.shape[0] < 2:
        return []
    closed = grip < grip_closed_below
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    obj_pos = {b: np.asarray(d.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)], np.float64)
               for b in object_bodies if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0}

    picks: List[Pick] = []
    used: set[str] = set()
    i, n = 0, len(closed)
    while i < n:
        if not closed[i]:
            i += 1
            continue
        start = i
        while i < n and closed[i]:
            i += 1
        end = i - 1  # last closed frame → release point
        # nearest unused object to the EE at the grasp instant
        p_grasp = ee[start]
        cand = [(np.linalg.norm(p_grasp - pos), b) for b, pos in obj_pos.items() if b not in used]
        if not cand:
            continue
        _, body = min(cand)
        used.add(body)
        grasp = obj_pos[body].copy()
        grasp[2] = max(grasp[2], _BLOCK / 2)
        # Preserve the DEMONSTRATED release height (floored at block-half so it never goes
        # underground): a place near the ground stays near the ground, but a block the human
        # lifted to build a tower keeps its height — otherwise stacking demos flatten to one
        # spot and modality A could never teach a stack.
        place = ee[end].copy()
        place[2] = max(float(place[2]), _BLOCK / 2)
        picks.append((grasp, place, body))
    return picks
