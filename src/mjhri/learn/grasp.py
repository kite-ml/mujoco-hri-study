"""Auto-grasp: a shared "sticky gripper" sim mechanic.

Task completion shouldn't depend on solving physical friction-grasping of small blocks
(unreliable) — but it also shouldn't be baked into one controller, or a cloned policy
could never reproduce it. ``AutoGrasp`` makes grasping a property of the *scene*: when
the gripper actuator closes and a graspable free body is within reach of the grasp
point, that body is kinematically attached to the gripper and rides along until the
gripper opens. Applied uniformly to every policy (scripted controller, cloned BC, live
teleop), so all of them succeed by the same rule — close the gripper near an object,
carry it, open to release.

Call ``update(model, data)`` once per physics step, after ``mj_step``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np


class AutoGrasp:
    def __init__(
        self,
        model: Any,
        *,
        ee_body: str,
        gripper_actuator: str,
        ee_offset: Sequence[float] = (0.0, 0.0, 0.0),
        graspable_bodies: Optional[Sequence[str]] = None,
        close_below: float = 0.5,
        radius: float = 0.05,
        hold_placed: bool = True,
        settle_delay: float = 0.25,
        place_targets: Optional[dict] = None,
        snap_radius: float = 0.045,
    ):
        import mujoco

        self._mj = mujoco
        self.model = model
        self._ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        self._grip_ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator)
        self._off = np.asarray(ee_offset, np.float64)
        self._close_below = float(close_below)
        self._r2 = float(radius) ** 2
        # Once a block is released it settles for ``settle_delay`` then FREEZES where it
        # landed, so a later carry can't knock a finished stack over. Re-grasping unfreezes
        # it. Harmless for place/sort (blocks freeze resting in their bin).
        self._hold_placed = bool(hold_placed)
        self._settle_delay = float(settle_delay)
        # If a block is released within ``snap_radius`` of its intended target it LOCKS
        # there (clean placement / a stable tower); released far, it just drops (a miss).
        # So success still depends on the policy getting the block near the goal.
        self._targets = {k: np.asarray(v, np.float64) for k, v in (place_targets or {}).items()}
        self._snap_r2 = float(snap_radius) ** 2
        self._releasing: dict[str, float] = {}   # body → release time (settling)
        self._frozen: dict[str, np.ndarray] = {}  # body → frozen world position

        # graspable = free-joint bodies (auto-detected unless named)
        self._grasp: dict[str, int] = {}
        if graspable_bodies is None:
            for j in range(int(model.njnt)):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    bid = int(model.jnt_bodyid[j])
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
                    if name:
                        self._grasp[name] = int(model.jnt_qposadr[j])
        else:
            for name in graspable_bodies:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                jadr = model.body_jntadr[bid] if bid >= 0 else -1
                if jadr >= 0 and model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                    self._grasp[name] = int(model.jnt_qposadr[jadr])
        self._held: Optional[str] = None

    def _grasp_point(self, data: Any) -> np.ndarray:
        mj = self._mj
        off = np.zeros(3)
        mj.mju_rotVecQuat(off, self._off, data.xquat[self._ee_bid])
        return np.asarray(data.xpos[self._ee_bid], np.float64) + off

    def _dof(self, model: Any, body: str) -> int:
        return int(model.jnt_dofadr[model.body_jntadr[
            self._mj.mj_name2id(model, self._mj.mjtObj.mjOBJ_BODY, body)]])

    def update(self, model: Any, data: Any, sim_time: float | None = None) -> None:
        t = float(sim_time) if sim_time is not None else 0.0
        gp = self._grasp_point(data)
        closed = float(data.ctrl[self._grip_ci]) < self._close_below

        if not closed and self._held is not None:  # release
            b = self._held
            tgt = self._targets.get(b)
            if tgt is not None and float(np.sum((data.qpos[self._grasp[b]:self._grasp[b] + 3] - tgt) ** 2)) < self._snap_r2:
                self._frozen[b] = tgt.copy()        # released near the goal → lock at it
            elif self._hold_placed:
                self._releasing[b] = t              # otherwise settle then freeze where it lands
            self._held = None
        if closed and self._held is None:  # look for an object to grab (frozen ones too)
            best, bd = None, self._r2
            for name, q in self._grasp.items():
                d2 = float(np.sum((data.qpos[q:q + 3] - gp) ** 2))
                if d2 < bd:
                    best, bd = name, d2
            self._held = best
            if best is not None:  # picking it back up unfreezes it
                self._frozen.pop(best, None)
                self._releasing.pop(best, None)
        if self._held is not None:  # carry it
            q = self._grasp[self._held]
            data.qpos[q:q + 3] = gp
            data.qpos[q + 3:q + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[self._dof(model, self._held):self._dof(model, self._held) + 6] = 0.0

        if self._hold_placed:  # settle → freeze placed blocks; pin frozen ones
            for name, tr in list(self._releasing.items()):
                if t - tr >= self._settle_delay:
                    self._frozen[name] = np.asarray(data.qpos[self._grasp[name]:self._grasp[name] + 3], np.float64).copy()
                    del self._releasing[name]
            for name, pos in self._frozen.items():
                q = self._grasp[name]
                data.qpos[q:q + 3] = pos
                data.qvel[self._dof(model, name):self._dof(model, name) + 6] = 0.0

    def reset(self) -> None:
        self._held = None
        self._releasing.clear()
        self._frozen.clear()
