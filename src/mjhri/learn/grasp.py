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
        radius_z: Optional[float] = None,
        hold_placed: bool = True,
        settle_delay: float = 0.25,
        place_targets: Optional[dict] = None,
        snap_radius: float = 0.045,
        grasp_point_geoms: Optional[Sequence[str]] = None,
    ):
        import mujoco

        self._mj = mujoco
        self.model = model
        self._ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        self._grip_ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator)
        self._off = np.asarray(ee_offset, np.float64)
        # Optional EXACT grasp point: the centroid of these geoms (e.g. the finger pads).
        # A fixed body offset points away from the fingers once the wrist tilts — the
        # invisible attach point then lands centimeters from where the gripper visibly
        # is, so closing "on" an object does nothing. The pad centroid is correct in
        # every orientation.
        self._gp_gids = []
        for name in (grasp_point_geoms or []):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._gp_gids.append(int(gid))
        self._close_below = float(close_below)
        # ANISOTROPIC grasp reach: ``radius`` is the HORIZONTAL (xy) tolerance — kept
        # tight so a close neighbour block isn't grabbed by mistake — while ``radius_z``
        # is the VERTICAL tolerance, made larger because the arm's pads bottom out a few
        # cm above a floor-height block (only full-IK wrist-bending dips lower). A block
        # is grabbed when ``(dxy/radius)² + (dz/radius_z)² < 1``. Defaults to isotropic.
        self._rxy = float(radius)
        self._rz = float(radius_z) if radius_z is not None else float(radius)
        self._r2 = float(radius) ** 2  # kept for any external reference
        # Once a block is released it settles for ``settle_delay`` then FREEZES where it
        # landed, so a later carry can't knock a finished stack over. Re-grasping unfreezes
        # it. Harmless for place/sort (blocks freeze resting in their bin).
        self._hold_placed = bool(hold_placed)
        self._settle_delay = float(settle_delay)
        # If a block is released near its intended target it GLIDES there (clean
        # placement / a stable tower); released far, it just drops (a miss). The catch
        # is ANISOTROPIC: tight horizontally (accuracy still matters) but generous
        # vertically — a block dropped from straight above the goal is clearly "on
        # target" even if let go high, which tall stacks do routinely.
        self._targets = {k: np.asarray(v, np.float64) for k, v in (place_targets or {}).items()}
        self._snap_r2 = float(snap_radius) ** 2
        self._snap_z = max(0.10, 2.0 * float(snap_radius))
        # Smoothed carry: the held block eases toward the grasp point with this time
        # constant (s) rather than snapping — smooth pick-up/transport, but short enough
        # that steady-state trailing doesn't spoil stacking precision at release.
        self._carry_tau = 0.03
        self._carry_pos = np.zeros(3)
        self._carry_locked = False
        self._glide_time = 0.15                   # snap-to-target ease duration (s)
        self._releasing: dict[str, float] = {}   # body → release time (settling)
        self._frozen: dict[str, np.ndarray] = {}  # body → frozen world position
        self._gliding: dict[str, tuple] = {}      # body → (start, target, t0) snap-glide
        # Blocks deliberately placed AT their target are done — excluded from the pick
        # search so a policy's trailing actions (a closing jaw hovering over the tower)
        # can't pluck the top block back off. Only populated when place_targets exist,
        # so free-form teleop (no targets) can always re-grab anything.
        self._placed: set[str] = set()

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
        if self._gp_gids:
            return np.mean(np.asarray(data.geom_xpos)[self._gp_gids], axis=0)
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
            dp = (data.qpos[self._grasp[b]:self._grasp[b] + 3] - tgt) if tgt is not None else None
            if dp is not None and float(dp[0] ** 2 + dp[1] ** 2) < self._snap_r2 and abs(float(dp[2])) < self._snap_z:
                # Released near the goal → GLIDE it there over ~0.3 s instead of
                # teleporting (the instant lock read as a violent, unnatural force,
                # worst when stacking). Frozen at the target once the glide lands.
                start = np.asarray(data.qpos[self._grasp[b]:self._grasp[b] + 3], np.float64).copy()
                self._gliding[b] = (start, tgt.copy(), t)
            elif self._hold_placed:
                self._releasing[b] = t              # otherwise settle then freeze where it lands
            self._held = None
        if closed and self._held is None:  # look for an object to grab (frozen ones too)
            best, bd = None, 1.0  # normalized ellipsoid metric; grab if < 1
            for name, q in self._grasp.items():
                if name in self._gliding or name in self._placed:
                    continue                     # mid-glide or already placed — not re-grabbable
                dp = data.qpos[q:q + 3] - gp
                metric = float((dp[0] ** 2 + dp[1] ** 2) / self._rxy ** 2 + dp[2] ** 2 / self._rz ** 2)
                if metric < bd:
                    best, bd = name, metric
            self._held = best
            if best is not None:  # picking it back up unfreezes it
                self._frozen.pop(best, None)
                self._releasing.pop(best, None)
                self._gliding.pop(best, None)
                # Start the smoothed pick-up AT the block's current pose so it EASES into
                # the gripper instead of teleporting (the visible "snap"). Once it has
                # caught up we lock to exact tracking, so carry/placement precision is
                # unchanged — only the initial grab is smoothed.
                self._carry_pos = np.asarray(data.qpos[self._grasp[best]:self._grasp[best] + 3], np.float64).copy()
                self._carry_locked = False
        if self._held is not None:  # carry it
            q = self._grasp[self._held]
            if self._carry_locked:
                self._carry_pos = gp
            else:
                dt = float(getattr(model.opt, "timestep", 0.002))
                alpha = 1.0 - np.exp(-dt / self._carry_tau)
                self._carry_pos = self._carry_pos + alpha * (gp - self._carry_pos)
                if float(np.sum((self._carry_pos - gp) ** 2)) < 9e-6:  # within 3 mm → lock
                    self._carry_pos = gp
                    self._carry_locked = True
            data.qpos[q:q + 3] = self._carry_pos
            data.qpos[q + 3:q + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[self._dof(model, self._held):self._dof(model, self._held) + 6] = 0.0

        # Snap-glide: ease released-near-goal blocks to their target, then freeze there.
        for name, (start, tgt, t0) in list(self._gliding.items()):
            q = self._grasp[name]
            frac = min(1.0, (t - t0) / self._glide_time)
            s = frac * frac * (3.0 - 2.0 * frac)                  # smoothstep ease
            data.qpos[q:q + 3] = start + s * (tgt - start)
            data.qvel[self._dof(model, name):self._dof(model, name) + 6] = 0.0
            if frac >= 1.0:
                del self._gliding[name]
                self._placed.add(name)
                if self._hold_placed:
                    self._frozen[name] = tgt.copy()

        if self._hold_placed:  # settle → freeze placed blocks; pin frozen ones
            for name, tr in list(self._releasing.items()):
                # Freeze only once the block has actually COME TO REST (or after a
                # 1.5 s cap) — freezing on a fixed timer used to pin blocks mid-bounce,
                # visibly hovering or intersecting a neighbour.
                if t - tr >= self._settle_delay:
                    dof = self._dof(model, name)
                    speed = float(np.linalg.norm(data.qvel[dof:dof + 3]))
                    if speed < 0.05 or t - tr >= 1.5:
                        self._frozen[name] = np.asarray(
                            data.qpos[self._grasp[name]:self._grasp[name] + 3], np.float64).copy()
                        del self._releasing[name]
            for name, pos in self._frozen.items():
                q = self._grasp[name]
                data.qpos[q:q + 3] = pos
                data.qvel[self._dof(model, name):self._dof(model, name) + 6] = 0.0

    def reset(self) -> None:
        self._held = None
        self._releasing.clear()
        self._frozen.clear()
        self._gliding.clear()
        self._placed.clear()
