"""Parametric pick-and-place controller — the shared execution engine.

A robot- and scene-agnostic controller that drives an arm's end-effector through a
reach → grasp → lift → transport → place cycle for each object in a *plan*, using
the package's DLS-IK (:class:`mjhri.teleop.EEController`). The grasp is **kinematic**
(``carry_update`` pins the held object to the scripted grasp path and releases it
over the target) rather than a physical friction grasp — reliable for small blocks
and dependency-free.

The three study teaching methods all drive this one controller, differing only in
how the plan / parameters are produced:

* imitation  — the plan (which object, where) is extracted from a demonstration
* reward     — the *parameters* are tuned to a reward (see :mod:`mjhri.learn.optimize`)
* generation — the plan comes from a language description (host-supplied)

It is decoupled from any live-session interface: ``act(qpos, t)`` returns a full
ctrl vector and ``carry_update(model, data, t)`` does the per-step kinematic grasp,
so a host can adapt it to whatever driving loop it has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..teleop.ik import EEController

Pick = Tuple[np.ndarray, np.ndarray, str]  # (grasp_pos, place_pos, body_name), world coords


@dataclass
class PickPlaceParams:
    """Tunable controller parameters (the search space for reward-conditioned tuning)."""
    hover_dz: float = 0.09       # height above a target for approach / lift (m)
    grasp_dz: float = 0.012      # grasp-point height above the object/target base (m)
    t_approach: float = 0.9
    t_descend: float = 0.5
    t_grasp: float = 0.25
    t_lift: float = 0.5
    t_transport: float = 1.0
    t_place: float = 0.5
    t_release: float = 0.4
    grip_open: float = 1.0
    grip_closed: float = 0.15

    def as_vector(self) -> np.ndarray:
        return np.array([self.hover_dz, self.grasp_dz, self.t_approach, self.t_descend,
                         self.t_lift, self.t_transport, self.t_place], dtype=np.float64)

    @classmethod
    def from_vector(cls, v: Sequence[float], base: "PickPlaceParams | None" = None) -> "PickPlaceParams":
        b = base or cls()
        return cls(hover_dz=float(v[0]), grasp_dz=float(v[1]), t_approach=float(v[2]),
                   t_descend=float(v[3]), t_lift=float(v[4]), t_transport=float(v[5]),
                   t_place=float(v[6]), t_grasp=b.t_grasp, t_release=b.t_release,
                   grip_open=b.grip_open, grip_closed=b.grip_closed)

    @staticmethod
    def bounds() -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([0.04, 0.004, 0.4, 0.3, 0.3, 0.5, 0.3])
        hi = np.array([0.14, 0.030, 1.6, 1.0, 1.0, 1.8, 1.0])
        return lo, hi


class _Seg:
    __slots__ = ("t0", "t1", "p0", "p1", "grip", "carry")

    def __init__(self, t0, t1, p0, p1, grip, carry):
        self.t0, self.t1, self.p0, self.p1, self.grip, self.carry = t0, t1, p0, p1, grip, carry


class PickPlaceController:
    def __init__(
        self,
        model: Any,
        *,
        ee_body: str,
        arm_joints: Sequence[str],
        gripper_actuator: str,
        ready_arm: Optional[Sequence[float]] = None,
        ee_offset: Sequence[float] = (0.0, 0.0, 0.0),
        ws_min: Sequence[float] = (-0.16, -0.24, 0.012),
        ws_max: Sequence[float] = (0.16, -0.05, 0.30),
        params: Optional[PickPlaceParams] = None,
        block_edge: float = 0.03,
        settle: float = 1.5,
    ):
        import mujoco

        self._mj = mujoco
        self.model = model
        self.p = params or PickPlaceParams()
        self._block = float(block_edge)
        # A warm-up that drives the arm to the ready pose BEFORE the IK segments start:
        # seeding IK from the ready pose (not qpos0) avoids a local minimum that would
        # otherwise leave the arm stuck ~20 cm from the workspace.
        self._settle = float(settle)
        self._ws_min = np.asarray(ws_min, dtype=np.float64)
        self._ws_max = np.asarray(ws_max, dtype=np.float64)
        self._ee_off = np.asarray(ee_offset, dtype=np.float64)

        self._arm_ci = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in arm_joints]
        self._grip_ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator)
        self._ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        self._ik = EEController(model, ee_body, list(arm_joints), ee_offset=ee_offset, max_iter=10, step=0.6)

        arm_qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in arm_joints]
        d0 = mujoco.MjData(model)
        if ready_arm is not None:
            for adr, v in zip(arm_qadr, ready_arm):
                d0.qpos[adr] = v
        mujoco.mj_forward(model, d0)
        self._base = np.array(
            [d0.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]] for a in range(model.nu)], dtype=np.float32)
        cr = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
        self._lo, self._hi = cr[:, 0], cr[:, 1]
        self._home = self._ik.ee_pos(d0.qpos).astype(np.float64)

        self._target = self._home.copy()
        self._grip = float(self.p.grip_open)
        self._segs: List[_Seg] = []
        self._place_of: dict[str, dict] = {}
        self._carry_adr: dict[str, dict] = {}
        self._t_end = 0.0
        self.duration = 0.0

    # -- plan → timeline ------------------------------------------------------

    def set_plan(self, picks: Sequence[Pick]) -> "PickPlaceController":
        mujoco = self._mj
        self._carry_adr = {}
        for _, _, body in picks:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            jadr = self.model.body_jntadr[bid] if bid >= 0 else -1
            if jadr >= 0 and self.model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                self._carry_adr[body] = {"q": int(self.model.jnt_qposadr[jadr]),
                                         "v": int(self.model.jnt_dofadr[jadr])}
        self._build_timeline(picks)
        return self

    def _build_timeline(self, picks: Sequence[Pick]) -> None:
        p = self.p
        segs: List[_Seg] = []
        place_of: dict[str, dict] = {}
        t = 0.0
        cur = self._home.copy()

        def seg(dt, p1, grip, carry):
            nonlocal t, cur
            segs.append(_Seg(t, t + dt, cur.copy(), np.asarray(p1, np.float64), grip, carry))
            t += dt
            cur = np.asarray(p1, np.float64)

        up = np.array([0.0, 0.0, p.hover_dz])
        for grasp, place, body in picks:
            above_g, at_g = grasp + up, grasp + np.array([0.0, 0.0, p.grasp_dz])
            above_p, at_p = place + up, place + np.array([0.0, 0.0, p.grasp_dz])
            seg(p.t_approach, above_g, p.grip_open, None)
            seg(p.t_descend, at_g, p.grip_open, None)
            place_of[body] = {"t_grasp": t, "place": np.asarray(place, np.float64)}
            seg(p.t_grasp, at_g, p.grip_closed, body)
            seg(p.t_lift, above_g, p.grip_closed, body)
            seg(p.t_transport, above_p, p.grip_closed, body)
            seg(p.t_place, at_p, p.grip_closed, body)
            # Open the jaw WHILE STATIONARY at the target, then withdraw. Previously the
            # release segment opened the gripper and rose to `above_p` in one motion: the
            # jaw needs time to physically clear the block, so by the time it did the
            # gripper had already climbed ~4 cm and the block was DROPPED rather than
            # placed. It then bounced and rolled 2-8 cm, which is why unaided placement
            # error was ~4.6 cm against a 2 cm stacking tolerance.
            seg(p.t_release, at_p, p.grip_open, None)
            seg(p.t_release, above_p, p.grip_open, None)
        seg(0.5, self._home, p.grip_open, None)
        self._segs, self._place_of = segs, place_of
        self._t_end = t                       # timeline length (excludes the settle warm-up)
        self.duration = self._settle + t

    # -- driving --------------------------------------------------------------

    def initial_ctrl(self) -> np.ndarray:
        c = self._base.copy()
        c[self._grip_ci] = self._grip
        return self._clip(c)

    def act(self, qpos: np.ndarray, sim_time: float) -> np.ndarray:
        if sim_time < self._settle:  # warm-up: hold the ready pose, gripper open
            self._grip = float(self.p.grip_open)
            c = self._base.copy()
            c[self._grip_ci] = self._grip
            return self._clip(c)
        st = sim_time - self._settle
        seg = self._seg_at(st)
        if seg is not None:
            a = min(max((st - seg.t0) / max(seg.t1 - seg.t0, 1e-3), 0.0), 1.0)
            self._target = np.clip(seg.p0 * (1 - a) + seg.p1 * a, self._ws_min, self._ws_max)
            self._grip = float(seg.grip)
        qpos = np.asarray(qpos, np.float64)
        qpos = qpos[0] if qpos.ndim == 2 else qpos
        arm = self._ik.solve(qpos, self._target)
        c = self._base.copy()
        for ci, v in zip(self._arm_ci, arm):
            c[ci] = v
        c[self._grip_ci] = self._grip
        return self._clip(c)

    def carry_update(self, model: Any, data: Any, sim_time: float) -> None:
        """Kinematic grasp: the currently-carried block tracks the scripted grasp path;
        already-placed blocks are held at their exact target so a later carry can't knock
        a finished stack over. Everything releases when the timeline ends."""
        st = sim_time - self._settle
        if st < 0 or st >= self._t_end:
            return
        seg = self._seg_at(st)
        carrying = seg.carry if seg is not None else None
        for body, adr in self._carry_adr.items():
            info = self._place_of.get(body)
            if info is None or sim_time < info["t_grasp"]:
                continue
            pos = self._target if body == carrying else info["place"]
            data.qpos[adr["q"]:adr["q"] + 3] = pos
            data.qpos[adr["q"] + 3:adr["q"] + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[adr["v"]:adr["v"] + 6] = 0.0

    # -- helpers --------------------------------------------------------------

    def _seg_at(self, t: float) -> Optional[_Seg]:
        """``t`` is the timeline clock (0-based, excludes the settle warm-up)."""
        for s in self._segs:
            if s.t0 <= t < s.t1:
                return s
        return self._segs[-1] if self._segs and t >= self._t_end else None

    def _clip(self, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, np.float32).reshape(-1)
        if c.shape[0] == self._lo.shape[0]:
            c = np.clip(c, self._lo, self._hi)
        return c.reshape(1, -1)
