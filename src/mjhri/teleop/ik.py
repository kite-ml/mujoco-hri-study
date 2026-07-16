"""Damped-least-squares end-effector IK for a named arm — robot-agnostic.

Given a set of actuated joints and an end-effector (a body plus a local offset, so
no scene edit is needed to place a grasp-point frame), ``EEController`` maps a world
target position to joint-position targets with a few DLS iterations seeded from the
live state. That makes it smooth for teleoperation: each control tick nudges the
target and re-solves incrementally from where the arm actually is.

Optionally it also keeps a chosen EE-local axis aligned to a world direction — e.g.
pointing the gripper straight down for a clean top-down grasp. With a 5-DOF arm that
3-position + 2-axis task is fully determined; the DLS handles any redundancy.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("mjhri.teleop requires mujoco (`pip install mujoco>=3.0`)") from exc


class EEController:
    def __init__(
        self,
        model,
        ee_body: str,
        joints: Sequence[str],
        *,
        ee_offset: Sequence[float] = (0.0, 0.0, 0.0),
        align_local_axis: Optional[Sequence[float]] = None,
        align_world_dir: Optional[Sequence[float]] = None,
        align_weight: float = 0.5,
        damping: float = 1e-3,
        step: float = 0.6,
        max_iter: int = 8,
        tol: float = 0.002,
    ):
        self.model = model
        self._data = mujoco.MjData(model)
        self._ee_bid = _require(model, mujoco.mjtObj.mjOBJ_BODY, ee_body, "body")
        self._ee_offset = np.asarray(ee_offset, dtype=np.float64)
        self._jids = [_require(model, mujoco.mjtObj.mjOBJ_JOINT, j, "joint") for j in joints]
        self._qadr = [int(model.jnt_qposadr[j]) for j in self._jids]
        self._dadr = [int(model.jnt_dofadr[j]) for j in self._jids]
        self._range = [tuple(model.jnt_range[j]) for j in self._jids]
        self.damping = damping
        self.step = step
        self.max_iter = max_iter
        self.tol = tol
        # optional orientation constraint: keep local axis aligned to a world dir
        self._align_axis = (
            np.asarray(align_local_axis, dtype=np.float64)
            if align_local_axis is not None else None
        )
        self._align_dir = (
            _unit(np.asarray(align_world_dir, dtype=np.float64))
            if align_world_dir is not None else None
        )
        self._align_w = float(align_weight)

    def ee_pos(self, qpos: np.ndarray) -> np.ndarray:
        """World position of the end-effector point for a given full qpos."""
        self._data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self._data)
        return self._point()

    def _rot(self) -> np.ndarray:
        return np.asarray(self._data.xmat[self._ee_bid], dtype=np.float64).reshape(3, 3)

    def _point(self) -> np.ndarray:
        return np.asarray(self._data.xpos[self._ee_bid], dtype=np.float64) + self._rot() @ self._ee_offset

    def solve(self, qpos: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return joint-position targets (one per controlled joint) that move the EE
        toward ``target`` (and, if configured, keep the gripper axis aligned), seeded
        from the live ``qpos``. Clipped to joint ranges."""
        self._data.qpos[:] = qpos
        target = np.asarray(target, dtype=np.float64)
        aligned = self._align_axis is not None and self._align_dir is not None
        rows = 6 if aligned else 3
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv)) if aligned else None
        for _ in range(self.max_iter):
            mujoco.mj_forward(self.model, self._data)
            perr = target - self._point()
            mujoco.mj_jac(self.model, self._data, jacp, jacr, self._point(), self._ee_bid)
            if aligned:
                cur = self._rot() @ self._align_axis
                oerr = np.cross(cur, self._align_dir) * self._align_w
                err = np.concatenate([perr, oerr])
                J = np.vstack([jacp[:, self._dadr], jacr[:, self._dadr]])
            else:
                if np.linalg.norm(perr) < self.tol:
                    break
                err = perr
                J = jacp[:, self._dadr]
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(rows), err)
            for i, delta in enumerate(dq):
                lo, hi = self._range[i]
                self._data.qpos[self._qadr[i]] = np.clip(
                    self._data.qpos[self._qadr[i]] + self.step * delta, lo, hi
                )
        return np.array([self._data.qpos[a] for a in self._qadr], dtype=np.float32)


def _require(model, objtype, name: str, kind: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise KeyError(f"no {kind} named {name!r} in model")
    return idx


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v
