"""Apply a ``ResetRandomization`` to object poses at episode reset.

Randomizing object start poses is what stops a demonstration policy from memorizing
one layout and makes a success rate meaningful. Operates on free-joint bodies by
name; leaves the robot and everything unnamed untouched.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("mjhri.tasks requires mujoco (`pip install mujoco>=3.0`)") from exc

from ..schemas.tasks import ResetRandomization


def _free_qposadr(model, body_name: str) -> Optional[int]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == bid and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[j])
    return None


def _yaw_quat(yaw: float) -> np.ndarray:
    half = yaw / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def apply_randomization(
    model, data, reset: Optional[ResetRandomization], rng: Optional[np.random.Generator] = None
) -> None:
    """Mutate ``data.qpos`` in place for each randomized object, then ``mj_forward``.

    ``delta`` mode jitters around the current pose; ``absolute`` sets x/y from the
    ranges. Bodies without a free joint (or not found) are skipped silently — the
    spec may name objects absent from a given scene variant.
    """
    if reset is None or not reset.objects:
        return
    if rng is None:
        rng = np.random.default_rng(reset.seed)

    for obj in reset.objects:
        adr = _free_qposadr(model, obj.object)
        if adr is None:
            continue
        dx = rng.uniform(obj.x[0], obj.x[1])
        dy = rng.uniform(obj.y[0], obj.y[1])
        dyaw = rng.uniform(obj.yaw[0], obj.yaw[1])
        if reset.mode == "delta":
            data.qpos[adr + 0] += dx
            data.qpos[adr + 1] += dy
        else:  # absolute
            data.qpos[adr + 0] = dx
            data.qpos[adr + 1] = dy
        if abs(dyaw) > 0:
            data.qpos[adr + 3 : adr + 7] = _yaw_quat(dyaw)

    mujoco.mj_forward(model, data)
