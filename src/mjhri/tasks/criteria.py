"""Geometric primitives for success scoring.

All checks take a ``mujoco.MjModel`` + ``MjData`` (already stepped/forwarded) and
reference bodies/sites **by name**, so they work on any scene. Kept dependency-free
beyond mujoco/numpy and side-effect-free (they read state, never step).
"""

from __future__ import annotations

import numpy as np

try:  # mujoco is a hard dependency, but keep the import error legible
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("mjhri.tasks requires mujoco (`pip install mujoco>=3.0`)") from exc


def body_id(model, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"no body named {name!r} in scene")
    return bid


def site_id(model, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"no site named {name!r} in scene")
    return sid


def _free_dofadr(model, bid: int) -> int | None:
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == bid and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_dofadr[j])
    return None


def object_speed(model, data, bid: int) -> float:
    """Linear speed (m/s) of a free body; 0 for a body with no free joint."""
    adr = _free_dofadr(model, bid)
    if adr is None:
        return 0.0
    return float(np.linalg.norm(data.qvel[adr : adr + 3]))


def centroid(data, bid: int) -> np.ndarray:
    return np.asarray(data.xpos[bid], dtype=float)


def up_axis(data, bid: int) -> np.ndarray:
    """The body's local +z expressed in world coordinates."""
    return np.asarray(data.xmat[bid], dtype=float).reshape(3, 3)[:, 2]


def point_in_site(model, data, point: np.ndarray, sid: int, margin: float = 0.0) -> bool:
    """Whether a world point lies inside a box/cylinder/sphere site, shrunk by margin."""
    pos = np.asarray(data.site_xpos[sid], dtype=float)
    rot = np.asarray(data.site_xmat[sid], dtype=float).reshape(3, 3)
    local = rot.T @ (point - pos)
    size = np.asarray(model.site_size[sid], dtype=float)
    stype = model.site_type[sid]
    if stype == mujoco.mjtGeom.mjGEOM_BOX:
        return bool(np.all(np.abs(local) <= np.maximum(size - margin, 0.0)))
    if stype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return bool(
            np.hypot(local[0], local[1]) <= max(size[0] - margin, 0.0)
            and abs(local[2]) <= size[1]
        )
    # sphere / other: bounding radius
    return bool(np.linalg.norm(local) <= max(size[0] - margin, 0.0))


def bodies_in_contact(model, data, b1: int, b2: int) -> bool:
    target = {b1, b2}
    for i in range(data.ncon):
        c = data.contact[i]
        if {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])} == target:
            return True
    return False


def check_in_region(
    model, data, *, obj: str, region: str, margin: float,
    require_resting: bool, rest_speed: float,
) -> tuple[bool, dict]:
    bid = body_id(model, obj)
    sid = site_id(model, region)
    inside = point_in_site(model, data, centroid(data, bid), sid, margin)
    speed = object_speed(model, data, bid)
    resting = (not require_resting) or (speed <= rest_speed)
    return (inside and resting), {"inside": inside, "speed": round(speed, 4), "resting": resting}


def check_stack(
    model, data, *, objects: list[str], xy_tol: float, require_contact: bool,
    upright_tol: float, min_layers: int | None,
) -> tuple[bool, dict]:
    ids = [body_id(model, o) for o in objects]
    # order bottom → top by centroid height
    order = sorted(ids, key=lambda b: centroid(data, b)[2])
    cos_thresh = float(np.cos(upright_tol))
    details: dict = {"layers": len(order), "pairs": []}

    # upright: every block's up-axis close to world +z
    upright_ok = True
    for b in order:
        cos_up = float(up_axis(data, b) @ np.array([0.0, 0.0, 1.0]))
        if cos_up < cos_thresh:
            upright_ok = False
    details["upright_ok"] = upright_ok

    aligned_ok = True
    contact_ok = True
    for lower, upper in zip(order, order[1:]):
        pl, pu = centroid(data, lower), centroid(data, upper)
        xy = float(np.hypot(pu[0] - pl[0], pu[1] - pl[1]))
        above = pu[2] > pl[2]
        touch = bodies_in_contact(model, data, lower, upper) if require_contact else True
        pair_ok = xy <= xy_tol and above and touch
        aligned_ok = aligned_ok and (xy <= xy_tol and above)
        contact_ok = contact_ok and touch
        details["pairs"].append({"xy": round(xy, 4), "above": above, "contact": touch})

    layers_ok = len(order) >= (min_layers if min_layers is not None else len(objects))
    success = upright_ok and aligned_ok and contact_ok and layers_ok
    details.update({"aligned_ok": aligned_ok, "contact_ok": contact_ok, "layers_ok": layers_ok})
    return success, details
