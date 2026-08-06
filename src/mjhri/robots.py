"""Robot profiles — the measured wiring that makes ``mjhri.learn`` work on an arm.

The learners in :mod:`mjhri.learn` are robot-agnostic: they ask for an end-effector
body, a list of arm joints, a gripper actuator, and a handful of *measured* offsets
and tolerances. Those numbers are not guessable — an ``ee_offset`` that is wrong by
a few centimetres puts the invisible grasp point off the finger pads, the gripper
closes on nothing, and every rollout fails at 0% with no error message.

So each supported arm gets a ``RobotProfile``: one object holding the numbers, plus
constructors that hand them to the controller / grasp / IK layers. Bring your own arm
by writing a new profile — nothing else in the package needs to change.

    from mjhri.robots import SO_ARM100
    SO_ARM100.require(model)                     # fail loudly on a mismatched scene
    ctrl  = SO_ARM100.controller(model).set_plan(picks)
    grasp = SO_ARM100.grasp(model, place_targets=targets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


class ProfileMismatch(RuntimeError):
    """A profile names bodies/joints/actuators the given MuJoCo model does not have."""


@dataclass(frozen=True)
class RobotProfile:
    """How one arm is wired into the learning + teleop layers.

    ``ee_offset`` is the vector from ``ee_body``'s frame to the point *between the
    finger pads* — the thing that must actually touch an object. ``grasp_radius`` is
    the horizontal capture tolerance (tight enough to disambiguate neighbouring
    objects) and ``grasp_radius_z`` the vertical one (loose enough to bridge the gap
    between the pads' lowest reach and an object resting on the floor).
    """

    name: str
    ee_body: str
    arm_joints: tuple[str, ...]
    gripper_actuator: str
    ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: joint targets for a pose that seeds IK inside the workspace (avoids a local
    #: minimum that leaves the arm stuck outside it); ``None`` seeds from qpos0.
    ready_arm: Optional[tuple[float, ...]] = None
    #: reachable box for planned waypoints, in world metres.
    ws_min: tuple[float, float, float] = (-0.16, -0.24, 0.012)
    ws_max: tuple[float, float, float] = (0.16, -0.05, 0.30)
    grasp_close_below: float = 0.5
    grasp_radius: float = 0.05
    grasp_radius_z: Optional[float] = None
    grasp_snap_radius: float = 0.045
    #: geoms whose centroid is the exact grasp point, when a fixed offset is too crude.
    grasp_point_geoms: Optional[tuple[str, ...]] = None
    notes: str = ""

    # -- validation --------------------------------------------------------

    def missing(self, model: Any) -> list[str]:
        """Names this profile needs that ``model`` does not define. Empty ⇒ compatible."""
        import mujoco

        def has(objtype, n):
            return mujoco.mj_name2id(model, objtype, n) >= 0

        out: list[str] = []
        if not has(mujoco.mjtObj.mjOBJ_BODY, self.ee_body):
            out.append(f"body {self.ee_body!r}")
        for j in self.arm_joints:
            if not has(mujoco.mjtObj.mjOBJ_JOINT, j):
                out.append(f"joint {j!r}")
        if not has(mujoco.mjtObj.mjOBJ_ACTUATOR, self.gripper_actuator):
            out.append(f"actuator {self.gripper_actuator!r}")
        for g in self.grasp_point_geoms or ():
            if not has(mujoco.mjtObj.mjOBJ_GEOM, g):
                out.append(f"geom {g!r}")
        return out

    def require(self, model: Any) -> None:
        """Raise :class:`ProfileMismatch` unless ``model`` defines everything needed.

        Call this once after loading a scene. It converts the silent failure mode —
        a policy that runs to completion and scores 0% — into a message naming the
        parts that do not line up.
        """
        miss = self.missing(model)
        if miss:
            raise ProfileMismatch(
                f"scene is not compatible with robot profile {self.name!r}; "
                f"missing: {', '.join(miss)}"
            )

    # -- constructors ------------------------------------------------------

    def controller(self, model: Any, *, params: Any = None, **kwargs):
        """A :class:`~mjhri.learn.controller.PickPlaceController` wired to this arm."""
        from .learn.controller import PickPlaceController

        return PickPlaceController(
            model,
            ee_body=self.ee_body,
            arm_joints=list(self.arm_joints),
            gripper_actuator=self.gripper_actuator,
            ready_arm=list(self.ready_arm) if self.ready_arm else None,
            ee_offset=self.ee_offset,
            ws_min=self.ws_min,
            ws_max=self.ws_max,
            params=params,
            **kwargs,
        )

    def grasp(self, model: Any, *, place_targets: Optional[dict] = None, **kwargs):
        """An :class:`~mjhri.learn.grasp.AutoGrasp` with this arm's measured tolerances."""
        from .learn.grasp import AutoGrasp

        return AutoGrasp(
            model,
            ee_body=self.ee_body,
            gripper_actuator=self.gripper_actuator,
            ee_offset=self.ee_offset,
            close_below=self.grasp_close_below,
            radius=self.grasp_radius,
            radius_z=self.grasp_radius_z,
            snap_radius=self.grasp_snap_radius,
            grasp_point_geoms=list(self.grasp_point_geoms) if self.grasp_point_geoms else None,
            place_targets=place_targets,
            **kwargs,
        )

    def ik(self, model: Any, **kwargs):
        """An :class:`~mjhri.teleop.ik.EEController` for teleoperating this arm."""
        from .teleop.ik import EEController

        return EEController(
            model, self.ee_body, list(self.arm_joints), ee_offset=self.ee_offset, **kwargs
        )


#: SO-ARM100 (MuJoCo Menagerie ``trs_so_arm100``). Offsets and tolerances measured
#: against the bundled study scenes; these are the numbers the teaching-trust study
#: ran on. ``ee_offset`` points from the Fixed_Jaw frame down between the pads.
SO_ARM100 = RobotProfile(
    name="so_arm100",
    ee_body="Fixed_Jaw",
    arm_joints=("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"),
    gripper_actuator="Jaw",
    ee_offset=(0.0, -0.10, 0.0),
    ready_arm=(0.0, -3.32, 2.1379, 1.66, -1.57),
    ws_min=(-0.16, -0.24, 0.012),
    ws_max=(0.16, -0.05, 0.30),
    grasp_close_below=0.5,
    grasp_radius=0.05,
    grasp_radius_z=0.11,
    grasp_snap_radius=0.075,
    notes="5-DoF + parallel jaw. Objects are reachable in a fan in front of the base (-y).",
)

_REGISTRY: dict[str, RobotProfile] = {SO_ARM100.name: SO_ARM100}


def register_robot(profile: RobotProfile) -> None:
    """Make a custom profile discoverable by name."""
    _REGISTRY[profile.name] = profile


def get_robot(name: str) -> RobotProfile:
    if name not in _REGISTRY:
        raise KeyError(f"no robot profile {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_robots() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "ProfileMismatch",
    "RobotProfile",
    "SO_ARM100",
    "available_robots",
    "get_robot",
    "register_robot",
]
