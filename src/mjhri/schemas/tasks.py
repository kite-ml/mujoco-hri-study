"""Task + success-criteria schema — robot- and scene-agnostic.

A ``TaskSpec`` names objects/regions in *some* MuJoCo scene by string; the scorer
(``mjhri.tasks.scorer``) resolves those names against whatever model it is handed.
Nothing here is specific to a particular robot or study.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class RegionCriterion(BaseModel):
    """An object's centroid must lie inside a named region site.

    Covers both Place (one object → one bin) and Sort (each coloured object → its
    own colour's bin): Sort is simply several ``RegionCriterion`` entries. The
    region is a ``box`` or ``cylinder`` site in the scene.
    """

    type: Literal["in_region"] = "in_region"
    object: str = Field(description="body name whose centroid is tested")
    region: str = Field(description="site name defining the region volume")
    margin: float = Field(0.0, description="shrink the region by this many metres (stricter)")
    require_resting: bool = Field(True, description="also require the object nearly at rest")
    rest_speed: float = Field(0.05, description="max object speed (m/s) to count as resting")


class StackCriterion(BaseModel):
    """A set of objects forms a stable, aligned, upright tower."""

    type: Literal["stack"] = "stack"
    objects: list[str] = Field(description="body names in the tower (order not required)")
    xy_tol: float = Field(0.02, description="max horizontal centroid offset between layers (m)")
    require_contact: bool = Field(True, description="require consecutive layers to touch")
    upright_tol: float = Field(0.26, description="max tilt of each object's up-axis from world +z (rad)")
    min_layers: Optional[int] = Field(None, description="required tower height; defaults to len(objects)")


Criterion = Annotated[Union[RegionCriterion, StackCriterion], Field(discriminator="type")]


class ObjectRandomization(BaseModel):
    """Uniform jitter for one object's planar pose at reset."""

    object: str
    x: tuple[float, float] = (0.0, 0.0)
    y: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (0.0, 0.0)


class ResetRandomization(BaseModel):
    """How object poses vary between episodes.

    ``delta`` (default) jitters around each object's scene pose; ``absolute`` sets
    the pose from the ranges directly. A per-episode ``seed`` makes it reproducible.
    """

    mode: Literal["delta", "absolute"] = "delta"
    objects: list[ObjectRandomization] = Field(default_factory=list)
    seed: Optional[int] = None


class TaskSpec(BaseModel):
    """A single task/variant: which scene, what counts as success, how it resets."""

    id: str
    name: str
    robot: str = "trs_so_arm100"
    scene: str = Field(description="path/ref to the MuJoCo XML; the host resolves it")
    ee_site: Optional[str] = Field(None, description="end-effector site name for teleop/IK")
    success_criteria: list[Criterion] = Field(default_factory=list)
    reset_randomization: Optional[ResetRandomization] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
