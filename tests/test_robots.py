"""Robot profiles: compatibility checking, and the wiring that makes learn work.

The end-to-end test here is the regression guard for the failure that motivated
profiles at all: with the wrong end-effector offset the whole pipeline runs to
completion and scores 0%, reporting nothing. If ``test_profile_solves_place`` ever
goes back to 0, the shipped numbers have drifted from the shipped scenes.
"""

import json
from pathlib import Path

import mujoco
import pytest

from mjhri.robots import (
    SO_ARM100,
    ProfileMismatch,
    RobotProfile,
    available_robots,
    get_robot,
    register_robot,
)
from mjhri.schemas.tasks import TaskSpec

STUDY = Path(__file__).parent.parent / "examples" / "teaching-trust-study"
ARM_XML = STUDY / "scenes" / "so_arm100.xml"

needs_assets = pytest.mark.skipif(
    not ARM_XML.is_file(),
    reason="run examples/teaching-trust-study/fetch_assets.py to enable scene tests",
)

_BARE_XML = """
<mujoco><worldbody>
  <body name="hand"><joint name="a" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".01 .01 .01"/></body>
</worldbody>
<actuator><position name="grip" joint="a"/></actuator></mujoco>
"""


def _load(task_id):
    spec = TaskSpec.model_validate(json.loads((STUDY / "tasks" / f"{task_id}.json").read_text()))
    return spec, mujoco.MjModel.from_xml_path(str(STUDY / spec.scene))


# -- registry ----------------------------------------------------------------


def test_so_arm100_is_registered():
    assert "so_arm100" in available_robots()
    assert get_robot("so_arm100") is SO_ARM100


def test_unknown_robot_lists_alternatives():
    with pytest.raises(KeyError, match="so_arm100"):
        get_robot("nonexistent_arm")


def test_register_custom_profile():
    profile = RobotProfile(name="_test_arm", ee_body="hand", arm_joints=("a",),
                           gripper_actuator="grip")
    register_robot(profile)
    assert get_robot("_test_arm") is profile


# -- compatibility checking --------------------------------------------------


def test_mismatch_names_every_missing_part():
    model = mujoco.MjModel.from_xml_string(_BARE_XML)
    missing = SO_ARM100.missing(model)
    assert any("Fixed_Jaw" in m for m in missing)
    assert any("Rotation" in m for m in missing)
    assert any("Jaw" in m for m in missing)


def test_require_raises_on_an_incompatible_scene():
    model = mujoco.MjModel.from_xml_string(_BARE_XML)
    with pytest.raises(ProfileMismatch, match="so_arm100"):
        SO_ARM100.require(model)


def test_require_is_quiet_on_a_compatible_scene():
    model = mujoco.MjModel.from_xml_string(_BARE_XML)
    profile = RobotProfile(name="_bare", ee_body="hand", arm_joints=("a",),
                           gripper_actuator="grip")
    assert profile.missing(model) == []
    profile.require(model)  # does not raise


# -- the shipped scenes ------------------------------------------------------


@needs_assets
@pytest.mark.parametrize("task_id", [
    "place_medium", "place_hard", "sort_medium", "sort_hard",
    "stack_medium", "stack_hard",
])
def test_every_example_scene_loads_and_matches_the_profile(task_id):
    spec, model = _load(task_id)
    SO_ARM100.require(model)
    for crit in spec.success_criteria:
        for body in getattr(crit, "objects", None) or [crit.object]:
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0, body
        region = getattr(crit, "region", None)
        if region:
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, region) >= 0, region


@needs_assets
def test_profile_builds_each_layer():
    _spec, model = _load("place_medium")
    assert SO_ARM100.controller(model) is not None
    assert SO_ARM100.grasp(model) is not None
    assert SO_ARM100.ik(model) is not None


# -- the regression guard ----------------------------------------------------


@needs_assets
def test_profile_solves_place():
    """The shipped profile must actually complete the shipped task.

    Wrong numbers do not raise — they score zero. This asserts the pipeline works.
    """
    from mjhri.learn import plan_from_taskspec, score_policy

    spec, model = _load("place_medium")
    SO_ARM100.require(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    picks = plan_from_taskspec(model, data, spec)
    assert len(picks) == 1
    targets = {body: list(place) for _grasp, place, body in picks}

    result = score_policy(
        model,
        lambda: SO_ARM100.controller(model).set_plan(picks),
        spec,
        grasp_factory=lambda: SO_ARM100.grasp(model, place_targets=targets),
        n_rollouts=3,
    )
    assert result["rate"] > 0.5, f"shipped profile no longer solves place_medium: {result}"
