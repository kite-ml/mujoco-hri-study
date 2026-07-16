"""Scorer: geometric success on real (tiny) MuJoCo scenes."""

import mujoco
import numpy as np
import pytest

from mjhri.schemas.tasks import RegionCriterion, StackCriterion, TaskSpec
from mjhri.tasks import Scorer

_REGION_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <body name="block_1" pos="0 0 0.015">
      <freejoint/>
      <geom type="box" size="0.015 0.015 0.015" mass="0.03"/>
    </body>
    <body name="bin" pos="0.2 0 0">
      <geom type="box" size="0.05 0.05 0.004" pos="0 0 0.004"/>
      <site name="bin_region" type="box" size="0.05 0.05 0.03" pos="0 0 0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


def _forward(model, data):
    mujoco.mj_forward(model, data)


def test_in_region_success_and_failure():
    model = mujoco.MjModel.from_xml_string(_REGION_XML)
    data = mujoco.MjData(model)
    spec = TaskSpec(
        id="place", name="Place", scene="inline",
        success_criteria=[RegionCriterion(object="block_1", region="bin_region")],
    )
    scorer = Scorer.from_spec(spec)

    # block outside the bin (at origin) → fail
    _forward(model, data)
    assert scorer.score(model, data).success is False

    # move the block over the bin → success
    adr = model.jnt_qposadr[0]
    data.qpos[adr : adr + 3] = [0.2, 0.0, 0.02]
    _forward(model, data)
    out = scorer.score(model, data)
    assert out.success is True
    assert out.criteria[0].detail["inside"] is True


def _stack_xml(n=3):
    bodies = "".join(
        f'<body name="block_{i+1}" pos="0.15 0 {0.015 + i*0.0298}">'
        f'<freejoint/><geom type="box" size="0.015 0.015 0.015" mass="0.03"/></body>'
        for i in range(n)
    )
    return f'<mujoco><worldbody><geom name="floor" type="plane" size="0 0 0.05"/>{bodies}</worldbody></mujoco>'


def test_stack_success():
    model = mujoco.MjModel.from_xml_string(_stack_xml(3))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    spec = TaskSpec(
        id="stack", name="Stack", scene="inline",
        success_criteria=[StackCriterion(objects=["block_1", "block_2", "block_3"], xy_tol=0.02)],
    )
    out = Scorer.from_spec(spec).score(model, data)
    assert out.success is True, out.criteria[0].detail


def test_stack_fails_when_toppled():
    model = mujoco.MjModel.from_xml_string(_stack_xml(3))
    data = mujoco.MjData(model)
    # shove the top block sideways, off the tower
    adr = model.jnt_qposadr[2]
    data.qpos[adr : adr + 3] = [0.30, 0.0, 0.015]
    mujoco.mj_forward(model, data)
    spec = TaskSpec(
        id="stack", name="Stack", scene="inline",
        success_criteria=[StackCriterion(objects=["block_1", "block_2", "block_3"], xy_tol=0.02)],
    )
    out = Scorer.from_spec(spec).score(model, data)
    assert out.success is False
