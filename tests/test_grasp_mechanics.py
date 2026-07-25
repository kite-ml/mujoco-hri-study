"""Regression guards for the AutoGrasp mechanics.

Locks in three behaviours that broke participant-facing UX when changed:
  * a block released NEAR its target GLIDES to it and freezes there (no teleport),
    and a placed block is never re-grabbed by trailing policy actions (this silently
    zeroed stack success — the jaw plucked the finished tower's top block back off);
  * a block released with ``hold_placed=False`` (teleop) is returned fully to
    physics — it falls instead of being pinned mid-air / inside a neighbour.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mjhri.learn.grasp import AutoGrasp

_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="ee" pos="0 0 0.2">
      <joint name="lift" type="slide" axis="0 0 1" range="-0.2 0.5"/>
      <geom name="ee_geom" type="sphere" size="0.01" contype="0" conaffinity="0"/>
    </body>
    <body name="blk" pos="0 0 0.015">
      <freejoint/>
      <geom name="blk_geom" type="box" size="0.015 0.015 0.015" mass="0.03"/>
    </body>
  </worldbody>
  <actuator>
    <position name="jaw" joint="lift" ctrlrange="0 1"/>
  </actuator>
</mujoco>
"""


def _setup(**kw):
    m = mujoco.MjModel.from_xml_string(_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    g = AutoGrasp(m, ee_body="ee", gripper_actuator="jaw",
                  radius=0.30, radius_z=0.30, **kw)
    return m, d, g


def _run(m, d, g, t0, seconds, ctrl):
    t = t0
    for _ in range(int(seconds / m.opt.timestep)):
        d.ctrl[0] = ctrl
        mujoco.mj_step(m, d)
        g.update(m, d, t)
        t += m.opt.timestep
    return t


def test_release_near_target_glides_and_freezes_at_target():
    target = np.array([0.0, 0.0, 0.12])  # within the anisotropic snap window of the release point
    m, d, g = _setup(place_targets={"blk": target}, snap_radius=0.075)
    jq = m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "blk")]]
    t = _run(m, d, g, 0.0, 0.05, ctrl=0.1)          # close → attach
    assert g._held == "blk"
    t = _run(m, d, g, t, 0.05, ctrl=0.9)            # open near target → glide starts
    assert "blk" in g._gliding or "blk" in g._frozen
    t = _run(m, d, g, t, 0.5, ctrl=0.9)             # glide completes
    assert "blk" in g._frozen
    assert np.allclose(d.qpos[jq:jq + 3], target, atol=1e-6), "block not frozen AT its target"


def test_placed_block_is_not_regrabbed_by_trailing_close():
    """After a block lands at its target, a policy's trailing jaw-close over it must
    NOT pick it back up — this exact re-grab silently zeroed stack/hard success."""
    target = np.array([0.0, 0.0, 0.12])
    m, d, g = _setup(place_targets={"blk": target}, snap_radius=0.075)
    t = _run(m, d, g, 0.0, 0.05, ctrl=0.1)          # attach
    t = _run(m, d, g, t, 0.6, ctrl=0.9)             # release → glide → placed+frozen
    assert "blk" in g._frozen and "blk" in g._placed
    t = _run(m, d, g, t, 0.2, ctrl=0.1)             # trailing CLOSE right above it
    assert g._held is None, "placed block was re-grabbed off its target"


def test_teleop_release_returns_block_to_physics():
    """hold_placed=False (teleop): a block released mid-air must FALL — never be
    pinned hovering or intersecting a neighbour (the 'unrealistic cubes' bug)."""
    m, d, g = _setup(hold_placed=False)
    jq = m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "blk")]]
    t = _run(m, d, g, 0.0, 0.05, ctrl=0.1)          # attach
    assert g._held == "blk"
    # lift the carrier: block rides to height
    d.qpos[0] = 0.3
    t = _run(m, d, g, t, 0.05, ctrl=0.1)
    high = float(d.qpos[jq + 2])
    assert high > 0.2, "block did not ride the gripper up"
    t = _run(m, d, g, t, 0.6, ctrl=0.9)             # release mid-air
    assert g._held is None and not g._frozen, "teleop release must not freeze-pin"
    assert float(d.qpos[jq + 2]) < 0.05, "released block did not fall back to the floor"
