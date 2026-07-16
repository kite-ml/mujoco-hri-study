"""Teleop: generic end-effector IK + keyboard mapping + LeRobot demo recording.

Optional layer (imports mujoco; recording imports lerobot). Lets a human drive any
MuJoCo arm's end-effector from keyboard deltas and record the result as a LeRobot
dataset — the same tooling the native runner and a web host (Kite) both use.
"""

from .ik import EEController

__all__ = ["EEController"]
