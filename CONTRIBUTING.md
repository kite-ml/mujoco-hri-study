# Contributing

Thanks for looking. This package is used to run real studies, so the bar is
"would I trust a result that came out of this" rather than "does it run".

## Setup

```bash
git clone https://github.com/kite-ml/mujoco-hri-study
cd mujoco-hri-study
pip install -e ".[dev]"
python examples/teaching-trust-study/fetch_assets.py   # enables the scene-backed tests
pytest -q
```

Without the assets step, the tests that load the bundled scenes skip rather than
fail — so a green run does not mean you exercised them. CI fetches them.

## What we're looking for

Most valuable, roughly in order:

- **New robot profiles.** A `RobotProfile` for an arm you have measured is the
  single highest-leverage contribution — it makes the whole `learn` stack usable
  on hardware we do not own. See [`src/mjhri/robots.py`](src/mjhri/robots.py).
- **New success criteria.** `in_region` and `stack` cover a lot but not
  everything. Criteria are small, self-contained, and testable against a
  ten-line MuJoCo scene.
- **Instruments.** Validated HRI scales as `Instrument` JSON. Every instrument must
  declare its provenance in `metadata` — see [Instruments](#instruments) below.
- **`EventLog` backends** for stores people actually use.

## House rules

- **Minimum, lean, modular.** Prefer the smallest change that fully solves the
  problem. New dependencies need a real argument; the core deliberately imports
  nothing but pydantic, and `mjhri.learn` deliberately has no torch.
- **Measure, don't reason.** Geometry constants (offsets, tolerances, workspace
  bounds) come from probing a scene, not from what the numbers ought to be. A
  plausible-looking wrong offset does not raise — it silently scores 0%.
- **Tests describe behaviour.** Name the property under test, not the function.
  Scene-backed tests build a minimal inline XML where they can.
- **Comments explain why.** The what is in the code.
- **The event log is the dataset.** Anything that changes event shape or meaning
  is a breaking change for somebody's analysis. Add fields; don't repurpose them.

## Adding a robot profile

```python
from mjhri.robots import RobotProfile, register_robot

MY_ARM = RobotProfile(
    name="my_arm",
    ee_body="gripper_base",
    arm_joints=("j1", "j2", "j3", "j4", "j5", "j6"),
    gripper_actuator="gripper",
    ee_offset=(0.0, 0.0, -0.08),   # measured to the point between the pads
)
register_robot(MY_ARM)
```

Then prove it: `MY_ARM.require(model)` on your scene, and a test in the shape of
`test_profile_solves_place` showing a planned pick-and-place actually succeeds.
A profile that has not completed a task is a guess.

## Instruments

Every built-in instrument declares where it came from, in `metadata`. This is
enforced by `tests/test_instrument_provenance.py`, for two reasons.

**Attribution.** Some scales are free to use *on the condition that the source is
acknowledged* — the System Usability Scale is the case in point (© Digital Equipment
Corporation, 1986). Shipping the items without that notice would put every
downstream user out of compliance without them knowing.

**Validity.** `builtin_ids()` lists a validated 10-item scale next to a two-item
measure written for one study, and they look equally official. A researcher has to
be able to tell which is which — and an adapted scale has to say what changed, because
psychometric properties do not transfer just because the items look familiar.

So a new instrument carries:

```jsonc
"metadata": {
  "instrument": "What it is. Add 'ADAPTED.' if you reworded a published scale,
                 or 'Study-defined … Not a published scale.' if you wrote it.",
  "citation":   "Full reference, if it derives from published work.",
  "attribution": "Any notice the licence requires you to carry.",
  "adaptation": "What you changed, and how to report it.",
  "scale":      "Response format — required for anything rated.",
  "scoring":    "How to aggregate, including reverse-coded items."
}
```

Then add its id to `PUBLISHED` or `STUDY_DEFINED` in the provenance test. That test
fails on any instrument it has not been told about, so nothing ships bare.

## Pull requests

Explain what the change lets a researcher do that they could not do before. If it
touches protocol, events, or scoring, say what it means for a study already in the
field. Keep unrelated changes out.
