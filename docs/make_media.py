#!/usr/bin/env python3
"""Regenerate the README figures from the bundled study — no hand-editing.

Every image in ``docs/media/`` that shows the simulator comes out of this script,
so the figures cannot drift from what the code actually does: the filmstrips are
real rollouts, and the SUCCESS/FAILURE badge is the task's own ``Scorer`` verdict,
not a caption.

    python examples/teaching-trust-study/fetch_assets.py   # the arm meshes
    pip install pillow
    python docs/make_media.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjhri import TaskSpec
from mjhri.learn import plan_from_taskspec
from mjhri.robots import SO_ARM100
from mjhri.tasks import Scorer

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "examples" / "teaching-trust-study"
OUT = ROOT / "docs" / "media"

W, H = 560, 460
# Azimuth 70 looks at the arm's FRONT — the side the gripper opens toward and the
# side the task objects are laid out on. The opposite view (250) frames the same
# scene but from behind the shoulder, where the wrist housings hide the jaws.
CAM = dict(azimuth=70, elevation=-34, distance=0.60, lookat=(0.0, -0.12, 0.04))
# The pose the arm holds while a scene is shown at rest — the same ready pose the
# controller seeds its IK from, so a still frame matches where a rollout starts.
READY = dict(zip(SO_ARM100.arm_joints, SO_ARM100.ready_arm or ()))

INK, MUTED = (15, 23, 42), (90, 100, 115)
GREEN, RED = (21, 128, 61), (185, 28, 28)


def font(size: int, bold: bool = True):
    path = ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf")
    for p in (path, "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
              % ("-Bold" if bold else "")):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load(task_id: str):
    spec = TaskSpec.model_validate(
        json.loads((STUDY / "tasks" / f"{task_id}.json").read_text()))
    model = mujoco.MjModel.from_xml_path(str(STUDY / spec.scene))
    SO_ARM100.require(model)
    return spec, model


def camera(**over):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cfg = {**CAM, **over}
    cam.azimuth, cam.elevation, cam.distance = cfg["azimuth"], cfg["elevation"], cfg["distance"]
    cam.lookat[:] = cfg["lookat"]
    return cam


# -- still frames of each task ------------------------------------------------


def still(task_id: str, size=(900, 620), **cam_over) -> Image.Image:
    _spec, model = load(task_id)
    data = mujoco.MjData(model)
    for name, q in READY.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = q
        if aid >= 0:
            data.ctrl[aid] = q
    mujoco.mj_forward(model, data)
    for _ in range(400):
        mujoco.mj_step(model, data)
    with mujoco.Renderer(model, height=size[1], width=size[0]) as r:
        r.update_scene(data, camera=camera(**cam_over))
        return Image.fromarray(r.render())


def task_strip(task_ids, labels, name="tasks.png", width=1400):
    imgs = [still(t, azimuth=70, elevation=-38, distance=0.66,
                  lookat=(0.0, -0.13, 0.03)) for t in task_ids]
    w, h, pad, cap = imgs[0].width, imgs[0].height, 8, 34
    strip = Image.new("RGB", (w * len(imgs) + pad * (len(imgs) - 1), h + cap), "white")
    d = ImageDraw.Draw(strip)
    for k, (im, lab) in enumerate(zip(imgs, labels)):
        x = k * (w + pad)
        strip.paste(im, (x, 0))
        d.text((x + (w - d.textlength(lab, font=font(22))) / 2, h + 6), lab,
               fill=INK, font=font(22))
    scale = width / strip.width
    strip = strip.resize((width, int(strip.height * scale)), Image.LANCZOS)
    strip.save(OUT / name, optimize=True)
    print(f"  {name}")


# -- filmstrip of a real, scored rollout --------------------------------------


def rollout_strip(task_id: str, name: str, n_frames=5, seed=0, width=1600):
    spec, model = load(task_id)
    data = mujoco.MjData(model)
    if spec.reset_randomization is not None:
        from mjhri.tasks.randomize import apply_randomization
        apply_randomization(model, data, spec.reset_randomization,
                            rng=np.random.default_rng(seed))
    mujoco.mj_forward(model, data)

    picks = plan_from_taskspec(model, data, spec)
    targets = {b: list(p) for _g, p, b in picks}
    policy = SO_ARM100.controller(model).set_plan(picks)
    grasp = SO_ARM100.grasp(model, place_targets=targets)
    grasp.reset()

    cam, opt = camera(), mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    dt, ci = float(model.opt.timestep), 1.0 / 30.0
    end = float(getattr(policy, "duration", 6.0)) + 1.0
    grab = [end * k / (n_frames - 1) for k in range(n_frames - 1)] + [end]
    frames, gi = [], 0

    with mujoco.Renderer(model, height=H, width=W) as r:
        def shot(t):
            r.update_scene(data, camera=cam, scene_option=opt)
            frames.append((t, Image.fromarray(r.render())))

        ctrl = np.asarray(policy.initial_ctrl()).reshape(-1)
        last, t = -1.0, 0.0
        while t < end:
            if t - last >= ci:
                ctrl = np.asarray(policy.act(data.qpos.copy(), t)).reshape(-1)
                last = t
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            grasp.update(model, data, t)
            if gi < len(grab) and t >= grab[gi]:
                shot(t)
                gi += 1
            t += dt
        for _ in range(200):          # settle, exactly as rollout_once does
            mujoco.mj_step(model, data)
            grasp.update(model, data, end)
        outcome = Scorer.from_spec(spec).score(model, data)
        shot(end)

    frames = frames[-n_frames:]
    pad, top, bot = 6, 40, 30
    strip = Image.new("RGB", (W * len(frames) + pad * (len(frames) - 1), H + top + bot), "white")
    d = ImageDraw.Draw(strip)
    for k, (t, im) in enumerate(frames):
        x = k * (W + pad)
        strip.paste(im, (x, top))
        d.text((x + 10, top + H + 6), f"t = {t:.1f}s", fill=MUTED, font=font(19, False))

    d.text((4, 8), spec.name, fill=INK, font=font(24))
    verdict = "SUCCESS" if outcome.success else "FAILURE"
    d.text((strip.width - d.textlength(verdict, font=font(24)) - 6, 8), verdict,
           fill=GREEN if outcome.success else RED, font=font(24))
    detail = " · ".join(f"{c.label}: {'ok' if c.success else 'no'}" for c in outcome.criteria)
    d.text((strip.width - d.textlength(detail, font=font(17, False)) - 6, top + H + 6),
           detail, fill=MUTED, font=font(17, False))

    scale = width / strip.width
    strip.resize((width, int(strip.height * scale)), Image.LANCZOS).save(OUT / name, optimize=True)
    print(f"  {name}  scored: {'SUCCESS' if outcome.success else 'FAILURE'}")
    return outcome.success


if __name__ == "__main__":
    if not (STUDY / "scenes" / "so_arm100.xml").is_file():
        raise SystemExit("run examples/teaching-trust-study/fetch_assets.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    print("rendering:")
    task_strip(["place_medium", "sort_medium", "stack_medium"], ["Place", "Sort", "Stack"])
    ok = [rollout_strip("place_medium", "rollout-place.png"),
          rollout_strip("stack_medium", "rollout-stack.png")]
    print("done" if all(ok) else "done — WARNING: a rollout failed, figures show it")
