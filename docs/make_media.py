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
import math
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

#: Filmstrip panels. Landscape, because the frame has to hold the arm's whole
#: horizontal sweep — a squarer panel just adds empty floor above and below it.
W, H = 700, 470
# True isometric: an orthographic projection at 45° azimuth and the isometric
# elevation, atan(1/√2) ≈ 35.264°, which is what makes the floor grid read as even
# diamonds and keeps equal lengths equal anywhere in frame. Azimuth 45 puts the
# gripper's opening toward the viewer with the task objects unobstructed; the other
# three isometric azimuths hide either the jaws or the objects behind the arm.
ISO_AZIMUTH = 45.0
ISO_ELEVATION = -math.degrees(math.atan(1 / math.sqrt(2)))
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


def visible_bounds(model, data):
    """World AABB of what will actually be drawn.

    Skips the ground plane (infinite, would swallow any framing) and the collision
    hulls, which are group 3 and invisible but sit outside the visual meshes.
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if model.geom_rgba[g][3] == 0 or model.geom_group[g] == 3:
            continue
        radius = float(np.max(model.geom_size[g])) or 0.01
        lo = np.minimum(lo, data.geom_xpos[g] - radius)
        hi = np.maximum(hi, data.geom_xpos[g] + radius)
    return lo, hi


def iso_camera(model, lo, hi, size, pad=1.06):
    """An isometric camera framed to contain ``lo``..``hi`` exactly.

    MuJoCo reads ``vis.global_.fovy`` as **degrees for a perspective camera but a
    length for an orthographic one** — it is the height of the view volume in metres,
    and it is what sets the zoom. ``cam.distance`` only positions the camera (so it
    still governs near/far clipping); changing it does not scale an ortho image.

    The fit is exact rather than a guess: project the eight corners of the box onto
    the camera's own axes and take the extremes, so nothing lands outside the frame.
    """
    model.vis.global_.orthographic = 1
    center = (lo + hi) / 2

    a, e = math.radians(ISO_AZIMUTH), math.radians(ISO_ELEVATION)
    forward = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - center
    half_w = float(np.max(np.abs(corners @ right)))
    half_h = float(np.max(np.abs(corners @ up)))
    aspect = size[0] / size[1]
    model.vis.global_.fovy = max(2 * half_h, 2 * half_w / aspect) * pad

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation = ISO_AZIMUTH, ISO_ELEVATION
    cam.lookat[:] = center
    cam.distance = float(np.max(hi - lo)) * 4      # clear of the scene; not a zoom
    return cam


# -- still frames of each task ------------------------------------------------


def settled(task_id: str):
    """The scene at rest with the arm in its ready pose."""
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
    return model, data


def still(task_id: str, size=(760, 640), bounds=None) -> Image.Image:
    model, data = settled(task_id)
    lo, hi = bounds if bounds is not None else visible_bounds(model, data)
    cam = iso_camera(model, lo, hi, size)
    with mujoco.Renderer(model, height=size[1], width=size[0]) as r:
        r.update_scene(data, camera=cam)
        return Image.fromarray(r.render())


def task_strip(task_ids, labels, name="tasks.png", width=1400):
    # One frame shared by every panel. Framed individually, a task with a tighter
    # layout would be drawn larger, and the tasks would stop being comparable at a
    # glance — the blocks are the same size in all three, so they must look it.
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for t in task_ids:
        a, b = visible_bounds(*settled(t))
        lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    imgs = [still(t, bounds=(lo, hi)) for t in task_ids]
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


def rollout_strip(task_id: str, name: str, n_frames=4, seed=0, width=1600):
    """Four panels, not more: the strip is ~880px wide once GitHub scales it, so
    every extra panel shrinks the robot. Four still tells it — at rest, reaching,
    carrying, done."""
    spec, model = load(task_id)

    def fresh():
        """A rollout's starting state — identical every time, so the measuring pass
        and the rendering pass see exactly the same motion."""
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
        return data, policy, grasp

    dt, ci = float(model.opt.timestep), 1.0 / 30.0

    def simulate(data, policy, grasp, end, at=None):
        """Step to ``end``, calling ``at(t)`` when each scheduled moment is reached."""
        ctrl = np.asarray(policy.initial_ctrl()).reshape(-1)
        last, t = -1.0, 0.0
        while t < end:
            if t - last >= ci:
                ctrl = np.asarray(policy.act(data.qpos.copy(), t)).reshape(-1)
                last = t
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            grasp.update(model, data, t)
            if at:
                at(t)
            t += dt
        for _ in range(200):          # settle, exactly as rollout_once does
            mujoco.mj_step(model, data)
            grasp.update(model, data, end)
        if at:
            at(end)

    # Pass 1 — physics only, to learn how far the arm travels. Framing each panel
    # independently would make the camera breathe between frames; framing on the
    # start pose alone would clip the arm at full reach. So measure the whole motion
    # once (cheap, no rendering) and hold that frame fixed for every panel.
    data, policy, grasp = fresh()
    end = float(getattr(policy, "duration", 6.0)) + 1.0
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)

    def measure(_t):
        nonlocal lo, hi
        a, b = visible_bounds(model, data)
        lo, hi = np.minimum(lo, a), np.maximum(hi, b)

    simulate(data, policy, grasp, end, at=measure)

    # Pass 2 — the same rollout again, rendered through that fixed camera.
    data, policy, grasp = fresh()
    cam = iso_camera(model, lo, hi, (W, H), pad=1.04)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    grab = [end * k / (n_frames - 1) for k in range(n_frames - 1)] + [end]
    frames, gi = [], 0

    with mujoco.Renderer(model, height=H, width=W) as r:
        def shot(t):
            r.update_scene(data, camera=cam, scene_option=opt)
            frames.append((t, Image.fromarray(r.render())))

        def maybe_shoot(t):
            nonlocal gi
            if gi < len(grab) and t >= grab[gi]:
                shot(t)
                gi += 1

        simulate(data, policy, grasp, end, at=maybe_shoot)
        outcome = Scorer.from_spec(spec).score(model, data)

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
