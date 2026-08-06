#!/usr/bin/env python3
"""A complete mjhri study in one file — run it, then read it.

Defines a 3-arm within-subjects design, assigns a participant, walks them through
every phase with the gates enforced, scores a robot policy on a real MuJoCo scene,
and exports the analysis CSVs. No robot hardware, no GPU, no network.

    pip install "mujoco-hri-study[yaml]"
    python examples/teaching-trust-study/fetch_assets.py     # one-time, for the scenes
    python examples/quickstart.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mjhri import (
    Condition,
    JsonlEventLog,
    PhaseSpec,
    ProtocolEngine,
    StudyEvent,
    StudyProtocol,
    TaskSpec,
    assign,
    get_instrument,
    summarize,
)
from mjhri.events.export import write_events_csv, write_summary_csv

HERE = Path(__file__).parent
STUDY = HERE / "teaching-trust-study"

# ---------------------------------------------------------------- 1. the design
# A study is data. Three teaching modalities, each run through the same phases.
protocol = StudyProtocol(
    id="teaching-trust",
    name="Does how you teach a robot change whether you trust it?",
    conditions=[
        Condition(id="A", label="Direct teleoperation", modality="teleop"),
        Condition(id="B", label="Reward specification", modality="reward_spec"),
        Condition(id="C", label="Natural-language description", modality="description"),
    ],
    phases=[
        PhaseSpec(name="teach", kind="teach",
                  requires_events=["spec_marked_deployable"]),
        PhaseSpec(name="review", kind="survey",
                  requires_instruments=["trust_scale"]),
        # group="deploy" runs after *every* condition's main phases — the two-visit
        # flow you need when training happens asynchronously between sessions.
        PhaseSpec(name="deploy", kind="decision", group="deploy",
                  requires_events=["deploy_decision"]),
    ],
    tasks=["place", "sort", "stack"],
    instruments=["trust_scale"],
)

# ------------------------------------------------------- 2. deterministic plan
# Condition order comes from a Williams square: balanced for first-order carryover,
# and a pure function of the participant id, so a refresh or crash re-derives it.
participant = "P7"
plan = assign(participant, protocol)
print(f"\n{participant} → conditions {plan.condition_order}")
for cond, spec in plan.task_by_condition.items():
    print(f"    {cond}: {spec['task']} ({spec['variant']})")

# ------------------------------------------------------------ 3. run the study
# The engine holds no state — it derives the current phase from the log every time.
log = JsonlEventLog(Path(tempfile.mkdtemp()) / "events.jsonl")
engine = ProtocolEngine(protocol, plan)
log.emit_many(engine.start(log.query(participant)))

trust = get_instrument("trust_scale")
print(f"\nrunning {len(engine.steps)} steps "
      f"({len(protocol.phases)} phases × {len(protocol.conditions)} conditions)")

while not engine.state(log.query(participant)).is_complete:
    state = engine.state(log.query(participant))

    # Whatever your UI does during this phase, it does here. We satisfy the gate.
    _, missing = engine.can_advance(log.query(participant))
    for gate in missing:
        kind, name = gate.split("'")[0], gate.split("'")[1]
        if kind.startswith("requires event"):
            payload = {"deploy": True, "confidence": 4} if name == "deploy_decision" else {}
            log.emit(StudyEvent(participant_id=participant, event_type=name,
                                phase=state.phase_name, condition=state.condition_id,
                                payload=payload))
        else:
            log.emit(StudyEvent(
                participant_id=participant, event_type="survey_submitted",
                phase=state.phase_name, condition=state.condition_id,
                payload={"instrument": name,
                         "responses": {item.id: 5 for item in trust.items}}))

    log.emit_many(engine.advance(log.query(participant)))
    print(f"  ✓ {state.condition_id}/{state.phase_name}")

# A gate you have not satisfied refuses to advance, rather than silently skipping:
try:
    engine.advance(log.query(participant))
except RuntimeError as exc:
    print(f"\ngate holds after completion: {exc}")

# ------------------------------------------------- 4. score a policy in MuJoCo
# Success is geometric and scene-agnostic: named bodies inside named region sites.
scene_ready = (STUDY / "scenes" / "so_arm100.xml").is_file()
if scene_ready:
    import mujoco

    from mjhri.learn import plan_from_taskspec, score_policy
    from mjhri.robots import SO_ARM100

    spec = TaskSpec.model_validate(
        json.loads((STUDY / "tasks" / "place_medium.json").read_text()))
    model = mujoco.MjModel.from_xml_path(str(STUDY / spec.scene))
    SO_ARM100.require(model)          # fails loudly if the scene and arm disagree

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    picks = plan_from_taskspec(model, data, spec)
    targets = {body: list(place) for _grasp, place, body in picks}

    result = score_policy(
        model,
        lambda: SO_ARM100.controller(model).set_plan(picks),
        spec,
        grasp_factory=lambda: SO_ARM100.grasp(model, place_targets=targets),
        n_rollouts=5,
    )
    print(f"\npolicy on '{spec.name}': {result['successes']}/{result['n']} "
          f"rollouts succeeded (rate {result['rate']})")
    log.emit(StudyEvent(participant_id=participant, event_type="eval_completed",
                        condition="A", payload={"success_rate": result["rate"]}))
else:
    print("\nskipping the MuJoCo rollout — run "
          "`python examples/teaching-trust-study/fetch_assets.py` first")

# ------------------------------------------------------------- 5. the dataset
events = log.query(participant)
summary = summarize(events)
print(f"\n{len(events)} events logged; complete={summary['complete']}; "
      f"{len(summary['surveys'])} surveys")

out = Path(tempfile.mkdtemp())
write_events_csv(events, out / "events.csv")
write_summary_csv({participant: events}, out / "summary.csv")
print(f"exported → {out}/events.csv, {out}/summary.csv")
print("\nThe log is the dataset: every measure is a pure function of these rows.")
