# Roadmap

`mjhri` exists so that running an HRI study in simulation is a matter of describing
the design, not building an experiment platform. Everything below is judged against
that: does it remove work from a researcher who wants to answer a question about
people and robots?

Dates are deliberately absent. Ordering within a milestone is roughly by priority.
If you need something here, open an issue — real demand reorders this list.

---

## 0.1 — today

Complete and used to run a real three-arm within-subjects study.

- **Protocol** — event-sourced phase machine, declarative gates, crash/refresh
  resume, `deploy` phase groups for two-visit asynchronous-training flows.
- **Counterbalancing** — Williams design (first-order carryover balanced),
  independent task rotation, pure function of the participant id.
- **Events** — one schema, extensible types, `EventLog` interface, JSONL + SQLite
  backends, derived measures, anonymized CSV export.
- **Instruments** — 7 built in (trust, NASA-TLX, SUS, control/steerability/effort,
  deploy confidence, 2 attention checks), renderer-agnostic JSON.
- **Scoring** — `in_region` and `stack` criteria against any MuJoCo scene, live or
  from a `qpos` snapshot; reset randomization.
- **Learn** — pick-and-place controller, auto-grasp, CEM tuning, nearest-state
  behaviour cloning, headless rollout scoring. numpy + mujoco, no torch.
- **Teleop** — damped-least-squares EE IK with axis alignment, LeRobot recording.
- **Robots** — `RobotProfile`, with SO-ARM100 measured against the bundled scenes.

**Known limits.** One robot profile ships. Success criteria cover placement and
stacking but not orientation, sequence, or timing. There is no analysis layer beyond
CSV export, and no interface — you bring your own.

---

## 0.2 — make somebody else's study easy

The theme is everything a second lab needs that we got away with not having.

- **A PyPI release.** Today installation is `pip install git+https://…`, which is
  fine for us and awkward for everyone else — and it means a study cannot pin an
  exact published version. `pip install mujoco-hri-study` is the 0.2 gate.
- **Protocol from YAML/JSON on disk.** A study should be a file you can diff, review,
  and attach to a preregistration — not a Python literal. Loader, validator, and a
  `mjhri validate my_study.yaml` CLI that catches a gate no event can satisfy, an
  instrument that is never shown, or a criterion naming a body the scene lacks.
- **More robot profiles.** Franka Panda, UR5e, and a parallel-jaw gripper on a
  generic 6-DoF arm, each proven by a passing pick-and-place test. This is where
  outside contributions help most.
- **A host-integration guide.** The protocol/event contract written down properly,
  with a worked example of driving the engine from a web backend.
- **Power and design tooling.** Given conditions, phases, and an expected effect
  size, how many participants — and does the counterbalancing actually balance at
  that N? A one-command report beats a spreadsheet.

## 0.3 — measure more of what happened

- **Richer success criteria.** Orientation (`upright`, `aligned_to`), ordering
  (this before that), dwell/timing, and negative criteria (nothing knocked off the
  table). Same discriminated-union shape, so old specs keep working.
- **Interaction-trace measures.** Beyond counting events: hesitation, correction and
  retry structure, time-to-first-action — the behavioural DVs an HRI paper wants
  next to the survey scores.
- **Instrument provenance.** Citations, subscale definitions, and reverse-scoring in
  the instrument JSON, with scoring helpers, so a trust score is computed one agreed
  way rather than re-derived per lab.
- **Export to long format.** Tidy per-trial rows for R/`lme4` and Python, alongside
  today's wide summary.

## 0.4 — reproducibility as a feature

- **Study bundles.** One archive containing the protocol, task specs, scene hashes,
  instrument versions, package version, and the event log — enough for someone else
  to re-run the analysis and get your numbers.
- **Replay.** Reconstruct a participant's session from the log alone, for coding
  behaviour, generating stimulus figures, or auditing a run.
- **Deterministic seeds end to end.** Reset randomization, CEM, and rollout scoring
  reproducible from the bundle.

## 1.0 — stability

- **Frozen event schema and protocol contract**, with a migration path for schema
  versions. The log is somebody's dataset; breaking it breaks published work.
- **Documented public API surface** with deprecation policy.
- **A published study** using the packaged engine end to end, cited from the README
  as a worked reference.

---

## Explicit non-goals

These are not "later" — they are things this package should not become.

- **A participant-facing UI.** Rendering, consent flows, and payment integration
  belong to your platform. The engine stays headless so it can serve a web app, a
  native runner, and a notebook equally.
- **A training framework.** `mjhri.learn` exists so a study can compare teaching
  modalities on a laptop. Real policy training belongs in LeRobot, SB3, or your own
  stack; the engine records that training happened and scores what came out.
- **A database.** `EventLog` is two methods on purpose.
- **A simulator abstraction layer.** This is MuJoCo. Supporting every simulator
  would cost the scene-agnostic scoring that makes the package useful.
- **Telemetry, accounts, or a hosted service.** The package never phones home.

---

## Where help goes furthest

1. **A robot profile for an arm you have measured** — this is the single highest-leverage
   contribution, because it makes the whole learn stack usable on hardware we do not own.
2. **A success criterion** your study needed and we did not have.
3. **An `EventLog` backend** for a store people actually use.
4. **Telling us your study did not fit.** The design constraint we got wrong is worth
   more than a patch.

See [CONTRIBUTING.md](CONTRIBUTING.md).
