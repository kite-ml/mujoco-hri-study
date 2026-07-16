# mujoco-hri-study (`mjhri`)

An experiment engine for **human-robot interaction studies in MuJoCo** — think
jsPsych/PsychoPy, but for robot-teaching studies.

A study is defined as **data** (YAML/JSON): conditions, phase sequences,
counterbalancing, survey instruments, tasks, and success criteria. The engine
runs the protocol; your platform (or the bundled native runner) supplies the UI.

## What it does

1. **Protocol** — phase state machine (teach → train → review, gates, practice
   blocks), deterministic condition/task assignment from a participant ID
   (Williams Latin square), crash/refresh resume.
2. **Events** — a standardized event log (`EventLog` interface; JSONL and SQLite
   backends built in), derived measures (phase durations, time-to-deployable-spec,
   effort counts), anonymized CSV export. The log *is* the analysis dataset.
3. **Instruments** — survey schemas plus built-in instruments (trust scale,
   NASA-TLX, control/steerability/effort, deploy confidence) as renderer-agnostic
   JSON.
4. **Scoring** — geometric task success (`in_region`, `stack`) evaluated against
   named bodies/sites of **any MuJoCo scene**, on live `MjData` or recorded qpos
   snapshots; plus reset randomization.
5. **Teleop** *(optional extra)* — keyboard end-effector control with generic
   damped-least-squares IK and LeRobot-format demonstration recording.

## What it deliberately does not do

- **No compute**: training/inference are host concerns; the engine only records
  that they happened.
- **No storage opinions**: implement `EventLog` for your database; built-in
  backends are plain local files.
- **No telemetry**: this package never sends data anywhere.

## Host integration

Implement `mjhri.events.EventLog`, drive `mjhri.protocol.ProtocolEngine` from
your app, validate surveys with `mjhri.instruments`, and score rollouts with
`mjhri.tasks.Scorer`. See `docs/` (host-integration guide, forthcoming).

License: Apache-2.0.
