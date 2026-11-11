# 0001: AgileX Nero + Inspire RH56 embodiment for LLM agents (nero_inspire)

Status: draft for review (2026-11-11)
Issue: RC-355 (Robocurve Linear), for the Pantograph robot safety hackathon (2026-11-14)

## Goal

Let an LLM agent drive a lent AgileX Nero 7-joint arm carrying an Inspire
RH56 five-finger hand on a custom mount, through Inspect Robots, with every
finger motor available. Hardware is not available before the event, so the
adapter must be testable in simulation and safe to bring up step by step.

## Contract

- `eef_delta_pos`, `frame="base"`, `gripper="continuous"`, 9-D: `hand_x`,
  `hand_y`, `hand_z` (metres per step, arm base frame), then one closure
  change per Inspire motor: `little_finger`, `ring_finger`, `middle_finger`,
  `index_finger`, `thumb_bend`, `thumb_rotate` (0 open, 1 closed).
- Bounds: `hand_speed / control_hz` (2 cm per step at the defaults) and
  `finger_rate / control_hz` (0.2 per step).
- State: `hand_pos` (3), `fingers` (6 measured closures), `arm_joints` (7).
  One camera, `scene_cam`.
- The hand orientation is held at its reset value: an LLM steers position and
  fingers, not wrist orientation.

## Real hardware (`backend=sdk`)

- Arm: pyAgxArm over CAN (AgileX's SDK for Piper and Nero; git install, not
  on PyPI). The embodiment owns a hand-point target, clamps it to a workspace
  box, converts it to a flange pose through `tcp_offset` and sends it to the
  arm's planner (`move_p`). Steps are speed-capped in any direction and the
  target never leads the measured hand by more than `max_target_lead_m`, so
  if the host stops, the arm finishes at most that short move.
- Startup: the operator confirms readiness before connecting (connecting
  enables the arm); reset refuses to home from far outside the box; any
  failure holds and closes.
- Stopping: never `disable()` or `reset()`, which pyAgxArm documents as
  dropping raised joints. `hold()` retargets the measured joints, refusing
  stale feedback and falling back to the electronic e-stop (a damped
  descent) when there is none.
- Hand: Inspire's own RS485 frame protocol (manual V1.0.9, section 2.2), with
  the manual's worked examples as test vectors. Each finger control maps
  linearly between `open_angles` and `closed_angles` (raw 0 to 1000).
- Declared runtime requirements (pyAgxArm, pyserial, OpenCV) and device slots
  (CAN channel, hand serial port, camera).

## Simulation

One station in `sim_layout.py`, used by both simulators: the Nero URDF from
AgileX `agx_arm_urdf` (MIT) joined to the right Inspire DFQ hand from Unitree
`unitree_ros` (BSD-3) by a guessed 3 cm mount, a table at z = 0, a red and a
blue block, and a fixed camera. The vendor models are not shipped;
`scripts/build_mujoco_scene.py` builds the scene and the joined URDF.

- `mujoco`: runs on a laptop. Reaching, pushing and finger moves work; grasps
  are unreliable with MuJoCo's convex collision hulls.
- `isaaclab`: the joined URDF converted to USD (finger couplings stay mimic
  joints), servoed with Isaac Lab's damped-least-squares IK. Grasps work.

Verified with an LLM agent (Claude Opus 5.5): pushing a block to a target in
MuJoCo, and picking up and holding a block in Isaac Lab.

## Open before hardware

- Who lends the arm, its firmware profile, and whether they provide their own
  embodiment or URDF.
- The mount: flange-to-hand transform (`tcp_offset`), mass, and how it angles
  the hand. The Nero wrist bends only about 45 degrees, so the mount decides
  which grasps are reachable.
- The Inspire model (DFX, DFTP or DFQ), its id, baud rate and finger
  directions, in particular thumb rotation.
- Workspace box and home position for the station's table.

## Testing

Hardware-free tests with injected backends, clock and operator, at 100
percent statement and branch coverage. The pyAgxArm, pyserial, OpenCV, MuJoCo
and Isaac Lab seams are `pragma: no cover` or omitted from coverage, and the
simulators are verified by running.
