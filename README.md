# inspect-robots-agilex-nero

Run [Inspect Robots](https://github.com/robocurve/inspect-robots) evals on an
AgileX Nero 7-joint arm with an Inspire RH56 five-finger hand mounted on it.
The package registers one embodiment, `nero_inspire`.

> [!NOTE]
> Alpha, written before we had the robot. The `kinematic` backend and the
> injected-driver tests are verified. The real backend follows pyAgxArm's
> documentation and the Inspire RH56 manual but has not run on hardware.

## Action and observation contract

Each action is one control step's change, `eef_delta_pos`:

| Label | Unit | Meaning |
|---|---|---|
| `hand_x` | m | hand point forward, away from the base |
| `hand_y` | m | hand point to the left |
| `hand_z` | m | hand point up |
| `little_finger`, `ring_finger`, `middle_finger`, `index_finger` | 0 to 1 | change in that finger's closure |
| `thumb_bend` | 0 to 1 | change in thumb bend |
| `thumb_rotate` | 0 to 1 | change in thumb rotation, out to the side (0) to across the palm (1) |

These are the Inspire RH56's six motors; each finger curls as a unit. The
bound is `hand_speed / control_hz` (default 0.10 m/s at 5 Hz: 2 cm per step)
and `finger_rate / control_hz` per finger (0.2 per step). With the `agent`
policy the LLM calls `move_by`, for example
`{"hand_z": -0.1, "thumb_rotate": 1.0, "index_finger": 0.6, "thumb_bend": 0.6}`.

Observations: one camera (`scene_cam`, OpenCV device `camera_device`), and
state `hand_pos` (m), `fingers` (6, 0 to 1) and `arm_joints` (7, rad).

## How it moves the arm

- The embodiment owns the hand target. Each step adds its displacement and
  clamps to a workspace box (`x_min` to `z_max`), then sends the matching
  flange pose to the arm's own planner (pyAgxArm `move_p`). The hand point is
  `tcp_offset` from the flange, set by the Inspire mount.
- The hand's orientation is held at whatever it was at reset.
- The arm holds its last target by itself, so there is no streaming deadline:
  a slow policy or a dead process leaves the arm still.
- Reset glides the hand to `home_pos` at hand speed and opens the fingers.
- Close holds the arm where it is and closes the connections. Nothing in
  this package calls pyAgxArm `disable()` or `reset()`, which drop raised
  joints. `emergency_stop()` uses the arm's electronic e-stop, which lowers it
  slowly with damping.

## Inspire hand

RS485 at 115200 baud, Inspire's own frame protocol (manual V1.0.9, section
2.2). Each finger control maps linearly between `open_angles` and
`closed_angles` (raw 0 to 1000; the manual calls 1000 open). Which way the
thumb rotation turns on the lent hand is unverified; swap its entries if
it is backwards. Reset sets the hand's speed
and force limits (`hand_speed_value`, `hand_force_value`, default 500 each).
Set `hand=none` for a station without a hand.

## Simulation (MuJoCo and Isaac Lab)

Two simulator backends run the same embodiment on the same station, defined
once in `sim_layout.py`: the Nero from AgileX's `agx_arm_urdf` (MIT) with the
right Inspire DFQ hand from Unitree's `unitree_ros` (BSD-3) on a guessed 3 cm
mount, a table at z = 0 with a red 4 cm block and a blue 5 cm block, and a
fixed scene camera. `sim_layout.SIM_SETTINGS` holds the run settings both use
(hand point, home, workspace box), so a trial means the same thing in either.
The vendor models are not shipped; build from your own checkouts:

```bash
python scripts/build_mujoco_scene.py --agx PATH/agx_arm_urdf --unitree PATH/unitree_ros \
    --out PATH/nero_scene.xml --urdf-out PATH/nero_inspire.urdf
# Isaac Lab only: convert the joined URDF (finger couplings stay mimic joints)
isaaclab.sh -p scripts/tools/convert_urdf.py PATH/nero_inspire.urdf PATH/nero_inspire.usd \
    --fix-base --headless --joint-stiffness 0 --joint-damping 0
```

- `backend=mujoco -E mujoco_scene=PATH/nero_scene.xml` runs on a laptop.
  Reaching, pushing and finger moves work; grasps are unreliable because
  MuJoCo's convex collision hulls push small blocks away.
- `backend=isaaclab -E isaac_robot_usd=PATH/nero_inspire.usd` needs a GPU.
  PhysX contacts make grasps work: a scripted pick lifted the red block 10 cm
  in 3 of 4 tries with the hand point on the block.

Both stand in for pyAgxArm's planner (6-D IK to the flange, then joint
position control; Isaac Lab servos with its damped-least-squares IK every
physics step) and the Inspire serial link (position control of the six
motors). Both run on simulation time. The arm starts palm down with the
fingers forward, because the Nero's wrist bends only about 45 degrees.

## To fill in on site

- `tcp_offset`: flange to hand point through the custom mount.
- `home_pos` and the workspace box: depend on where the table is.
- `firmware`: the Nero's firmware profile (`default`, `v111` to `v121`).
- `hand_port`, `hand_id`: the lent unit's serial port and id.
- `closed_angles`: how far each motor may close, if the full range is too much.
- `camera_device`: the station camera.

## First run on a real Nero

Nothing here has run on the physical arm yet. Please go step by step and send
us the output of each step.

1. Bring up CAN and set `firmware` to the arm's profile. Measure and set
   `tcp_offset` (flange to the point you want to steer, through the mount),
   then `home_pos` and the workspace box for the table.
2. **Arm only, slowly:** `-E hand=none -E speed_percent=10`, home well above
   the table, `unattended` off so the operator prompts appear. Reset glides
   the hand home in 2 cm steps; check it goes where you expect.
3. **Then the hand:** set `hand_port` and `hand_id`, keep `hand_force_value`
   low at first, and check each finger's direction (swap `open_angles` and
   `closed_angles` entries if one runs backwards).

Stopping: Ctrl-C leaves the arm holding its last target. To lower it, use the
arm's electronic e-stop (`emergency_stop()`), which descends slowly. Never
call pyAgxArm `disable()` or `reset()` with the arm raised: they drop it.

## Run

```bash
git clone https://github.com/robocurve/inspect-robots-agilex-nero && cd inspect-robots-agilex-nero
uv venv && source .venv/bin/activate
uv pip install inspect-robots inspect-robots-agent -e ".[hardware]"
pip install "pyAgxArm @ git+https://github.com/agilexrobotics/pyAgxArm"
sudo ip link set can0 up type can bitrate 1000000

inspect-robots doctor --embodiment nero_inspire
inspect-robots run --policy agent --embodiment nero_inspire \
    -E backend=kinematic -E unattended=true \
    -P model=anthropic/claude-opus-5-5 \
    --instruction "move the hand down 10 cm and pinch with thumb and index finger"
```

## Develop

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
uv venv && uv pip install -e ".[dev]"
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest --cov
```

## License

MIT
