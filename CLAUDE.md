# inspect-robots-agilex-nero agent guide

Inspect Robots embodiment `nero_inspire`: an AgileX Nero 7-joint arm with an
Inspire RH56 hand on a custom mount, steered by per-step hand displacement
`(hand_x, hand_y, hand_z)` plus a change in each of the six Inspire finger
motors.

## Layout

- `config.py`: frozen config and spaces. Station geometry is all config.
- `inspire.py`: Inspire frame protocol over RS485, tested against the
  manual's worked examples.
- `nero.py`: backend protocols, kinematic stand-ins, the pyAgxArm driver.
- `embodiment.py`: target ownership, workspace clamp, homing, teardown.

## Working here

- Set `UV_CACHE_DIR=$PWD/.uv-cache` for uv commands.
- Gates: ruff check, ruff format check, strict mypy, pytest at 100 percent
  statement and branch coverage.
- Keep pyAgxArm, pyserial and OpenCV imports lazy. Construction is inert.

## Safety invariants

- Never call pyAgxArm `disable()` or `reset()`: they drop raised joints.
  The stop path is `hold()`; `emergency_stop()` uses the electronic e-stop.
- Every action is clamped to its bound, and the target to the workspace box.
- A failing step holds the arm, then closes.
- `close()` holds, then closes every connection, and never disables motors.

## Writing style

- No em dashes in prose. No decorative emoji. Plain headers.
