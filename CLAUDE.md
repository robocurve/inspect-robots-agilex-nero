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
- `check.py`: `inspect-robots-nero-check`, the read-only pre-motion check. It
  must never enable the arm, command it, or write to the hand.
- `embodiment.py`: target ownership, workspace clamp, homing, teardown.

## Working here

- Set `UV_CACHE_DIR=$PWD/.uv-cache` for uv commands.
- Gates: ruff check, ruff format check, strict mypy, pytest at 100 percent
  statement and branch coverage.
- Keep pyAgxArm, pyserial and OpenCV imports lazy. Construction is inert.

## Safety invariants

- Never call pyAgxArm `disable()` or `reset()`: they drop raised joints.
  The stop path is `hold()`; `emergency_stop()` uses the electronic e-stop.
- Every action is clamped to its bound and speed-capped in any direction; the
  target is clamped to the workspace box and to `max_target_lead_m` ahead of
  the measured hand.
- Readiness is confirmed before connecting, because connecting enables the arm.
- Any failure in reset, step or park holds the arm, then closes.
- `hold()` never sends stale joint angles: without fresh feedback it uses the
  electronic e-stop.
- `close()` holds, then closes every connection, and never disables motors.

## CI and releases

- CI installs from `uv.lock`; `ci-ok` needs every blocking job, including
  import hygiene (the package imports with only inspect-robots and numpy).
- Versions come from git tags through hatch-vcs. Design notes live in
  `plans/` (`0001-nero-inspire-design.md`).

## Writing style

- No em dashes in prose. No decorative emoji. Plain headers.
