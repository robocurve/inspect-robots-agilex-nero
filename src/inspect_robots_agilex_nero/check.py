"""Read-only Nero + Inspire check: listen, report, never move anything.

Run it on the machine wired to the arm, before the first reset::

    inspect-robots-nero-check -E can_channel=can0 -E hand_port=/dev/ttyUSB0

It connects to the arm without enabling it, so the motors are never powered
for motion and nothing is commanded. It then reports, for every feedback
group the adapter relies on, whether it arrived, its raw timestamp, which
clock that timestamp appears to use and how old it is, and whether the
adapter will accept it. It also reads the Inspire hand's finger angles (a
read only: no speed, force or angle is written) and one camera frame. The
exit code is 1 if the adapter would refuse anything it needs.

Why this matters: when the adapter cannot trust the arm's feedback it stops
the arm with the electronic e-stop, which lowers it slowly. This check shows
any mismatch (a renamed feedback group, a different timestamp unit) before
anything can move.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from inspect_robots_agilex_nero.config import NeroConfig
from inspect_robots_agilex_nero.inspire import ANGLE_ACT, InspireHand
from inspect_robots_agilex_nero.nero import (
    JOINT_GROUPS,
    POSE_GROUPS,
    PyAgxArmDriver,
    stamp_age,
    stamp_clock,
)


@dataclass(frozen=True)
class Check:
    """One line of the report. ``level`` is ``ok``, ``warn`` or ``fail``."""

    name: str
    level: str
    detail: str


@dataclass
class Connections:
    """What the check reads from. Any part may be missing (None) with a reason."""

    robot: Any = None
    robot_error: str | None = None
    hand: InspireHand | None = None
    hand_error: str | None = None
    camera: Callable[[], Any] | None = None
    camera_error: str | None = None
    closers: Sequence[Callable[[], None]] = ()


def _stamp_detail(stamp: Any, stale_s: float) -> tuple[str, str]:
    """(level, text) for one raw timestamp."""
    clock = stamp_clock(stamp)
    age = stamp_age(stamp)
    if clock is None or age is None:
        return "fail", f"timestamp {stamp!r} is not a usable time"
    text = f"timestamp {float(stamp):.3f} ({clock} clock), age {age:+.3f} s"
    if age > stale_s:
        return "fail", text + f": older than {stale_s} s"
    if -age > stale_s:
        return "fail", text + ": ahead of the clock"
    return "ok", text


def assess_groups(robot: Any, stale_s: float) -> list[Check]:
    """One line per internal feedback group the adapter checks."""
    parser = getattr(robot, "_parser", None)
    if parser is None:
        return [
            Check(
                "feedback groups",
                "fail",
                "pyAgxArm has no _parser: this version caches feedback differently, so the "
                "adapter cannot check freshness and will stop the arm with the e-stop",
            )
        ]
    checks = []
    for group in (*JOINT_GROUPS, *POSE_GROUPS):
        message = getattr(parser, group, None)
        if message is None:
            checks.append(Check(group, "fail", "no feedback received for this group"))
            continue
        level, text = _stamp_detail(getattr(message, "timestamp", None), stale_s)
        checks.append(Check(group, level, text))
    return checks


def assess_readings(robot: Any, stale_s: float) -> list[Check]:
    """The combined joint and pose readings, and whether the adapter accepts them.

    Accepted means what hold() and the embodiment require: fresh, plus seven
    finite joint angles and a finite six-value pose.
    """
    driver = PyAgxArmDriver(robot, stale_s=stale_s)
    checks = []

    def joints() -> None:
        q = driver.read_joints()
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise RuntimeError(f"not 7 finite joint angles: {q!r}")

    def pose() -> None:
        flange = driver.read_flange()
        parts = (np.asarray(flange.position), np.asarray(flange.rpy))
        if any(part.shape != (3,) or not np.all(np.isfinite(part)) for part in parts):
            raise RuntimeError(f"not a finite 6-value pose: {flange!r}")

    for name, getter, read in (
        ("joint angles", "get_joint_angles", joints),
        ("flange pose", "get_flange_pose", pose),
    ):
        try:
            reply = getattr(robot, getter)()
            rate = getattr(reply, "hz", None)
            values = getattr(reply, "msg", None)
            shown = (
                np.round(np.asarray(values, dtype=np.float64), 3).tolist()
                if values is not None
                else None
            )
            read()
        except Exception as exc:
            checks.append(Check(name, "fail", f"the adapter refuses it: {exc}"))
            continue
        checks.append(Check(name, "ok", f"{shown} at {rate} Hz; the adapter accepts it"))
    return checks


def assess_hand(cfg: NeroConfig, connections: Connections) -> Check:
    """One read of the Inspire hand's finger angles."""
    if cfg.hand == "none":
        return Check("hand", "ok", "not used (hand=none)")
    if connections.hand is None:
        return Check("hand", "fail", connections.hand_error or "hand not opened")
    try:
        angles = connections.hand.read_block(ANGLE_ACT)
    except Exception as exc:
        return Check("hand", "fail", f"no answer from hand id {cfg.hand_id}: {exc}")
    return Check(
        "hand", "ok", f"id {cfg.hand_id} on {cfg.hand_port}: finger angles {angles} (0 to 1000)"
    )


def assess_camera(cfg: NeroConfig, connections: Connections) -> Check:
    """One frame from the station camera."""
    if connections.camera is None:
        return Check("camera", "fail", connections.camera_error or "camera not opened")
    try:
        frame = np.asarray(connections.camera())
    except Exception as exc:
        return Check("camera", "fail", f"{cfg.camera_device!r}: {exc}")
    return Check("camera", "ok", f"{cfg.camera_device!r}: frame {tuple(frame.shape)}")


def assess(cfg: NeroConfig, connections: Connections) -> list[Check]:
    """Every check, in report order."""
    if connections.robot is None:
        checks = [Check("arm", "fail", connections.robot_error or "arm not connected")]
    else:
        checks = [
            *assess_groups(connections.robot, cfg.feedback_stale_s),
            *assess_readings(connections.robot, cfg.feedback_stale_s),
        ]
    return [*checks, assess_hand(cfg, connections), assess_camera(cfg, connections)]


def format_report(checks: list[Check]) -> str:
    """Human-readable report, one line per check, with a verdict last."""
    lines = [f"{check.level.upper():<5} {check.name:<14} {check.detail}" for check in checks]
    failed = [check.name for check in checks if check.level == "fail"]
    verdict = (
        "NOT READY: " + ", ".join(failed)
        if failed
        else "READY: the adapter will accept this arm's feedback (nothing was moved)"
    )
    return "\n".join([*lines, verdict])


ConnectFn = Callable[[NeroConfig, float], Connections]


def _parse_options(items: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"-E expects key=value; got {item!r}")
        options[key] = value
    return options


def main(argv: list[str] | None = None, *, connect_fn: ConnectFn | None = None) -> int:
    """Listen to the arm, hand and camera, print the report, return 1 if not ready."""
    parser = argparse.ArgumentParser(
        prog="inspect-robots-nero-check",
        description="Read-only check of a real Nero + Inspire before the first reset. "
        "Never enables or moves the arm.",
    )
    parser.add_argument(
        "-E", dest="options", action="append", default=[], help="nero_inspire option key=value"
    )
    parser.add_argument("--seconds", type=float, default=2.0, help="listening window")
    parser.add_argument("--json", action="store_true", help="print the checks as JSON")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive")
    options = _parse_options(args.options)
    options["backend"] = "sdk"
    cfg = NeroConfig.from_kwargs(**options)
    connections = (connect_fn or connect_read_only)(cfg, args.seconds)
    try:
        checks = assess(cfg, connections)
    finally:
        for close in connections.closers:
            try:
                close()
            except Exception as exc:
                print(f"warning: cleanup failed: {exc}")
    if args.json:
        payload = [{"name": c.name, "level": c.level, "detail": c.detail} for c in checks]
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(checks))
    return 1 if any(check.level == "fail" for check in checks) else 0


def connect_read_only(cfg: NeroConfig, seconds: float) -> Connections:  # pragma: no cover
    """Connect the arm without enabling it, open the hand port and the camera, listen."""
    import time

    from inspect_robots_agilex_nero.embodiment import _opencv_reader
    from inspect_robots_agilex_nero.inspire import open_inspire

    connections = Connections()
    closers: list[Callable[[], None]] = []
    try:
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

        config = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=getattr(NeroFW, cfg.firmware.upper()),
            channel=cfg.can_channel,
        )
        robot = AgxArmFactory.create_arm(config)
        robot.connect()  # listen only: no enable(), no mode change, no motion
        closers.append(robot.disconnect)
        connections.robot = robot
    except Exception as exc:
        connections.robot_error = f"could not connect on {cfg.can_channel}: {exc!r}"
    if cfg.hand != "none":
        try:
            hand = open_inspire(
                cfg.hand_port, cfg.hand_baud, cfg.hand_id, cfg.open_angles, cfg.closed_angles
            )
            closers.append(hand.close)
            connections.hand = hand
        except Exception as exc:
            connections.hand_error = f"could not open {cfg.hand_port}: {exc!r}"
    try:
        connections.camera = _opencv_reader(cfg)
    except Exception as exc:
        connections.camera_error = repr(exc)
    time.sleep(seconds)
    connections.closers = tuple(closers)
    return connections
