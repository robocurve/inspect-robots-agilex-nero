"""The read-only Nero check: what it reports, and that it never writes or moves anything."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from inspect_robots_agilex_nero.check import (
    Connections,
    assess,
    format_report,
    main,
)
from inspect_robots_agilex_nero.config import NeroConfig
from inspect_robots_agilex_nero.inspire import InspireHand
from test_inspire import FakePort
from test_pyagxarm_driver import FakeRobot, message

OPEN = (1000,) * 6
CLOSED = (0,) * 6


def healthy(cfg: NeroConfig | None = None) -> tuple[Connections, FakeRobot, FakePort]:
    robot = FakeRobot()
    port = FakePort()
    hand = InspireHand(port, hand_id=1, open_angles=OPEN, closed_angles=CLOSED)
    closed: list[str] = []
    connections = Connections(
        robot=robot,
        hand=hand,
        camera=lambda: np.zeros((480, 640, 3), dtype=np.uint8),
        closers=(lambda: closed.append("arm"), lambda: closed.append("hand")),
    )
    return connections, robot, port


def levels(checks: list[Any]) -> dict[str, str]:
    return {check.name: check.level for check in checks}


def test_a_healthy_station_is_ready_and_nothing_is_commanded() -> None:
    cfg = NeroConfig()
    connections, robot, port = healthy()
    checks = assess(cfg, connections)
    assert all(check.level == "ok" for check in checks), format_report(checks)
    report = format_report(checks)
    assert report.endswith("READY: the adapter will accept this arm's feedback (nothing was moved)")
    assert "wall clock" in report and "joint_7" in report and "end_pose_ryrz" in report
    assert "accepts it" in report and "frame (480, 640, 3)" in report
    assert robot.calls == []  # no move, hold, e-stop or disconnect from the assessment
    assert all(frame[4] == 0x11 for frame in port.written)  # hand: reads only


def test_a_renamed_or_missing_cache_explains_the_e_stop_risk() -> None:
    connections, robot, _ = healthy()
    del robot._parser
    checks = assess(NeroConfig(), connections)
    first = checks[0]
    assert first.level == "fail" and "e-stop" in first.detail
    assert "NOT READY" in format_report(checks)


@pytest.mark.parametrize(
    ("change", "name", "text"),
    [
        (lambda r: setattr(r._parser, "joint_34", None), "joint_34", "no feedback"),
        (lambda r: setattr(r._parser, "joint_12", message(object(), age=3.0)), "joint_12", "older"),
        (lambda r: setattr(r._parser, "joint_7", message(object(), age=-60.0)), "joint_7", "ahead"),
        (
            lambda r: setattr(r._parser, "end_pose_xy", SimpleNamespace(timestamp="x")),
            "end_pose_xy",
            "not a usable time",
        ),
        (lambda r: setattr(r, "joints", message([0.1] * 7, hz=0.0)), "joint angles", "refuses"),
        (lambda r: setattr(r, "joints", message([0.1] * 6, hz=200.0)), "joint angles", "not 7"),
        (
            lambda r: setattr(r, "joints", message([0.1] * 6 + [np.nan], hz=200.0)),
            "joint angles",
            "not 7 finite",
        ),
        (
            lambda r: setattr(r, "pose", message([0.3, 0.0, np.inf, 3.1, 0.0, 0.0], hz=200.0)),
            "flange pose",
            "not a finite",
        ),
    ],
)
def test_each_problem_is_named(change: Any, name: str, text: str) -> None:
    connections, robot, _ = healthy()
    change(robot)
    checks = {check.name: check for check in assess(NeroConfig(), connections)}
    assert checks[name].level == "fail" and text in checks[name].detail


def test_monotonic_timestamps_and_a_looser_threshold() -> None:
    connections, robot, _ = healthy()
    robot._parser.joint_12 = message(object(), age=0.8, clock="monotonic")
    assert levels(assess(NeroConfig(), connections))["joint_12"] == "fail"
    looser = NeroConfig(feedback_stale_s=1.0)
    checks = assess(looser, connections)
    assert levels(checks)["joint_12"] == "ok"
    assert "monotonic clock" in format_report(checks)


def test_missing_arm_hand_and_camera_are_reported() -> None:
    connections = Connections(robot_error="no can0", hand_error="no /dev/ttyUSB0")
    checks = levels(assess(NeroConfig(), connections))
    assert checks == {"arm": "fail", "hand": "fail", "camera": "fail"}
    assert levels(assess(NeroConfig(hand="none"), Connections()))["hand"] == "ok"


def test_hand_and_camera_errors_are_reported() -> None:
    connections, _, port = healthy()
    port.reply_override = b""
    connections.camera = lambda: (_ for _ in ()).throw(RuntimeError("no frame"))
    checks = {check.name: check for check in assess(NeroConfig(), connections)}
    assert checks["hand"].level == "fail" and "no answer from hand id 1" in checks["hand"].detail
    assert checks["camera"].level == "fail" and "no frame" in checks["camera"].detail


def test_main_prints_closes_and_sets_the_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[tuple[NeroConfig, float]] = []
    closed: list[str] = []

    def connect(cfg: NeroConfig, seconds: float) -> Connections:
        seen.append((cfg, seconds))
        connections, _, _ = healthy()
        connections.closers = (lambda: closed.append("arm"),)
        return connections

    assert main(["-E", "can_channel=can1", "-E", "hand_port=/dev/ttyUSB3"], connect_fn=connect) == 0
    cfg, seconds = seen[0]
    assert cfg.backend == "sdk" and cfg.can_channel == "can1" and seconds == 2.0
    assert "READY" in capsys.readouterr().out and closed == ["arm"]

    def broken(cfg: NeroConfig, seconds: float) -> Connections:
        def fail() -> None:
            raise OSError("bus gone")

        return Connections(robot_error="no can0", closers=(fail,))

    assert main(["--json", "--seconds", "1"], connect_fn=broken) == 1
    out = capsys.readouterr().out
    assert "warning: cleanup failed: bus gone" in out
    payload = json.loads(out[out.index("[") :])
    assert payload[0]["name"] == "arm" and payload[0]["level"] == "fail"


@pytest.mark.parametrize("argv", [["-E", "can_channel"], ["--seconds", "0"]])
def test_main_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        main(argv, connect_fn=lambda cfg, seconds: Connections())
