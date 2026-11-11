"""The real-arm driver's feedback checks, against a fake shaped like pyAgxArm's messages."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from inspect_robots_agilex_nero.nero import JOINT_GROUPS, POSE_GROUPS, PyAgxArmDriver


def message(payload: Any, *, age: float = 0.0, hz: float = 0.0, clock: str = "epoch") -> Any:
    """Like pyAgxArm's MessageAbstract: parser caches leave hz at 0."""
    now = time.time() if clock == "epoch" else time.monotonic()
    return SimpleNamespace(msg=payload, timestamp=now - age, hz=hz)


class FakeRobot:
    def __init__(self) -> None:
        self._parser = SimpleNamespace(
            **{group: message(object()) for group in (*JOINT_GROUPS, *POSE_GROUPS)}
        )
        self.joints = message([0.1] * 7, hz=200.0)
        self.pose = message([0.3, 0.0, 0.4, 3.1, 0.0, 0.0], hz=200.0)
        self.calls: list[tuple[str, Any]] = []
        self.move_j_error: BaseException | None = None

    def get_joint_angles(self) -> Any:
        return self.joints

    def get_flange_pose(self) -> Any:
        return self.pose

    def move_p(self, pose: Any) -> None:
        self.calls.append(("move_p", pose))

    def move_j(self, q: Any) -> None:
        if self.move_j_error is not None:
            raise self.move_j_error
        self.calls.append(("move_j", q))

    def electronic_emergency_stop(self) -> None:
        self.calls.append(("estop", None))

    def disconnect(self) -> None:
        self.calls.append(("disconnect", None))


def test_healthy_feedback_is_accepted_even_though_parser_groups_have_no_rate() -> None:
    robot = FakeRobot()
    driver = PyAgxArmDriver(robot)
    pose = driver.read_flange()
    np.testing.assert_allclose(pose.position, (0.3, 0.0, 0.4))
    np.testing.assert_allclose(driver.read_joints(), 0.1)
    driver.hold()
    assert robot.calls == [("move_j", [0.1] * 7)]  # held, no e-stop
    # A monotonic-clock timestamp is also accepted.
    robot.joints = message([0.2] * 7, hz=200.0, clock="monotonic")
    np.testing.assert_allclose(driver.read_joints(), 0.2)


@pytest.mark.parametrize(
    ("break_it", "match"),
    [
        (lambda r: setattr(r._parser, "joint_34", None), "no joint_34 feedback"),
        (lambda r: setattr(r._parser, "joint_12", message(object(), age=2.0)), "joint_12"),
        (lambda r: delattr(r, "_parser"), "no _parser"),
        (lambda r: setattr(r, "joints", message([0.1] * 7, hz=0.0)), "stopped"),
        (lambda r: setattr(r, "joints", SimpleNamespace(msg=None)), "no feedback"),
        (lambda r: setattr(r, "joints", SimpleNamespace(msg=[0.1] * 7, hz=9.0)), "timestamp"),
        (lambda r: setattr(r, "joints", message([np.nan] * 7, hz=200.0)), "invalid"),
    ],
)
def test_bad_joint_feedback_is_refused_and_hold_uses_the_estop(break_it: Any, match: str) -> None:
    robot = FakeRobot()
    break_it(robot)
    driver = PyAgxArmDriver(robot)
    with pytest.raises(RuntimeError, match="used the e-stop") as raised:
        driver.hold()
    assert match in str(raised.value.__cause__)
    assert robot.calls == [("estop", None)]


def test_stale_pose_group_is_refused() -> None:
    robot = FakeRobot()
    robot._parser.end_pose_zrx = message(object(), age=2.0)
    with pytest.raises(RuntimeError, match="end_pose_zrx"):
        PyAgxArmDriver(robot).read_flange()


def test_a_failed_hold_command_falls_back_to_the_estop() -> None:
    robot = FakeRobot()
    robot.move_j_error = OSError("CAN write failed")
    with pytest.raises(RuntimeError, match="used the e-stop"):
        PyAgxArmDriver(robot).hold()
    assert robot.calls == [("estop", None)]


def test_move_estop_and_disconnect() -> None:
    robot = FakeRobot()
    driver = PyAgxArmDriver(robot)
    driver.move_flange(np.asarray((0.3, 0.0, 0.4)), np.asarray((3.1, 0.0, 0.0)))
    driver.emergency_stop()
    driver.disconnect()
    assert [name for name, _ in robot.calls] == ["move_p", "estop", "disconnect"]
    assert robot.calls[0][1] == [0.3, 0.0, 0.4, 3.1, 0.0, 0.0]
    PyAgxArmDriver(SimpleNamespace()).disconnect()  # no disconnect method: nothing to do
