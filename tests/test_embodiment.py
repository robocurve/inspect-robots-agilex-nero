"""Reset, stepping, the workspace clamp, teardown and a full eval on the kinematic arm."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from inspect_robots import eval as robots_eval
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.scorer import episode_length
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation

from conftest import FakeClock, Rig
from inspect_robots_agilex_nero.config import FINGERS_KEY, HAND_KEY, JOINTS_KEY, NeroConfig
from inspect_robots_agilex_nero.nero import (
    ArmDriver,
    FlangePose,
    HandDriver,
    KinematicArm,
    KinematicHand,
    rpy_matrix,
)

SCENE = Scene(id="s", instruction="pick up the cup")


def act(*values: float) -> Action:
    return Action(data=np.asarray(values, dtype=np.float64))


def test_stand_ins_satisfy_the_protocols() -> None:
    assert isinstance(KinematicArm(clock=FakeClock()), ArmDriver)
    assert isinstance(KinematicHand(), HandDriver)


def test_rpy_matrix_is_a_rotation() -> None:
    rotation = rpy_matrix((0.3, -0.2, 1.1))
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rpy_matrix((math.pi, 0, 0)) @ (0, 0, 1), (0, 0, -1), atol=1e-12)


def test_reset_glides_home_opens_the_hand_and_observes() -> None:
    rig = Rig()
    rig.hand.fingers[:] = 0.7  # type: ignore[union-attr]
    obs = rig.emb.reset(SCENE)
    np.testing.assert_allclose(obs.state[HAND_KEY], rig.cfg.home_pos, atol=1e-9)
    assert obs.state[FINGERS_KEY].tolist() == [0.0] * 6
    assert obs.state[JOINTS_KEY].shape == (7,)
    assert obs.instruction == "pick up the cup"
    assert obs.images["scene_cam"].shape == (48, 64, 3)
    # Start (0.30, 0, 0.25) to home (0.35, 0, 0.30): 5 cm at 2 cm per step is 3 steps.
    assert rig.clock.sleeps[:3] == [0.2, 0.2, 0.2]
    assert len(rig.atexit.registered) == 1
    rig.emb.close()
    assert rig.atexit.registered == [] and rig.arm.connected is False
    assert rig.hand.closed  # type: ignore[union-attr]


def test_steps_move_the_hand_clamp_to_the_box_and_grip() -> None:
    rig = Rig()
    rig.emb.reset(SCENE)
    # x clamps to the 2 cm bound; the index finger to the 0.2 bound.
    result = rig.emb.step(act(0.5, 0.0, -0.01, 0.0, 0.0, 0.0, 0.9, 0.15, -0.1))
    np.testing.assert_allclose(result.observation.state[HAND_KEY], (0.37, 0.0, 0.29), atol=1e-9)
    np.testing.assert_allclose(result.observation.state[FINGERS_KEY], (0, 0, 0, 0.2, 0.15, 0))
    for _ in range(20):
        result = rig.emb.step(act(0.02, 0.0, -0.02, *([0.2] * 6)))
    np.testing.assert_allclose(result.observation.state[HAND_KEY], (0.55, 0.0, 0.10), atol=1e-9)
    np.testing.assert_allclose(result.observation.state[FINGERS_KEY], 1.0)
    assert rig.emb.num_steps == 21 and result.terminated is False
    with pytest.raises(ValueError, match="9 values"):
        rig.emb.step(act(0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        rig.emb.step(act(math.nan, *([0.0] * 8)))
    rig.emb.close()


def test_tcp_offset_is_applied_through_the_held_orientation() -> None:
    # Flange points down (roll pi), so a 0.1 m offset along flange z lies 0.1 m below it.
    rig = Rig(tcp_offset=(0.0, 0.0, 0.1))
    rig.emb.reset(SCENE)
    np.testing.assert_allclose(rig.arm.read_flange().position, (0.35, 0.0, 0.40), atol=1e-9)
    rig.emb.close()


def test_no_hand_station() -> None:
    rig = Rig(hand=False)
    obs = rig.emb.reset(SCENE)
    assert obs.state[FINGERS_KEY].tolist() == [0.0] * 6
    rig.emb.step(act(*([0.0] * 9)))
    rig.emb.close()


def test_operator_ready_prompt_and_success_verdict() -> None:
    rig = Rig(unattended=False)
    rig.emb.reset(SCENE)
    assert "Clear the table" in rig.inputs[0]
    rig.ended = True
    result = rig.emb.step(act(*([0.0] * 9)))
    assert result.terminated and result.termination_reason == "success"
    rig.emb.close()


def test_step_before_reset_and_health() -> None:
    rig = Rig()
    with pytest.raises(RuntimeError, match="before reset"):
        rig.emb.step(act(*([0.0] * 9)))
    assert rig.emb.health().checks == ()
    rig.emb.park()  # not connected: a no-op
    rig.emb.emergency_stop()  # not connected: a no-op
    rig.emb.reset(SCENE)
    assert rig.emb.health().ok
    rig.emb.emergency_stop()
    assert rig.arm.stopped
    rig.emb.step(act(-0.02, *([0.0] * 8)))
    rig.emb.park()
    np.testing.assert_allclose(rig.emb._hand_point(), rig.cfg.home_pos, atol=1e-9)
    rig.emb.close()
    rig.emb.close()  # idempotent


class BrokenArm(KinematicArm):
    def __init__(self, clock: Any, **faults: Any) -> None:
        super().__init__(clock=clock)
        self.faults = faults
        self.held = 0

    def read_flange(self) -> FlangePose:
        fault = self.faults.get("pose")
        if fault == "raise":
            raise RuntimeError("CAN timeout")
        pose = super().read_flange()
        if fault == "shape":
            return FlangePose(pose.position[:2], pose.rpy)
        if fault == "nan":
            return FlangePose(pose.position * np.nan, pose.rpy)
        return pose

    def read_joints(self) -> Any:
        return np.zeros(6) if self.faults.get("joints") else super().read_joints()

    def hold(self) -> None:
        self.held += 1
        if self.faults.get("hold"):
            raise RuntimeError("hold failed")
        super().hold()

    def move_flange(self, position: Any, rpy: Any) -> None:
        if self.faults.get("move"):
            raise RuntimeError("move failed")
        super().move_flange(position, rpy)


@pytest.mark.parametrize(
    ("faults", "message"),
    [
        ({"pose": "shape"}, "wrong shape"),
        ({"pose": "nan"}, "non-finite"),
        ({"joints": True}, "joint angles"),
    ],
)
def test_bad_arm_data_fails_reset_and_disconnects(faults: dict[str, Any], message: str) -> None:
    rig = Rig()
    rig.arm = BrokenArm(rig.clock, **faults)
    with pytest.raises(ValueError, match=message):
        rig.emb.reset(SCENE)
    assert rig.arm.connected is False


def test_a_failing_step_holds_the_arm_and_closes() -> None:
    rig = Rig()
    arm = BrokenArm(rig.clock)
    rig.arm = arm
    rig.emb.reset(SCENE)
    arm.faults["move"] = True
    with pytest.raises(RuntimeError, match="move failed"):
        rig.emb.step(act(0.01, *([0.0] * 8)))
    assert arm.held == 2  # once on the failure, once in close
    assert arm.connected is False


def test_health_reports_a_broken_state_read() -> None:
    rig = Rig()
    arm = BrokenArm(rig.clock)
    rig.arm = arm
    rig.emb.reset(SCENE)
    arm.faults["pose"] = "raise"
    report = rig.emb.health()
    assert not report.ok and "CAN timeout" in report.checks[0].detail
    arm.faults["pose"] = None
    rig.emb.close()


def test_close_tries_everything_and_reports_the_first_error() -> None:
    rig = Rig()
    arm = BrokenArm(rig.clock)
    rig.arm = arm
    rig.emb.reset(SCENE)
    arm.faults["hold"] = True
    with pytest.raises(RuntimeError, match="hold failed"):
        rig.emb.close()
    assert arm.connected is False and rig.hand.closed  # type: ignore[union-attr]
    assert rig.atexit.registered == []
    rig.emb._backstop()  # safe to run after close


def test_backstop_holds_and_disconnects_and_reset_reuses_backends() -> None:
    rig = Rig()
    rig.emb.reset(SCENE)
    rig.emb.reset(SCENE)  # second reset keeps the same connection
    assert len(rig.atexit.registered) == 1
    rig.atexit.registered[0]()
    assert rig.arm.connected is False and rig.atexit.registered == []


def test_dead_stdin_at_the_ready_prompt_is_a_fault() -> None:
    from inspect_robots.errors import EmbodimentFault

    rig = Rig(unattended=False)

    def dead(prompt: str) -> str:
        raise EOFError

    rig.emb._operator.input_fn = dead
    with pytest.raises(EmbodimentFault, match="stdin"):
        rig.emb.reset(SCENE)
    assert rig.arm.connected is False


def test_unreachable_home_times_out() -> None:
    rig = Rig()
    rig.arm = KinematicArm(clock=rig.clock, speed_m_s=0.001)
    with pytest.raises(RuntimeError, match="did not reach home"):
        rig.emb.reset(SCENE)


def test_bad_camera_frame_and_grip_are_refused() -> None:
    rig = Rig()
    rig.frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="camera frame shape"):
        rig.emb.reset(SCENE)
    rig = Rig()
    rig.hand.read_fingers = lambda: np.full(6, math.nan)  # type: ignore[method-assign,union-attr]
    with pytest.raises(ValueError, match="invalid finger closures"):
        rig.emb.reset(SCENE)


class ReachDownAndGrip:
    def __init__(self, cfg: NeroConfig) -> None:
        self.info = PolicyInfo(
            name="reach",
            action_space=cfg.make_action_box(),
            observation_space=cfg.make_observation_space(),
        )
        self.config = PolicyConfig(action_horizon=5)
        self.seen: list[Any] = []

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        self.seen.append(observation.state[HAND_KEY].copy())
        return ActionChunk(actions=[act(0.0, 0.01, -0.02, *([0.2] * 6))] * 5)


def test_eval_reaches_down_and_grips(tmp_path: Path) -> None:
    rig = Rig()
    task = Task(
        name="reach",
        scenes=[Scene(id="r", instruction="reach down and grip")],
        scorer=episode_length(),
        max_steps=10,
    )
    policy = ReachDownAndGrip(rig.cfg)
    logs = robots_eval(task, policy, rig.emb, sinks=[], seed=0, log_dir=str(tmp_path))
    assert logs[0].status == "success", logs[0].error
    # After five steps the hand is 5 cm left and 10 cm lower.
    np.testing.assert_allclose(policy.seen[1], (0.35, 0.05, 0.20), atol=1e-9)
    np.testing.assert_allclose(rig.hand.fingers, 1.0)  # type: ignore[union-attr]
    # eval() parks the arm afterwards: the hand glides home, still connected.
    np.testing.assert_allclose(rig.emb._hand_point(), rig.cfg.home_pos, atol=1e-9)
    assert rig.arm.connected is True
    rig.emb.close()


def test_a_simulator_backend_supplies_clock_sleep_and_camera() -> None:
    from inspect_robots_agilex_nero.nero import Backends

    rig = Rig()
    sim_clock = FakeClock()
    frames: list[int] = []

    def sim_camera() -> Any:
        frames.append(1)
        return rig.frame

    arm = KinematicArm(clock=sim_clock)
    emb = type(rig.emb)(
        rig.cfg,
        backend_factory=lambda _: Backends(
            arm, KinematicHand(), clock=sim_clock, sleep=sim_clock.sleep, camera=sim_camera
        ),
        atexit_module=rig.atexit,
    )
    emb.reset(SCENE)
    assert sim_clock.sleeps and frames  # paced and rendered by the simulator
    emb.close()
    assert emb._clock is not sim_clock and emb._camera_reader is None
