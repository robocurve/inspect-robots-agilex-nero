"""AgileX Nero arm with an Inspire hand, steered by small hand displacements.

Each action moves the hand point (the ``tcp_offset`` point on the mount) by a
few centimetres in the arm base frame and changes the grip. The embodiment
owns the target: it adds the step, clamps it to the workspace box, converts
it to a flange target with the orientation held from reset, and hands it to
the arm's own planner. The arm holds its last target on its own, so a slow
policy or a dead process leaves it still, not moving.
"""

from __future__ import annotations

import atexit
import contextlib
import math
import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo, HealthCheck, HealthReport
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_agilex_nero.config import GRIP_KEY, HAND_KEY, JOINTS_KEY, NeroConfig
from inspect_robots_agilex_nero.nero import (
    BackendFactory,
    Backends,
    FlangePose,
    default_backend_factory,
    rpy_matrix,
)
from inspect_robots_agilex_nero.operator import OperatorIO, default_poll_end

Vec = npt.NDArray[np.float64]
CameraReader = Callable[[], npt.NDArray[np.uint8]]

_DOCS = """An AgileX Nero 7-joint arm with an Inspire five-finger hand on its end.
You move the hand; the arm works out its own joints and keeps the hand's
orientation fixed. Each action is how far things move during one control step:
- hand_x: metres the hand moves forward, away from the arm's base.
- hand_y: metres the hand moves to the left, seen from behind the base.
- hand_z: metres the hand moves up.
- grip: change in grip, 0 open to 1 closed (all fingers close together).
hand_pos in the state is the hand point (x, y, z) in metres from the arm's
base, x forward, y left, z up. grip is the measured grip. arm_joints are the
seven joint angles in radians. The hand stays inside a box set for this
station; a move past it stops at the edge, and hand_pos shows where the hand
really is. A move can be larger than one step's bound; it is split into
steps. Close the hand slowly around an object and check grip: fingers stop
when they meet resistance, so a grip well below 1 means something is held.
If the grip bound is zero this station has no working hand."""


class NeroEmbodiment:
    """Inspect Robots embodiment ``nero_inspire``."""

    NAME: ClassVar[str] = "nero_inspire"
    READY_PROMPT: ClassVar[str] = (
        "The Nero will move its hand to the home position and open the fingers. "
        "Clear the table around it, keep the stop within reach, then press Enter..."
    )
    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {}

    def __init__(
        self,
        config: NeroConfig | None = None,
        *,
        backend_factory: BackendFactory | None = None,
        camera_reader: CameraReader | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        atexit_module: Any = atexit,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else NeroConfig.from_kwargs(**flat)
        if not isinstance(self._cfg, NeroConfig):
            raise TypeError(f"NeroEmbodiment needs a NeroConfig; got {type(self._cfg).__name__}")
        self._backend_factory = backend_factory or default_backend_factory
        self._camera_reader = camera_reader
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end = poll_end or default_poll_end
        self._clock = clock or time.monotonic
        self._sleep = sleep_fn or time.sleep
        self._atexit = atexit_module
        self._backends: Backends | None = None
        self._rpy: Vec | None = None
        self._offset_world = np.zeros(3, dtype=np.float64)
        self._target = np.asarray(self._cfg.home_pos, dtype=np.float64)
        self._grip = 0.0
        self._instruction: str | None = None
        self.num_steps = 0
        docs = _DOCS
        if self._cfg.docs_extra.strip():
            docs += "\n\n" + self._cfg.docs_extra.strip()
        self.info = EmbodimentInfo(
            name=self.NAME,
            action_space=self._cfg.make_action_box(),
            observation_space=self._cfg.make_observation_space(),
            control_hz=self._cfg.control_hz,
            is_simulated=self._cfg.backend != "sdk",
            capabilities=frozenset({SELF_PACED}),
            docs=docs,
            morphology="agilex-nero-inspire",
            environment_id=f"nero:{self._cfg.backend}",
        )

    # ------------------------------------------------------------------ lifecycle

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect lazily, wait for the operator, glide home, open the hand, observe."""
        del seed
        if self._backends is None:
            self._backends = self._backend_factory(self._cfg)
            self._atexit.register(self._backstop)
        try:
            if not self._cfg.unattended:
                self._operator.wait_ready(self.READY_PROMPT)
            backends = self._require()
            start = self._read_flange()
            self._rpy = start.rpy.copy()
            self._offset_world = rpy_matrix(start.rpy) @ np.asarray(self._cfg.tcp_offset)
            self._target = start.position + self._offset_world
            self._glide_home()
            self._grip = 0.0
            if backends.hand is not None:
                backends.hand.set_grip(0.0)
            self._instruction = scene.instruction
            self.num_steps = 0
            return self._observe()
        except BaseException:
            self.close()
            raise

    def step(self, action: Action) -> StepResult:
        """Clamp, move the target, command the arm and hand, pace, observe."""
        backends = self._require()
        command = np.asarray(action.data, dtype=np.float64).reshape(-1)
        if command.shape != self._cfg.low.shape:
            raise ValueError(f"Nero action must have 4 values; got {command.shape}")
        if not np.isfinite(command).all():
            raise ValueError("action must contain only finite values")
        clamped = np.clip(command, self._cfg.low, self._cfg.high)
        try:
            self._target = np.clip(
                self._target + clamped[:3], self._cfg.box_low, self._cfg.box_high
            )
            self._command_arm()
            if backends.hand is not None:
                self._grip = float(np.clip(self._grip + clamped[3], 0.0, 1.0))
                backends.hand.set_grip(self._grip)
            self._sleep(1.0 / self._cfg.control_hz)
            self.num_steps += 1
            observation = self._observe()
        except BaseException:
            self._safe_hold()
            self.close()
            raise
        if not self._cfg.unattended and self._poll_end():
            self._safe_hold()
            success = self._operator.confirm_success()
            return StepResult(
                observation=observation,
                terminated=True,
                termination_reason="success" if success else "failure",
                info={"operator_confirmed": success},
            )
        return StepResult(observation=observation, terminated=False)

    def health(self) -> HealthReport:
        """Report whether the arm state reads cleanly."""
        if self._backends is None:
            return HealthReport()
        try:
            self._read_flange()
        except Exception as exc:
            return HealthReport((HealthCheck("arm_state", False, str(exc)),))
        return HealthReport((HealthCheck("arm_state", True),))

    def park(self) -> None:
        """Glide the hand home. Never disables the arm."""
        if self._backends is None:
            return
        self._glide_home()

    def emergency_stop(self) -> None:
        """Lower the arm slowly with damping (pyAgxArm's electronic e-stop)."""
        if self._backends is not None:
            self._backends.arm.emergency_stop()

    def close(self) -> None:
        """Hold the arm where it is and close both connections. Never disables motors."""
        backends, self._backends = self._backends, None
        if backends is None:
            return
        error: BaseException | None = None
        for action in (
            backends.arm.hold,
            backends.arm.disconnect,
            *((backends.hand.close,) if backends.hand is not None else ()),
        ):
            try:
                action()
            except BaseException as exc:
                error = error or exc
        self._atexit.unregister(self._backstop)
        if error is not None:
            raise error

    def __del__(self) -> None:
        """Best-effort hold when normal teardown was skipped."""
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------ helpers

    def _require(self) -> Backends:
        if self._backends is None:
            raise RuntimeError("step() called before reset() (or after close())")
        return self._backends

    def _backstop(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def _safe_hold(self) -> None:
        with contextlib.suppress(Exception):
            self._require().arm.hold()

    def _read_flange(self) -> FlangePose:
        pose = self._require().arm.read_flange()
        position = np.asarray(pose.position, dtype=np.float64)
        rpy = np.asarray(pose.rpy, dtype=np.float64)
        if position.shape != (3,) or rpy.shape != (3,):
            raise ValueError("arm returned a flange pose of the wrong shape")
        if not (np.all(np.isfinite(position)) and np.all(np.isfinite(rpy))):
            raise ValueError("arm returned a non-finite flange pose")
        return FlangePose(position, rpy)

    def _command_arm(self) -> None:
        assert self._rpy is not None
        self._require().arm.move_flange(self._target - self._offset_world, self._rpy.copy())

    def _hand_point(self) -> Vec:
        return self._read_flange().position + self._offset_world

    def _glide_home(self) -> None:
        """Move the hand point to ``home_pos`` in hand_speed-sized steps, then settle."""
        home = np.asarray(self._cfg.home_pos, dtype=np.float64)
        start = self._hand_point()
        step = self._cfg.hand_speed / self._cfg.control_hz
        count = max(1, math.ceil(float(np.max(np.abs(home - start))) / step))
        period = 1.0 / self._cfg.control_hz
        for index in range(1, count + 1):
            self._target = start + (home - start) * (index / count)
            self._command_arm()
            self._sleep(period)
        deadline = self._clock() + self._cfg.home_timeout_s
        while float(np.linalg.norm(self._hand_point() - home)) > self._cfg.settle_tolerance_m:
            if self._clock() > deadline:
                raise RuntimeError(
                    f"Nero hand did not reach home {tuple(home)} within "
                    f"{self._cfg.home_timeout_s} s; is home_pos reachable?"
                )
            self._sleep(period)

    def _observe(self) -> Observation:
        backends = self._require()
        hand = self._hand_point()
        joints = np.asarray(backends.arm.read_joints(), dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("arm returned invalid joint angles")
        grip = backends.hand.read_grip() if backends.hand is not None else 0.0
        if not math.isfinite(grip):
            raise ValueError("hand returned a non-finite grip")
        reader = self._camera_reader or _opencv_reader(self._cfg)
        self._camera_reader = reader
        frame = np.asarray(reader(), dtype=np.uint8)
        expected = (self._cfg.camera_height, self._cfg.camera_width, 3)
        if frame.shape != expected:
            raise ValueError(f"camera frame shape {frame.shape} != declared {expected}")
        observed_at = self._clock()
        name = self._cfg.camera_name
        return Observation(
            images={name: frame},
            state={
                HAND_KEY: hand,
                GRIP_KEY: np.asarray((grip,), dtype=np.float64),
                JOINTS_KEY: joints,
            },
            instruction=self._instruction,
            image_times={name: observed_at},
            state_time=observed_at,
        )


def _opencv_reader(cfg: NeroConfig) -> CameraReader:  # pragma: no cover - live camera
    """Read the station camera with OpenCV, resized to the declared shape."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'The station camera needs OpenCV: pip install "inspect-robots-agilex-nero[hardware]"'
        ) from exc
    device: int | str = int(cfg.camera_device) if cfg.camera_device.isdigit() else cfg.camera_device
    capture = cv2.VideoCapture(device)

    def reader() -> npt.NDArray[np.uint8]:
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"camera {cfg.camera_device!r} returned no frame")
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (cfg.camera_height, cfg.camera_width):
            frame = cv2.resize(frame, (cfg.camera_width, cfg.camera_height))
        return np.asarray(frame, dtype=np.uint8)

    return reader
