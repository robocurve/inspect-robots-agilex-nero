"""AgileX Nero arm with an Inspire hand, steered by small hand displacements.

Each action moves the hand point (the ``tcp_offset`` point on the mount) by a
few centimetres in the arm base frame and opens or closes each finger motor. The embodiment
owns the target: it adds the step (capped in any direction), keeps it in the
workspace box and within a short lead of the measured hand, converts it
to a flange target with the orientation held from reset, and hands it to the
arm's own planner. If this process stops, the arm finishes its last short
move and stays there.
"""

from __future__ import annotations

import atexit
import contextlib
import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from inspect_robots.conformance import DeviceSlot
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo, HealthCheck, HealthReport
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_agilex_nero.config import FINGERS_KEY, HAND_KEY, JOINTS_KEY, NeroConfig
from inspect_robots_agilex_nero.nero import (
    PYAGXARM_INSTALL,
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
- little_finger: change in little finger closure, 0 straight to 1 curled.
- ring_finger: change in ring finger closure, 0 straight to 1 curled.
- middle_finger: change in middle finger closure, 0 straight to 1 curled.
- index_finger: change in index finger closure, 0 straight to 1 curled.
- thumb_bend: change in thumb bend, 0 straight to 1 bent in.
- thumb_rotate: change in thumb rotation, 0 out to the side to 1 across
  the palm, facing the fingers.
Each finger curls as a unit. A whole-hand grasp closes all five and swings
the thumb across first; a pinch uses thumb_rotate, thumb_bend and
index_finger only. hand_pos in the state is the hand point (x, y, z) in
metres from the arm's base, x forward, y left, z up. fingers is the measured
closure of each of the six, in the same order. arm_joints are the seven joint
angles in radians. Hand targets stay inside a box set for this station and
within a few centimetres of where the hand is; a move past the box stops at its
edge, and hand_pos shows where the hand really is. Only targets are bounded:
the arm plans its own path between them, which is not checked and can pass
outside the box, so keep clear of obstacles near its edges. A
move can be larger than one step's bound; it is split into steps. Fingers
stop when they meet resistance, so a measured closure well below its command
means the finger is pressing on something. If the finger bounds are zero
this station has no working hand."""


class NeroEmbodiment:
    """Inspect Robots embodiment ``nero_inspire``."""

    NAME: ClassVar[str] = "nero_inspire"
    READY_PROMPT: ClassVar[str] = (
        "The Nero will move its hand to the home position and open the fingers. "
        "Clear the table around it, keep the stop within reach, then press Enter..."
    )
    # What the real arm, hand and camera need (simulator backends need their own).
    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "pyAgxArm": PYAGXARM_INSTALL,
        "serial": 'pip install "inspect-robots-agilex-nero[hardware]"',
        "cv2": 'pip install "inspect-robots-agilex-nero[hardware]"',
    }
    DEVICE_SLOTS: ClassVar[tuple[DeviceSlot, ...]] = (
        DeviceSlot(arg="can_channel", kind="can", label="Nero arm CAN channel"),
        DeviceSlot(arg="hand_port", kind="serial", label="Inspire hand RS485 adapter"),
        DeviceSlot(arg="camera_device", kind="v4l2", label="station camera"),
    )

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
        self._own_clock, self._own_sleep = self._clock, self._sleep
        self._own_camera = camera_reader
        self._atexit = atexit_module
        self._backends: Backends | None = None
        self._rpy: Vec | None = None
        self._offset_world = np.zeros(3, dtype=np.float64)
        self._target = np.asarray(self._cfg.home_pos, dtype=np.float64)
        self._fingers = np.zeros(6, dtype=np.float64)
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
        """Wait for the operator, connect lazily, home, open the hand, observe.

        The operator confirms readiness before anything connects, because
        connecting enables the arm. A simulator's scene is reset on every reset
        after the first. ``seed`` is unused: the station is deterministic.
        """
        del seed
        try:
            if not self._cfg.unattended:
                self._operator.wait_ready(self.READY_PROMPT)
            if self._backends is None:
                self._connect()
            elif self._backends.reset_scene is not None:
                self._backends.reset_scene()
            backends = self._require()
            start = self._read_flange()
            self._rpy = start.rpy.copy()
            self._offset_world = rpy_matrix(start.rpy) @ np.asarray(self._cfg.tcp_offset)
            self._target = start.position + self._offset_world
            self._check_start(self._target)
            self._glide_home()
            self._fingers = np.zeros(6, dtype=np.float64)
            if backends.hand is not None:
                backends.hand.set_fingers(self._fingers.copy())
            self._instruction = scene.instruction
            self.num_steps = 0
            return self._observe()
        except BaseException:
            self._fail_safe()
            raise

    def step(self, action: Action) -> StepResult:
        """Clamp, move the target, command the arm and hand, pace, observe.

        Any failure, a malformed action included, holds the arm and closes.
        """
        backends = self._require()
        try:
            command = np.asarray(action.data, dtype=np.float64).reshape(-1)
            if command.shape != self._cfg.low.shape:
                size = self._cfg.low.size
                raise ValueError(f"Nero action must have {size} values; got {command.shape}")
            if not np.isfinite(command).all():
                raise ValueError("action must contain only finite values")
            clamped = np.clip(command, self._cfg.low, self._cfg.high)
            self._target = self._next_target(clamped[:3])
            self._command_arm()
            if backends.hand is not None:
                self._fingers = np.clip(self._fingers + clamped[3:], 0.0, 1.0)
                backends.hand.set_fingers(self._fingers.copy())
            self._sleep(1.0 / self._cfg.control_hz)
            self.num_steps += 1
            observation = self._observe()
            if not self._cfg.unattended and self._poll_end():
                backends.arm.hold()
                success = self._operator.confirm_success()
                return StepResult(
                    observation=observation,
                    terminated=True,
                    termination_reason="success" if success else "failure",
                    info={"operator_confirmed": success},
                )
        except BaseException:
            self._fail_safe()
            raise
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
        """Glide the hand home. Never disables the arm; a failure holds it and closes."""
        if self._backends is None:
            return
        try:
            self._glide_home()
        except BaseException:
            self._fail_safe()
            raise

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
        self._clock, self._sleep = self._own_clock, self._own_sleep
        self._camera_reader = self._own_camera
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

    def _connect(self) -> None:
        self._backends = self._backend_factory(self._cfg)
        self._atexit.register(self._backstop)
        if self._backends.clock is not None and self._backends.sleep is not None:
            self._clock, self._sleep = self._backends.clock, self._backends.sleep
        if self._camera_reader is None and self._backends.camera is not None:
            self._camera_reader = self._backends.camera

    def _fail_safe(self) -> None:
        """Hold and close after a failure, keeping the original error the one raised."""
        with contextlib.suppress(BaseException):
            self.close()

    def _check_start(self, hand: Vec) -> None:
        """Refuse to move from well outside the workspace box (home or park)."""
        margin = self._cfg.home_margin_m
        if np.any(hand < self._cfg.box_low - margin) or np.any(hand > self._cfg.box_high + margin):
            raise RuntimeError(
                f"the hand is at {tuple(np.round(hand, 3))}, outside the workspace box by more "
                f"than {margin} m. Move the arm into the workspace by hand (teach mode or the "
                "AgileX tools), then reset again; this adapter will not move it from there."
            )

    def _safe_target(self, proposed: Vec, hand: Vec) -> Vec:
        """The point nearest ``proposed`` that is in the box and within the lead of the hand.

        Both sets are convex. The exact projection onto their intersection is
        ``clip((proposed + w * hand) / (1 + w))`` for the smallest ``w >= 0`` that
        lands within the lead, found by bisection. A hand more than
        ``home_margin_m`` outside the box, or too far from it for the two sets to
        meet, is a fault rather than a silent violation of either limit.
        """
        self._check_start(hand)
        low, high = self._cfg.box_low, self._cfg.box_high
        lead = self._cfg.max_target_lead_m
        gap = float(np.linalg.norm(np.clip(hand, low, high) - hand))
        if gap >= lead - 1e-9:  # the sets do not meet with room to spare
            raise RuntimeError(
                f"the hand is {gap:.3f} m outside the workspace box, beyond the "
                f"{lead} m target lead: no safe target exists"
            )

        proposed = np.asarray(proposed, dtype=np.float64)

        def candidate(weight: float) -> Vec:
            # The projection is clip((proposed + weight * hand) / (1 + weight)) for the
            # smallest weight >= 0 that brings it within the lead (Lagrange multiplier
            # of the ball constraint); its distance to the hand falls as weight grows.
            blended = (proposed + weight * hand) / (1.0 + weight)
            return np.asarray(np.clip(blended, low, high), dtype=np.float64)

        point = candidate(0.0)
        if float(np.linalg.norm(point - hand)) > lead:
            below, above = 0.0, 1.0
            while float(np.linalg.norm(candidate(above) - hand)) > lead:
                # Terminates: as w grows the candidate tends to clip(hand), within the lead.
                below, above = above, above * 2.0
            for _ in range(80):
                middle = 0.5 * (below + above)
                if float(np.linalg.norm(candidate(middle) - hand)) > lead:
                    below = middle
                else:
                    above = middle
            point = candidate(above)
        return point

    def _capped(self, delta: Vec) -> Vec:
        """A step no longer than ``hand_speed / control_hz`` in any direction."""
        cap = self._cfg.hand_speed / self._cfg.control_hz
        norm = float(np.linalg.norm(delta))
        return delta * (cap / norm) if norm > cap else delta

    def _next_target(self, delta: Vec) -> Vec:
        """Add a capped step to the target, then keep it in the box and near the hand."""
        return self._safe_target(self._target + self._capped(delta), self._hand_point())

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
        """The measured hand point: the flange pose with its measured orientation."""
        pose = self._read_flange()
        offset = rpy_matrix(pose.rpy) @ np.asarray(self._cfg.tcp_offset)
        return np.asarray(pose.position + offset, dtype=np.float64)

    def _glide_home(self) -> None:
        """Step the hand point to ``home_pos`` under the same limits as any step.

        Each step reads the hand, moves the target at most one capped step toward
        home, keeps it in the box and within the lead of the hand, then commands
        it. One deadline covers the whole move.
        """
        home = np.asarray(self._cfg.home_pos, dtype=np.float64)
        period = 1.0 / self._cfg.control_hz
        deadline = self._clock() + self._cfg.home_timeout_s
        hand = self._hand_point()
        # Replace whatever target the arm still has with where it is now, so an
        # earlier move cannot carry on even if the hand is already near home.
        self._target = self._safe_target(hand, hand)
        self._command_arm()
        while float(np.linalg.norm(home - hand)) > self._cfg.settle_tolerance_m:
            if self._clock() > deadline:
                raise RuntimeError(
                    f"Nero hand did not reach home {tuple(home)} within "
                    f"{self._cfg.home_timeout_s} s; is home_pos reachable?"
                )
            proposed = self._target + self._capped(home - self._target)
            self._target = self._safe_target(proposed, hand)
            self._command_arm()
            self._sleep(period)
            hand = self._hand_point()

    def _observe(self) -> Observation:
        backends = self._require()
        hand = self._hand_point()
        joints = np.asarray(backends.arm.read_joints(), dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("arm returned invalid joint angles")
        fingers = (
            np.asarray(backends.hand.read_fingers(), dtype=np.float64)
            if backends.hand is not None
            else np.zeros(6, dtype=np.float64)
        )
        if fingers.shape != (6,) or not np.all(np.isfinite(fingers)):
            raise ValueError("hand returned invalid finger closures")
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
                FINGERS_KEY: fingers,
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
