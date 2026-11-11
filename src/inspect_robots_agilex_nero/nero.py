"""Nero arm and hand backends: the protocol, a kinematic stand-in, and pyAgxArm.

The arm is driven by flange pose targets through the arm's own planner
(``move_p``): each command replaces the last target, and the arm holds
whatever target it last received, including when this process dies. So
there is no streaming deadline to keep, unlike a walking robot.

pyAgxArm (https://github.com/agilexrobotics/pyAgxArm, docs/nero/nero_api.md)
warns that ``disable()`` and ``reset()`` drop raised joints immediately,
while ``electronic_emergency_stop()`` lowers the arm slowly with damping.
This package never calls ``disable()`` or ``reset()``.

Not yet run on hardware: the pyAgxArm calls below follow its documentation
and the pose convention (``[x, y, z, roll, pitch, yaw]``, metres and
radians, fixed-axis roll-pitch-yaw) is assumed from it.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from inspect_robots_agilex_nero.config import NeroConfig
from inspect_robots_agilex_nero.inspire import InspireHand, open_inspire

Vec = npt.NDArray[np.float64]

PYAGXARM_INSTALL = (
    'pip install "pyAgxArm @ git+https://github.com/agilexrobotics/pyAgxArm", '
    "then bring up CAN: sudo ip link set can0 up type can bitrate 1000000"
)


@dataclass(frozen=True)
class FlangePose:
    """Flange position (m) and fixed-axis roll, pitch, yaw (rad), arm base frame."""

    position: Vec
    rpy: Vec


@runtime_checkable
class ArmDriver(Protocol):
    """What the embodiment needs from an arm backend."""

    def read_flange(self) -> FlangePose:
        """Measured flange pose."""
        ...

    def read_joints(self) -> Vec:
        """Measured joint angles ``(7,)`` in radians."""
        ...

    def move_flange(self, position: Vec, rpy: Vec) -> None:
        """Replace the flange target. Returns at once; the arm keeps moving."""
        ...

    def hold(self) -> None:
        """Stop where the arm is now and keep holding it there."""
        ...

    def emergency_stop(self) -> None:
        """Lower the arm slowly with damping. Never drops it."""
        ...

    def disconnect(self) -> None:
        """Close the connection without disabling the motors."""
        ...


@runtime_checkable
class HandDriver(Protocol):
    """What the embodiment needs from a hand backend."""

    def set_fingers(self, closure: Vec) -> None:
        """Command six closures, 0 open to 1 closed."""
        ...

    def read_fingers(self) -> Vec:
        """Measured closures ``(6,)``."""
        ...

    def close(self) -> None:
        """Release the connection; fingers hold their last command."""
        ...


def rpy_matrix(rpy: npt.ArrayLike) -> Vec:
    """Fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = (float(v) for v in np.asarray(rpy).reshape(3))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


class KinematicArm:
    """A stand-in arm whose flange moves straight toward its target at a fixed speed."""

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        speed_m_s: float = 0.3,
        start: FlangePose | None = None,
    ) -> None:
        self._clock = clock
        self._speed = speed_m_s
        pose = start or FlangePose(np.asarray((0.30, 0.0, 0.25)), np.asarray((math.pi, 0.0, 0.0)))
        self._position = pose.position.astype(np.float64).copy()
        self._rpy = pose.rpy.astype(np.float64).copy()
        self._target = self._position.copy()
        self._time = clock()
        self._lock = threading.Lock()
        self.stopped = False
        self.connected = True

    def _advance_locked(self) -> None:
        now = self._clock()
        travel = self._speed * max(0.0, now - self._time)
        self._time = now
        delta = self._target - self._position
        distance = float(np.linalg.norm(delta))
        if distance <= travel:
            self._position = self._target.copy()
        else:
            self._position = self._position + delta * (travel / distance)

    def read_flange(self) -> FlangePose:
        """Current flange pose."""
        with self._lock:
            self._advance_locked()
            return FlangePose(self._position.copy(), self._rpy.copy())

    def read_joints(self) -> Vec:
        """Placeholder joint angles: the stand-in has no arm model."""
        return np.zeros(7, dtype=np.float64)

    def move_flange(self, position: Vec, rpy: Vec) -> None:
        """Set a new target; orientation snaps."""
        with self._lock:
            self._advance_locked()
            self._target = np.asarray(position, dtype=np.float64).copy()
            self._rpy = np.asarray(rpy, dtype=np.float64).copy()

    def hold(self) -> None:
        """Freeze at the current position."""
        with self._lock:
            self._advance_locked()
            self._target = self._position.copy()

    def emergency_stop(self) -> None:
        """Record the stop and freeze."""
        self.hold()
        self.stopped = True

    def disconnect(self) -> None:
        """Mark disconnected."""
        self.connected = False


class KinematicHand:
    """A stand-in hand whose fingers reach their command instantly."""

    def __init__(self) -> None:
        self.fingers = np.zeros(6, dtype=np.float64)
        self.closed = False

    def set_fingers(self, closure: Vec) -> None:
        """Snap to the closures."""
        self.fingers = np.clip(np.asarray(closure, dtype=np.float64), 0.0, 1.0)

    def read_fingers(self) -> Vec:
        """Current closures."""
        return self.fingers.copy()

    def close(self) -> None:
        """Mark closed."""
        self.closed = True


@dataclass
class Backends:
    """The arm and, unless ``hand=none``, the hand.

    A simulator also supplies its own clock, sleep (which steps physics) and
    camera; the embodiment then runs on simulation time.
    """

    arm: ArmDriver
    hand: HandDriver | None
    clock: Callable[[], float] | None = None
    sleep: Callable[[float], None] | None = None
    camera: Callable[[], npt.NDArray[np.uint8]] | None = None


BackendFactory = Callable[[NeroConfig], Backends]


def default_backend_factory(cfg: NeroConfig) -> Backends:  # pragma: no cover - backends
    """Kinematic stand-ins, or the real arm and hand."""
    if cfg.backend in ("mujoco", "isaaclab"):
        sim: Any
        if cfg.backend == "mujoco":
            from inspect_robots_agilex_nero._mujoco import MujocoNero

            sim = MujocoNero(cfg)
        else:
            from inspect_robots_agilex_nero._isaaclab import IsaacLabNero

            sim = IsaacLabNero(cfg)
        return Backends(
            sim,
            sim if cfg.fingers_enabled else None,
            clock=sim.sim_time,
            sleep=sim.advance,
            camera=sim.read_frame,
        )
    if cfg.backend == "kinematic":
        return Backends(
            KinematicArm(clock=time.monotonic), KinematicHand() if cfg.fingers_enabled else None
        )
    arm = PyAgxArmDriver.connect(cfg)
    hand: InspireHand | None = None
    if cfg.hand == "inspire":
        try:
            hand = open_inspire(
                cfg.hand_port, cfg.hand_baud, cfg.hand_id, cfg.open_angles, cfg.closed_angles
            )
            hand.configure(cfg.hand_speed_value, cfg.hand_force_value)
        except BaseException:
            arm.disconnect()
            raise
    return Backends(arm, hand)


class PyAgxArmDriver:  # pragma: no cover - hardware
    """The real Nero over pyAgxArm. Calls follow pyAgxArm's docs; untested on hardware."""

    def __init__(self, robot: Any) -> None:
        self._robot = robot

    @classmethod
    def connect(cls, cfg: NeroConfig) -> PyAgxArmDriver:
        """Create, connect and enable the arm, and set its speed."""
        try:
            from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(f"pyAgxArm is missing. {PYAGXARM_INSTALL}") from exc
        firmware = getattr(NeroFW, cfg.firmware.upper())
        config = create_agx_arm_config(
            robot=ArmModel.NERO, firmeware_version=firmware, channel=cfg.can_channel
        )
        robot = AgxArmFactory.create_arm(config)
        robot.connect()
        deadline = time.monotonic() + cfg.enable_timeout_s
        while not robot.enable(255):
            if time.monotonic() > deadline:
                robot.disconnect()
                raise RuntimeError("Nero did not enable; check power, CAN and firmware profile")
            time.sleep(0.05)
        robot.set_speed_percent(cfg.speed_percent)
        return cls(robot)

    @staticmethod
    def _message(reply: Any) -> Any:
        return getattr(reply, "msg", reply)

    def read_flange(self) -> FlangePose:
        """Measured flange pose from ``get_flange_pose``."""
        pose = np.asarray(self._message(self._robot.get_flange_pose()), dtype=np.float64)
        return FlangePose(pose[:3].copy(), pose[3:6].copy())

    def read_joints(self) -> Vec:
        """Measured joints from ``get_joint_angles``."""
        return np.asarray(self._message(self._robot.get_joint_angles()), dtype=np.float64)

    def move_flange(self, position: Vec, rpy: Vec) -> None:
        """Planned move to the flange pose with ``move_p``."""
        self._robot.move_p([*map(float, position), *map(float, rpy)])

    def hold(self) -> None:
        """Re-target the measured joints with ``move_j`` so the arm stops in place."""
        self._robot.move_j([float(q) for q in self.read_joints()])

    def emergency_stop(self) -> None:
        """``electronic_emergency_stop``: slow damped descent."""
        self._robot.electronic_emergency_stop()

    def disconnect(self) -> None:
        """Close CAN. Motors stay enabled and holding."""
        disconnect = getattr(self._robot, "disconnect", None)
        if disconnect is not None:
            disconnect()
