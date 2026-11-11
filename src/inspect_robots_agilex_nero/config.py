"""Frozen Nero + Inspire configuration and the spaces derived from it.

The action is a per-control-step displacement of the hand point (``hand_x``,
``hand_y``, ``hand_z`` in metres, arm base frame) plus a change in each of the
Inspire hand's six motors (0 open to 1 closed). The hand's orientation is held at whatever it was at
reset. Workspace bounds, home position and the mount offset are guesses until
someone measures the real station; every one of them is a config field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)

Vec = npt.NDArray[np.float64]

BACKENDS: tuple[str, ...] = ("sdk", "kinematic")
HANDS: tuple[str, ...] = ("inspire", "none")

#: The Inspire RH56's six motors, in its register order.
FINGER_LABELS: tuple[str, ...] = (
    "little_finger",
    "ring_finger",
    "middle_finger",
    "index_finger",
    "thumb_bend",
    "thumb_rotate",
)
DIM_LABELS: tuple[str, ...] = ("hand_x", "hand_y", "hand_z", *FINGER_LABELS)
ACTION_DIM = len(DIM_LABELS)

#: State keys: the hand point in the arm base frame (m), the six finger
#: motors (0 open, 1 closed) and the seven measured arm joint angles (rad).
HAND_KEY = "hand_pos"
FINGERS_KEY = "fingers"
JOINTS_KEY = "arm_joints"

ACTION_SEMANTICS = ActionSemantics(
    control_mode="eef_delta_pos",
    frame="base",
    gripper="continuous",
    dim_labels=DIM_LABELS,
)

#: NeroFW profiles in pyAgxArm (docs/nero/first_time_user_guide_can.md).
FIRMWARE_PROFILES: tuple[str, ...] = ("default", "v111", "v112", "v120", "v121")


@dataclass(frozen=True)
class NeroConfig:
    """Everything that varies between stations, rigs and runs.

    ``tcp_offset`` is the hand point relative to the arm's flange, in the
    flange frame (metres): the custom Inspire mount sets it. ``home_pos`` is
    where the hand point goes at reset. ``speed_percent`` is the arm's own
    speed setting; it must be fast enough to keep up with ``hand_speed``.
    Each finger control runs 0 (open) to 1 (closed) and maps linearly onto the
    hand's raw 0 to 1000 units between ``open_angles`` and ``closed_angles``
    (little, ring, middle, index, thumb bend, thumb rotation). The manual
    calls 1000 open for every motor; which way the thumb rotation turns on
    the lent hand is unverified. ``finger_rate`` is the fastest change per
    second of any finger control.
    """

    backend: str = "sdk"
    control_hz: float = 5.0
    hand_speed: float = 0.10
    finger_rate: float = 1.0
    x_min: float = 0.20
    x_max: float = 0.55
    y_min: float = -0.35
    y_max: float = 0.35
    z_min: float = 0.10
    z_max: float = 0.50
    home_pos: tuple[float, float, float] = (0.35, 0.0, 0.30)
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Nero arm (pyAgxArm over CAN).
    can_channel: str = "can0"
    firmware: str = "default"
    speed_percent: int = 20
    enable_timeout_s: float = 5.0
    settle_tolerance_m: float = 0.01
    home_timeout_s: float = 15.0
    # Inspire RH56 hand (RS485, Inspire frame protocol).
    hand: str = "inspire"
    hand_port: str = "/dev/ttyUSB0"
    hand_baud: int = 115200
    hand_id: int = 1
    hand_speed_value: int = 500
    hand_force_value: int = 500
    open_angles: tuple[int, ...] = (1000, 1000, 1000, 1000, 1000, 1000)
    closed_angles: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    # Camera.
    camera_name: str = "scene_cam"
    camera_height: int = 480
    camera_width: int = 640
    camera_device: str = "0"
    # Run.
    unattended: bool = False
    docs_extra: str = ""

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}; got {self.backend!r}")
        if self.hand not in HANDS:
            raise ValueError(f"hand must be one of {HANDS}; got {self.hand!r}")
        if self.firmware not in FIRMWARE_PROFILES:
            raise ValueError(f"firmware must be one of {FIRMWARE_PROFILES}; got {self.firmware!r}")
        for name in (
            "control_hz",
            "hand_speed",
            "finger_rate",
            "enable_timeout_s",
            "settle_tolerance_m",
            "home_timeout_s",
        ):
            _require_positive(name, getattr(self, name))
        for low, high in (("x_min", "x_max"), ("y_min", "y_max"), ("z_min", "z_max")):
            if not getattr(self, low) < getattr(self, high):
                raise ValueError(f"{low} must be below {high}")
        if len(self.home_pos) != 3 or len(self.tcp_offset) != 3:
            raise ValueError("home_pos and tcp_offset need three values")
        if not all(math.isfinite(v) for v in (*self.home_pos, *self.tcp_offset)):
            raise ValueError("home_pos and tcp_offset must be finite")
        home = np.asarray(self.home_pos)
        if np.any(home < self.box_low) or np.any(home > self.box_high):
            raise ValueError("home_pos must be inside the workspace box")
        if not 1 <= self.speed_percent <= 100:
            raise ValueError("speed_percent must be 1 to 100")
        if not 1 <= self.hand_id <= 254:
            raise ValueError("hand_id must be 1 to 254")
        for name in ("hand_speed_value", "hand_force_value"):
            if not 0 <= getattr(self, name) <= 1000:
                raise ValueError(f"{name} must be 0 to 1000")
        for name in ("open_angles", "closed_angles"):
            values = getattr(self, name)
            if len(values) != 6 or any(not 0 <= v <= 1000 for v in values):
                raise ValueError(f"{name} needs six values from 0 to 1000")
        for name in ("camera_height", "camera_width", "hand_baud"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("can_channel", "hand_port", "camera_name"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        """Build from CLI-style kwargs, coercing strings and rejecting unknown keys."""
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(kwargs) - known)
        if unknown:
            raise TypeError(f"unknown {cls.__name__} option(s): {', '.join(unknown)}")
        defaults = {item.name: item.default for item in fields(cls)}
        return cls(**{key: _coerce(key, value, defaults[key]) for key, value in kwargs.items()})

    @property
    def fingers_enabled(self) -> bool:
        """Whether the finger dimensions do anything."""
        return self.hand != "none"

    @property
    def box_low(self) -> Vec:
        """Lower corner of the hand workspace, base frame."""
        return np.asarray((self.x_min, self.y_min, self.z_min), dtype=np.float64)

    @property
    def box_high(self) -> Vec:
        """Upper corner of the hand workspace, base frame."""
        return np.asarray((self.x_max, self.y_max, self.z_max), dtype=np.float64)

    @property
    def high(self) -> Vec:
        """Per-step upper bounds: three hand axes, then six fingers."""
        finger = self.finger_rate / self.control_hz if self.fingers_enabled else 0.0
        hand = self.hand_speed / self.control_hz
        return np.asarray((hand, hand, hand, *([finger] * len(FINGER_LABELS))), dtype=np.float64)

    @property
    def low(self) -> Vec:
        """Per-step lower bounds: three hand axes, then six fingers."""
        return -self.high

    def make_action_box(self) -> Box:
        """The 9-D hand and finger displacement box."""
        return Box(shape=(ACTION_DIM,), low=self.low, high=self.high, semantics=ACTION_SEMANTICS)

    def make_observation_space(self) -> ObservationSpace:
        """One camera, hand position, finger closures and joint angles."""
        return ObservationSpace(
            cameras=(
                CameraSpec(
                    name=self.camera_name,
                    height=self.camera_height,
                    width=self.camera_width,
                    channels=3,
                ),
            ),
            state=StateSpec(
                fields=(
                    StateField(key=HAND_KEY, shape=(3,), unit="m"),
                    StateField(key=FINGERS_KEY, shape=(6,), unit="normalized"),
                    StateField(key=JOINTS_KEY, shape=(7,), unit="rad"),
                )
            ),
        )


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")


_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def _coerce(key: str, value: Any, default: Any) -> Any:
    """Coerce a CLI string to the type implied by the field's default."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if isinstance(default, bool):
        lowered = text.lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
        raise ValueError(f"{key} must be a boolean; got {value!r}")
    if isinstance(default, tuple):
        kind = type(default[0])
        return tuple(kind(part) for part in text.split(","))
    if isinstance(default, int):
        return int(text)
    if isinstance(default, float):
        return float(text)
    return value
