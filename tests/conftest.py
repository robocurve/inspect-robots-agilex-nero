from __future__ import annotations

from typing import Any

import numpy as np

from inspect_robots_agilex_nero.config import NeroConfig
from inspect_robots_agilex_nero.embodiment import NeroEmbodiment
from inspect_robots_agilex_nero.nero import Backends, KinematicArm, KinematicHand
from inspect_robots_agilex_nero.operator import OperatorIO


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.now += duration


class FakeAtexit:
    def __init__(self) -> None:
        self.registered: list[Any] = []

    def register(self, fn: Any) -> None:
        self.registered.append(fn)

    def unregister(self, fn: Any) -> None:
        self.registered.remove(fn)


class Rig:
    """A kinematic arm and hand wired into an embodiment with a fake clock."""

    def __init__(self, hand: bool = True, **cfg: Any) -> None:
        cfg.setdefault("unattended", True)
        cfg.setdefault("camera_height", 48)
        cfg.setdefault("camera_width", 64)
        if not hand:
            cfg["hand"] = "none"
        self.cfg = NeroConfig(**cfg)
        self.clock = FakeClock()
        self.arm = KinematicArm(clock=self.clock)
        self.hand = KinematicHand() if hand else None
        self.atexit = FakeAtexit()
        self.inputs: list[str] = []
        self.answers = iter(["", "y"])
        self.ended = False
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.emb = NeroEmbodiment(
            self.cfg,
            backend_factory=lambda _: Backends(self.arm, self.hand),
            camera_reader=lambda: self.frame,
            operator=OperatorIO(input_fn=self._input, output_fn=lambda _: None),
            poll_end=lambda: self.ended,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
            atexit_module=self.atexit,
        )

    def _input(self, prompt: str) -> str:
        self.inputs.append(prompt)
        return next(self.answers)
