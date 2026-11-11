"""Inspect Robots embodiment ``nero_inspire``: an AgileX Nero arm with an Inspire hand.

Each action is a per-step hand displacement ``(hand_x, hand_y, hand_z)`` in
metres in the arm base frame plus a grip change. Construction is inert;
backends connect on the first ``reset()``.
"""

from importlib.metadata import PackageNotFoundError, version

from inspect_robots_agilex_nero.config import DIM_LABELS, NeroConfig
from inspect_robots_agilex_nero.embodiment import NeroEmbodiment
from inspect_robots_agilex_nero.inspire import InspireHand
from inspect_robots_agilex_nero.nero import KinematicArm, KinematicHand

try:
    __version__ = version("inspect-robots-agilex-nero")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = [
    "DIM_LABELS",
    "InspireHand",
    "KinematicArm",
    "KinematicHand",
    "NeroConfig",
    "NeroEmbodiment",
    "__version__",
]
