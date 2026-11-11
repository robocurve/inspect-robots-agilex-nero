"""Config validation and the contract as the framework and the LLM agent see it."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from inspect_robots.conformance import assert_embodiment_conformant, check_evaluation_interface
from inspect_robots.guard import DeltaLimitGuard
from inspect_robots.registry import resolve
from inspect_robots.tools import control_tools

import inspect_robots_agilex_nero as package
from inspect_robots_agilex_nero.config import DIM_LABELS, NeroConfig
from inspect_robots_agilex_nero.embodiment import _DOCS, NeroEmbodiment


def test_defaults_bounds_and_hand_none() -> None:
    cfg = NeroConfig()
    np.testing.assert_allclose(cfg.high, (0.02, 0.02, 0.02, *([0.2] * 6)))
    np.testing.assert_allclose(cfg.low, -cfg.high)
    np.testing.assert_allclose(NeroConfig(hand="none").high[3:], 0.0)
    assert len(DIM_LABELS) == 9


def test_flat_kwargs_are_coerced() -> None:
    cfg = NeroConfig.from_kwargs(
        backend="kinematic",
        control_hz="10",
        speed_percent="30",
        home_pos="0.3,0.1,0.25",
        closed_angles="0,0,0,0,200,800",
        unattended="yes",
        firmware="v121",
        hand_port="/dev/ttyUSB1",
    )
    assert cfg.control_hz == 10.0 and cfg.speed_percent == 30
    assert cfg.home_pos == (0.3, 0.1, 0.25)
    assert cfg.closed_angles == (0, 0, 0, 0, 200, 800)
    assert cfg.unattended is True and cfg.hand_port == "/dev/ttyUSB1"
    assert NeroConfig.from_kwargs(unattended="off", hand_speed=0.05).unattended is False
    with pytest.raises(ValueError, match="boolean"):
        NeroConfig.from_kwargs(unattended="maybe")
    with pytest.raises(TypeError, match="unknown NeroConfig option"):
        NeroConfig.from_kwargs(wheels=4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "isaac"}, "backend must be"),
        ({"hand": "dex3"}, "hand must be"),
        ({"firmware": "v2"}, "firmware must be"),
        ({"control_hz": 0}, "control_hz must be"),
        ({"x_min": 0.9}, "x_min must be below x_max"),
        ({"home_pos": (0.3, 0.0)}, "three values"),
        ({"tcp_offset": (0.0, float("nan"), 0.0)}, "finite"),
        ({"home_pos": (0.9, 0.0, 0.3)}, "inside the workspace"),
        ({"speed_percent": 0}, "speed_percent"),
        ({"hand_id": 0}, "hand_id"),
        ({"hand_force_value": 1001}, "hand_force_value"),
        ({"closed_angles": (0, 0, 0)}, "six values"),
        ({"open_angles": (0, 0, 0, 0, 0, 2000)}, "six values"),
        ({"camera_width": 0}, "camera_width"),
        ({"can_channel": " "}, "can_channel"),
    ],
)
def test_invalid_config_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NeroConfig(**kwargs)


def test_conformance_agent_and_guard() -> None:
    emb = NeroEmbodiment()
    assert_embodiment_conformant(emb.info)
    assert check_evaluation_interface(emb).ok
    assert emb.info.is_simulated is False
    assert NeroEmbodiment(backend="kinematic").info.environment_id == "nero:kinematic"
    tools = control_tools(emb.info.action_space, emb.info.observation_space, emb.info.control_hz)
    assert tools.library.names[0] == "move_by"
    assert "hand_z: [-0.02, 0.02]" in tools.bounds_text
    DeltaLimitGuard(emb.info.action_space)
    emb.close()  # nothing connected: a no-op
    with pytest.raises(TypeError, match="needs a NeroConfig"):
        NeroEmbodiment(config=object())  # type: ignore[arg-type]


def test_docs_name_every_dimension_once_and_take_extras() -> None:
    bullets = [line for line in _DOCS.splitlines() if line.startswith("- ")]
    for label in DIM_LABELS:
        assert sum(line.startswith(f"- {label}:") for line in bullets) == 1
    assert NeroEmbodiment(NeroConfig(docs_extra=" table at z=0 ")).info.docs == (
        _DOCS + "\n\ntable at z=0"
    )


def test_entry_point_and_public_api() -> None:
    emb = resolve("embodiment", "nero_inspire", backend="kinematic", hand_speed="0.05")
    assert isinstance(emb, NeroEmbodiment)
    assert np.asarray(emb.info.action_space.high)[0] == pytest.approx(0.01)
    assert "NeroEmbodiment" in package.__all__


def test_runtime_requirements_and_device_slots() -> None:
    from inspect_robots.conformance import device_slots, missing_runtime_requirements

    assert set(NeroEmbodiment.RUNTIME_REQUIREMENTS) == {"pyAgxArm", "serial", "cv2"}
    assert (
        "git+https://github.com/agilexrobotics/pyAgxArm"
        in (NeroEmbodiment.RUNTIME_REQUIREMENTS["pyAgxArm"])
    )
    assert "pyAgxArm" in missing_runtime_requirements(NeroEmbodiment)
    slots = {slot.arg: slot.kind for slot in device_slots(NeroEmbodiment)}
    assert slots == {"can_channel": "can", "hand_port": "serial", "camera_device": "v4l2"}
    fields = set(NeroConfig.__dataclass_fields__)
    assert set(slots) <= fields
