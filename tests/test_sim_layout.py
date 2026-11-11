"""The shared simulated station: geometry helpers and the joined robot URDF."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from inspect_robots_agilex_nero.nero import rpy_matrix
from inspect_robots_agilex_nero.sim_layout import (
    BLOCKS,
    CAMERA_EYE,
    CAMERA_TARGET,
    HAND_URDF,
    SIM_SETTINGS,
    TCP_OFFSET,
    camera_axes,
    combined_urdf,
    matrix_rpy,
    mimic_pairs,
    start_rotation,
)


def test_start_rotation_is_palm_down_fingers_forward() -> None:
    rotation = start_rotation()
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    fingers, palm_up = rotation[:, 2], rotation[:, 0]
    assert fingers[0] > 0.98 and fingers[2] < 0  # forward, tilted down
    assert palm_up[2] > 0.98  # flange -x (the palm) faces down
    np.testing.assert_allclose(rpy_matrix(matrix_rpy(rotation)), rotation, atol=1e-12)


def test_camera_looks_at_its_target() -> None:
    right, up, back = camera_axes()
    view = np.asarray(CAMERA_TARGET) - np.asarray(CAMERA_EYE)
    np.testing.assert_allclose(-back, view / np.linalg.norm(view))
    assert abs(right[2]) < 1e-12 and up[2] > 0
    np.testing.assert_allclose(np.cross(right, up), back, atol=1e-12)


def test_settings_and_blocks_are_consistent() -> None:
    assert SIM_SETTINGS["tcp_offset"] == ",".join(map(str, TCP_OFFSET))
    for block in BLOCKS:
        assert math.isclose(block.pos[2], block.half[2])  # resting on the table


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_combined_urdf_joins_the_hand_to_link7(tmp_path: Path) -> None:
    agx, unitree = tmp_path / "agx", tmp_path / "unitree"
    _write(
        agx / "nero/urdf/nero_description.urdf",
        '<robot name="nero"><link name="link7"><visual><geometry>'
        '<mesh filename="package://x/nero/meshes/dae/link7.dae"/></geometry></visual></link></robot>',
    )
    _write(
        unitree / HAND_URDF,
        '<robot name="g1"><link name="torso_link"/>'
        '<link name="R_hand_base_link"><visual><geometry><mesh filename="meshes/R.STL"/>'
        '</geometry></visual></link><link name="R_index_proximal"/>'
        '<link name="R_index_intermediate"/>'
        '<joint name="R_base_link_joint" type="fixed"><parent link="x"/>'
        '<child link="R_hand_base_link"/></joint>'
        '<joint name="R_index_proximal_joint" type="revolute"><parent link="R_hand_base_link"/>'
        '<child link="R_index_proximal"/></joint>'
        '<joint name="R_index_intermediate_joint" type="revolute"><parent link="R_index_proximal"/>'
        '<child link="R_index_intermediate"/>'
        '<mimic joint="R_index_proximal_joint" multiplier="1.6"/>'
        '</joint><joint name="waist" type="fixed"/></robot>',
    )
    root = combined_urdf(agx, unitree)
    links = {link.get("name") for link in root.findall("link")}
    assert {"link7", "mount_spacer", "R_hand_base_link", "R_index_proximal"} <= links
    assert "torso_link" not in links
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    assert "R_base_link_joint" not in joints and "waist" not in joints
    mount = joints["hand_mount"]
    assert mount.find("parent").get("link") == "link7"  # type: ignore[union-attr]
    meshes = [mesh.get("filename", "") for mesh in root.iter("mesh")]
    assert meshes[0].endswith("nero/meshes/link7.stl")
    assert meshes[1].endswith("robots/g1_description/meshes/R.STL")
    assert mimic_pairs(unitree) == [("R_index_intermediate_joint", "R_index_proximal_joint", 1.6)]
    ET.tostring(root)  # serialises


@pytest.mark.parametrize("pitch", [math.pi / 2, -math.pi / 2])
def test_matrix_rpy_round_trips_at_gimbal_lock(pitch: float) -> None:
    rotation = rpy_matrix((0.4, pitch, 1.1))
    np.testing.assert_allclose(rpy_matrix(matrix_rpy(rotation)), rotation, atol=1e-9)
