"""One simulated station, shared by every simulator backend (MuJoCo, Isaac Lab).

The robot is the Nero from AgileX ``agx_arm_urdf`` (MIT) with the right
Inspire DFQ hand from Unitree ``unitree_ros`` (BSD-3), bolted to the flange by
a guessed mount. The table top is the plane z = 0 in the arm base frame, with
a red and a blue block and a fixed scene camera. Keeping this in one place
means a trial means the same thing in either simulator.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

HAND_URDF = "robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
ARM_JOINTS: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
#: Inspire control name to the joint its motor drives in Unitree's model.
FINGER_JOINTS: dict[str, str] = {
    "little_finger": "R_pinky_proximal_joint",
    "ring_finger": "R_ring_proximal_joint",
    "middle_finger": "R_middle_proximal_joint",
    "index_finger": "R_index_proximal_joint",
    "thumb_bend": "R_thumb_proximal_pitch_joint",
    "thumb_rotate": "R_thumb_proximal_yaw_joint",
}
#: Guessed mount: 3 cm out from the flange, fingers along flange +z.
MOUNT_XYZ = (0.0, 0.0, 0.03)
MOUNT_RPY = (-math.pi / 2, 0.0, 0.0)
#: Hand point in the hand base frame: just in front of the palm (-x), level
#: with the finger roots (-y, fingers extend along -y), centred across (z).
GRASP_IN_HAND = (-0.025, -0.125, -0.01)
#: The matching ``tcp_offset`` (flange frame), as printed by the scene builder.
TCP_OFFSET = (-0.025, -0.01, 0.155)
#: Run settings every simulator uses: home, workspace box, and the table.
SIM_SETTINGS: dict[str, str] = {
    "tcp_offset": ",".join(str(v) for v in TCP_OFFSET),
    "home_pos": "0.45,0.05,0.22",
    "x_min": "0.38",
    "x_max": "0.65",
    "z_min": "0.005",
    "z_max": "0.40",
}
SIM_DOCS = (
    "Simulation. The table top is at z = 0. The red block is a 4 cm cube and the blue "
    "block a 5 cm cube. The palm faces down and the fingers point forward (+x); the "
    "thumb is on the left (+y) side."
)


@dataclass(frozen=True)
class Block:
    """A loose box on the table: half extents and centre in metres, colour, mass."""

    name: str
    half: tuple[float, float, float]
    pos: tuple[float, float, float]
    rgb: tuple[float, float, float]
    mass: float = 0.05


BLOCKS: tuple[Block, ...] = (
    Block("red_block", (0.02, 0.02, 0.02), (0.52, -0.06, 0.02), (0.85, 0.15, 0.12)),
    Block("blue_block", (0.025, 0.025, 0.025), (0.48, 0.20, 0.025), (0.15, 0.3, 0.85)),
)
CAMERA_NAME = "scene_cam"
CAMERA_EYE = (0.95, -0.55, 0.65)
CAMERA_TARGET = (0.45, 0.05, 0.05)
CAMERA_FOVY_DEG = 55.0


def start_rotation() -> Vec:
    """Flange orientation at the start: palm down, fingers forward and 10 degrees down.

    In the mount the fingers lie along flange +z and the palm faces flange -x.
    The Nero's wrist (joint 6) bends only about 45 degrees, so the hand points
    roughly along the forearm; the tilt keeps away from the roll-pitch-yaw
    singularity at 90 degrees of pitch.
    """
    tilt = math.radians(10.0)
    fingers = np.asarray((math.cos(tilt), 0.0, -math.sin(tilt)))
    palm_normal = np.asarray((-math.sin(tilt), 0.0, -math.cos(tilt)))
    x_axis = -palm_normal
    return np.column_stack((x_axis, np.cross(fingers, x_axis), fingers))


def matrix_rpy(rotation: npt.ArrayLike) -> Vec:
    """Fixed-axis roll, pitch, yaw of a rotation matrix (inverse of ``rpy_matrix``)."""
    r = np.asarray(rotation, dtype=np.float64)
    if abs(float(r[2, 0])) > 1.0 - 1e-9:
        # Gimbal lock (pitch at +/-90 degrees): only roll -/+ yaw is observable.
        # Put it all in yaw, with roll zero, so the rotation round-trips exactly.
        pitch = -math.copysign(math.pi / 2, float(r[2, 0]))
        yaw = math.atan2(-float(r[0, 1]), float(r[1, 1]))
        return np.asarray((0.0, pitch, yaw), dtype=np.float64)
    pitch = -math.asin(max(-1.0, min(1.0, float(r[2, 0]))))
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


def camera_axes() -> tuple[Vec, Vec, Vec]:
    """Camera right, up and backward unit vectors (OpenGL convention) for the scene camera."""
    eye, target = np.asarray(CAMERA_EYE), np.asarray(CAMERA_TARGET)
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross(np.asarray((0.0, 0.0, 1.0)), back)
    right /= np.linalg.norm(right)
    return right, np.cross(back, right), back


def combined_urdf(agx: Path, unitree: Path) -> ET.Element:
    """The Nero URDF with the right Inspire hand fixed to link7 by the guessed mount."""
    nero = ET.parse(agx / "nero/urdf/nero_description.urdf").getroot()
    for mesh in nero.iter("mesh"):
        name = Path(mesh.get("filename", "")).name.replace(".dae", ".stl")
        mesh.set("filename", str((agx / "nero/meshes" / name).resolve()))
    hand_root = ET.parse(unitree / HAND_URDF).getroot()
    mesh_dir = (unitree / "robots/g1_description").resolve()
    for link in hand_root.findall("link"):
        if link.get("name", "").startswith("R_"):
            for mesh in link.iter("mesh"):
                mesh.set("filename", str(mesh_dir / mesh.get("filename", "")))
            nero.append(link)
    for joint in hand_root.findall("joint"):
        name = joint.get("name", "")
        if name.startswith("R_") and name != "R_base_link_joint":
            nero.append(joint)
    # A 6 cm round spacer stands in for the custom mount.
    spacer = ET.SubElement(nero, "link", name="mount_spacer")
    inertial = ET.SubElement(spacer, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0.015", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="0.05")
    ET.SubElement(
        inertial, "inertia", ixx="1.5e-5", ixy="0", ixz="0", iyy="1.5e-5", iyz="0", izz="2.3e-5"
    )
    for tag in ("visual", "collision"):
        part = ET.SubElement(spacer, tag)
        ET.SubElement(part, "origin", xyz="0 0 0.015", rpy="0 0 0")
        ET.SubElement(ET.SubElement(part, "geometry"), "cylinder", radius="0.03", length="0.03")
    spacer_joint = ET.SubElement(nero, "joint", name="spacer_mount", type="fixed")
    ET.SubElement(spacer_joint, "parent", link="link7")
    ET.SubElement(spacer_joint, "child", link="mount_spacer")
    ET.SubElement(spacer_joint, "origin", xyz="0 0 0", rpy="0 0 0")
    mount = ET.SubElement(nero, "joint", name="hand_mount", type="fixed")
    ET.SubElement(mount, "parent", link="link7")
    ET.SubElement(mount, "child", link="R_hand_base_link")
    ET.SubElement(
        mount,
        "origin",
        xyz=" ".join(str(v) for v in MOUNT_XYZ),
        rpy=" ".join(str(v) for v in MOUNT_RPY),
    )
    return nero


def mimic_pairs(unitree: Path) -> list[tuple[str, str, float]]:
    """(follower, leader, multiplier) for the hand's coupled joints."""
    root = ET.parse(unitree / HAND_URDF).getroot()
    pairs = []
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if joint.get("name", "").startswith("R_") and mimic is not None:
            pairs.append(
                (joint.get("name", ""), mimic.get("joint", ""), float(mimic.get("multiplier", "1")))
            )
    return pairs
