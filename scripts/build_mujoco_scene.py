"""Build a MuJoCo scene: the Nero arm with an Inspire RH56 hand, a table and objects.

Inputs are the vendors' own models, which this script does not ship:

- AgileX ``agx_arm_urdf`` (MIT): ``nero/urdf/nero_description.urdf`` and meshes.
  https://github.com/agilexrobotics/agx_arm_urdf
- Unitree ``unitree_ros`` (BSD-3): the right Inspire DFQ hand links from
  ``robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf``.
  https://github.com/unitreerobotics/unitree_ros

The real lender's mount is unknown, so the hand is bolted to the flange with
a guessed 3 cm spacer, fingers pointing out along the flange axis. Usage::

    python scripts/build_mujoco_scene.py --agx PATH/agx_arm_urdf \
        --unitree PATH/unitree_ros --out PATH/nero_scene.xml

It prints the ``tcp_offset`` to use with ``-E backend=mujoco``.
"""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HAND_URDF = "robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
#: Guessed mount: 3 cm out from the flange, hand fingers along flange +z.
MOUNT_XYZ = "0 0 0.03"
MOUNT_RPY = "-1.5707963 0 0"
#: Grasp point in the hand base frame, measured from where the fingers and
#: thumb close: in front of the palm (-x), level with the middle finger
#: joints (-y, fingers extend along -y) and centred across the palm (z).
GRASP_IN_HAND = (-0.025, -0.125, -0.01)

#: (control name, driving joint) for the six Inspire motors.
FINGER_JOINTS = (
    ("little_finger", "R_pinky_proximal_joint"),
    ("ring_finger", "R_ring_proximal_joint"),
    ("middle_finger", "R_middle_proximal_joint"),
    ("index_finger", "R_index_proximal_joint"),
    ("thumb_bend", "R_thumb_proximal_pitch_joint"),
    ("thumb_rotate", "R_thumb_proximal_yaw_joint"),
)


def combined_urdf(agx: Path, unitree: Path) -> ET.Element:
    """The Nero URDF with the right Inspire hand fixed to link7."""
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
    mount = ET.SubElement(nero, "joint", name="hand_mount", type="fixed")
    ET.SubElement(mount, "parent", link="link7")
    ET.SubElement(mount, "child", link="R_hand_base_link")
    ET.SubElement(mount, "origin", xyz=MOUNT_XYZ, rpy=MOUNT_RPY)
    extension = ET.SubElement(nero, "mujoco")
    ET.SubElement(
        extension,
        "compiler",
        balanceinertia="true",
        discardvisual="false",
        fusestatic="false",
        strippath="false",
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


def decorate(mjcf: Path, unitree: Path) -> None:
    """Add the table, objects, camera, sites, actuators and finger couplings."""
    tree = ET.parse(mjcf)
    root = tree.getroot()
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    asset = root.find("asset")
    assert asset is not None
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        width="512",
        height="512",
        rgb1="0.82 0.78 0.70",
        rgb2="0.76 0.72 0.64",
    )
    ET.SubElement(asset, "material", name="table", texture="grid", texrepeat="8 8")
    world = root.find("worldbody")
    assert world is not None
    for body in world.iter("body"):
        body.set("gravcomp", "1")
        # Robot parts touch the table and objects but not each other: their
        # convex collision hulls overlap at the joints.
        for geom in body.iter("geom"):
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
    ET.SubElement(world, "light", pos="0.4 -0.4 1.5", dir="0 0.3 -1", diffuse="0.9 0.9 0.9")
    ET.SubElement(world, "light", pos="-0.5 0.5 1.2", dir="0.4 -0.4 -1", diffuse="0.4 0.4 0.4")
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="plane",
        size="1 1 0.02",
        material="table",
        friction="1 0.01 0.001",
        contype="1",
        conaffinity="3",
    )
    objects = (
        ("red_block", "box", "0.02 0.02 0.02", "0.52 -0.06 0.02", "0.85 0.15 0.12 1"),
        ("blue_block", "box", "0.025 0.025 0.025", "0.48 0.20 0.025", "0.15 0.3 0.85 1"),
    )
    for name, kind, size, pos, rgba in objects:
        body = ET.SubElement(world, "body", name=name, pos=pos)
        ET.SubElement(body, "freejoint")
        ET.SubElement(
            body,
            "geom",
            type=kind,
            size=size,
            rgba=rgba,
            mass="0.05",
            friction="1.5 0.02 0.001",
            condim="4",
            contype="1",
            conaffinity="3",
        )
    ET.SubElement(
        world,
        "camera",
        name="scene_cam",
        pos="0.95 -0.55 0.65",
        xyaxes="0.5 0.866 0 -0.45 0.26 0.85",
        fovy="55",
    )
    for body in world.iter("body"):
        if body.get("name") == "link7":
            ET.SubElement(body, "site", name="flange", pos="0 0 0", size="0.005")
            ET.SubElement(
                body,
                "geom",
                name="mount_spacer",
                type="cylinder",
                size="0.03 0.015",
                pos="0 0 0.015",
                rgba="0.25 0.25 0.28 1",
                mass="0.05",
                contype="2",
                conaffinity="1",
            )
        if body.get("name") == "R_hand_base_link":
            ET.SubElement(
                body,
                "site",
                name="tcp",
                pos=" ".join(map(str, GRASP_IN_HAND)),
                size="0.008",
                rgba="1 1 0 0.6",
            )
    actuator = ET.SubElement(root, "actuator")
    for index in range(1, 8):
        ET.SubElement(
            actuator,
            "position",
            name=f"joint{index}",
            joint=f"joint{index}",
            kp="400",
            dampratio="1",
            forcerange="-60 60",
        )
    for name, joint in FINGER_JOINTS:
        ET.SubElement(
            actuator,
            "position",
            name=name,
            joint=joint,
            kp="10",
            dampratio="1",
            forcerange="-5 5",
        )
    equality = ET.SubElement(root, "equality")
    for follower, leader, multiplier in mimic_pairs(unitree):
        ET.SubElement(
            equality,
            "joint",
            joint1=follower,
            joint2=leader,
            polycoef=f"0 {multiplier} 0 0 0",
        )
    for joint in root.iter("joint"):
        if joint.get("name", "").startswith("joint"):
            joint.set("damping", "2")
            joint.set("armature", "0.05")
    tree.write(mjcf)


def main() -> None:
    """Build the scene and print the tcp_offset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agx", type=Path, required=True)
    parser.add_argument("--unitree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        urdf = Path(tmp) / "nero_inspire.urdf"
        ET.ElementTree(combined_urdf(args.agx, args.unitree)).write(urdf)
        mujoco.MjModel.from_xml_path(str(urdf))
        mujoco.mj_saveLastXML(str(args.out), mujoco.MjModel.from_xml_path(str(urdf)))
    decorate(args.out, args.unitree)
    model = mujoco.MjModel.from_xml_path(str(args.out))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    flange = model.site("flange").id
    tcp = model.site("tcp").id
    offset = data.site_xmat[flange].reshape(3, 3).T @ (data.site_xpos[tcp] - data.site_xpos[flange])
    print(f"wrote {args.out}: {model.nq} qpos, {model.nu} actuators")
    print("tcp_offset=" + ",".join(f"{v:.4f}" for v in np.round(offset, 4)))


if __name__ == "__main__":
    main()
