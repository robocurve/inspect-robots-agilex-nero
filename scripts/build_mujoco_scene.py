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

from inspect_robots_agilex_nero.sim_layout import (
    BLOCKS,
    CAMERA_EYE,
    CAMERA_FOVY_DEG,
    CAMERA_NAME,
    FINGER_JOINTS,
    GRASP_IN_HAND,
    camera_axes,
    combined_urdf,
    mimic_pairs,
)


def mujoco_urdf(agx: Path, unitree: Path) -> ET.Element:
    """The shared combined URDF plus MuJoCo's compiler settings."""
    root = combined_urdf(agx, unitree)
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(
        extension,
        "compiler",
        balanceinertia="true",
        discardvisual="false",
        fusestatic="false",
        strippath="false",
    )
    return root


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
    for block in BLOCKS:
        body = ET.SubElement(world, "body", name=block.name, pos=" ".join(map(str, block.pos)))
        ET.SubElement(body, "freejoint")
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(map(str, block.half)),
            rgba=" ".join(map(str, (*block.rgb, 1))),
            mass=str(block.mass),
            friction="1.5 0.02 0.001",
            condim="4",
            contype="1",
            conaffinity="3",
        )
    right, up, _ = camera_axes()
    ET.SubElement(
        world,
        "camera",
        name=CAMERA_NAME,
        pos=" ".join(map(str, CAMERA_EYE)),
        xyaxes=" ".join(f"{v:.4f}" for v in (*right, *up)),
        fovy=str(CAMERA_FOVY_DEG),
    )
    for body in world.iter("body"):
        if body.get("name") == "link7":
            ET.SubElement(body, "site", name="flange", pos="0 0 0", size="0.005")
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
    for name, joint in FINGER_JOINTS.items():
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
    parser.add_argument("--urdf-out", type=Path, help="also write the combined URDF (for Isaac)")
    args = parser.parse_args()
    if args.urdf_out:
        ET.ElementTree(combined_urdf(args.agx, args.unitree)).write(args.urdf_out)
    with tempfile.TemporaryDirectory() as tmp:
        urdf = Path(tmp) / "nero_inspire.urdf"
        ET.ElementTree(mujoco_urdf(args.agx, args.unitree)).write(urdf)
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
