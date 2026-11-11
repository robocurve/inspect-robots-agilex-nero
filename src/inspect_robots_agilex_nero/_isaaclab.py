"""Isaac Lab backend: the shared simulated station (``sim_layout``) on PhysX.

The robot is the same joined Nero + Inspire URDF the MuJoCo backend uses,
converted to USD with Isaac Lab's ``scripts/tools/convert_urdf.py --fix-base``
(the finger couplings stay PhysX mimic joints). ``move_flange`` sets a pose
target and every physics step servos toward it with Isaac Lab's
damped-least-squares IK, standing in for the arm's own planner. Finger
controls drive the six Inspire motor joints with the stiffness Isaac Lab
uses for its own Inspire hand. Gravity is off for the robot, as the real
arm compensates for it. Time is simulation time.

Not unit tested (needs Isaac Sim and a GPU); verified by running.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots_agilex_nero.config import FINGER_LABELS, NeroConfig
from inspect_robots_agilex_nero.nero import FlangePose, rpy_matrix
from inspect_robots_agilex_nero.sim_layout import (
    ARM_JOINTS,
    BLOCKS,
    CAMERA_EYE,
    CAMERA_FOVY_DEG,
    CAMERA_TARGET,
    FINGER_JOINTS,
    matrix_rpy,
    start_rotation,
)

Vec = npt.NDArray[np.float64]

_APP: Any = None
PHYSICS_DT = 1.0 / 200.0
#: Start of the arm before it servos to the start pose (same seed as MuJoCo).
SEED_Q = (0.0, 0.6, 0.0, 1.2, 0.0, 0.6, 0.0)


def _launch_app() -> None:
    global _APP
    if _APP is None:
        from isaaclab.app import AppLauncher

        _APP = AppLauncher(headless=True, enable_cameras=True).app


class IsaacLabNero:
    """Arm, hand, clock and camera over one Isaac Lab scene."""

    def __init__(self, cfg: NeroConfig) -> None:
        if not cfg.isaac_robot_usd:
            raise ValueError("backend=isaaclab needs -E isaac_robot_usd=PATH (see README)")
        _launch_app()
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils import configclass

        self._torch = torch
        self._cfg = cfg
        leaders = [FINGER_JOINTS[name] for name in FINGER_LABELS]

        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=cfg.isaac_robot_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=dict(zip(ARM_JOINTS, SEED_Q, strict=True))
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["joint[1-7]"],
                    effort_limit_sim=60.0,
                    velocity_limit_sim=3.0,
                    stiffness=400.0,
                    damping=40.0,
                ),
                "hand": ImplicitActuatorCfg(
                    joint_names_expr=["R_.*"],
                    effort_limit_sim=30.0,
                    velocity_limit_sim=10.0,
                    stiffness=10.0,
                    damping=0.2,
                ),
            },
        )
        blocks = {
            block.name: RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + block.name,
                init_state=RigidObjectCfg.InitialStateCfg(pos=block.pos),
                spawn=sim_utils.CuboidCfg(
                    size=tuple(2 * h for h in block.half),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=block.mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.5, dynamic_friction=1.2
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=block.rgb),
                ),
            )
            for block in BLOCKS
        }
        focal = 24.0
        vertical = 2 * focal * float(np.tan(np.radians(CAMERA_FOVY_DEG) / 2))
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/scene_cam",
            update_period=0.0,
            height=cfg.camera_height,
            width=cfg.camera_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal,
                horizontal_aperture=vertical * cfg.camera_width / cfg.camera_height,
                vertical_aperture=vertical,
                clipping_range=(0.02, 20.0),
            ),
        )

        @configclass
        class _SceneCfg(InteractiveSceneCfg):  # type: ignore[misc]
            ground = AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.8
                    )
                ),
            )
            light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
            )

        scene_cfg = _SceneCfg(num_envs=1, env_spacing=2.0)
        scene_cfg.robot = robot
        scene_cfg.scene_cam = camera
        for name, block_cfg in blocks.items():
            setattr(scene_cfg, name, block_cfg)

        self._sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=PHYSICS_DT, device="cuda:0")
        )
        self._scene = InteractiveScene(scene_cfg)
        self._sim.reset()
        self._robot = self._scene["robot"]
        self._camera = self._scene["scene_cam"]
        device = self._sim.device
        self._camera.set_world_poses_from_view(
            torch.tensor([CAMERA_EYE], device=device), torch.tensor([CAMERA_TARGET], device=device)
        )
        names = self._robot.joint_names
        self._arm_ids = [names.index(n) for n in ARM_JOINTS]
        self._finger_ids = [names.index(n) for n in leaders]
        limits = self._robot.data.joint_pos_limits[0].cpu().numpy()
        self._finger_max = limits[self._finger_ids, 1].astype(np.float64)
        self._flange_body = self._robot.body_names.index("link7")
        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs=1,
            device=device,
        )
        self._targets = self._robot.data.default_joint_pos.clone()
        self._finger_cmd = np.zeros(len(leaders), dtype=np.float64)
        self._sim_time = 0.0
        self.stopped = False
        # Servo to the start pose: hand a little above home, palm down.
        start_rpy = matrix_rpy(start_rotation())
        offset = rpy_matrix(start_rpy) @ np.asarray(cfg.tcp_offset)
        start = np.asarray(cfg.home_pos) + np.asarray((0.0, 0.0, 0.05)) - offset
        self.move_flange(start, start_rpy)
        self.advance(2.0)

    # ------------------------------------------------------------------ time

    def sim_time(self) -> float:
        """Simulation time in seconds."""
        return self._sim_time

    def advance(self, seconds: float) -> None:
        """Step physics for ``seconds``, servoing the arm toward its pose target."""
        torch = self._torch
        robot = self._robot
        for _ in range(max(1, round(seconds / PHYSICS_DT))):
            pos = robot.data.body_pos_w[:, self._flange_body]
            quat = robot.data.body_quat_w[:, self._flange_body]
            jacobian = robot.root_physx_view.get_jacobians()[
                :, self._flange_body - 1, :, self._arm_ids
            ]
            joint_pos = robot.data.joint_pos[:, self._arm_ids]
            arm_q = self._ik.compute(pos, quat, jacobian, joint_pos)
            self._targets[:, self._arm_ids] = arm_q
            self._targets[:, self._finger_ids] = torch.tensor(
                self._finger_cmd * self._finger_max, dtype=torch.float32, device=self._sim.device
            )
            robot.set_joint_position_target(self._targets)
            self._scene.write_data_to_sim()
            self._sim.step(render=False)
            self._scene.update(PHYSICS_DT)
            self._sim_time += PHYSICS_DT

    # ------------------------------------------------------------------ ArmDriver

    def read_flange(self) -> FlangePose:
        """Pose of link7 (the flange)."""
        from isaaclab.utils.math import matrix_from_quat

        data = self._robot.data
        pos = data.body_pos_w[0, self._flange_body].cpu().numpy().astype(np.float64)
        rotation = matrix_from_quat(data.body_quat_w[0, self._flange_body]).cpu().numpy()
        return FlangePose(pos, matrix_rpy(rotation))

    def read_joints(self) -> Vec:
        """Arm joint angles."""
        q = self._robot.data.joint_pos[0, self._arm_ids].cpu().numpy()
        return np.asarray(q, dtype=np.float64)

    def move_flange(self, position: Vec, rpy: Vec) -> None:
        """Set the pose target the IK servos toward."""
        from isaaclab.utils.math import quat_from_matrix

        torch = self._torch
        device = self._sim.device
        rotation = torch.tensor(rpy_matrix(rpy), dtype=torch.float32, device=device)
        quat = quat_from_matrix(rotation)
        target = torch.tensor(np.asarray(position), dtype=torch.float32, device=device)
        self._ik.set_command(torch.cat((target, quat)).unsqueeze(0))

    def hold(self) -> None:
        """Retarget the arm to where it is now."""
        flange = self.read_flange()
        self.move_flange(flange.position, flange.rpy)

    def emergency_stop(self) -> None:
        """Hold and record the stop."""
        self.hold()
        self.stopped = True

    def disconnect(self) -> None:
        """Nothing to release; the app stays up for the process."""

    # ------------------------------------------------------------------ HandDriver

    def set_fingers(self, closure: Vec) -> None:
        """Closure 0 to 1 onto each motor's 0 to upper-limit range."""
        self._finger_cmd = np.clip(np.asarray(closure, dtype=np.float64), 0.0, 1.0)

    def read_fingers(self) -> Vec:
        """Measured closures."""
        q = self._robot.data.joint_pos[0, self._finger_ids].cpu().numpy().astype(np.float64)
        return np.asarray(np.clip(q / self._finger_max, 0.0, 1.0), dtype=np.float64)

    def close(self) -> None:
        """Nothing to release for the hand."""

    # ------------------------------------------------------------------ camera

    def read_frame(self) -> npt.NDArray[np.uint8]:
        """Render and return the scene camera's RGB frame."""
        self._sim.render()
        self._camera.update(PHYSICS_DT, force_recompute=True)
        rgb = self._camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def object_position(self, name: str) -> Vec:
        """World position of a block."""
        return np.asarray(self._scene[name].data.root_pos_w[0].cpu().numpy(), dtype=np.float64)
