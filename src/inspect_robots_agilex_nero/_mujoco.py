"""MuJoCo backend: the Nero + Inspire scene from ``scripts/build_mujoco_scene.py``.

It stands in for pyAgxArm and the Inspire serial link. ``move_flange`` solves
6-D damped least squares IK on the flange site and sets the arm's position
actuators, the way the real arm's planner turns a pose into joint motion.
Finger controls drive the six Inspire motors; the coupled joints follow
through equality constraints. Time is simulation time: ``advance`` steps
physics, so a run is as fast as the machine allows.

Not unit tested (needs MuJoCo and the vendor models); verified by running.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots_agilex_nero.config import FINGER_LABELS, NeroConfig
from inspect_robots_agilex_nero.nero import FlangePose, rpy_matrix
from inspect_robots_agilex_nero.sim_layout import (
    ARM_JOINTS,
    CAMERA_NAME,
    FINGER_JOINTS,
    matrix_rpy,
    start_rotation,
)

Vec = npt.NDArray[np.float64]


class MujocoNero:
    """Arm, hand, clock and camera over one MuJoCo model."""

    def __init__(self, cfg: NeroConfig) -> None:
        import mujoco

        if not cfg.mujoco_scene:
            raise ValueError("backend=mujoco needs -E mujoco_scene=PATH (see scripts/)")
        self._mj = mujoco
        self._cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.mujoco_scene)
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)
        model = self.model
        self._qadr = np.asarray([model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self._dadr = np.asarray([model.joint(n).dofadr[0] for n in ARM_JOINTS])
        self._lower = np.asarray([model.joint(n).range[0] for n in ARM_JOINTS])
        self._upper = np.asarray([model.joint(n).range[1] for n in ARM_JOINTS])
        self._arm_act = np.asarray([model.actuator(n).id for n in ARM_JOINTS])
        self._finger_act = np.asarray([model.actuator(n).id for n in FINGER_LABELS])
        self._finger_q = np.asarray(
            [model.joint(FINGER_JOINTS[n]).qposadr[0] for n in FINGER_LABELS]
        )
        self._finger_max = np.asarray(
            [model.joint(FINGER_JOINTS[n]).range[1] for n in FINGER_LABELS]
        )
        self._flange = model.site("flange").id
        self._renderer: Any = None
        self.stopped = False
        # Start with the hand a little above home, palm down.
        start_rpy = matrix_rpy(start_rotation())
        offset = rpy_matrix(start_rpy) @ np.asarray(cfg.tcp_offset)
        start = np.asarray(cfg.home_pos) + np.asarray((0.0, 0.0, 0.05)) - offset
        q = self._ik(start, start_rpy, np.asarray((0.0, 0.6, 0.0, 1.2, 0.0, 0.6, 0.0)))
        self.data.qpos[self._qadr] = q
        self.data.ctrl[self._arm_act] = q
        mujoco.mj_forward(model, self.data)
        self.advance(0.5)

    # ------------------------------------------------------------------ time

    def sim_time(self) -> float:
        """Simulation time in seconds."""
        return float(self.data.time)

    def advance(self, seconds: float) -> None:
        """Step physics for ``seconds``."""
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            self._mj.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ IK

    def _ik(self, position: Vec, rpy: Vec, seed: Vec) -> Vec:
        mj, model, data = self._mj, self.model, self._scratch
        target_r = rpy_matrix(rpy)
        q = np.clip(np.asarray(seed, dtype=np.float64), self._lower, self._upper)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        for _ in range(200):
            data.qpos[self._qadr] = q
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            current_r = data.site_xmat[self._flange].reshape(3, 3)
            error_p = position - data.site_xpos[self._flange]
            error_r = 0.5 * sum(np.cross(current_r[:, i], target_r[:, i]) for i in range(3))
            error = np.concatenate((error_p, 0.3 * error_r))
            if float(np.linalg.norm(error)) < 1e-5:
                break
            mj.mj_jacSite(model, data, jacp, jacr, self._flange)
            jac = np.vstack((jacp[:, self._dadr], 0.3 * jacr[:, self._dadr]))
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            q = np.clip(q + np.clip(step, -0.2, 0.2), self._lower, self._upper)
        return q

    # ------------------------------------------------------------------ ArmDriver

    def read_flange(self) -> FlangePose:
        """Flange site pose."""
        rotation = self.data.site_xmat[self._flange].reshape(3, 3)
        return FlangePose(self.data.site_xpos[self._flange].copy(), matrix_rpy(rotation))

    def read_joints(self) -> Vec:
        """Arm joint angles."""
        return np.array(self.data.qpos[self._qadr], dtype=np.float64)

    def move_flange(self, position: Vec, rpy: Vec) -> None:
        """Solve IK from the current command and retarget the arm actuators."""
        seed = self.data.ctrl[self._arm_act].astype(np.float64)
        self.data.ctrl[self._arm_act] = self._ik(np.asarray(position), np.asarray(rpy), seed)

    def hold(self) -> None:
        """Command the arm to stay where it is."""
        self.data.ctrl[self._arm_act] = self.data.qpos[self._qadr]

    def emergency_stop(self) -> None:
        """Hold and record the stop."""
        self.hold()
        self.stopped = True

    def disconnect(self) -> None:
        """Release the renderer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ HandDriver

    def set_fingers(self, closure: Vec) -> None:
        """Closure 0 to 1 onto each motor's 0 to upper-limit range."""
        alpha = np.clip(np.asarray(closure, dtype=np.float64), 0.0, 1.0)
        self.data.ctrl[self._finger_act] = alpha * self._finger_max

    def read_fingers(self) -> Vec:
        """Measured closures."""
        closure = np.asarray(self.data.qpos[self._finger_q], dtype=np.float64) / self._finger_max
        return np.asarray(np.clip(closure, 0.0, 1.0), dtype=np.float64)

    def close(self) -> None:
        """Nothing to release for the hand."""

    # ------------------------------------------------------------------ camera

    def read_frame(self) -> npt.NDArray[np.uint8]:
        """Render the scene camera."""
        if self._renderer is None:
            self._renderer = self._mj.Renderer(
                self.model, self._cfg.camera_height, self._cfg.camera_width
            )
        self._renderer.update_scene(self.data, camera=CAMERA_NAME)
        return np.asarray(self._renderer.render(), dtype=np.uint8)

    def object_position(self, name: str) -> Vec:
        """World position of a named body (for scoring and checks)."""
        return np.array(self.data.body(name).xpos, dtype=np.float64)
