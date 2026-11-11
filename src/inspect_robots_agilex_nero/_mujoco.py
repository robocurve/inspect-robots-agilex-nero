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
#: IK convergence: position (m) and orientation (rad) error accepted as reached.
IK_POS_TOL = 1e-3
IK_ROT_TOL = 0.02


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
        self.ik_rejections = 0
        self._go_to_start()

    def _go_to_start(self) -> None:
        """Place the hand a little above home, palm down, and let it settle."""
        cfg = self._cfg
        start_rpy = matrix_rpy(start_rotation())
        offset = rpy_matrix(start_rpy) @ np.asarray(cfg.tcp_offset)
        start = np.asarray(cfg.home_pos) + np.asarray((0.0, 0.0, 0.05)) - offset
        q = self._ik(start, start_rpy, np.asarray((0.0, 0.6, 0.0, 1.2, 0.0, 0.6, 0.0)))
        if q is None:
            raise RuntimeError("MuJoCo: the start pose above home_pos is unreachable")
        self.data.qpos[self._qadr] = q
        self.data.ctrl[self._arm_act] = q
        self._mj.mj_forward(self.model, self.data)
        self.advance(0.5)

    def reset_scene(self) -> None:
        """Put the robot, blocks and time back to their start state."""
        self._mj.mj_resetData(self.model, self.data)
        self.stopped = False
        self._go_to_start()

    # ------------------------------------------------------------------ time

    def sim_time(self) -> float:
        """Simulation time in seconds."""
        return float(self.data.time)

    def advance(self, seconds: float) -> None:
        """Step physics for ``seconds``."""
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            self._mj.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ IK

    def _ik(self, position: Vec, rpy: Vec, seed: Vec) -> Vec | None:
        """6-D damped least squares; ``None`` if it does not converge.

        The orientation error is the rotation vector between the current and
        target orientations (``mju_subQuat``, rotated into the world frame),
        which stays correct up to 180 degrees.
        """
        mj, model, data = self._mj, self.model, self._scratch
        target_q = np.zeros(4)
        mj.mju_mat2Quat(target_q, rpy_matrix(rpy).reshape(-1))
        current_q = np.zeros(4)
        error_local = np.zeros(3)
        error_r = np.zeros(3)
        error_p = np.zeros(3)
        q = np.clip(np.asarray(seed, dtype=np.float64), self._lower, self._upper)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        for _ in range(300):
            data.qpos[self._qadr] = q
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            mj.mju_mat2Quat(current_q, data.site_xmat[self._flange])
            mj.mju_subQuat(error_local, target_q, current_q)
            # mju_subQuat gives the error in the flange frame; the Jacobian is in the world frame.
            error_r = data.site_xmat[self._flange].reshape(3, 3) @ error_local
            error_p = position - data.site_xpos[self._flange]
            if np.linalg.norm(error_p) < IK_POS_TOL and np.linalg.norm(error_r) < IK_ROT_TOL / 10:
                return q
            mj.mj_jacSite(model, data, jacp, jacr, self._flange)
            jac = np.vstack((jacp[:, self._dadr], 0.3 * jacr[:, self._dadr]))
            error = np.concatenate((error_p, 0.3 * error_r))
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            q = np.clip(q + np.clip(step, -0.2, 0.2), self._lower, self._upper)
        converged = (
            np.linalg.norm(error_p) < 5 * IK_POS_TOL and np.linalg.norm(error_r) < IK_ROT_TOL
        )
        return q if converged else None

    # ------------------------------------------------------------------ ArmDriver

    def read_flange(self) -> FlangePose:
        """Flange site pose."""
        rotation = self.data.site_xmat[self._flange].reshape(3, 3)
        return FlangePose(self.data.site_xpos[self._flange].copy(), matrix_rpy(rotation))

    def read_joints(self) -> Vec:
        """Arm joint angles."""
        return np.array(self.data.qpos[self._qadr], dtype=np.float64)

    def move_flange(self, position: Vec, rpy: Vec) -> None:
        """Solve IK from the current command and retarget the arm actuators.

        An unreachable pose is rejected, as a planner would: the arm keeps its
        previous target and ``hand_pos`` shows where it really is.
        """
        seed = self.data.ctrl[self._arm_act].astype(np.float64)
        q = self._ik(np.asarray(position), np.asarray(rpy), seed)
        if q is None:
            self.ik_rejections += 1
            return
        self.data.ctrl[self._arm_act] = q

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
