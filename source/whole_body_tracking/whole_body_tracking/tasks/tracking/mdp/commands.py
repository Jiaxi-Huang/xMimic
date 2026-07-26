from __future__ import annotations

import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from motrix_envs.managers import CommandTerm, CommandTermCfg
from motrix_envs.torch.adapter.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from motrix_envs.torch.adapter.articulation import Articulation
    from motrix_envs.torch.manager_based_env import ManagerBasedTorchEnv as ManagerBasedRLEnv

_NPZ_BODY_NAMES = (
    "pelvis",
    "hip_pitch_l_link", "hip_pitch_r_link", "imu_waist_link", "waist_yaw_link",
    "hip_roll_l_link", "hip_roll_r_link", "waist_roll_link",
    "hip_yaw_l_link", "hip_yaw_r_link", "waist_pitch_link",
    "knee_pitch_l_link", "knee_pitch_r_link", "camera_body_front_link",
    "head_yaw_link", "imu_head_link", "radar_head_link",
    "shoulder_pitch_l_link", "shoulder_pitch_r_link",
    "ankle_pitch_l_link", "ankle_pitch_r_link", "head_pitch_link",
    "shoulder_roll_l_link", "shoulder_roll_r_link",
    "ankle_roll_l_link", "ankle_roll_r_link", "camera_head_link",
    "shoulder_yaw_l_link", "shoulder_yaw_r_link",
    "elbow_pitch_l_link", "elbow_pitch_r_link",
    "elbow_yaw_l_link", "elbow_yaw_r_link",
    "wrist_pitch_l_link", "wrist_pitch_r_link",
    "wrist_roll_l_link", "wrist_roll_r_link", "left_tcp_link", "right_tcp_link",
)

_NPZ_JOINT_NAMES = (
    "hip_pitch_l_joint", "hip_pitch_r_joint", "waist_yaw_joint",
    "hip_roll_l_joint", "hip_roll_r_joint", "waist_roll_joint",
    "hip_yaw_l_joint", "hip_yaw_r_joint", "waist_pitch_joint",
    "knee_pitch_l_joint", "knee_pitch_r_joint",
    "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",
    "ankle_pitch_l_joint", "ankle_pitch_r_joint",
    "shoulder_roll_l_joint", "shoulder_roll_r_joint",
    "ankle_roll_l_joint", "ankle_roll_r_joint",
    "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_l_joint", "elbow_pitch_r_joint",
)


class MotionLoader:
    def __init__(
        self, motion_file: str, joint_indexes: Sequence[int], body_indexes: Sequence[int], device: str = "cpu"
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        self.joint_pos = torch.tensor(data["joint_pos"][:, joint_indexes], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"][:, joint_indexes], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.device = env.device

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        motion_joint_indexes = [_NPZ_JOINT_NAMES.index(name) for name in self.robot.joint_names]
        motion_body_indexes = [_NPZ_BODY_NAMES.index(name) for name in self.cfg.body_names]
        self.motion = MotionLoader(
            self.cfg.motion_file, motion_joint_indexes, motion_body_indexes, device=self.device
        )
        motion_steps_per_control_step = self.motion.fps * env.step_dt
        self.motion_step_stride = max(int(round(motion_steps_per_control_step)), 1)
        if not np.isclose(motion_steps_per_control_step, self.motion_step_stride):
            raise ValueError(
                "Motion FPS must be an integer multiple of the environment control frequency: "
                f"fps={self.motion.fps}, ctrl_dt={env.step_dt}."
            )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        # Include the raw (unaligned) reference root pose so policies can directly consume the
        # commanded root state in addition to the DOF targets.
        # root_pos = self.anchor_pos_w
        # root_rot_mat = matrix_from_quat(self.anchor_quat_w)
        # root_rot = root_rot_mat[..., :2].reshape(root_rot_mat.shape[0], -1)
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        self.time_steps[env_ids] = (phase * (self.motion.time_step_total - 1)).long()

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        root_state = torch.cat(
            [root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1
        )
        self.robot.write_root_and_joint_state_to_sim(
            root_state,
            joint_pos[env_ids],
            joint_vel[env_ids],
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += self.motion_step_stride
        # Clip completion is handled as a normal episode truncation. Keep the
        # final valid reference here until the environment resets the command;
        # resampling during a step update would otherwise teleport the robot
        # between the transition reward and its next observation.
        self.time_steps.clamp_(max=self.motion.time_step_total - 1)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

@dataclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = ""

    motion_file: str = ""
    anchor_body: str = ""
    body_names: list[str] = field(default_factory=list)

    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
