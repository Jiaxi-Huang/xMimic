from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from motrix_envs.managers import CommandTerm, CommandTermCfg, ManagerRuntimePhase
from motrix_envs.motion import AdaptiveTimestepsSampler
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
    from motrix_envs.torch.manager_based_env import (
        ManagerBasedTorchEnv as ManagerBasedRLEnv,
    )

_NPZ_BODY_NAMES = (
    "pelvis",
    "hip_pitch_l_link",
    "hip_pitch_r_link",
    "imu_waist_link",
    "waist_yaw_link",
    "hip_roll_l_link",
    "hip_roll_r_link",
    "waist_roll_link",
    "hip_yaw_l_link",
    "hip_yaw_r_link",
    "waist_pitch_link",
    "knee_pitch_l_link",
    "knee_pitch_r_link",
    "camera_body_front_link",
    "head_yaw_link",
    "imu_head_link",
    "radar_head_link",
    "shoulder_pitch_l_link",
    "shoulder_pitch_r_link",
    "ankle_pitch_l_link",
    "ankle_pitch_r_link",
    "head_pitch_link",
    "shoulder_roll_l_link",
    "shoulder_roll_r_link",
    "ankle_roll_l_link",
    "ankle_roll_r_link",
    "camera_head_link",
    "shoulder_yaw_l_link",
    "shoulder_yaw_r_link",
    "elbow_pitch_l_link",
    "elbow_pitch_r_link",
    "elbow_yaw_l_link",
    "elbow_yaw_r_link",
    "wrist_pitch_l_link",
    "wrist_pitch_r_link",
    "wrist_roll_l_link",
    "wrist_roll_r_link",
    "left_tcp_link",
    "right_tcp_link",
)

_NPZ_JOINT_NAMES = (
    "hip_pitch_l_joint",
    "hip_pitch_r_joint",
    "waist_yaw_joint",
    "hip_roll_l_joint",
    "hip_roll_r_joint",
    "waist_roll_joint",
    "hip_yaw_l_joint",
    "hip_yaw_r_joint",
    "waist_pitch_joint",
    "knee_pitch_l_joint",
    "knee_pitch_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "ankle_pitch_l_joint",
    "ankle_pitch_r_joint",
    "shoulder_roll_l_joint",
    "shoulder_roll_r_joint",
    "ankle_roll_l_joint",
    "ankle_roll_r_joint",
    "shoulder_yaw_l_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_l_joint",
    "elbow_pitch_r_joint",
)


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        device: str = "cpu",
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if "joint_names" in data.files and "body_names" in data.files:
            motion_joint_names = [str(name) for name in data["joint_names"]]
            motion_body_names = [str(name) for name in data["body_names"]]
            joint_indexes = [motion_joint_names.index(name) for name in joint_names]
            body_indexes = [motion_body_names.index(name) for name in body_names]
            body_quat_w = np.asarray(data["body_quat_w"])[..., (3, 0, 1, 2)]
        else:
            joint_indexes = [_NPZ_JOINT_NAMES.index(name) for name in joint_names]
            body_indexes = [_NPZ_BODY_NAMES.index(name) for name in body_names]
            body_quat_w = data["body_quat_w"]
        self.joint_pos = torch.tensor(
            data["joint_pos"][:, joint_indexes], dtype=torch.float32, device=device
        )
        self.joint_vel = torch.tensor(
            data["joint_vel"][:, joint_indexes], dtype=torch.float32, device=device
        )
        self._body_pos_w = torch.tensor(
            data["body_pos_w"], dtype=torch.float32, device=device
        )
        self._body_quat_w = torch.tensor(
            body_quat_w, dtype=torch.float32, device=device
        )
        self._body_lin_vel_w = torch.tensor(
            data["body_lin_vel_w"], dtype=torch.float32, device=device
        )
        self._body_ang_vel_w = torch.tensor(
            data["body_ang_vel_w"], dtype=torch.float32, device=device
        )
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
        self._device = env.device

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self._device,
        )

        self.motion = MotionLoader(
            self.cfg.motion_file,
            self.robot.joint_names,
            self.cfg.body_names,
            device=self._device,
        )
        motion_steps_per_control_step = self.motion.fps * env.step_dt
        self.motion_step_stride = max(int(round(motion_steps_per_control_step)), 1)
        if not np.isclose(motion_steps_per_control_step, self.motion_step_stride):
            raise ValueError(
                "Motion FPS must be an integer multiple of the environment control frequency: "
                f"fps={self.motion.fps}, ctrl_dt={env.step_dt}."
            )
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._device
        )
        self._adaptive_timestep_sampler = self._build_adaptive_timestep_sampler()

        self.metrics["error_anchor_pos"] = torch.zeros(
            self.num_envs, device=self._device
        )
        self.metrics["error_anchor_rot"] = torch.zeros(
            self.num_envs, device=self._device
        )
        self.metrics["error_anchor_lin_vel"] = torch.zeros(
            self.num_envs, device=self._device
        )
        self.metrics["error_anchor_ang_vel"] = torch.zeros(
            self.num_envs, device=self._device
        )
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self._device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self._device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self._device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self._device)
        if self._adaptive_timestep_sampler is not None:
            for name, value in self._adaptive_timestep_sampler.stats().items():
                self.metrics[name] = torch.full(
                    (self.num_envs,), value, device=self._device
                )

    def _build_adaptive_timestep_sampler(
        self,
    ) -> AdaptiveTimestepsSampler | None:
        if not self.cfg.adaptive_timestep_sampling:
            return None
        return AdaptiveTimestepsSampler(
            motion_time_step_total=self.motion.time_step_total,
            env_fps=int(round(1.0 / self._env.step_dt)),
            kernel_size=self.cfg.adaptive_kernel_size,
            kernel_lambda=self.cfg.adaptive_kernel_lambda,
            uniform_ratio=self.cfg.adaptive_uniform_ratio,
            alpha=self.cfg.adaptive_alpha,
        )

    def update_adaptive_timestep_sampler(self, failed: torch.Tensor) -> None:
        """Record current-frame failures before reset sampling, as direct WBT does."""
        sampler = self._adaptive_timestep_sampler
        if sampler is None:
            return
        failed_steps = self.time_steps[failed].detach().cpu().numpy()
        sampler.record_failures(failed_steps)
        sampler.update()
        for name, value in sampler.stats().items():
            self.metrics[name].fill_(value)

    @property
    def command(
        self,
    ) -> torch.Tensor:  # TODO Consider again if this is the best observation
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
        return (
            self.motion.body_pos_w[self.time_steps]
            + self._env.scene.env_origins[:, None, :]
        )

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
        return (
            self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[
            self.time_steps, self.motion_anchor_body_index
        ]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[
            self.time_steps, self.motion_anchor_body_index
        ]

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

    def _relative_body_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Align motion bodies against the robot's current anchor state."""
        num_bodies = len(self.cfg.body_names)
        anchor_pos_w = self.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        anchor_quat_w = self.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_pos_w = self.robot_anchor_pos_w[:, None, :].repeat(
            1, num_bodies, 1
        )
        robot_anchor_quat_w = self.robot_anchor_quat_w[:, None, :].repeat(
            1, num_bodies, 1
        )

        delta_pos_w = robot_anchor_pos_w.clone()
        delta_pos_w[..., 2] = anchor_pos_w[..., 2]
        delta_quat_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))
        body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w)
        body_pos_relative_w = delta_pos_w + quat_apply(
            delta_quat_w, self.body_pos_w - anchor_pos_w
        )
        return body_pos_relative_w, body_quat_relative_w

    @property
    def body_pos_relative_w(self) -> torch.Tensor:
        # This target depends on the post-physics robot anchor, so keep it
        # derived from the current state instead of caching it across steps.
        return self._relative_body_targets()[0]

    @property
    def body_quat_relative_w(self) -> torch.Tensor:
        return self._relative_body_targets()[1]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    def _resample_command(self, env_ids: Sequence[int]):
        num_resets = len(env_ids)
        sampler = self._adaptive_timestep_sampler
        if sampler is None:
            phase = sample_uniform(0.0, 1.0, (num_resets,), device=self._device)
            num_control_steps = (
                self.motion.time_step_total - 1
            ) // self.motion_step_stride
            sampled_steps = (phase * num_control_steps).long() * self.motion_step_stride
        else:
            sampled_steps = torch.as_tensor(
                sampler.sample(
                    num_resets,
                    max_step_exclusive=self.motion.time_step_total - 1,
                ),
                dtype=torch.long,
                device=self._device,
            )
            sampled_steps = (
                sampled_steps // self.motion_step_stride
            ) * self.motion_step_stride
        if self.cfg.start_at_timestep_zero_prob >= 1.0:
            sampled_steps.zero_()
        elif self.cfg.start_at_timestep_zero_prob > 0.0:
            start_at_zero = (
                sample_uniform(0.0, 1.0, (num_resets,), device=self._device)
                < self.cfg.start_at_timestep_zero_prob
            )
            sampled_steps[start_at_zero] = 0
        self.time_steps[env_ids] = sampled_steps

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self._device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self._device
        )
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self._device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self._device
        )
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos[env_ids] += sample_uniform(
            *self.cfg.joint_position_range,
            (num_resets, joint_pos.shape[-1]),
            joint_pos.device,
        )
        model_joint_ids = torch.as_tensor(
            self.robot.joint_ids, dtype=torch.long, device=self._device
        )
        hard_joint_pos_limits = torch.as_tensor(
            np.asarray(self._env.model.joint_limits, dtype=np.float32),
            dtype=joint_pos.dtype,
            device=self._device,
        )[:, model_joint_ids].T
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids],
            hard_joint_pos_limits[:, 0],
            hard_joint_pos_limits[:, 1],
        )
        root_state = torch.cat(
            [
                root_pos[env_ids],
                root_ori[env_ids],
                root_lin_vel[env_ids],
                root_ang_vel[env_ids],
            ],
            dim=-1,
        )
        self.robot.write_root_and_joint_state_to_sim(
            root_state,
            joint_pos[env_ids],
            joint_vel[env_ids],
            env_ids=env_ids,
        )
    def _update_command(self):
        self.time_steps += self.motion_step_stride
        if self.cfg.hold_at_clip_end:
            self.time_steps.clamp_(max=self.motion.time_step_total - 1)
            return
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if env_ids.numel():
            self._resample_command(env_ids)
            # Direct WBT starts the new clip with a zero action history.
            self._env.action_manager.reset(env_ids=env_ids, state=self._env.state)


@dataclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand
    update_phase: ManagerRuntimePhase | str = ManagerRuntimePhase.PRE_REWARD

    asset_name: str = ""

    motion_file: str = ""
    anchor_body: str = ""
    body_names: list[str] = field(default_factory=list)

    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    start_at_timestep_zero_prob: float = 0.0
    hold_at_clip_end: bool = False
    adaptive_timestep_sampling: bool = False
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_kernel_size: int = 1
    adaptive_kernel_lambda: float = 0.8
