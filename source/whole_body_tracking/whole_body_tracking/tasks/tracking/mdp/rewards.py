from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from motrix_envs.managers import SceneEntityCfg
from motrix_envs.torch.adapter.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from motrix_envs.torch.adapter.sensors import ContactSensor
    from motrix_envs.torch.adapter.articulation import Articulation
    from motrix_envs.torch.manager_based_env import ManagerBasedTorchEnv as ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def torque_sum_excess(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Compute summed torque excess above a threshold (zero when under the limit)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)

    torque_sum = torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return torch.clamp(torque_sum - threshold, min=0.0)


def joint_position_limit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.9,
    cap: float = 5.0,
) -> torch.Tensor:
    """Match the direct WBT soft-limit penalty, including its per-env cap."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = (
        slice(None)
        if asset_cfg.joint_ids is None
        else torch.as_tensor(asset_cfg.joint_ids, dtype=torch.long, device=env.device)
    )
    position = asset.data.joint_pos[:, joint_ids]
    model_joint_ids = torch.as_tensor(
        asset.joint_ids, dtype=torch.long, device=env.device
    )
    hard_limits = torch.as_tensor(
        np.asarray(env.model.joint_limits, dtype=np.float32),
        dtype=position.dtype,
        device=position.device,
    )[:, model_joint_ids].T[joint_ids]
    middle = hard_limits.mean(dim=-1)
    half_range = 0.5 * (hard_limits[:, 1] - hard_limits[:, 0]) * soft_ratio
    lower = middle - half_range
    upper = middle + half_range
    penalty = torch.clamp(lower - position, min=0.0) + torch.clamp(
        position - upper, min=0.0
    )
    return torch.clamp(torch.sum(penalty, dim=-1), max=cap)


def direct_wbt_undesired_floor_contacts(
    env: ManagerBasedRLEnv,
    ground_name_token: str = "floor",
    allowed_name_tokens: tuple[str, ...] = ("foot", "hand", "wrist", "ankle"),
) -> torch.Tensor:
    """Reproduce direct WBT's geom-name-based undesired-contact selection."""
    ground_geom_ids = [
        geom.index
        for geom in env.model.geoms
        if geom.name and ground_name_token in geom.name
    ]
    if not ground_geom_ids:
        raise ValueError(f"No ground geom contains name token {ground_name_token!r}.")
    undesired_geom_ids = [
        geom.index
        for geom in env.model.geoms
        if geom.name
        and ground_name_token not in geom.name
        and not any(token in geom.name for token in allowed_name_tokens)
    ]
    if not undesired_geom_ids:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    pairs = np.asarray(
        [
            (geom_id, ground_id)
            for geom_id in undesired_geom_ids
            for ground_id in ground_geom_ids
        ],
        dtype=np.uint32,
    )
    colliding = env.model.get_contact_query(env.state.data).is_colliding(pairs)
    values = np.sum(colliding, axis=-1).astype(np.float32)
    return torch.as_tensor(values, dtype=torch.float32, device=env.device)
