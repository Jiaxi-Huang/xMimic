from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Any

import motrix_envs.torch.adapter.utils.math as math_utils

if TYPE_CHECKING:
    from motrix_envs.torch.adapter.articulation import Articulation
    from motrix_envs.torch.manager_based_env import ManagerBasedTorchEnv as ManagerBasedRLEnv
from motrix_envs.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


RigidObject = Any


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def bad_dof_pos(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate non-finite joints or hard-limit violations above ``threshold``."""
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
    violation = torch.clamp(hard_limits[:, 0] - position, min=0.0)
    violation += torch.clamp(position - hard_limits[:, 1], min=0.0)
    finite = torch.all(torch.isfinite(position), dim=-1)
    return (~finite) | (torch.amax(violation, dim=-1) > threshold)


def bad_dof_vel(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate non-finite joints or absolute velocity spikes."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = (
        slice(None)
        if asset_cfg.joint_ids is None
        else torch.as_tensor(asset_cfg.joint_ids, dtype=torch.long, device=env.device)
    )
    velocity = asset.data.joint_vel[:, joint_ids]
    finite = torch.all(torch.isfinite(velocity), dim=-1)
    max_abs = torch.amax(
        torch.nan_to_num(torch.abs(velocity), nan=torch.inf, posinf=torch.inf), dim=-1
    )
    return (~finite) | (max_abs > threshold)


def bad_dof_vel_and_update_motion_sampling(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    failure_term_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Finish direct-WBT failure feedback after all preceding failure terms."""
    bad_velocity = bad_dof_vel(env, threshold=threshold, asset_cfg=asset_cfg)
    failed = bad_velocity.clone()
    for name in failure_term_names:
        values = env.termination_manager.buffers.get(name)
        if values is None:
            raise RuntimeError(
                f"Adaptive motion sampling requires preceding termination term {name!r}."
            )
        failed |= values
    command: MotionCommand = env.command_manager.get_term(command_name)
    command.update_adaptive_timestep_sampler(failed)
    return bad_velocity
