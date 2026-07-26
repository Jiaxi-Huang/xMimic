from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from motrix_envs.mdp.torch.events import randomize_prop_by_op as _randomize_prop_by_op
from motrix_envs.managers import SceneEntityCfg

if TYPE_CHECKING:
    from motrix_envs.torch.adapter.articulation import Articulation
    from motrix_envs.torch.manager_based_env import ManagerBasedTorchEnv as ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    env_ids = env.resolve_env_ids(env_ids)

    # resolve joint indices
    if asset_cfg.joint_names is None:
        raise ValueError("randomize_joint_default_pos requires resolved joint_names.")
    joint_ids = asset.find_joints(asset_cfg.joint_names, preserve_order=True)[0]

    if pos_distribution_params is not None:
        pos = _randomize_prop_by_op(
            asset.data.default_joint_pos.clone(),
            pos_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
            rng=env.get_rng("event"),
        )[env_ids][:, joint_ids]
        asset.set_default_joint_positions(asset_cfg.joint_names, pos, env_ids=env_ids)


def align_stairs_with_envs(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    # stairs_low_offset: tuple[float, float, float],
    stairs_high_offset: tuple[float, float, float],
    # stairs_low_name: str = "stairs_low",
    stairs_high_name: str = "stairs_high",
    stairs_rot: tuple[float, float, float, float] = (1 ,0 ,0, 0),
):
    """Place stair rigid objects at each env origin so they follow terrain generator offsets."""
    origins = env.scene.env_origins
    if env_ids is None:
        env_ids = torch.arange(origins.shape[0], device=origins.device)
    else:
        env_ids = env_ids.to(origins.device)

    def _place(name: str, offset: tuple[float, float, float]):
        if name is None:
            return
        try:
            stairs = env.scene[name]
        except KeyError:
            return
        root_state = torch.zeros((len(env_ids), 13), device=origins.device)
        root_state[:, 0:3] = origins[env_ids] + torch.tensor(offset, device=origins.device, dtype=origins.dtype)
        root_state[:, 3:7] = torch.tensor(stairs_rot, device=origins.device, dtype=origins.dtype)
        # velocities stay zero
        stairs.write_root_state_to_sim(root_state, env_ids=env_ids)

    # _place(stairs_low_name, stairs_low_offset)
    _place(stairs_high_name, stairs_high_offset)


def align_chair_with_envs(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    chair_offset: tuple[float, float, float],
    chair_name: str = "chair",
    chair_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
):
    """Place chair rigid objects at each env origin so they follow terrain generator offsets."""
    origins = env.scene.env_origins
    if env_ids is None:
        env_ids = torch.arange(origins.shape[0], device=origins.device)
    else:
        env_ids = env_ids.to(origins.device)

    try:
        chair = env.scene[chair_name]
    except KeyError:
        return

    root_state = torch.zeros((len(env_ids), 13), device=origins.device)
    root_state[:, 0:3] = origins[env_ids] + torch.tensor(chair_offset, device=origins.device, dtype=origins.dtype)
    root_state[:, 3:7] = torch.tensor(chair_rot, device=origins.device, dtype=origins.dtype)
    chair.write_root_state_to_sim(root_state, env_ids=env_ids)


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids is None:
        raise ValueError("randomize_rigid_body_com requires resolved body_ids.")
    env_ids = env.resolve_env_ids(env_ids)
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)
    offsets = _randomize_prop_by_op(
        torch.zeros((len(env_ids), 3), device=env.device),
        (ranges[:, 0], ranges[:, 1]),
        None,
        slice(None),
        operation="abs",
        distribution="uniform",
        rng=env.get_rng("event"),
    )
    asset.set_center_of_mass_offsets(asset_cfg.body_ids, offsets, env_ids=env_ids)
