"""xMimic boundaries for generic MotrixLab runtimes."""

from __future__ import annotations

import torch

from motrix_envs.np.env import NpEnvState, NpObs
from motrix_envs.torch.env import TorchEnvState
from motrix_envs.torch.manager_based_env import ManagerBasedTorchEnv
from motrix_rl.rslrl.torch import RslrlNpEnvWrap


class TorchManagerNpCompat:
    """Expose Torch manager results through the existing ``NpEnv`` contract.

    This adapter is used only at the xMimic-to-MotrixRL boundary. The wrapped
    environment and all task managers continue to operate on Torch tensors.
    """

    def __init__(self, env: ManagerBasedTorchEnv):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def init_state(self) -> NpEnvState:
        return _to_np_state(self._env.init_state())

    def step(self, actions) -> NpEnvState:
        tensor_actions = torch.as_tensor(actions, dtype=torch.float32, device=self._env.device)
        return _to_np_state(self._env.step(tensor_actions))


class TrackingManagerBasedTorchEnv(ManagerBasedTorchEnv):
    """Expose per-step reward terms to framework-neutral training loggers.

    The Torch manager records episode summaries under ``info["log"]`` for
    RSL-RL. FastSAC instead consumes per-step reward components from
    ``info["Reward"]``. Publishing the reward manager's already-scaled buffers
    here lets both trainers report the same task terms without recomputing any
    rewards or coupling the task to a specific RL algorithm.
    """

    def step(self, actions):
        state = super().step(actions)
        state.info["Reward"] = dict(self.reward_manager.buffers)
        return state


class RslRlTorchManagerWrap(RslrlNpEnvWrap):
    """Keep manager episode logs when adapting the environment to RSL-RL.

    ``motrix_rl`` 0.3.0 only forwards ``time_outs`` from ``NpEnvState.info``.
    RSL-RL consumes manager metrics from ``extras["log"]``, so dropping that
    entry removes all Episode_Reward, Metrics, and Episode_Termination series
    from TensorBoard.
    """

    def step(self, actions: torch.Tensor):
        manager_episode_length = self._env.episode_length_buf
        rsl_episode_length = self.episode_length_buf.to(manager_episode_length.device)
        if not torch.equal(rsl_episode_length, manager_episode_length):
            manager_episode_length.copy_(rsl_episode_length)
        obs, rewards, dones, extras = super().step(actions)
        episode_log = self._state.info.get("log")
        if episode_log:
            extras["log"] = episode_log
        return obs, rewards, dones, extras


def _to_np_state(state: TorchEnvState) -> NpEnvState:
    info = {
        name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        for name, value in state.info.items()
    }
    return NpEnvState(
        data=state.data,
        obs=NpObs(
            policy=state.obs.policy.detach().cpu().numpy(),
            value=None if state.obs.value is None else state.obs.value.detach().cpu().numpy(),
        ),
        reward=state.reward.detach().cpu().numpy(),
        terminated=state.terminated.detach().cpu().numpy(),
        truncated=state.truncated.detach().cpu().numpy(),
        info=info,
    )


__all__ = ["RslRlTorchManagerWrap", "TorchManagerNpCompat", "TrackingManagerBasedTorchEnv"]
