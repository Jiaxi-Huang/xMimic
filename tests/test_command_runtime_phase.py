from pathlib import Path

import torch

from motrix_envs.managers import ManagerRuntimePhase
from motrix_envs.torch.manager_based_env import ManagerBasedTorchEnv
from whole_body_tracking.tasks.tracking.config.dex_evt import make_env_config


ENV_NAME = "Tracking-Flat-DexEVT-Wo-State-Estimation-v0"
MOTION_FILE = Path(__file__).resolve().parents[1] / "motion_example/dance1_easy.npz"


def test_motion_command_advances_at_pre_reward_without_reordering_manager_step():
    cfg = make_env_config(ENV_NAME)
    cfg.commands.motion.motion_file = str(MOTION_FILE)
    cfg.commands.motion.start_at_timestep_zero_prob = 1.0
    env = ManagerBasedTorchEnv(cfg, num_envs=2)
    env.init_state()
    motion = env.command_manager.get_term("motion")
    initial_steps = motion.time_steps.clone()
    observed = {}

    def capture(phase):
        def callback(context):
            del context
            observed[phase] = motion.time_steps.clone()

        return callback

    env.manager_runtime.register(
        ManagerRuntimePhase.PRE_TERMINATION,
        capture("pre_termination"),
        name="test.capture.pre_termination",
    )
    env.manager_runtime.register(
        ManagerRuntimePhase.PRE_REWARD,
        capture("pre_reward"),
        name="test.capture.pre_reward",
        priority=100,
    )

    env.step(torch.zeros((env.num_envs, env.action_manager.action_dim), device=env.device))

    torch.testing.assert_close(observed["pre_termination"], initial_steps)
    torch.testing.assert_close(
        observed["pre_reward"],
        initial_steps + motion.motion_step_stride,
    )
    env.close()
