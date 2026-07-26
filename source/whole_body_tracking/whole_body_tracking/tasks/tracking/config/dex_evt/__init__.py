from motrix_envs import registry
from motrix_rl import registry as rl_registry
from whole_body_tracking.utils.motrix import TrackingManagerBasedTorchEnv

from . import flat_env_cfg
from .agents import fastsac_async_cfg, rsl_rl_ppo_cfg

##
# Register Gym environments for Dex-V3.
##

_TASKS = {
    "Tracking-Flat-DexEVT-v0": (
        flat_env_cfg.DexEVTFlatEnvConfig,
        rsl_rl_ppo_cfg.DexEVTFlatPPORunnerCfg,
        fastsac_async_cfg.DexEVTFlatAsyncFastSacCfg,
    ),
    "Tracking-Flat-DexEVT-Wo-State-Estimation-v0": (
        flat_env_cfg.DexEVTFlatWoStateEstimationEnvCfg,
        rsl_rl_ppo_cfg.DexEVTFlatPPORunnerCfg,
        fastsac_async_cfg.DexEVTFlatAsyncFastSacCfg,
    ),
    "Tracking-Flat-DexEVT-Low-Freq-v0": (
        flat_env_cfg.DexEVTFlatLowFreqEnvCfg,
        rsl_rl_ppo_cfg.DexEVTFlatLowFreqPPORunnerCfg,
        fastsac_async_cfg.DexEVTFlatAsyncFastSacCfg,
    ),
}

for task_name, (env_cfg_cls, rslrl_cfg_cls, fastsac_cfg_cls) in _TASKS.items():
    registry.register_env_config(task_name, env_cfg_cls)
    registry.register_env(task_name, TrackingManagerBasedTorchEnv)
    rl_registry.rlcfg(task_name, backend="torch")(rslrl_cfg_cls)
    rl_registry.rlcfg(task_name, backend="torch")(fastsac_cfg_cls)


def make_env_config(task_name: str):
    """Build an xMimic task config before applying launcher-only overrides."""
    try:
        env_cfg_cls = _TASKS[task_name][0]
    except KeyError as error:
        raise ValueError(f"Unknown xMimic task '{task_name}'.") from error
    return env_cfg_cls()
