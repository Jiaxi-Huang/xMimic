from dataclasses import dataclass, field

from motrix_rl.rslrl.cfg import RslRlActorCfg, RslRlCriticCfg, RslRlPpoAlgorithmCfg, RslrlCfg, RslrlRunnerCfg


@dataclass
class DexEVTRslrlRunnerCfg(RslrlRunnerCfg):
    seed: int = 42
    device: str = "cpu"
    num_steps_per_env: int = 24
    max_iterations: int = 100000
    save_interval: int = 500
    experiment_name: str = "dex_evt_flat"
    obs_groups: dict[str, list[str]] = field(
        default_factory=lambda: {"actor": ["policy"], "critic": ["value"]}
    )
    actor: RslRlActorCfg = field(default_factory=lambda: RslRlActorCfg(
        init_noise_std=1.0,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    ))
    critic: RslRlCriticCfg = field(default_factory=lambda: RslRlCriticCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    ))
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=lambda: RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ))

    def to_dict(self) -> dict:
        """Translate MotrixRL's policy fields to RSL-RL's model schema."""
        result = super().to_dict()
        actor = result["actor"]
        actor.pop("stochastic")
        actor.pop("state_dependent_std")
        actor["distribution_cfg"] = {
            "class_name": "rsl_rl.modules:GaussianDistribution",
            "init_std": actor.pop("init_noise_std"),
            "std_type": actor.pop("noise_std_type"),
        }
        result["critic"].pop("stochastic")
        return result


@dataclass
class DexEVTFlatPPORunnerCfg(RslrlCfg):
    num_envs: int = 16
    play_num_envs: int = 1
    runner: DexEVTRslrlRunnerCfg = field(default_factory=DexEVTRslrlRunnerCfg)


LOW_FREQ_SCALE = 0.5


@dataclass
class DexEVTFlatLowFreqPPORunnerCfg(DexEVTFlatPPORunnerCfg):
    def __post_init__(self):
        self.runner.num_steps_per_env = round(self.runner.num_steps_per_env * LOW_FREQ_SCALE)
        self.runner.algorithm.gamma = self.runner.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.runner.algorithm.lam = self.runner.algorithm.lam ** (1 / LOW_FREQ_SCALE)
