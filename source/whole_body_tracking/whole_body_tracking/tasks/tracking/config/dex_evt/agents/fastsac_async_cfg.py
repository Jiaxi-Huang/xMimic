from dataclasses import dataclass

from motrix_rl.tasks.wbt import WbtAsyncFastSac


@dataclass
class DexEVTFlatAsyncFastSacCfg(WbtAsyncFastSac):
    """DexEVT FastSAC configuration aligned with morphos-lab WBT tuning."""

    num_envs: int = 2048

    def __post_init__(self) -> None:
        super().__post_init__()
        # Original unbounded-action configuration:
        # # Match the original Isaac Lab PPO control semantics: raw policy
        # # actions are unbounded and D3_ACTION_SCALE converts them to joint
        # # position offsets. A tanh actor would restrict every joint to only
        # # +/-0.25 rad around its default pose.
        # self.runner.agent.use_tanh = False
        # Keep the default tanh actor to match the motrixlab branch.
        self.utd_mode = "strict"
        self.device = "cuda"
