from dataclasses import dataclass

from motrix_rl.tasks.wbt import WbtAsyncFastSac


@dataclass
class DexEVTFlatAsyncFastSacCfg(WbtAsyncFastSac):
    """DexEVT FastSAC configuration aligned with morphos-lab WBT tuning."""

    num_envs: int = 2048

    def __post_init__(self) -> None:
        super().__post_init__()
        self.utd_mode = "strict"