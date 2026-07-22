from dataclasses import dataclass

from motrix_envs.managers.adapter import ImplicitActuatorCfg


@dataclass(kw_only=True)
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """DexEVT PD actuator group with command-delay metadata.

    MotrixLab uses the PD fields to construct the URDF actuators. The delay
    bounds remain task metadata until delay sampling is supported by the
    generic Torch action runtime.
    """

    min_delay: int = 0
    max_delay: int = 0
