import copy
from dataclasses import dataclass, replace

from motrix_envs.managers import (
    EventTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from whole_body_tracking.robots.dex_evt import D3_ACTION_SCALE, DEX_EVT_CFG
from whole_body_tracking.tasks.tracking.config.dex_evt.agents.rsl_rl_ppo_cfg import (
    LOW_FREQ_SCALE,
)
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


_MORPHOS_TRACKED_BODY_NAMES = [
    "pelvis",
    "hip_roll_l_link",
    "knee_pitch_l_link",
    "ankle_roll_l_link",
    "hip_roll_r_link",
    "knee_pitch_r_link",
    "ankle_roll_r_link",
    "waist_pitch_link",
    "shoulder_roll_l_link",
    "elbow_pitch_l_link",
    "shoulder_roll_r_link",
    "elbow_pitch_r_link",
]

# Temporary parity values for morphos-lab's DexEvtWbtControlConfig. That
# environment maps normalized [-1, 1] actions across the maximum distance from
# the default pose to either hard joint limit.
_MORPHOS_ACTION_SCALE = {
    "hip_pitch_l_joint": 3.1297900676727295,
    "hip_roll_l_joint": 2.617990016937256,
    "hip_yaw_l_joint": 4.537859916687012,
    "knee_pitch_l_joint": 2.0307300090789795,
    "ankle_pitch_l_joint": 0.9717299938201904,
    "ankle_roll_l_joint": 0.5235990285873413,
    "hip_pitch_r_joint": 3.1297900676727295,
    "hip_roll_r_joint": 2.617990016937256,
    "hip_yaw_r_joint": 4.537859916687012,
    "knee_pitch_r_joint": 2.0307300090789795,
    "ankle_pitch_r_joint": 0.9717299938201904,
    "ankle_roll_r_joint": 0.5235990285873413,
    "waist_yaw_joint": 3.2288599014282227,
    "waist_roll_joint": 0.43633198738098145,
    "waist_pitch_joint": 0.9599310159683228,
    "shoulder_pitch_l_joint": 2.967060089111328,
    "shoulder_roll_l_joint": 3.1033899784088135,
    "shoulder_yaw_l_joint": 2.967060089111328,
    "elbow_pitch_l_joint": 2.3179900646209717,
    "shoulder_pitch_r_joint": 2.967060089111328,
    "shoulder_roll_r_joint": 3.1033899784088135,
    "shoulder_yaw_r_joint": 2.967060089111328,
    "elbow_pitch_r_joint": 2.3179900646209717,
}


class DexEVTFlatEnvConfig(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Update the robot configuration
        self.scene.robot = DEX_EVT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos = replace(self.actions.joint_pos)
        self.actions.joint_pos.scale = D3_ACTION_SCALE
        self.actions.joint_pos.clip = None

        # Set the anchor body for motion commands
        self.commands.motion = replace(self.commands.motion)
        self.commands.motion.anchor_body = "pelvis"

        # Define the body names based on the MJCF model structure
        self.commands.motion.body_names = [
            "pelvis",
            "hip_pitch_l_link",
            "hip_roll_l_link",
            "hip_yaw_l_link",
            "knee_pitch_l_link",
            "ankle_pitch_l_link",
            "ankle_roll_l_link",
            "hip_pitch_r_link",
            "hip_roll_r_link",
            "hip_yaw_r_link",
            "knee_pitch_r_link",
            "ankle_pitch_r_link",
            "ankle_roll_r_link",
            "waist_yaw_link",
            "waist_roll_link",
            "waist_pitch_link",
            "shoulder_pitch_l_link",
            "shoulder_roll_l_link",
            "shoulder_yaw_l_link",
            "elbow_pitch_l_link",
            # "elbow_yaw_l_link",
            # "wrist_pitch_l_link",
            # "wrist_roll_l_link",
            "shoulder_pitch_r_link",
            "shoulder_roll_r_link",
            "shoulder_yaw_r_link",
            "elbow_pitch_r_link",
            # "elbow_yaw_r_link",
            # "wrist_pitch_r_link",
            # "wrist_roll_r_link"
        ]
        self._align_with_morphos_wbt()

    def _align_with_morphos_wbt(self):
        # Timing and motion/reference selection.
        self.decimation = 4
        self.episode_length_s = 10.0
        self.commands.motion.anchor_body = "waist_pitch_link"
        self.commands.motion.body_names = list(_MORPHOS_TRACKED_BODY_NAMES)
        self.commands.motion.adaptive_timestep_sampling = True
        self.commands.motion.adaptive_uniform_ratio = 0.1
        self.commands.motion.adaptive_alpha = 0.001
        self.commands.motion.adaptive_kernel_size = 1
        self.commands.motion.adaptive_kernel_lambda = 0.8

        # Match the direct WBT control surface and remove manager-only delay.
        robot_cfg = copy.deepcopy(self.scene.robot)
        for actuator_cfg in robot_cfg.actuators.values():
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 0
        self.scene.robot = robot_cfg
        self.actions.joint_pos = replace(
            self.actions.joint_pos,
            scale=dict(_MORPHOS_ACTION_SCALE),
            clip=(-1.0, 1.0),
        )

        # The direct WBT comparator has reset noise but no manager domain
        # randomization, interval pushes, or actuator-gain randomization.
        for name in (
            "randomize_joint_params",
            "physics_material",
            "add_joint_default_pos",
            "base_com",
            "randomize_rigid_body_mass_others",
            "push_robot",
            "randomize_actuator_gains",
        ):
            term = getattr(self.events, name)
            if not isinstance(term, EventTermCfg):
                raise TypeError(
                    f"Expected event term '{name}', got {type(term).__name__}."
                )
            setattr(self.events, name, replace(term, enable=False))

        # Reward names differ between the two implementations; functions,
        # weights, sigmas, dt scaling, and the joint-limit cap are aligned here.
        self.rewards.motion_global_anchor_pos = replace(
            self.rewards.motion_global_anchor_pos, weight=2.0
        )
        self.rewards.motion_global_anchor_ori = replace(
            self.rewards.motion_global_anchor_ori, weight=0.5
        )
        self.rewards.motion_body_pos = replace(self.rewards.motion_body_pos, weight=2.0)
        self.rewards.motion_body_ori = replace(self.rewards.motion_body_ori, weight=1.0)
        self.rewards.motion_body_lin_vel = replace(
            self.rewards.motion_body_lin_vel, weight=1.0
        )
        self.rewards.motion_body_ang_vel = replace(
            self.rewards.motion_body_ang_vel, weight=1.0
        )
        self.rewards.action_rate_l2 = replace(self.rewards.action_rate_l2, weight=-1.0)
        self.rewards.joint_torque_l2 = replace(
            self.rewards.joint_torque_l2, enable=False
        )
        self.rewards.joint_vel_limit = replace(
            self.rewards.joint_vel_limit, enable=False
        )
        self.rewards.joint_limit = RewardTermCfg(
            func=mdp.joint_position_limit_penalty,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "soft_ratio": 0.9,
                "cap": 5.0,
            },
        )
        self.rewards.undesired_contacts = RewardTermCfg(
            func=mdp.direct_wbt_undesired_floor_contacts,
            weight=-0.1,
        )

        # Match direct-WBT failure thresholds and its extra numerical guards.
        self.terminations.anchor_pos = replace(
            self.terminations.anchor_pos,
            params={"command_name": "motion", "threshold": 0.5},
        )
        self.terminations.ee_body_pos = replace(
            self.terminations.ee_body_pos,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": [
                    "ankle_roll_l_link",
                    "ankle_roll_r_link",
                    "elbow_pitch_l_link",
                    "elbow_pitch_r_link",
                ],
            },
        )
        self.terminations.bad_dof_pos = TerminationTermCfg(
            func=mdp.bad_dof_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "threshold": 0.5,
            },
        )
        self.terminations.bad_dof_vel = TerminationTermCfg(
            func=mdp.bad_dof_vel_and_update_motion_sampling,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "threshold": 100.0,
                "command_name": "motion",
                "failure_term_names": (
                    "anchor_pos",
                    "anchor_ori",
                    "ee_body_pos",
                    "bad_dof_pos",
                ),
            },
        )

    def for_play(self) -> "DexEVTFlatEnvConfig":
        """Return the deterministic playback variant used by direct Dex-EVT WBT."""
        cfg = copy.deepcopy(self)
        cfg.commands.motion = replace(
            cfg.commands.motion,
            pose_range={
                key: (0.0, 0.0)
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            },
            velocity_range={
                key: (0.0, 0.0)
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            },
            joint_position_range=(0.0, 0.0),
            start_at_timestep_zero_prob=1.0,
            hold_at_clip_end=True,
            adaptive_timestep_sampling=False,
        )
        cfg.terminations.time_out = replace(
            cfg.terminations.time_out,
            enable=False,
        )
        return cfg


@dataclass
class DexEVTFlatWoStateEstimationEnvCfg(DexEVTFlatEnvConfig):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = replace(self.observations.policy.motion_anchor_pos_b, enable=False)
        self.observations.policy.base_lin_vel = replace(self.observations.policy.base_lin_vel, enable=False)


@dataclass
class DexEVTFlatLowFreqEnvCfg(DexEVTFlatEnvConfig):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
