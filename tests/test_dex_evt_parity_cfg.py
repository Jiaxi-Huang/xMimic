# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pytest

from whole_body_tracking.tasks.tracking.config.dex_evt import make_env_config


DIRECT_PARITY_EVENTS = (
    "randomize_joint_params",
    "physics_material",
    "add_joint_default_pos",
    "base_com",
    "randomize_rigid_body_mass_others",
    "push_robot",
    "randomize_actuator_gains",
)


@pytest.mark.parametrize(
    ("task_name", "step_dt"),
    (
        ("Tracking-Flat-DexEVT-v0", 0.02),
        ("Tracking-Flat-DexEVT-Wo-State-Estimation-v0", 0.02),
        ("Tracking-Flat-DexEVT-Low-Freq-v0", 0.04),
    ),
)
def test_all_dex_evt_tasks_apply_direct_parity(task_name: str, step_dt: float):
    cfg = make_env_config(task_name)

    assert cfg.step_dt == pytest.approx(step_dt)
    assert cfg.commands.motion.anchor_body == "waist_pitch_link"
    assert len(cfg.commands.motion.body_names) == 12
    assert cfg.commands.motion.adaptive_timestep_sampling
    assert cfg.actions.joint_pos.clip == (-1.0, 1.0)
    assert max(cfg.actions.joint_pos.scale.values()) > 4.0
    assert all(
        actuator.min_delay == actuator.max_delay == 0
        for actuator in cfg.scene.robot.actuators.values()
    )
    assert all(not getattr(cfg.events, name).enable for name in DIRECT_PARITY_EVENTS)
