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

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import whole_body_tracking  # noqa: F401 - register xMimic tasks

from motrix_envs.base import ABEnv
from motrix_envs.np.env import NpEnv
from motrix_envs.np.renderer import NpRenderer
from motrix_envs.torch.manager_based_env import ManagerBasedTorchEnv
from whole_body_tracking.tasks.tracking.config.dex_evt import make_env_config

DEFAULT_MOTION_FILE = Path(__file__).resolve().parents[2] / "motion_example" / "dance1_easy.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="View a MotrixSim environment with random actions."
    )
    parser.add_argument(
        "--env",
        default="Tracking-Flat-DexEVT-v0",
        help="Environment registry name.",
    )
    parser.add_argument("--sim-backend", default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--motion-file",
        type=Path,
        help="xMimic .npz tracking motion (default: motion_example/dance1_easy.npz).",
    )
    return parser


def _sample_random_action_np(env: ABEnv) -> np.ndarray:
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Box):
        size = (env.num_envs, *action_space.shape)
        low = action_space.low
        high = action_space.high
        low = np.where(np.isneginf(low), -1e6, low)
        high = np.where(np.isposinf(high), 1e6, high)
        return np.random.uniform(low=low, high=high, size=size).astype(action_space.dtype)
    else:
        raise NotImplementedError("Only Box action space is supported")


def _run_np(env: NpEnv):
    renderer = NpRenderer(env)
    env_dt = env.step_dt
    while True:
        t0 = time.monotonic()
        actions = _sample_random_action_np(env)
        env.step(actions)
        renderer.render()
        real_dt = time.monotonic() - t0
        sleep_dt = env_dt - real_dt
        if sleep_dt > 0:
            time.sleep(sleep_dt)


def run(args: argparse.Namespace) -> None:
    env_name = args.env
    env_cfg = make_env_config(env_name)
    motion_command = getattr(getattr(env_cfg, "commands", None), "motion", None)
    motion_file = args.motion_file
    if motion_file is None and motion_command is not None and not motion_command.motion_file:
        motion_file = DEFAULT_MOTION_FILE
    if motion_file is not None:
        motion_file = motion_file.expanduser().resolve()
        if not motion_file.is_file():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        if motion_command is None:
            raise ValueError(f"Environment '{env_name}' does not accept a motion file.")
        motion_command.motion_file = str(motion_file)
    if args.sim_backend not in (None, "torch"):
        raise ValueError("xMimic tracking tasks currently require sim_backend='torch'.")
    env = ManagerBasedTorchEnv(env_cfg, num_envs=args.num_envs)
    try:
        _run_np(env)
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


console_main = main


if __name__ == "__main__":
    main()
