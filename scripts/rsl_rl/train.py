from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import torch
import whole_body_tracking  # noqa: F401 - register xMimic tasks
from rsl_rl.runners import OnPolicyRunner
from motrix_envs.torch.manager_based_env import ManagerBasedTorchEnv
from motrix_rl import registry as rl_registry
from whole_body_tracking.tasks.tracking.config.dex_evt import make_env_config
from whole_body_tracking.utils.motrix import RslRlTorchManagerWrap, TorchManagerNpCompat

import motrix_envs  # noqa: F401 - load the simulation package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an RL agent on MotrixSim.")
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--video_interval", type=int, default=2000)
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=int,
        default=None,
        help="Override the number of environments; by default use the RL config value.",
    )
    parser.add_argument("--torch_threads", type=int, default=None)
    parser.add_argument("--task", type=str, default="Tracking-Flat-DexEVT-v0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--motion_file", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument(
        "--algo",
        type=str,
        default="rslrl.ppo",
        help="RL algorithm in <rllib>.<algo> form: rslrl.ppo, fastsac.async, fastsac.sync, ...",
    )
    parser.add_argument("--logging_interval", type=int, default=None, help="FastSAC: logging interval (iters).")
    parser.add_argument("--learning_starts", type=int, default=None, help="FastSAC: num steps before first update.")
    return parser


def _resolve_checkpoint(log_root: Path, run_pattern: str, checkpoint_pattern: str) -> Path:
    runs = sorted(path for path in log_root.glob(run_pattern) if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run matching '{run_pattern}' under {log_root}.")
    checkpoints = sorted(runs[-1].glob(checkpoint_pattern))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint matching '{checkpoint_pattern}' under {runs[-1]}."
        )
    return checkpoints[-1]


def _dump_configs(log_dir: Path, env_cfg, agent_cfg) -> None:
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("env", env_cfg), ("agent", agent_cfg)):
        pickle_path = params_dir / f"{name}.pkl"
        try:
            with pickle_path.open("wb") as stream:
                pickle.dump(value, stream)
        except (pickle.PicklingError, AttributeError, TypeError) as error:
            pickle_path.unlink(missing_ok=True)
            warnings.warn(f"Skipping non-picklable {name} config: {error}", stacklevel=2)
        serializable = asdict(value) if is_dataclass(value) else value
        (params_dir / f"{name}.json").write_text(
            json.dumps(serializable, indent=2, default=str), encoding="utf-8"
        )


def _check_cpu_batch_size(num_envs: int, steps_per_env: int) -> int:
    transitions_per_iteration = num_envs * steps_per_env
    if num_envs > 512:
        warnings.warn(
            f"Using {num_envs} environments on the CPU-only MotrixSim backend creates "
            f"{transitions_per_iteration:,} transitions per PPO iteration and can make each "
            "iteration take minutes. Start with 16 environments and benchmark up to 256.",
            stacklevel=2,
        )
    return transitions_per_iteration


def _fastsac_cfg_override(args: argparse.Namespace) -> dict:
    """Build only the FastSAC overrides explicitly requested on the CLI."""
    cfg_override: dict = {}
    if args.num_envs is not None:
        cfg_override["num_envs"] = args.num_envs
    if args.seed is not None:
        cfg_override["runner.seed"] = args.seed
    if args.logging_interval is not None:
        cfg_override["runner.trainer.logging_interval"] = args.logging_interval
    if args.learning_starts is not None:
        cfg_override["runner.agent.learning_starts"] = args.learning_starts
    return cfg_override


def train_fastsac(args: argparse.Namespace) -> Path:
    """Train via motrix_rl runner (fastsac.async / fastsac.sync / skrl.* / ...).

    Shares the same CLI flags as the RSL-RL path. The env is built inside
    motrix_rl from the registered task spec; we only forward the overrides.
    """
    from motrix_rl import runner as motrix_runner
    from motrix_rl.method import RlMethod

    motion_file = Path(args.motion_file).expanduser().resolve()
    if not motion_file.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    parts = args.algo.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"--algo must be '<rllib>.<algo>', got '{args.algo}'")
    method = RlMethod(rllib=parts[0], algo=parts[1])

    cfg_override = _fastsac_cfg_override(args)
    # run_name is not a FastSacRunnerCfg field; motrix_rl derives the run dir
    # from experiment_name + timestamp automatically.

    # Resolve env into a spawn-safe spec (env_cls + env_cfg), like RSL-RL
    # constructs its env config in the caller.
    from motrix_envs import registry as env_registry

    from whole_body_tracking.tasks.tracking.config.dex_evt import make_env_config

    env_cfg = make_env_config(args.task)
    env_cfg.commands.motion.motion_file = str(motion_file)
    env_spec = env_registry.resolve(args.task, env_cfg=env_cfg)

    num_envs_source = args.num_envs if args.num_envs is not None else "RL config"
    print(
        f"Training setup: algo={args.algo}, task={args.task}, "
        f"num_envs={num_envs_source}, motion={motion_file.name}"
    )
    resume_from = None
    if args.resume:
        # FastSAC resume expects a checkpoint path; --load_run/--load_checkpoint
        # are RSL-RL-style patterns that don't apply here. User should pass the
        # checkpoint path directly via --load_checkpoint for now.
        if args.load_checkpoint:
            resume_from = str(Path(args.load_checkpoint).resolve())
        else:
            raise ValueError("--resume for FastSAC requires --load_checkpoint <path>")
    motrix_runner.train(
        motrix_runner.TrainRequest(
            env_name=args.task,
            method=method,
            requested_train_backend="torch",
            cfg_override=cfg_override,
            env_spec=env_spec,
            resume_from=resume_from,
        )
    )
    return Path("logs")


def train(args: argparse.Namespace, *, runner_cls=OnPolicyRunner):
    if args.video:
        raise NotImplementedError("Video recording is not yet available in the MotrixSim training launcher.")

    env_cfg = make_env_config(args.task)
    rl_cfg = rl_registry.default_rl_cfg(args.task, "rslrl", train_backend="torch", algo="ppo")
    agent_cfg = rl_cfg.runner
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name
    if args.device is not None:
        agent_cfg.device = args.device
    train_device = torch.device(agent_cfg.device)

    motion_file = Path(args.motion_file).expanduser().resolve()
    if not motion_file.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    env_cfg.commands.motion.motion_file = str(motion_file)
    env_cfg.seed = agent_cfg.seed

    num_envs = args.num_envs if args.num_envs is not None else rl_cfg.num_envs
    transitions_per_iteration = _check_cpu_batch_size(
        num_envs, agent_cfg.num_steps_per_env
    )
    print(
        f"Training setup: num_envs={num_envs}, steps_per_env={agent_cfg.num_steps_per_env}, "
        f"transitions_per_iteration={transitions_per_iteration:,}, device={agent_cfg.device}"
    )
    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = (log_root / run_name).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(agent_cfg.seed)
    env = ManagerBasedTorchEnv(env_cfg, num_envs=num_envs)
    vec_env = RslRlTorchManagerWrap(TorchManagerNpCompat(env), train_device)
    runner = runner_cls(vec_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=train_device)
    runner.add_git_repo_to_log(__file__)

    if args.resume:
        checkpoint = _resolve_checkpoint(
            log_root.resolve(),
            args.load_run or ".*",
            args.load_checkpoint or "model_.*.pt",
        )
        runner.load(str(checkpoint))

    _dump_configs(log_dir, env_cfg, agent_cfg)
    try:
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        env.close()
    return log_dir


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.algo != "rslrl.ppo":
        return train_fastsac(args)
    return train(args)


if __name__ == "__main__":
    main()
