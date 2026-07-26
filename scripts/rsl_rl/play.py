"""Play RSL-RL PPO and FastSAC PyTorch checkpoints in MotrixSim."""

from __future__ import annotations

import argparse
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import whole_body_tracking  # noqa: F401 - register xMimic tasks
from rsl_rl.runners import OnPolicyRunner

from motrix_envs.np.renderer import NpRenderer
from motrix_envs.torch.manager_based_env import ManagerBasedTorchEnv
from motrix_rl import registry as rl_registry
from motrix_rl.rslrl.torch import wrap_env
from motrix_rl.runs import find_metadata_for_policy
from whole_body_tracking.tasks.tracking.config.dex_evt import make_env_config
from whole_body_tracking.utils.motrix import TorchManagerNpCompat


SUPPORTED_ALGOS = ("rslrl.ppo", "fastsac.async", "fastsac.sync")
DEFAULT_ENV = "Tracking-Flat-DexEVT-v0"
DEFAULT_MOTION_FILE = (
    Path(__file__).resolve().parents[2] / "motion_example" / "dance1_easy.npz"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="PyTorch .pt checkpoint path"
    )
    parser.add_argument(
        "--algo",
        choices=SUPPORTED_ALGOS,
        help="Policy algorithm (default: infer from run metadata, else rslrl.ppo)",
    )
    parser.add_argument(
        "--env",
        help=f"Environment name (default: infer from run metadata, else {DEFAULT_ENV})",
    )
    parser.add_argument(
        "--motion-file",
        "--motion_file",
        dest="motion_file",
        type=Path,
        help="Tracking motion override; needed when replaying a different motion",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        help="Saved env.pkl override (PPO default: RUN/params/env.pkl)",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        help="Saved agent.pkl override (PPO default: RUN/params/agent.pkl)",
    )
    parser.add_argument("--sim-backend", default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", help="Inference device override, e.g. cpu or cuda")
    parser.add_argument("--activation", choices=("elu", "relu", "tanh"))
    parser.add_argument(
        "--action-clip", type=float, help="Clamp policy actions to this magnitude"
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not pace simulation to control dt",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without opening a viewer"
    )
    parser.add_argument("--steps", type=int, help="Stop after this many control steps")
    parser.add_argument(
        "--report", action="store_true", help="Print rollout and tracking statistics"
    )
    return parser


def _load_pickle(path: Path, description: str):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    with path.open("rb") as stream:
        return pickle.load(stream)


def _saved_config_path(run_dir: Path, override: Path | None, name: str) -> Path:
    return override if override is not None else run_dir / "params" / f"{name}.pkl"


def _resolve_run(
    checkpoint: Path,
) -> tuple[Path, object | None]:
    found = find_metadata_for_policy(checkpoint)
    if found is not None:
        return found
    return checkpoint.parent, None


def _resolve_algo_and_env(
    args: argparse.Namespace, metadata: object | None
) -> tuple[str, str]:
    metadata_algo = None
    metadata_env = None
    if metadata is not None:
        metadata_algo = f"{metadata.rllib}.{metadata.algo}"
        metadata_env = metadata.env_name
    algo = args.algo or metadata_algo or "rslrl.ppo"
    env_name = args.env or metadata_env or DEFAULT_ENV
    if algo not in SUPPORTED_ALGOS:
        choices = ", ".join(SUPPORTED_ALGOS)
        raise ValueError(f"Unsupported algorithm '{algo}'; choose one of: {choices}")
    return algo, env_name


def _set_motion_file(env_cfg, motion_file: Path | None, *, use_default: bool) -> None:
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_file is None and motion_cfg is not None and not motion_cfg.motion_file:
        if use_default:
            motion_file = DEFAULT_MOTION_FILE
            print(
                "FastSAC run has no saved env.pkl; using default motion "
                f"{motion_file}. Pass --motion-file to override it."
            )
        else:
            return
    if motion_file is not None:
        motion_file = motion_file.expanduser().resolve()
        if not motion_file.is_file():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        if motion_cfg is None:
            raise ValueError("The selected environment does not accept a motion file.")
        motion_cfg.motion_file = str(motion_file)
    elif motion_cfg is not None and motion_cfg.motion_file:
        saved_motion = Path(motion_cfg.motion_file).expanduser()
        if not saved_motion.is_file():
            raise FileNotFoundError(
                f"Saved motion file not found: {saved_motion}. "
                "Pass --motion-file with an existing .npz file."
            )


def _load_env_cfg(
    args: argparse.Namespace,
    run_dir: Path,
    env_name: str,
    *,
    require_saved: bool,
):
    path = _saved_config_path(run_dir, args.env_config, "env")
    if path.expanduser().is_file() or args.env_config is not None:
        env_cfg = _load_pickle(path, "Environment config")
        reconstructed = False
    elif require_saved:
        raise FileNotFoundError(f"Environment config not found: {path.resolve()}")
    else:
        env_cfg = make_env_config(env_name)
        reconstructed = True
    _set_motion_file(env_cfg, args.motion_file, use_default=reconstructed)
    return env_cfg


class _RolloutReport:
    def __init__(self, num_envs: int):
        self.episode_returns = torch.zeros(num_envs)
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long)
        self.completed_returns: list[float] = []
        self.completed_steps: list[int] = []
        self.termination_counts: dict[str, int] = defaultdict(int)
        self.metric_samples: dict[str, list[np.ndarray]] = defaultdict(list)
        self.action_samples: list[np.ndarray] = []

    def update(self, env, state, actions: torch.Tensor) -> None:
        rewards = state.reward.detach().cpu()
        self.episode_returns += rewards
        self.episode_steps += 1
        self.action_samples.append(actions.detach().abs().cpu().numpy().reshape(-1))

        done = state.done.detach().cpu()
        surviving = ~done
        for group_name, group in env.command_manager.metrics.items():
            for metric_name, values in group.items():
                key = f"{group_name}/{metric_name}"
                samples = values.detach().cpu()[surviving]
                if samples.numel():
                    self.metric_samples[key].append(samples.numpy().reshape(-1))

        done_ids = torch.nonzero(done, as_tuple=False).flatten()
        for env_id in done_ids.tolist():
            self.completed_returns.append(float(self.episode_returns[env_id]))
            self.completed_steps.append(int(self.episode_steps[env_id]))
        if done_ids.numel():
            self.episode_returns[done_ids] = 0.0
            self.episode_steps[done_ids] = 0

        for name, values in env.termination_manager.buffers.items():
            self.termination_counts[name] += int(
                torch.count_nonzero(values & state.done).item()
            )

    def print(self, ctrl_dt: float) -> None:
        print("\nRollout report")
        print(f"  completed episodes: {len(self.completed_steps)}")
        if self.completed_steps:
            steps = np.asarray(self.completed_steps)
            returns = np.asarray(self.completed_returns)
            print(
                f"  episode duration: mean={steps.mean() * ctrl_dt:.3f}s, "
                f"min={steps.min() * ctrl_dt:.3f}s, max={steps.max() * ctrl_dt:.3f}s"
            )
            print(
                f"  episode return: mean={returns.mean():.4f}, "
                f"min={returns.min():.4f}, max={returns.max():.4f}"
            )
        if self.termination_counts:
            summary = ", ".join(
                f"{name}={count}" for name, count in self.termination_counts.items()
            )
            print(f"  terminations: {summary}")
        if self.action_samples:
            actions = np.concatenate(self.action_samples)
            print(
                f"  |action|: mean={actions.mean():.4f}, "
                f"p95={np.percentile(actions, 95):.4f}, max={actions.max():.4f}"
            )
        for name, samples in sorted(self.metric_samples.items()):
            values = np.concatenate(samples)
            print(
                f"  {name}: mean={values.mean():.4f}, "
                f"p95={np.percentile(values, 95):.4f}, max={values.max():.4f}"
            )


def _renderer_open(renderer) -> bool:
    return renderer is None or not renderer._render.is_closed


def _pace(ctrl_dt: float, start: float, no_realtime: bool) -> None:
    if no_realtime:
        return
    remaining = ctrl_dt - (time.monotonic() - start)
    if remaining > 0:
        time.sleep(remaining)


def _run_rslrl(
    args: argparse.Namespace,
    checkpoint: Path,
    run_dir: Path,
    env_name: str,
) -> None:
    env_cfg = _load_env_cfg(args, run_dir, env_name, require_saved=True)
    agent_cfg = _load_pickle(
        _saved_config_path(run_dir, args.agent_config, "agent"), "Agent config"
    )
    if args.activation is not None:
        agent_cfg.actor.activation = args.activation
        agent_cfg.critic.activation = args.activation
    if args.device is not None:
        agent_cfg.device = args.device
    device = torch.device(agent_cfg.device)
    env = ManagerBasedTorchEnv(env_cfg, num_envs=args.num_envs)
    vec_env = wrap_env(TorchManagerNpCompat(env), device)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(
        str(checkpoint),
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    renderer = None if args.headless else NpRenderer(env)
    observations = vec_env.get_observations()
    expected_obs = int(observations["policy"].shape[-1])
    expected_actions = int(env.action_manager.action_dim)
    ctrl_dt = env.step_dt
    print(
        f"Playing {checkpoint}: algo=rslrl.ppo, env={env_name}, "
        f"num_envs={args.num_envs}, observations={expected_obs}, "
        f"actions={expected_actions}; Ctrl+C to stop."
    )

    report = _RolloutReport(args.num_envs) if args.report else None
    step = 0
    try:
        while _renderer_open(renderer) and (args.steps is None or step < args.steps):
            start = time.monotonic()
            with torch.inference_mode():
                actions = policy(observations)
                if args.action_clip is not None:
                    actions = torch.clamp(actions, -args.action_clip, args.action_clip)
            observations, _, _, _ = vec_env.step(actions)
            if report is not None:
                report.update(env, env._state, actions)
            if renderer is not None:
                renderer.render()
            step += 1
            _pace(ctrl_dt, start, args.no_realtime)
    except KeyboardInterrupt:
        pass
    finally:
        if report is not None:
            report.print(ctrl_dt)
        env.close()


def _plain_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if any(key.startswith(prefix) for key in state_dict):
        return {key.removeprefix(prefix): value for key, value in state_dict.items()}
    return state_dict


def _fastsac_agent_cfg(args: argparse.Namespace, env_name: str):
    if args.agent_config is not None:
        saved_cfg = _load_pickle(args.agent_config, "Agent config")
        if hasattr(saved_cfg, "runner"):
            return saved_cfg.runner.agent, getattr(saved_cfg, "device", None)
        return saved_cfg, None

    # Async and sync FastSAC checkpoints share the same actor/state format. The
    # xMimic task currently registers its common FastSAC defaults under async.
    rl_cfg = rl_registry.default_rl_cfg(
        env_name, "fastsac", train_backend="torch", algo="async"
    )
    return rl_cfg.runner.agent, rl_cfg.device


def _run_fastsac(
    args: argparse.Namespace,
    checkpoint: Path,
    run_dir: Path,
    algo: str,
    env_name: str,
) -> None:
    from motrix_rl.fastsac.buffer import EmpiricalNormalization
    from motrix_rl.fastsac.networks import Actor

    env_cfg = _load_env_cfg(args, run_dir, env_name, require_saved=False)
    agent_cfg, configured_device = _fastsac_agent_cfg(args, env_name)
    requested_device = args.device or configured_device
    if requested_device is None:
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU for FastSAC inference.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    env = ManagerBasedTorchEnv(env_cfg, num_envs=args.num_envs)
    state = env.init_state()
    observations = state.obs.policy.to(device=device, dtype=torch.float32)
    obs_dim = int(observations.shape[-1])
    act_dim = int(env.action_manager.action_dim)
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)
    actor = Actor(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=agent_cfg.actor_hidden_dim,
        log_std_max=agent_cfg.log_std_max,
        log_std_min=agent_cfg.log_std_min,
        use_tanh=agent_cfg.use_tanh,
        use_layer_norm=agent_cfg.use_layer_norm,
        action_scale=(action_high - action_low) / 2.0,
        action_bias=(action_high + action_low) / 2.0,
        device=device,
    )
    normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(_plain_state_dict(checkpoint_data["actor"]))
    normalizer.load_state_dict(checkpoint_data["obs_normalizer"])
    actor.eval()
    normalizer.eval()

    renderer = None if args.headless else NpRenderer(env)
    ctrl_dt = env.step_dt
    print(
        f"Playing {checkpoint}: algo={algo}, env={env_name}, "
        f"num_envs={args.num_envs}, observations={obs_dim}, actions={act_dim}, "
        f"device={device}; Ctrl+C to stop."
    )
    report = _RolloutReport(args.num_envs) if args.report else None
    step = 0
    try:
        while _renderer_open(renderer) and (args.steps is None or step < args.steps):
            start = time.monotonic()
            with torch.inference_mode():
                actions = actor.explore(
                    normalizer(observations, update=False), deterministic=True
                )
                if args.action_clip is not None:
                    actions = torch.clamp(actions, -args.action_clip, args.action_clip)
            state = env.step(actions.to(device=env.device, dtype=torch.float32))
            observations = state.obs.policy.to(device=device, dtype=torch.float32)
            if report is not None:
                report.update(env, state, actions)
            if renderer is not None:
                renderer.render()
            step += 1
            _pace(ctrl_dt, start, args.no_realtime)
    except KeyboardInterrupt:
        pass
    finally:
        if report is not None:
            report.print(ctrl_dt)
        env.close()


def run(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.suffix != ".pt":
        raise ValueError(
            f"play.py expects a .pt checkpoint, got {checkpoint}. "
            "Use sim2sim_play.py for ONNX."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    if args.steps is not None and args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.sim_backend not in (None, "torch"):
        raise ValueError("xMimic checkpoints require the MotrixLab Torch backend.")

    run_dir, metadata = _resolve_run(checkpoint)
    algo, env_name = _resolve_algo_and_env(args, metadata)
    if algo == "rslrl.ppo":
        _run_rslrl(args, checkpoint, run_dir, env_name)
    else:
        _run_fastsac(args, checkpoint, run_dir, algo, env_name)


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
