import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gymn
import numpy as np


# D4RL expert demonstrations for Gym-MuJoCo tasks.
D4RL_EXPERT_DATASETS = {
    "Hopper-v4": "hopper-expert-v2",
    "Walker2d-v4": "walker2d-expert-v2",
    "HalfCheetah-v4": "halfcheetah-expert-v2",
    "Ant-v4": "ant-expert-v2",
}


@dataclass
class DemoBatch:
    online_env_id: str
    dataset_id: str
    observations: np.ndarray
    actions: np.ndarray
    online_obs_dim: int
    online_act_dim: int
    dataset_obs_dim: int
    dataset_act_dim: int
    shape_match: bool


def _make_online_env(online_env_id: str):
    env = gymn.make(online_env_id)
    if not isinstance(env.observation_space, gymn.spaces.Box):
        raise TypeError(f"{online_env_id}: expected Box observation space.")
    if not isinstance(env.action_space, gymn.spaces.Box):
        raise TypeError(f"{online_env_id}: expected Box action space.")
    return env


def load_d4rl_expert_data(online_env_id: str, strict_match: bool = True) -> DemoBatch:
    """
    Load D4RL expert demonstration data and sanity-check shape compatibility
    with the online Gymnasium environment.
    """
    if online_env_id not in D4RL_EXPERT_DATASETS:
        supported = ", ".join(sorted(D4RL_EXPERT_DATASETS))
        raise KeyError(f"Unsupported env_id {online_env_id!r}. Supported: {supported}")

    dataset_id = D4RL_EXPERT_DATASETS[online_env_id]
    obs = None
    acts = None

    # Preferred path: lightweight downloader/loader.
    try:
        from just_d4rl import d4rl_offline_dataset

        dataset = d4rl_offline_dataset(dataset_id)
        obs = dataset["observations"].astype(np.float32)
        acts = dataset["actions"].astype(np.float32)
    except ImportError:
        pass

    # Fallback path: legacy d4rl + gym loader.
    if obs is None or acts is None:
        try:
            import d4rl  # noqa: F401  (registers datasets)
            import gym as old_gym
        except ImportError as exc:
            raise ImportError(
                "Could not load D4RL expert data. Install either `just-d4rl` "
                "or (`d4rl` + legacy `gym`) before running demos.py."
            ) from exc

        d4rl_env = old_gym.make(dataset_id)
        dataset = d4rl_env.get_dataset()
        obs = dataset["observations"].astype(np.float32)
        acts = dataset["actions"].astype(np.float32)
        d4rl_env.close()

    online_env = _make_online_env(online_env_id)
    online_obs_dim = int(np.prod(online_env.observation_space.shape))
    online_act_dim = int(np.prod(online_env.action_space.shape))
    online_env.close()

    if obs.ndim != 2 or acts.ndim != 2:
        raise ValueError(
            f"{dataset_id}: expected 2D arrays, got observations {obs.shape}, actions {acts.shape}"
        )
    dataset_obs_dim = int(obs.shape[-1])
    dataset_act_dim = int(acts.shape[-1])
    shape_match = (dataset_obs_dim == online_obs_dim) and (dataset_act_dim == online_act_dim)
    if strict_match and not shape_match:
        raise ValueError(
            f"Shape mismatch for {online_env_id}: dataset {dataset_id} has "
            f"obs={dataset_obs_dim}, act={dataset_act_dim}; online env expects "
            f"obs={online_obs_dim}, act={online_act_dim}"
        )

    return DemoBatch(
        online_env_id=online_env_id,
        dataset_id=dataset_id,
        observations=obs,
        actions=acts,
        online_obs_dim=online_obs_dim,
        online_act_dim=online_act_dim,
        dataset_obs_dim=dataset_obs_dim,
        dataset_act_dim=dataset_act_dim,
        shape_match=shape_match,
    )


def save_demo_batch(batch: DemoBatch, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env_slug = batch.online_env_id.replace("-v", "_v").replace("-", "_").lower()
    out_path = out_dir / f"{env_slug}__{batch.dataset_id}.npz"
    np.savez_compressed(
        out_path,
        online_env_id=batch.online_env_id,
        dataset_id=batch.dataset_id,
        online_obs_dim=batch.online_obs_dim,
        online_act_dim=batch.online_act_dim,
        dataset_obs_dim=batch.dataset_obs_dim,
        dataset_act_dim=batch.dataset_act_dim,
        shape_match=batch.shape_match,
        observations=batch.observations,
        actions=batch.actions,
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch/cache D4RL expert demonstrations.")
    parser.add_argument(
        "--env-id",
        action="append",
        dest="env_ids",
        choices=sorted(D4RL_EXPERT_DATASETS.keys()),
        help="Gymnasium online env id(s). May be provided multiple times.",
    )
    parser.add_argument(
        "--include-ant",
        action="store_true",
        help="Include Ant-v4. By default, Ant is excluded due to potential version-space mismatch risk.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/datasets/d4rl_expert"),
        help="Directory for cached .npz expert demonstration files.",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Cache dataset even if obs/action dims do not match online env (useful for Ant diagnostics).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_ids:
        env_ids = list(dict.fromkeys(args.env_ids))
    else:
        env_ids = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
        if args.include_ant:
            env_ids.append("Ant-v4")

    for env_id in env_ids:
        batch = load_d4rl_expert_data(env_id, strict_match=not args.allow_mismatch)
        out_path = save_demo_batch(batch, args.out_dir)
        status = "ok" if batch.shape_match else "warn:mismatch"
        print(
            f"[{status}] {env_id} <- {batch.dataset_id} "
            f"obs={batch.observations.shape} act={batch.actions.shape} "
            f"online_obs={batch.online_obs_dim} online_act={batch.online_act_dim} "
            f"cached={out_path}"
        )


if __name__ == "__main__":
    main()
