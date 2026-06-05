import argparse
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch

EXP0_DIR = Path(__file__).resolve().parent.parent / "exp0"
sys.path.insert(0, str(EXP0_DIR))

from demos import D4RL_EXPERT_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a starter policy can transfer across MuJoCo envs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--envs", nargs="+", default=["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4"])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def env_shape(env_id: str) -> tuple[int, int]:
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    env.close()
    return obs_dim, action_dim


def checkpoint_shape(checkpoint: dict) -> tuple[str | None, int | None, int | None]:
    args = checkpoint.get("args") or {}
    source_env = checkpoint.get("env_id") or args.get("env_id")
    if source_env in D4RL_EXPERT_DATASETS:
        obs_dim, action_dim = env_shape(source_env)
        return source_env, obs_dim, action_dim
    return source_env, None, None


def main() -> None:
    args = parse_args()
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    source_env, source_obs_dim, source_action_dim = checkpoint_shape(checkpoint)
    rows = []
    for env_id in args.envs:
        obs_dim, action_dim = env_shape(env_id)
        rows.append(
            {
                "target_env": env_id,
                "target_obs_dim": obs_dim,
                "target_action_dim": action_dim,
                "compatible": obs_dim == source_obs_dim and action_dim == source_action_dim,
            }
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "source_env": source_env,
        "source_obs_dim": source_obs_dim,
        "source_action_dim": source_action_dim,
        "targets": rows,
        "recommendation": "Use per-target distillation unless obs/action dimensions match exactly.",
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
