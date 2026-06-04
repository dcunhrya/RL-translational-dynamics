import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

EXP0_DIR = Path(__file__).resolve().parent.parent / "exp0"
sys.path.insert(0, str(EXP0_DIR))

from demos import D4RL_EXPERT_DATASETS
from train_sac import Actor, SoftQNetwork, evaluate, grad_norm, layer_init, seed_everything, write_metric

from diagnostics import require_known_phase


@dataclass
class Args:
    env_id: str = "Hopper-v4"
    seed: int = 0
    total_updates: int = 100_000
    batch_size: int = 256
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    temperature: float = 3.0
    max_weight: float = 100.0
    max_grad_norm: float = 100.0
    eval_interval: int = 5_000
    num_eval_episodes: int = 5
    save_interval: int = 25_000
    max_dataset_samples: int = 0
    save_dir: str = "results/raw/abhinav_task/iql_pretrain"
    track: bool = False
    wandb_project: str = "rl-translational-dynamics"
    wandb_entity: str | None = None
    wandb_group: str = "abhinav_iql_pretrain"
    cuda: bool = True


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    for field_name, field_def in Args.__dataclass_fields__.items():
        default = field_def.default
        arg_name = f"--{field_name.replace('_', '-')}"
        if isinstance(default, bool):
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(arg_name, type=type(default) if default is not None else str, default=default)
    return Args(**vars(parser.parse_args()))


def load_dataset(env_id: str) -> tuple[str, dict[str, np.ndarray]]:
    dataset_id = D4RL_EXPERT_DATASETS[env_id]
    try:
        from just_d4rl import d4rl_offline_dataset

        raw = d4rl_offline_dataset(dataset_id)
    except ImportError:
        import d4rl  # noqa: F401
        import gym as old_gym

        d4rl_env = old_gym.make(dataset_id)
        raw = d4rl_env.get_dataset()
        d4rl_env.close()

    observations = raw["observations"].astype(np.float32)
    actions = raw["actions"].astype(np.float32)
    rewards = raw["rewards"].astype(np.float32).reshape(-1, 1)
    if "next_observations" in raw:
        next_observations = raw["next_observations"].astype(np.float32)
    else:
        next_observations = np.concatenate([observations[1:], observations[-1:]], axis=0).astype(np.float32)
    terminals = raw.get("terminals", raw.get("dones", np.zeros(len(observations), dtype=np.float32))).astype(np.float32)
    timeouts = raw.get("timeouts", np.zeros(len(observations), dtype=np.float32)).astype(np.float32)
    dones = np.logical_or(terminals > 0.0, timeouts > 0.0).astype(np.float32).reshape(-1, 1)
    return dataset_id, {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "dones": dones,
    }


def truncate_dataset(data: dict[str, np.ndarray], max_samples: int) -> dict[str, np.ndarray]:
    if max_samples <= 0:
        return data
    limit = min(max_samples, len(data["observations"]))
    return {key: value[:limit] for key, value in data.items()}


def actor_log_prob(actor: Actor, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    scaled = torch.clamp((actions - actor.action_bias) / actor.action_scale, -0.999999, 0.999999)
    raw_actions = torch.atanh(scaled)
    mean, log_std = actor(obs)
    normal = Normal(mean, log_std.exp())
    log_prob = normal.log_prob(raw_actions)
    log_prob -= torch.log(actor.action_scale * (1 - torch.tanh(raw_actions).pow(2)) + 1e-6)
    return log_prob.sum(dim=1, keepdim=True)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff.pow(2)).mean()


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def save_checkpoint(
    path: Path,
    args: Args,
    actor: Actor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    value_net: ValueNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    value_optimizer: optim.Optimizer,
    update: int,
    dataset_id: str,
    dataset_size: int,
    eval_return_mean: float | None,
) -> None:
    torch.save(
        {
            "algorithm": "iql",
            "args": asdict(args),
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "value_net": value_net.state_dict(),
            "qf1_target": qf1_target.state_dict(),
            "qf2_target": qf2_target.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "value_optimizer": value_optimizer.state_dict(),
            "seed": args.seed,
            "env_id": args.env_id,
            "dataset_id": dataset_id,
            "offline_dataset_size": dataset_size,
            "offline_updates": update,
            "gradient_updates": update,
            "env_steps": 0,
            "phase": require_known_phase("iql"),
            "eval_return_mean": eval_return_mean,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    dataset_id, data = load_dataset(args.env_id)
    data = truncate_dataset(data, args.max_dataset_samples)
    dataset_size = len(data["observations"])

    env = gym.make(args.env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    if data["observations"].shape[-1] != obs_dim or data["actions"].shape[-1] != action_dim:
        raise ValueError(
            f"{dataset_id} shape mismatch: dataset obs/action {data['observations'].shape[-1]}/"
            f"{data['actions'].shape[-1]}, env obs/action {obs_dim}/{action_dim}"
        )

    actor = Actor(obs_dim, env.action_space).to(device)
    qf1 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, action_dim).to(device)
    value_net = ValueNetwork(obs_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    env.close()

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.critic_learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.value_learning_rate)

    env_slug = args.env_id.replace("-v", "_v").replace("-", "_")
    run_name = f"iql__{env_slug}__seed_{args.seed}__updates_{args.total_updates}__{int(time.time())}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    wandb_run = None
    if args.track:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type="iql_pretrain",
            name=run_name,
            config={**asdict(args), "dataset_id": dataset_id, "offline_dataset_size": dataset_size},
            save_code=True,
            tags=["abhinav_task", "iql", args.env_id],
        )

    start_time = time.time()
    eval_mean, eval_std = evaluate(actor, args.env_id, args.seed + 10_000, device, args.num_eval_episodes)
    write_metric(
        metrics_path,
        {
            "algorithm": "iql",
            "env": args.env_id,
            "seed": args.seed,
            "env_steps": 0,
            "gradient_updates": 0,
            "offline_updates": 0,
            "wall_clock_sec": 0.0,
            "phase": require_known_phase("iql"),
            "switched": False,
            "switch_step": None,
            "switch_reason": None,
            "trigger_value": None,
            "dataset_id": dataset_id,
            "offline_dataset_size": dataset_size,
            "expectile": args.expectile,
            "temperature": args.temperature,
            "eval_return_mean": eval_mean,
            "eval_return_std": eval_std,
        },
        wandb_run,
    )
    last_eval_mean = eval_mean

    obs_np = data["observations"]
    next_obs_np = data["next_observations"]
    actions_np = data["actions"]
    rewards_np = data["rewards"]
    dones_np = data["dones"]

    for update in range(1, args.total_updates + 1):
        batch_indices = np.random.randint(0, dataset_size, size=min(args.batch_size, dataset_size))
        obs = torch.as_tensor(obs_np[batch_indices], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(next_obs_np[batch_indices], dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions_np[batch_indices], dtype=torch.float32, device=device)
        rewards = torch.as_tensor(rewards_np[batch_indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor(dones_np[batch_indices], dtype=torch.float32, device=device)

        with torch.no_grad():
            target_q = torch.min(qf1_target(obs, actions), qf2_target(obs, actions))
        value = value_net(obs)
        value_loss = expectile_loss(target_q - value, args.expectile)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_grad_norm = grad_norm(value_net.parameters())
        nn.utils.clip_grad_norm_(value_net.parameters(), args.max_grad_norm)
        value_optimizer.step()

        with torch.no_grad():
            next_v = value_net(next_obs)
            backup = rewards + (1 - dones) * args.gamma * next_v
        q1 = qf1(obs, actions)
        q2 = qf2(obs, actions)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        critic_optimizer.zero_grad()
        q_loss.backward()
        critic_grad_norm = grad_norm(list(qf1.parameters()) + list(qf2.parameters()))
        nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), args.max_grad_norm)
        critic_optimizer.step()

        with torch.no_grad():
            policy_target_q = torch.min(qf1_target(obs, actions), qf2_target(obs, actions))
            advantage = policy_target_q - value_net(obs)
            weights = torch.exp(advantage / args.temperature).clamp(max=args.max_weight)
        log_prob = actor_log_prob(actor, obs, actions)
        actor_loss = -(weights * log_prob).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = grad_norm(actor.parameters())
        nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        actor_optimizer.step()

        soft_update(qf1, qf1_target, args.tau)
        soft_update(qf2, qf2_target, args.tau)

        finite_values = [
            actor_loss.item(),
            q_loss.item(),
            value_loss.item(),
            weights.mean().item(),
            weights.max().item(),
        ]
        if not np.isfinite(finite_values).all():
            raise FloatingPointError(f"Non-finite IQL metric at update {update}: {finite_values}")

        if update % args.eval_interval == 0 or update == args.total_updates:
            eval_mean, eval_std = evaluate(actor, args.env_id, args.seed + 10_000 + update, device, args.num_eval_episodes)
            last_eval_mean = eval_mean
            metrics = {
                "algorithm": "iql",
                "env": args.env_id,
                "seed": args.seed,
                "env_steps": 0,
                "gradient_updates": update,
                "offline_updates": update,
                "wall_clock_sec": time.time() - start_time,
                "phase": require_known_phase("iql"),
                "switched": False,
                "switch_step": None,
                "switch_reason": None,
                "trigger_value": None,
                "dataset_id": dataset_id,
                "offline_dataset_size": dataset_size,
                "expectile": args.expectile,
                "temperature": args.temperature,
                "iql_actor_loss": float(actor_loss.item()),
                "iql_q_loss": float(q_loss.item()),
                "iql_value_loss": float(value_loss.item()),
                "iql_advantage_mean": float(advantage.mean().item()),
                "iql_advantage_std": float(advantage.std().item()) if advantage.numel() > 1 else 0.0,
                "iql_weight_mean": float(weights.mean().item()),
                "iql_weight_max": float(weights.max().item()),
                "iql_q_mean": float(policy_target_q.mean().item()),
                "iql_value_mean": float(value.mean().item()),
                "iql_actor_grad_norm": actor_grad_norm,
                "iql_critic_grad_norm": critic_grad_norm,
                "iql_value_grad_norm": value_grad_norm,
                "eval_return_mean": eval_mean,
                "eval_return_std": eval_std,
            }
            write_metric(metrics_path, metrics, wandb_run)

        if update % args.save_interval == 0 or update == args.total_updates:
            save_checkpoint(
                save_dir / f"checkpoint_update_{update}.pt",
                args,
                actor,
                qf1,
                qf2,
                value_net,
                qf1_target,
                qf2_target,
                actor_optimizer,
                critic_optimizer,
                value_optimizer,
                update,
                dataset_id,
                dataset_size,
                last_eval_mean,
            )

    save_checkpoint(
        save_dir / "iql_policy.pt",
        args,
        actor,
        qf1,
        qf2,
        value_net,
        qf1_target,
        qf2_target,
        actor_optimizer,
        critic_optimizer,
        value_optimizer,
        args.total_updates,
        dataset_id,
        dataset_size,
        last_eval_mean,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
