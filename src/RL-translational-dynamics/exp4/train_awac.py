import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

EXP0_DIR = Path(__file__).resolve().parent.parent / "exp0"
sys.path.insert(0, str(EXP0_DIR))

from demos import D4RL_EXPERT_DATASETS
from train_sac import Actor, SoftQNetwork, evaluate, grad_norm, seed_everything, write_metric

from diagnostics import require_known_phase


@dataclass
class Args:
    env_id: str = "Hopper-v4"
    seed: int = 0
    total_updates: int = 100_000
    batch_size: int = 256
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    awac_lambda: float = 1.0
    max_weight: float = 20.0
    eval_interval: int = 5_000
    num_eval_episodes: int = 5
    save_interval: int = 25_000
    save_dir: str = "results/raw/abhinav_task/awac_pretrain"
    track: bool = False
    wandb_project: str = "rl-translational-dynamics"
    wandb_entity: str | None = None
    wandb_group: str = "abhinav_awac_pretrain"
    cuda: bool = True


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


def actor_log_prob(actor: Actor, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    scaled = torch.clamp((actions - actor.action_bias) / actor.action_scale, -0.999999, 0.999999)
    raw_actions = torch.atanh(scaled)
    mean, log_std = actor(obs)
    normal = Normal(mean, log_std.exp())
    log_prob = normal.log_prob(raw_actions)
    log_prob -= torch.log(actor.action_scale * (1 - torch.tanh(raw_actions).pow(2)) + 1e-6)
    return log_prob.sum(dim=1, keepdim=True)


def save_checkpoint(
    path: Path,
    args: Args,
    actor: Actor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    update: int,
    dataset_id: str,
    dataset_size: int,
    eval_return_mean: float | None,
) -> None:
    torch.save(
        {
            "algorithm": "awac",
            "args": asdict(args),
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(),
            "qf2_target": qf2_target.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "seed": args.seed,
            "env_id": args.env_id,
            "dataset_id": dataset_id,
            "offline_dataset_size": dataset_size,
            "offline_updates": update,
            "gradient_updates": update,
            "env_steps": 0,
            "phase": require_known_phase("awac"),
            "eval_return_mean": eval_return_mean,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    dataset_id, data = load_dataset(args.env_id)
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
    qf1_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    env.close()

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.critic_learning_rate)

    env_slug = args.env_id.replace("-v", "_v").replace("-", "_")
    run_name = f"awac__{env_slug}__seed_{args.seed}__updates_{args.total_updates}__{int(time.time())}"
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
            job_type="awac_pretrain",
            name=run_name,
            config={**asdict(args), "dataset_id": dataset_id, "offline_dataset_size": dataset_size},
            save_code=True,
            tags=["abhinav_task", "awac", args.env_id],
        )

    start_time = time.time()
    eval_mean, eval_std = evaluate(actor, args.env_id, args.seed + 10_000, device, args.num_eval_episodes)
    write_metric(
        metrics_path,
        {
            "algorithm": "awac",
            "env": args.env_id,
            "seed": args.seed,
            "env_steps": 0,
            "gradient_updates": 0,
            "offline_updates": 0,
            "wall_clock_sec": 0.0,
            "phase": require_known_phase("awac"),
            "switched": False,
            "switch_step": None,
            "switch_reason": None,
            "trigger_value": None,
            "dataset_id": dataset_id,
            "offline_dataset_size": dataset_size,
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
            next_actions, _, _ = actor.get_action(next_obs)
            target_q = torch.min(qf1_target(next_obs, next_actions), qf2_target(next_obs, next_actions))
            backup = rewards + (1 - dones) * args.gamma * target_q

        q1 = qf1(obs, actions)
        q2 = qf2(obs, actions)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = grad_norm(list(qf1.parameters()) + list(qf2.parameters()))
        critic_optimizer.step()

        with torch.no_grad():
            sampled_actions, _, _ = actor.get_action(obs)
            value_estimate = torch.min(qf1(obs, sampled_actions), qf2(obs, sampled_actions))
            dataset_q = torch.min(qf1(obs, actions), qf2(obs, actions))
            weights = torch.exp((dataset_q - value_estimate) / args.awac_lambda).clamp(max=args.max_weight)

        log_prob = actor_log_prob(actor, obs, actions)
        actor_loss = -(weights * log_prob).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = grad_norm(actor.parameters())
        actor_optimizer.step()

        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        if update % args.eval_interval == 0 or update == args.total_updates:
            eval_mean, eval_std = evaluate(actor, args.env_id, args.seed + 10_000 + update, device, args.num_eval_episodes)
            last_eval_mean = eval_mean
            metrics = {
                "algorithm": "awac",
                "env": args.env_id,
                "seed": args.seed,
                "env_steps": 0,
                "gradient_updates": update,
                "offline_updates": update,
                "wall_clock_sec": time.time() - start_time,
                "phase": require_known_phase("awac"),
                "switched": False,
                "switch_step": None,
                "switch_reason": None,
                "trigger_value": None,
                "dataset_id": dataset_id,
                "offline_dataset_size": dataset_size,
                "awac_actor_loss": float(actor_loss.item()),
                "awac_critic_loss": float(critic_loss.item()),
                "awac_weight_mean": float(weights.mean().item()),
                "awac_weight_max": float(weights.max().item()),
                "awac_actor_grad_norm": actor_grad_norm,
                "awac_critic_grad_norm": critic_grad_norm,
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
                qf1_target,
                qf2_target,
                actor_optimizer,
                critic_optimizer,
                update,
                dataset_id,
                dataset_size,
                last_eval_mean,
            )

    save_checkpoint(
        save_dir / "awac_policy.pt",
        args,
        actor,
        qf1,
        qf2,
        qf1_target,
        qf2_target,
        actor_optimizer,
        critic_optimizer,
        args.total_updates,
        dataset_id,
        dataset_size,
        last_eval_mean,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
