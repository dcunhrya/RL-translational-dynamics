import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


LOG_STD_MAX = 2
LOG_STD_MIN = -5


@dataclass
class Args:
    env_id: str = "Hopper-v4"
    seed: int = 0
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5_000
    policy_frequency: int = 2
    target_network_frequency: int = 1
    train_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    eval_interval: int = 5_000
    num_eval_episodes: int = 5
    save_interval: int = 25_000
    save_dir: str = "results/raw/experiment_0"
    track: bool = False
    wandb_project: str = "rl-translational-dynamics"
    wandb_entity: str | None = None
    wandb_group: str = "experiment_0"
    capture_video: bool = False
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


def make_env(env_id: str, seed: int, capture_video: bool, run_name: str) -> gym.Env:
    render_mode = "rgb_array" if capture_video else None
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic_torch


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=1))


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_space: gym.spaces.Box):
        super().__init__()
        action_dim = int(np.prod(action_space.shape))
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.fc_mean = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(obs)
        mean = self.fc_mean(hidden)
        log_std = self.fc_logstd(hidden)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = mean if deterministic else normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        if deterministic:
            return action, None, mean
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, mean


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int, device: torch.device):
        self.observations = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: float, done: bool) -> None:
        self.observations[self.ptr] = obs
        self.next_observations[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_idx = self.size if self.full else self.ptr
        idxs = np.random.randint(0, max_idx, size=batch_size)
        return (
            torch.as_tensor(self.observations[idxs], device=self.device),
            torch.as_tensor(self.next_observations[idxs], device=self.device),
            torch.as_tensor(self.actions[idxs], device=self.device),
            torch.as_tensor(self.rewards[idxs], device=self.device),
            torch.as_tensor(self.dones[idxs], device=self.device),
        )


def grad_norm(parameters) -> float:
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def scalarize(metrics: dict) -> dict:
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            clean[key] = value.item()
        elif isinstance(value, torch.Tensor):
            clean[key] = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
        else:
            clean[key] = value
    return clean


def write_metric(metrics_path: Path, metrics: dict, wandb_run=None) -> None:
    metrics = scalarize(metrics)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, sort_keys=True) + "\n")
    if wandb_run is not None:
        wandb_run.log(metrics, step=int(metrics["env_steps"]))


def evaluate(actor: Actor, env_id: str, seed: int, device: torch.device, episodes: int) -> tuple[float, float]:
    env = make_env(env_id, seed, capture_video=False, run_name="eval")
    returns = []
    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        done = False
        episode_return = 0.0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = actor.get_action(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            episode_return += float(reward)
        returns.append(episode_return)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def save_checkpoint(
    path: Path,
    args: Args,
    actor: Actor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    alpha_optimizer: optim.Optimizer | None,
    global_step: int,
    gradient_updates: int,
    log_alpha: torch.Tensor | None,
) -> None:
    torch.save(
        {
            "algorithm": "sac",
            "args": asdict(args),
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(),
            "qf2_target": qf2_target.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "q_optimizer": q_optimizer.state_dict(),
            "alpha_optimizer": alpha_optimizer.state_dict() if alpha_optimizer is not None else None,
            "log_alpha": log_alpha.detach().cpu() if log_alpha is not None else None,
            "seed": args.seed,
            "env_id": args.env_id,
            "global_step": global_step,
            "env_steps": global_step,
            "gradient_updates": gradient_updates,
            "phase": "sac",
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    run_name = f"sac__{args.env_id}__seed_{args.seed}__{int(time.time())}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    wandb_run = None
    if args.track:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            config=asdict(args),
            save_code=True,
        )

    env = make_env(args.env_id, args.seed, args.capture_video, run_name)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    actor = Actor(obs_dim, env.action_space).to(device)
    qf1 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size, device)

    if args.autotune:
        target_entropy = -float(action_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate)
        alpha = float(log_alpha.exp().item())
    else:
        target_entropy = 0.0
        log_alpha = None
        alpha_optimizer = None
        alpha = args.alpha

    start_time = time.time()
    gradient_updates = 0
    initial_eval_mean, initial_eval_std = evaluate(actor, args.env_id, args.seed + 10_000, device, args.num_eval_episodes)
    write_metric(
        metrics_path,
        {
            "algorithm": "sac",
            "env": args.env_id,
            "seed": args.seed,
            "env_steps": 0,
            "gradient_updates": 0,
            "wall_clock_sec": 0.0,
            "eval_return_mean": initial_eval_mean,
            "eval_return_std": initial_eval_std,
            "phase": "sac",
            "switched": False,
            "switch_step": None,
            "switch_reason": None,
            "trigger_value": None,
        },
        wandb_run,
    )

    for global_step in range(1, args.total_timesteps + 1):
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _, _ = actor.get_action(obs_tensor)
            action = action_tensor.cpu().numpy()[0]

        next_obs, reward, terminated, truncated, info = env.step(action)
        real_done = bool(terminated)
        episode_done = bool(terminated or truncated)
        replay_buffer.add(obs, next_obs, action, float(reward), real_done)
        obs = next_obs

        if "episode" in info:
            episode_metrics = {
                "algorithm": "sac",
                "env": args.env_id,
                "seed": args.seed,
                "env_steps": global_step,
                "gradient_updates": gradient_updates,
                "wall_clock_sec": time.time() - start_time,
                "episode_return": float(info["episode"]["r"]),
                "episode_length": int(info["episode"]["l"]),
                "phase": "sac",
                "switched": False,
                "switch_step": None,
                "switch_reason": None,
                "trigger_value": None,
            }
            write_metric(metrics_path, episode_metrics, wandb_run)

        if episode_done:
            obs, _ = env.reset()

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            b_obs, b_next_obs, b_actions, b_rewards, b_dones = replay_buffer.sample(args.batch_size)
            with torch.no_grad():
                next_actions, next_log_pi, _ = actor.get_action(b_next_obs)
                qf1_next_target = qf1_target(b_next_obs, next_actions)
                qf2_next_target = qf2_target(b_next_obs, next_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_pi
                next_q_value = b_rewards + (1 - b_dones) * args.gamma * min_qf_next_target

            qf1_a_values = qf1(b_obs, b_actions)
            qf2_a_values = qf2(b_obs, b_actions)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            q_optimizer.zero_grad()
            qf_loss.backward()
            critic_grad_norm = grad_norm(list(qf1.parameters()) + list(qf2.parameters()))
            q_optimizer.step()

            actor_loss_value = None
            alpha_loss_value = None
            actor_grad_norm = None
            policy_entropy = None
            if global_step % args.policy_frequency == 0:
                pi, log_pi, _ = actor.get_action(b_obs)
                qf1_pi = qf1(b_obs, pi)
                qf2_pi = qf2(b_obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = grad_norm(actor.parameters())
                actor_optimizer.step()
                actor_loss_value = float(actor_loss.item())
                policy_entropy = float((-log_pi).mean().item())

                if args.autotune and log_alpha is not None and alpha_optimizer is not None:
                    with torch.no_grad():
                        _, log_pi_for_alpha, _ = actor.get_action(b_obs)
                    alpha_loss = (-log_alpha.exp() * (log_pi_for_alpha + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    alpha = float(log_alpha.exp().item())
                    alpha_loss_value = float(alpha_loss.item())

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            gradient_updates += 1
            train_metrics = {
                "algorithm": "sac",
                "env": args.env_id,
                "seed": args.seed,
                "env_steps": global_step,
                "gradient_updates": gradient_updates,
                "wall_clock_sec": time.time() - start_time,
                "sac_actor_loss": actor_loss_value,
                "sac_critic_loss": float(qf_loss.item()),
                "sac_qf1_loss": float(qf1_loss.item()),
                "sac_qf2_loss": float(qf2_loss.item()),
                "sac_alpha_loss": alpha_loss_value,
                "sac_alpha": alpha,
                "sac_policy_entropy": policy_entropy,
                "sac_qf1_mean": float(qf1_a_values.mean().item()),
                "sac_qf2_mean": float(qf2_a_values.mean().item()),
                "sac_target_q_mean": float(next_q_value.mean().item()),
                "sac_actor_grad_norm": actor_grad_norm,
                "sac_critic_grad_norm": critic_grad_norm,
                "actor_lr": actor_optimizer.param_groups[0]["lr"],
                "critic_lr": q_optimizer.param_groups[0]["lr"],
                "phase": "sac",
                "switched": False,
                "switch_step": None,
                "switch_reason": None,
                "trigger_value": None,
            }
            write_metric(metrics_path, train_metrics, wandb_run)
            if not np.isfinite([v for v in train_metrics.values() if isinstance(v, (int, float))]).all():
                raise FloatingPointError(f"Non-finite SAC metric at step {global_step}: {train_metrics}")

        if global_step % args.eval_interval == 0 or global_step == args.total_timesteps:
            eval_mean, eval_std = evaluate(actor, args.env_id, args.seed + 10_000 + global_step, device, args.num_eval_episodes)
            write_metric(
                metrics_path,
                {
                    "algorithm": "sac",
                    "env": args.env_id,
                    "seed": args.seed,
                    "env_steps": global_step,
                    "gradient_updates": gradient_updates,
                    "wall_clock_sec": time.time() - start_time,
                    "eval_return_mean": eval_mean,
                    "eval_return_std": eval_std,
                    "phase": "sac",
                    "switched": False,
                    "switch_step": None,
                    "switch_reason": None,
                    "trigger_value": None,
                },
                wandb_run,
            )

        if global_step % args.save_interval == 0 or global_step == args.total_timesteps:
            save_checkpoint(
                save_dir / f"checkpoint_step_{global_step}.pt",
                args,
                actor,
                qf1,
                qf2,
                qf1_target,
                qf2_target,
                actor_optimizer,
                q_optimizer,
                alpha_optimizer,
                global_step,
                gradient_updates,
                log_alpha,
            )

    env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
