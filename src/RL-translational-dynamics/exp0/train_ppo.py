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


@dataclass
class Args:
    env_id: str = "Hopper-v4"
    seed: int = 0
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    num_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    eval_interval: int = 5_000
    num_eval_episodes: int = 5
    save_interval: int = 25_000
    save_dir: str = "results/raw/experiment_0"
    bc_policy_path: str | None = None
    offline_policy_source: str = "bc"
    bc_distill_steps: int = 0
    bc_distill_batch_size: int = 1024
    bc_distill_learning_rate: float = 1e-3
    track: bool = False
    wandb_project: str = "rl-translational-dynamics"
    wandb_entity: str | None = None
    wandb_group: str = "experiment_0"
    capture_video: bool = False
    cuda: bool = True
    anneal_lr: bool = True


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    for field_name, field_def in Args.__dataclass_fields__.items():
        default = field_def.default
        arg_name = f"--{field_name.replace('_', '-')}"
        if isinstance(default, bool):
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction, default=default)
        elif field_name == "target_kl":
            parser.add_argument(arg_name, type=float, default=default)
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


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_space: gym.spaces.Box):
        super().__init__()
        action_dim = int(np.prod(action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        raw_action = action_mean if deterministic else probs.sample()
        if action is not None:
            raw_action = torch.atanh(torch.clamp((action - self.action_bias) / self.action_scale, -0.999999, 0.999999))
        squashed = torch.tanh(raw_action)
        scaled_action = squashed * self.action_scale + self.action_bias
        log_prob = probs.log_prob(raw_action) - torch.log(self.action_scale * (1 - squashed.pow(2)) + 1e-6)
        return scaled_action, log_prob.sum(1), probs.entropy().sum(1), self.critic(obs)


def grad_norm(parameters) -> float:
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    return float(np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y)


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


def evaluate(agent: Agent, env_id: str, seed: int, device: torch.device, episodes: int) -> tuple[float, float]:
    env = make_env(env_id, seed, capture_video=False, run_name="eval")
    returns = []
    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        done = False
        episode_return = 0.0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            episode_return += float(reward)
        returns.append(episode_return)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def algorithm_name(args: Args) -> str:
    if args.bc_policy_path:
        return f"{args.offline_policy_source}_to_ppo"
    return "ppo"


def base_metadata(args: Args, phase: str) -> dict:
    has_bc_policy = bool(args.bc_policy_path)
    return {
        "algorithm": algorithm_name(args),
        "phase": phase,
        "switched": False,
        "switch_step": None,
        "switch_reason": None,
        "trigger_value": None,
        "policy_init": "distill" if has_bc_policy else "random",
        "policy_source": args.offline_policy_source if has_bc_policy else "none",
        "value_init": "random" if has_bc_policy else "native",
        "value_source": "none",
        "bc_policy_path": args.bc_policy_path,
    }


def load_bc_actor(path: str, obs_dim: int, action_space: gym.spaces.Box, device: torch.device):
    from train_sac import Actor as SourceActor

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    actor_state = checkpoint.get("actor") or checkpoint.get("bc_actor") or checkpoint.get("sac_actor")
    if actor_state is None:
        raise KeyError(f"{path} does not contain an actor, bc_actor, or sac_actor state dict.")
    actor = SourceActor(obs_dim, action_space).to(device)
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


def load_bc_observations(env_id: str) -> np.ndarray:
    from demos import load_d4rl_expert_data

    batch = load_d4rl_expert_data(env_id)
    return batch.observations


def distill_ppo_actor_from_bc(
    agent: Agent,
    bc_actor,
    observations: np.ndarray,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> float | None:
    if steps <= 0:
        return None
    if len(observations) == 0:
        raise ValueError("Cannot distill from BC policy without observations.")
    optimizer = optim.Adam(agent.actor_mean.parameters(), lr=learning_rate)
    last_loss = None
    for _ in range(steps):
        batch_indices = np.random.randint(0, len(observations), size=min(batch_size, len(observations)))
        obs_batch = torch.as_tensor(observations[batch_indices], dtype=torch.float32, device=device)
        with torch.no_grad():
            bc_actions, _, _ = bc_actor.get_action(obs_batch, deterministic=True)
        ppo_actions, _, _, _ = agent.get_action_and_value(obs_batch, deterministic=True)
        loss = F.mse_loss(ppo_actions, bc_actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    return last_loss


def save_checkpoint(
    path: Path,
    args: Args,
    agent: Agent,
    optimizer: optim.Optimizer,
    global_step: int,
    gradient_updates: int,
) -> None:
    torch.save(
        {
            "algorithm": algorithm_name(args),
            "args": asdict(args),
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": args.seed,
            "env_id": args.env_id,
            "global_step": global_step,
            "env_steps": global_step,
            "gradient_updates": gradient_updates,
            "phase": "ppo",
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    env_slug = args.env_id.replace("-v", "_v").replace("-", "_")
    horizon_k = int(args.total_timesteps / 1000)
    algo = algorithm_name(args)
    run_name = f"{algo}__{env_slug}__seed_{args.seed}__{horizon_k}k__{int(time.time())}"
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
            job_type=algo,
            name=run_name,
            config=asdict(args),
            save_code=True,
            tags=["experiment_0", algo, args.env_id],
        )

    env = make_env(args.env_id, args.seed, args.capture_video, run_name)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    agent = Agent(obs_dim, env.action_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_buf = torch.zeros((args.num_steps, obs_dim), device=device)
    actions_buf = torch.zeros((args.num_steps, action_dim), device=device)
    logprobs_buf = torch.zeros(args.num_steps, device=device)
    rewards_buf = torch.zeros(args.num_steps, device=device)
    dones_buf = torch.zeros(args.num_steps, device=device)
    values_buf = torch.zeros(args.num_steps, device=device)

    start_time = time.time()
    global_step = 0
    gradient_updates = 0
    num_updates = math.ceil(args.total_timesteps / args.num_steps)
    if args.bc_policy_path:
        bc_actor = load_bc_actor(args.bc_policy_path, obs_dim, env.action_space, device)
        bc_observations = load_bc_observations(args.env_id)
        bc_distill_loss = distill_ppo_actor_from_bc(
            agent,
            bc_actor,
            bc_observations,
            args.bc_distill_steps,
            args.bc_distill_batch_size,
            args.bc_distill_learning_rate,
            device,
        )
        bc_eval_mean, bc_eval_std = evaluate(agent, args.env_id, args.seed + 50_000, device, args.num_eval_episodes)
        distill_metrics = {
            "algorithm": algo,
            "env": args.env_id,
            "seed": args.seed,
            "env_steps": 0,
            "gradient_updates": 0,
            "wall_clock_sec": time.time() - start_time,
            "eval_return_mean": bc_eval_mean,
            "eval_return_std": bc_eval_std,
            "bc_pre_finetune_eval_return_mean": bc_eval_mean,
            "bc_pre_finetune_eval_return_std": bc_eval_std,
            "bc_distill_loss": bc_distill_loss,
            "bc_distill_steps": args.bc_distill_steps,
            "offline_pretrain_updates": None,
        }
        distill_metrics.update(base_metadata(args, "distill"))
        write_metric(metrics_path, distill_metrics, wandb_run)

    initial_eval_mean, initial_eval_std = evaluate(agent, args.env_id, args.seed + 10_000, device, args.num_eval_episodes)
    initial_metrics = {
        "algorithm": algo,
        "env": args.env_id,
        "seed": args.seed,
        "env_steps": 0,
        "gradient_updates": 0,
        "wall_clock_sec": time.time() - start_time,
        "eval_return_mean": initial_eval_mean,
        "eval_return_std": initial_eval_std,
    }
    initial_metrics.update(base_metadata(args, "ppo"))
    write_metric(metrics_path, initial_metrics, wandb_run)

    next_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    next_done = torch.zeros((), device=device)
    last_eval_step = 0
    last_save_step = 0

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            if global_step >= args.total_timesteps:
                break
            global_step += 1
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values_buf[step] = value.flatten()
            actions_buf[step] = action.squeeze(0)
            logprobs_buf[step] = logprob.squeeze(0)

            obs_np, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            episode_done = bool(terminated or truncated)
            # PPO treats time-limit truncation as an episode boundary in Experiment 0; this is explicit and conservative.
            next_done = torch.tensor(float(episode_done), device=device)
            rewards_buf[step] = float(reward)

            if "episode" in info:
                episode_metrics = {
                    "algorithm": algo,
                    "env": args.env_id,
                    "seed": args.seed,
                    "env_steps": global_step,
                    "gradient_updates": gradient_updates,
                    "wall_clock_sec": time.time() - start_time,
                    "episode_return": float(info["episode"]["r"]),
                    "episode_length": int(info["episode"]["l"]),
                }
                episode_metrics.update(base_metadata(args, "ppo"))
                write_metric(metrics_path, episode_metrics, wandb_run)

            if episode_done:
                obs_np, _ = env.reset()
            next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

            if global_step - last_eval_step >= args.eval_interval or global_step == args.total_timesteps:
                eval_mean, eval_std = evaluate(agent, args.env_id, args.seed + 10_000 + global_step, device, args.num_eval_episodes)
                eval_metrics = {
                    "algorithm": algo,
                    "env": args.env_id,
                    "seed": args.seed,
                    "env_steps": global_step,
                    "gradient_updates": gradient_updates,
                    "wall_clock_sec": time.time() - start_time,
                    "eval_return_mean": eval_mean,
                    "eval_return_std": eval_std,
                }
                eval_metrics.update(base_metadata(args, "ppo"))
                write_metric(metrics_path, eval_metrics, wandb_run)
                last_eval_step = global_step

            if global_step - last_save_step >= args.save_interval or global_step == args.total_timesteps:
                save_checkpoint(save_dir / f"checkpoint_step_{global_step}.pt", args, agent, optimizer, global_step, gradient_updates)
                last_save_step = global_step

        rollout_steps = min(args.num_steps, args.total_timesteps - (update - 1) * args.num_steps)
        if rollout_steps <= 0:
            break

        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1)
            advantages = torch.zeros(rollout_steps, device=device)
            lastgaelam = 0.0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf[:rollout_steps]

        b_obs = obs_buf[:rollout_steps]
        b_logprobs = logprobs_buf[:rollout_steps]
        b_actions = actions_buf[:rollout_steps]
        b_advantages = advantages
        b_returns = returns
        b_values = values_buf[:rollout_steps]
        b_inds = np.arange(rollout_steps)
        minibatch_size = max(1, rollout_steps // args.num_minibatches)
        clipfracs = []
        approx_kl_value = 0.0
        pg_loss_value = 0.0
        v_loss_value = 0.0
        entropy_loss_value = 0.0
        grad_norm_value = 0.0

        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, rollout_steps, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                grad_norm_value = grad_norm(agent.parameters())
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                gradient_updates += 1
                approx_kl_value = float(approx_kl.item())
                pg_loss_value = float(pg_loss.item())
                v_loss_value = float(v_loss.item())
                entropy_loss_value = float(entropy_loss.item())

            if args.target_kl is not None and approx_kl_value > args.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        train_metrics = {
            "algorithm": algo,
            "env": args.env_id,
            "seed": args.seed,
            "env_steps": global_step,
            "gradient_updates": gradient_updates,
            "wall_clock_sec": time.time() - start_time,
            "ppo_policy_loss": pg_loss_value,
            "ppo_value_loss": v_loss_value,
            "ppo_entropy": entropy_loss_value,
            "ppo_approx_kl": approx_kl_value,
            "ppo_clip_fraction": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "ppo_explained_variance": explained_variance(y_pred, y_true),
            "ppo_advantage_mean": float(b_advantages.mean().item()),
            "ppo_advantage_std": float(b_advantages.std().item()) if len(b_advantages) > 1 else 0.0,
            "ppo_return_mean": float(b_returns.mean().item()),
            "ppo_return_std": float(b_returns.std().item()) if len(b_returns) > 1 else 0.0,
            "ppo_old_approx_kl": float(old_approx_kl.item()),
            "ppo_grad_norm": grad_norm_value,
            "actor_value_lr": optimizer.param_groups[0]["lr"],
            "ppo_time_limit_bootstrap": False,
        }
        train_metrics.update(base_metadata(args, "ppo"))
        write_metric(metrics_path, train_metrics, wandb_run)
        finite_values = [v for v in train_metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not np.isfinite(finite_values).all():
            raise FloatingPointError(f"Non-finite PPO metric at step {global_step}: {train_metrics}")

    env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
