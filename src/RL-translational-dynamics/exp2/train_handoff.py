import argparse
import math
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

EXP0_DIR = Path(__file__).resolve().parent.parent / "exp0"
sys.path.insert(0, str(EXP0_DIR))

from train_ppo import Agent as PPOAgent
from train_ppo import explained_variance
from train_ppo import evaluate as evaluate_ppo
from train_sac import Actor as SACActor
from train_sac import ReplayBuffer
from train_sac import SoftQNetwork
from train_sac import evaluate as evaluate_sac
from train_sac import grad_norm
from train_sac import make_env
from train_sac import seed_everything
from train_sac import write_metric

from handoff_utils import build_metadata as build_handoff_metadata
from handoff_utils import compute_switch_step
from handoff_utils import transfer_arm_name
from handoff_utils import validate_switch_fraction
from handoff_utils import validate_transfer_config


@dataclass
class Args:
    env_id: str = "Hopper-v4"
    seed: int = 0
    total_timesteps: int = 1_000_000
    switch_fraction: float = 0.5
    sac_learning_rate: float = 3e-4
    sac_buffer_size: int = 1_000_000
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_batch_size: int = 256
    sac_learning_starts: int = 5_000
    sac_policy_frequency: int = 2
    sac_target_network_frequency: int = 1
    sac_train_frequency: int = 1
    sac_alpha: float = 0.2
    sac_autotune: bool = True
    ppo_learning_rate: float = 3e-4
    ppo_num_steps: int = 2048
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_num_minibatches: int = 32
    ppo_update_epochs: int = 10
    ppo_norm_adv: bool = True
    ppo_clip_coef: float = 0.2
    ppo_clip_vloss: bool = True
    ppo_ent_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_target_kl: float | None = None
    ppo_anneal_lr: bool = True
    policy_init: str = "distill"
    value_init: str = "self-warmup"
    policy_source: str | None = "sac"
    value_source: str | None = None
    distill_steps: int = 500
    distill_batch_size: int = 1024
    distill_learning_rate: float = 1e-3
    value_warmup_updates: int = 2
    source_value_warmup_steps: int = 500
    source_value_warmup_batch_size: int = 1024
    source_value_warmup_learning_rate: float = 3e-4
    eval_interval: int = 5_000
    num_eval_episodes: int = 5
    save_interval: int = 100_000
    save_dir: str = "results/raw/experiment_2"
    track: bool = False
    wandb_project: str = "rl-translational-dynamics"
    wandb_entity: str | None = None
    wandb_group: str = "experiment_2"
    capture_video: bool = False
    cuda: bool = True


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    for field_name, field_def in Args.__dataclass_fields__.items():
        default = field_def.default
        arg_name = f"--{field_name.replace('_', '-')}"
        if isinstance(default, bool):
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction, default=default)
        elif field_name == "ppo_target_kl":
            parser.add_argument(arg_name, type=float, default=default)
        else:
            parser.add_argument(arg_name, type=type(default) if default is not None else str, default=default)
    return Args(**vars(parser.parse_args()))


def build_metadata(args: Args, phase: str, switch_step: int, switched: bool) -> dict:
    return build_handoff_metadata(
        args.switch_fraction,
        phase,
        switch_step,
        switched,
        policy_init=args.policy_init,
        value_init=args.value_init,
        policy_source=args.policy_source,
        value_source=args.value_source,
    )


def save_checkpoint(
    path: Path,
    args: Args,
    phase: str,
    env_steps: int,
    gradient_updates: int,
    switch_step: int,
    sac_actor: SACActor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    sac_actor_optimizer: optim.Optimizer,
    sac_q_optimizer: optim.Optimizer,
    sac_alpha_optimizer: optim.Optimizer | None,
    sac_log_alpha: torch.Tensor | None,
    ppo_agent: PPOAgent,
    value_warmup_optimizer: optim.Optimizer | None,
    ppo_optimizer: optim.Optimizer | None,
) -> None:
    torch.save(
        {
            "algorithm": "sac_to_ppo",
            "args": asdict(args),
            "phase": phase,
            "seed": args.seed,
            "env_id": args.env_id,
            "env_steps": env_steps,
            "global_step": env_steps,
            "gradient_updates": gradient_updates,
            "switch_step": switch_step,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "sac_actor": sac_actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(),
            "qf2_target": qf2_target.state_dict(),
            "sac_actor_optimizer": sac_actor_optimizer.state_dict(),
            "sac_q_optimizer": sac_q_optimizer.state_dict(),
            "sac_alpha_optimizer": sac_alpha_optimizer.state_dict() if sac_alpha_optimizer is not None else None,
            "sac_log_alpha": sac_log_alpha.detach().cpu() if sac_log_alpha is not None else None,
            "ppo_agent": ppo_agent.state_dict(),
            "value_warmup_optimizer": value_warmup_optimizer.state_dict() if value_warmup_optimizer is not None else None,
            "ppo_optimizer": ppo_optimizer.state_dict() if ppo_optimizer is not None else None,
        },
        path,
    )


def replay_observations(replay_buffer: ReplayBuffer) -> np.ndarray:
    max_idx = replay_buffer.size if replay_buffer.full else replay_buffer.ptr
    return replay_buffer.observations[:max_idx]


def policy_retention_metrics(
    ppo_agent: PPOAgent,
    sac_actor: SACActor,
    obs_batch: torch.Tensor,
) -> dict:
    with torch.no_grad():
        ppo_actions, _, _, _ = ppo_agent.get_action_and_value(obs_batch, deterministic=True)
        sac_actions, _, _ = sac_actor.get_action(obs_batch, deterministic=True)
        ppo_mean = ppo_agent.actor_mean(obs_batch)
        ppo_logstd = ppo_agent.actor_logstd.expand_as(ppo_mean)
        ppo_var = torch.exp(2 * ppo_logstd)
        sac_mean, sac_logstd = sac_actor(obs_batch)
        sac_var = torch.exp(2 * sac_logstd)
        approx_kl = 0.5 * ((ppo_var + (ppo_mean - sac_mean).pow(2)) / sac_var - 1 + 2 * (sac_logstd - ppo_logstd))
    return {
        "policy_retention_action_mse": float(F.mse_loss(ppo_actions, sac_actions).item()),
        "policy_retention_approx_kl": float(approx_kl.sum(dim=1).mean().item()),
    }


def distill_ppo_policy_from_sac(
    ppo_agent: PPOAgent,
    sac_actor: SACActor,
    observations: np.ndarray,
    args: Args,
    device: torch.device,
) -> float | None:
    if args.policy_init == "random":
        return None

    distill_optimizer = optim.Adam(ppo_agent.actor_mean.parameters(), lr=args.distill_learning_rate)
    distill_loss_value = 0.0
    for _ in range(1, args.distill_steps + 1):
        batch_indices = np.random.randint(0, len(observations), size=min(args.distill_batch_size, len(observations)))
        obs_batch = torch.as_tensor(observations[batch_indices], dtype=torch.float32, device=device)
        with torch.no_grad():
            sac_actions, _, _ = sac_actor.get_action(obs_batch, deterministic=True)
        ppo_actions, _, _, _ = ppo_agent.get_action_and_value(obs_batch, deterministic=True)
        distill_loss = F.mse_loss(ppo_actions, sac_actions)
        distill_optimizer.zero_grad()
        distill_loss.backward()
        distill_optimizer.step()
        distill_loss_value = float(distill_loss.item())
    return distill_loss_value


def source_aligned_value_warmup(
    ppo_agent: PPOAgent,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    observations: np.ndarray,
    args: Args,
    device: torch.device,
) -> tuple[float | None, float | None, int, bool]:
    if args.source_value_warmup_steps <= 0:
        return None, None, 0, True

    optimizer = optim.Adam(
        ppo_agent.critic.parameters(),
        lr=args.source_value_warmup_learning_rate,
        eps=1e-5,
    )
    initial_loss = None
    final_loss = None
    for step in range(1, args.source_value_warmup_steps + 1):
        batch_indices = np.random.randint(
            0,
            len(observations),
            size=min(args.source_value_warmup_batch_size, len(observations)),
        )
        obs_batch = torch.as_tensor(observations[batch_indices], dtype=torch.float32, device=device)
        with torch.no_grad():
            ppo_actions, _, _, _ = ppo_agent.get_action_and_value(obs_batch)
            source_values = torch.min(qf1(obs_batch, ppo_actions), qf2(obs_batch, ppo_actions)).view(-1)
        predicted_values = ppo_agent.get_value(obs_batch).view(-1)
        warmup_loss = F.mse_loss(predicted_values, source_values)
        optimizer.zero_grad()
        warmup_loss.backward()
        optimizer.step()
        final_loss = float(warmup_loss.item())
        if initial_loss is None:
            initial_loss = final_loss
        if not np.isfinite(final_loss):
            return initial_loss, final_loss, step, False

    converged = final_loss is not None and initial_loss is not None and final_loss <= max(initial_loss * 1.5, 1e-6)
    return initial_loss, final_loss, args.source_value_warmup_steps, converged


def main() -> None:
    args = parse_args()
    validate_transfer_config(args.policy_init, args.value_init, args.policy_source, args.value_source)
    switch_step = compute_switch_step(args.total_timesteps, args.switch_fraction)
    arm_name = transfer_arm_name(args.policy_init, args.value_init, args.policy_source, args.value_source)
    run_name = (
        f"handoff__{args.env_id}__seed_{args.seed}__frac_{args.switch_fraction:.2f}"
        f"__{arm_name}__switch_{switch_step}__{int(time.time())}"
    )
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

    sac_actor = SACActor(obs_dim, env.action_space).to(device)
    qf1 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    sac_actor_optimizer = optim.Adam(sac_actor.parameters(), lr=args.sac_learning_rate)
    sac_q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.sac_learning_rate)
    replay_buffer = ReplayBuffer(obs_dim, action_dim, args.sac_buffer_size, device)

    if args.sac_autotune:
        target_entropy = -float(action_dim)
        sac_log_alpha = torch.zeros(1, requires_grad=True, device=device)
        sac_alpha_optimizer = optim.Adam([sac_log_alpha], lr=args.sac_learning_rate)
        sac_alpha = float(sac_log_alpha.exp().item())
    else:
        target_entropy = 0.0
        sac_log_alpha = None
        sac_alpha_optimizer = None
        sac_alpha = args.sac_alpha

    ppo_agent = PPOAgent(obs_dim, env.action_space).to(device)
    value_warmup_optimizer = None
    ppo_optimizer = None

    start_time = time.time()
    gradient_updates = 0
    last_eval_step = 0
    last_save_step = 0

    initial_eval_mean, initial_eval_std = evaluate_sac(
        sac_actor, args.env_id, args.seed + 10_000, device, args.num_eval_episodes
    )
    initial_metrics = {
        "env": args.env_id,
        "seed": args.seed,
        "env_steps": 0,
        "gradient_updates": 0,
        "wall_clock_sec": 0.0,
        "eval_return_mean": initial_eval_mean,
        "eval_return_std": initial_eval_std,
    }
    initial_metrics.update(build_metadata(args, "sac", switch_step, switched=False))
    write_metric(metrics_path, initial_metrics, wandb_run)

    for global_step in range(1, switch_step + 1):
        if global_step < args.sac_learning_starts:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _, _ = sac_actor.get_action(obs_tensor)
            action = action_tensor.cpu().numpy()[0]

        next_obs, reward, terminated, truncated, info = env.step(action)
        real_done = bool(terminated)
        episode_done = bool(terminated or truncated)
        replay_buffer.add(obs, next_obs, action, float(reward), real_done)
        obs = next_obs

        if "episode" in info:
            episode_metrics = {
                "env": args.env_id,
                "seed": args.seed,
                "env_steps": global_step,
                "gradient_updates": gradient_updates,
                "wall_clock_sec": time.time() - start_time,
                "episode_return": float(info["episode"]["r"]),
                "episode_length": int(info["episode"]["l"]),
            }
            episode_metrics.update(build_metadata(args, "sac", switch_step, switched=False))
            write_metric(metrics_path, episode_metrics, wandb_run)

        if episode_done:
            obs, _ = env.reset()

        if global_step > args.sac_learning_starts and global_step % args.sac_train_frequency == 0:
            b_obs, b_next_obs, b_actions, b_rewards, b_dones = replay_buffer.sample(args.sac_batch_size)
            with torch.no_grad():
                next_actions, next_log_pi, _ = sac_actor.get_action(b_next_obs)
                qf1_next_target = qf1_target(b_next_obs, next_actions)
                qf2_next_target = qf2_target(b_next_obs, next_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - sac_alpha * next_log_pi
                next_q_value = b_rewards + (1 - b_dones) * args.sac_gamma * min_qf_next_target

            qf1_a_values = qf1(b_obs, b_actions)
            qf2_a_values = qf2(b_obs, b_actions)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            sac_q_optimizer.zero_grad()
            qf_loss.backward()
            critic_grad_norm = grad_norm(list(qf1.parameters()) + list(qf2.parameters()))
            sac_q_optimizer.step()

            actor_loss_value = None
            alpha_loss_value = None
            actor_grad_norm = None
            policy_entropy = None
            if global_step % args.sac_policy_frequency == 0:
                pi, log_pi, _ = sac_actor.get_action(b_obs)
                qf1_pi = qf1(b_obs, pi)
                qf2_pi = qf2(b_obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((sac_alpha * log_pi) - min_qf_pi).mean()
                sac_actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = grad_norm(sac_actor.parameters())
                sac_actor_optimizer.step()
                actor_loss_value = float(actor_loss.item())
                policy_entropy = float((-log_pi).mean().item())

                if args.sac_autotune and sac_log_alpha is not None and sac_alpha_optimizer is not None:
                    with torch.no_grad():
                        _, log_pi_for_alpha, _ = sac_actor.get_action(b_obs)
                    alpha_loss = (-sac_log_alpha.exp() * (log_pi_for_alpha + target_entropy)).mean()
                    sac_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    sac_alpha_optimizer.step()
                    sac_alpha = float(sac_log_alpha.exp().item())
                    alpha_loss_value = float(alpha_loss.item())

            if global_step % args.sac_target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.sac_tau * param.data + (1 - args.sac_tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.sac_tau * param.data + (1 - args.sac_tau) * target_param.data)

            gradient_updates += 1
            train_metrics = {
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
                "sac_alpha": sac_alpha,
                "sac_policy_entropy": policy_entropy,
                "sac_qf1_mean": float(qf1_a_values.mean().item()),
                "sac_qf2_mean": float(qf2_a_values.mean().item()),
                "sac_target_q_mean": float(next_q_value.mean().item()),
                "sac_actor_grad_norm": actor_grad_norm,
                "sac_critic_grad_norm": critic_grad_norm,
                "actor_lr": sac_actor_optimizer.param_groups[0]["lr"],
                "critic_lr": sac_q_optimizer.param_groups[0]["lr"],
            }
            train_metrics.update(build_metadata(args, "sac", switch_step, switched=False))
            write_metric(metrics_path, train_metrics, wandb_run)
            finite_values = [v for v in train_metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not np.isfinite(finite_values).all():
                raise FloatingPointError(f"Non-finite SAC metric at step {global_step}: {train_metrics}")

        if global_step % args.eval_interval == 0 or global_step == switch_step:
            eval_mean, eval_std = evaluate_sac(
                sac_actor, args.env_id, args.seed + 10_000 + global_step, device, args.num_eval_episodes
            )
            eval_metrics = {
                "env": args.env_id,
                "seed": args.seed,
                "env_steps": global_step,
                "gradient_updates": gradient_updates,
                "wall_clock_sec": time.time() - start_time,
                "eval_return_mean": eval_mean,
                "eval_return_std": eval_std,
            }
            eval_metrics.update(build_metadata(args, "sac", switch_step, switched=False))
            write_metric(metrics_path, eval_metrics, wandb_run)
            last_eval_step = global_step

        if global_step % args.save_interval == 0 or global_step == switch_step:
            save_checkpoint(
                save_dir / f"checkpoint_step_{global_step}.pt",
                args,
                "sac",
                global_step,
                gradient_updates,
                switch_step,
                sac_actor,
                qf1,
                qf2,
                qf1_target,
                qf2_target,
                sac_actor_optimizer,
                sac_q_optimizer,
                sac_alpha_optimizer,
                sac_log_alpha,
                ppo_agent,
                value_warmup_optimizer,
                ppo_optimizer,
            )
            last_save_step = global_step

    observations = replay_observations(replay_buffer)
    if len(observations) == 0:
        raise RuntimeError("Cannot distill PPO actor because the SAC replay buffer is empty.")

    distill_loss_value = distill_ppo_policy_from_sac(ppo_agent, sac_actor, observations, args, device)
    retention_obs = torch.as_tensor(
        observations[np.random.randint(0, len(observations), size=min(args.distill_batch_size, len(observations)))],
        dtype=torch.float32,
        device=device,
    )
    retention_metrics = policy_retention_metrics(ppo_agent, sac_actor, retention_obs)

    handoff_metrics = {
        "env": args.env_id,
        "seed": args.seed,
        "env_steps": switch_step,
        "gradient_updates": gradient_updates,
        "wall_clock_sec": time.time() - start_time,
        "handoff_distill_loss": distill_loss_value,
        "handoff_distill_steps": args.distill_steps if args.policy_init == "distill" else 0,
    }
    handoff_metrics.update(retention_metrics)
    handoff_metrics.update(build_metadata(args, "handoff", switch_step, switched=True))
    write_metric(metrics_path, handoff_metrics, wandb_run)

    post_distill_eval_mean, post_distill_eval_std = evaluate_ppo(
        ppo_agent, args.env_id, args.seed + 20_000 + switch_step, device, args.num_eval_episodes
    )
    post_distill_eval_metrics = {
        "env": args.env_id,
        "seed": args.seed,
        "env_steps": switch_step,
        "gradient_updates": gradient_updates,
        "wall_clock_sec": time.time() - start_time,
        "eval_return_mean": post_distill_eval_mean,
        "eval_return_std": post_distill_eval_std,
        "handoff_distill_loss": distill_loss_value,
    }
    post_distill_eval_metrics.update(retention_metrics)
    post_distill_eval_metrics.update(build_metadata(args, "handoff", switch_step, switched=True))
    write_metric(metrics_path, post_distill_eval_metrics, wandb_run)
    last_eval_step = switch_step

    next_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    next_done = torch.zeros((), device=device)
    remaining_updates = math.ceil((args.total_timesteps - switch_step) / args.ppo_num_steps)
    warmup_updates = min(args.value_warmup_updates, remaining_updates) if args.value_init == "self-warmup" else 0
    source_value_warmup_loss_initial = None
    source_value_warmup_loss_final = None
    source_value_warmup_converged = None
    source_value_warmup_steps_done = 0
    value_warmup_fallback = False
    if args.value_init == "source-aligned":
        (
            source_value_warmup_loss_initial,
            source_value_warmup_loss_final,
            source_value_warmup_steps_done,
            source_value_warmup_converged,
        ) = source_aligned_value_warmup(ppo_agent, qf1, qf2, observations, args, device)
        value_warmup_fallback = not bool(source_value_warmup_converged)
        warmup_updates = min(args.value_warmup_updates, remaining_updates) if value_warmup_fallback else 0
        source_warmup_metrics = {
            "env": args.env_id,
            "seed": args.seed,
            "env_steps": switch_step,
            "gradient_updates": gradient_updates,
            "wall_clock_sec": time.time() - start_time,
            "source_value_warmup_loss_initial": source_value_warmup_loss_initial,
            "source_value_warmup_loss_final": source_value_warmup_loss_final,
            "source_value_warmup_steps": source_value_warmup_steps_done,
            "source_value_warmup_converged": bool(source_value_warmup_converged),
            "value_warmup_fallback_to_self": value_warmup_fallback,
        }
        source_warmup_metrics.update(build_metadata(args, "source_value_warmup", switch_step, switched=True))
        write_metric(metrics_path, source_warmup_metrics, wandb_run)
    value_warmup_optimizer = optim.Adam(ppo_agent.critic.parameters(), lr=args.ppo_learning_rate, eps=1e-5)

    obs_buf = torch.zeros((args.ppo_num_steps, obs_dim), device=device)
    actions_buf = torch.zeros((args.ppo_num_steps, action_dim), device=device)
    logprobs_buf = torch.zeros(args.ppo_num_steps, device=device)
    rewards_buf = torch.zeros(args.ppo_num_steps, device=device)
    dones_buf = torch.zeros(args.ppo_num_steps, device=device)
    values_buf = torch.zeros(args.ppo_num_steps, device=device)

    global_step = switch_step
    old_approx_kl_value = 0.0
    ppo_update_index = 0
    while global_step < args.total_timesteps:
        ppo_update_index += 1
        warmup_only = ppo_update_index <= warmup_updates
        current_phase = "ppo_value_warmup" if warmup_only else "ppo"
        if not warmup_only and ppo_optimizer is None:
            ppo_optimizer = optim.Adam(ppo_agent.parameters(), lr=args.ppo_learning_rate, eps=1e-5)
            optimizer_reset_metrics = {
                "env": args.env_id,
                "seed": args.seed,
                "env_steps": global_step,
                "gradient_updates": gradient_updates,
                "wall_clock_sec": time.time() - start_time,
                "optimizer_reset": 1,
            }
            optimizer_reset_metrics.update(build_metadata(args, current_phase, switch_step, switched=True))
            write_metric(metrics_path, optimizer_reset_metrics, wandb_run)

        active_optimizer = value_warmup_optimizer if warmup_only else ppo_optimizer
        assert active_optimizer is not None
        if args.ppo_anneal_lr:
            frac = 1.0 - (ppo_update_index - 1.0) / max(1, remaining_updates)
            active_optimizer.param_groups[0]["lr"] = frac * args.ppo_learning_rate

        rollout_steps = min(args.ppo_num_steps, args.total_timesteps - global_step)
        for step in range(rollout_steps):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = ppo_agent.get_action_and_value(next_obs.unsqueeze(0))
                values_buf[step] = value.flatten()
            actions_buf[step] = action.squeeze(0)
            logprobs_buf[step] = logprob.squeeze(0)

            obs_np, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            episode_done = bool(terminated or truncated)
            next_done = torch.tensor(float(episode_done), device=device)
            rewards_buf[step] = float(reward)
            global_step += 1

            if "episode" in info:
                episode_metrics = {
                    "env": args.env_id,
                    "seed": args.seed,
                    "env_steps": global_step,
                    "gradient_updates": gradient_updates,
                    "wall_clock_sec": time.time() - start_time,
                    "episode_return": float(info["episode"]["r"]),
                    "episode_length": int(info["episode"]["l"]),
                }
                episode_metrics.update(build_metadata(args, current_phase, switch_step, switched=True))
                write_metric(metrics_path, episode_metrics, wandb_run)

            if episode_done:
                obs_np, _ = env.reset()
                next_done = torch.zeros((), device=device)
            next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

            if global_step - last_eval_step >= args.eval_interval or global_step == args.total_timesteps:
                eval_mean, eval_std = evaluate_ppo(
                    ppo_agent, args.env_id, args.seed + 20_000 + global_step, device, args.num_eval_episodes
                )
                eval_metrics = {
                    "env": args.env_id,
                    "seed": args.seed,
                    "env_steps": global_step,
                    "gradient_updates": gradient_updates,
                    "wall_clock_sec": time.time() - start_time,
                    "eval_return_mean": eval_mean,
                    "eval_return_std": eval_std,
                }
                eval_metrics.update(policy_retention_metrics(ppo_agent, sac_actor, retention_obs))
                eval_metrics.update(build_metadata(args, current_phase, switch_step, switched=True))
                write_metric(metrics_path, eval_metrics, wandb_run)
                last_eval_step = global_step

            if global_step - last_save_step >= args.save_interval or global_step == args.total_timesteps:
                save_checkpoint(
                    save_dir / f"checkpoint_step_{global_step}.pt",
                    args,
                    current_phase,
                    global_step,
                    gradient_updates,
                    switch_step,
                    sac_actor,
                    qf1,
                    qf2,
                    qf1_target,
                    qf2_target,
                    sac_actor_optimizer,
                    sac_q_optimizer,
                    sac_alpha_optimizer,
                    sac_log_alpha,
                    ppo_agent,
                    value_warmup_optimizer,
                    ppo_optimizer,
                )
                last_save_step = global_step

        with torch.no_grad():
            next_value = ppo_agent.get_value(next_obs.unsqueeze(0)).reshape(1)
            advantages = torch.zeros(rollout_steps, device=device)
            lastgaelam = 0.0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + args.ppo_gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.ppo_gamma * args.ppo_gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf[:rollout_steps]

        b_obs = obs_buf[:rollout_steps]
        b_logprobs = logprobs_buf[:rollout_steps]
        b_actions = actions_buf[:rollout_steps]
        b_advantages = advantages
        b_returns = returns
        b_values = values_buf[:rollout_steps]
        b_inds = np.arange(rollout_steps)
        minibatch_size = max(1, rollout_steps // args.ppo_num_minibatches)
        clipfracs = []
        approx_kl_value = 0.0
        pg_loss_value = None
        v_loss_value = 0.0
        entropy_loss_value = None
        grad_norm_value = 0.0

        for _ in range(args.ppo_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, rollout_steps, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                if warmup_only:
                    newvalue = ppo_agent.get_value(b_obs[mb_inds]).view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    active_optimizer.zero_grad()
                    v_loss.backward()
                    grad_norm_value = grad_norm(ppo_agent.critic.parameters())
                    active_optimizer.step()
                    gradient_updates += 1
                    v_loss_value = float(v_loss.item())
                    continue

                _, newlogprob, entropy, newvalue = ppo_agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.ppo_clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.ppo_norm_adv and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.ppo_clip_coef, 1 + args.ppo_clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.ppo_clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.ppo_clip_coef, args.ppo_clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ppo_ent_coef * entropy_loss + args.ppo_vf_coef * v_loss
                active_optimizer.zero_grad()
                loss.backward()
                grad_norm_value = grad_norm(ppo_agent.parameters())
                torch.nn.utils.clip_grad_norm_(ppo_agent.parameters(), args.ppo_max_grad_norm)
                active_optimizer.step()

                gradient_updates += 1
                approx_kl_value = float(approx_kl.item())
                old_approx_kl_value = float(old_approx_kl.item())
                pg_loss_value = float(pg_loss.item())
                v_loss_value = float(v_loss.item())
                entropy_loss_value = float(entropy_loss.item())

            if not warmup_only and args.ppo_target_kl is not None and approx_kl_value > args.ppo_target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        train_metrics = {
            "env": args.env_id,
            "seed": args.seed,
            "env_steps": global_step,
            "gradient_updates": gradient_updates,
            "wall_clock_sec": time.time() - start_time,
            "ppo_policy_loss": pg_loss_value,
            "ppo_value_loss": v_loss_value,
            "ppo_entropy": entropy_loss_value,
            "ppo_approx_kl": approx_kl_value if not warmup_only else None,
            "ppo_clip_fraction": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "ppo_explained_variance": explained_variance(y_pred, y_true),
            "ppo_advantage_mean": float(b_advantages.mean().item()),
            "ppo_advantage_std": float(b_advantages.std().item()) if len(b_advantages) > 1 else 0.0,
            "ppo_return_mean": float(b_returns.mean().item()),
            "ppo_return_std": float(b_returns.std().item()) if len(b_returns) > 1 else 0.0,
            "ppo_old_approx_kl": old_approx_kl_value if not warmup_only else None,
            "ppo_grad_norm": grad_norm_value,
            "actor_value_lr": active_optimizer.param_groups[0]["lr"],
            "ppo_time_limit_bootstrap": False,
            "source_value_warmup_loss_initial": source_value_warmup_loss_initial,
            "source_value_warmup_loss_final": source_value_warmup_loss_final,
            "source_value_warmup_steps": source_value_warmup_steps_done,
            "source_value_warmup_converged": source_value_warmup_converged,
            "value_warmup_fallback_to_self": value_warmup_fallback,
        }
        train_metrics.update(policy_retention_metrics(ppo_agent, sac_actor, b_obs))
        train_metrics.update(build_metadata(args, current_phase, switch_step, switched=True))
        write_metric(metrics_path, train_metrics, wandb_run)
        finite_values = [v for v in train_metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not np.isfinite(finite_values).all():
            raise FloatingPointError(f"Non-finite PPO metric at step {global_step}: {train_metrics}")

    save_checkpoint(
        save_dir / f"checkpoint_step_{args.total_timesteps}.pt",
        args,
        "ppo",
        args.total_timesteps,
        gradient_updates,
        switch_step,
        sac_actor,
        qf1,
        qf2,
        qf1_target,
        qf2_target,
        sac_actor_optimizer,
        sac_q_optimizer,
        sac_alpha_optimizer,
        sac_log_alpha,
        ppo_agent,
        value_warmup_optimizer,
        ppo_optimizer,
    )

    env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
