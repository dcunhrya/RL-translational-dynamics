import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

EXP0_DIR = Path(__file__).resolve().parent.parent / "exp0"
sys.path.insert(0, str(EXP0_DIR))

from demos import load_d4rl_expert_data, save_demo_batch
from train_sac import Actor, evaluate, make_env, seed_everything, write_metric

from diagnostics import require_known_phase


@dataclass
class Args:
    env_id: str = "Hopper-v4"
    seed: int = 0
    total_updates: int = 50_000
    batch_size: int = 1024
    learning_rate: float = 1e-3
    eval_interval: int = 5_000
    num_eval_episodes: int = 5
    save_interval: int = 25_000
    dataset_cache_dir: str = "results/datasets/d4rl_expert"
    save_dir: str = "results/raw/abhinav_task/bc_pretrain"
    max_demo_samples: int = 0
    track: bool = False
    wandb_project: str = "rl-translational-dynamics"
    wandb_entity: str | None = None
    wandb_group: str = "abhinav_bc_pretrain"
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


def save_checkpoint(
    path: Path,
    args: Args,
    actor: Actor,
    optimizer: optim.Optimizer,
    update: int,
    dataset_id: str,
    dataset_size: int,
    eval_return_mean: float | None,
) -> None:
    torch.save(
        {
            "algorithm": "bc",
            "args": asdict(args),
            "actor": actor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": args.seed,
            "env_id": args.env_id,
            "dataset_id": dataset_id,
            "offline_dataset_size": dataset_size,
            "offline_updates": update,
            "gradient_updates": update,
            "env_steps": 0,
            "phase": require_known_phase("bc"),
            "eval_return_mean": eval_return_mean,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_slug = args.env_id.replace("-v", "_v").replace("-", "_")
    run_name = f"bc__{env_slug}__seed_{args.seed}__updates_{args.total_updates}__{int(time.time())}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    demo_batch = load_d4rl_expert_data(args.env_id)
    cached_dataset_path = save_demo_batch(demo_batch, Path(args.dataset_cache_dir))
    observations = demo_batch.observations
    actions = demo_batch.actions
    if args.max_demo_samples > 0:
        observations = observations[: args.max_demo_samples]
        actions = actions[: args.max_demo_samples]

    env = make_env(args.env_id, args.seed, capture_video=False, run_name=run_name)
    obs_dim = int(np.prod(env.observation_space.shape))
    if observations.shape[-1] != obs_dim:
        raise ValueError(f"Dataset obs dim {observations.shape[-1]} does not match env obs dim {obs_dim}.")
    actor = Actor(obs_dim, env.action_space).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    env.close()

    wandb_run = None
    if args.track:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type="bc_pretrain",
            name=run_name,
            config={**asdict(args), "dataset_id": demo_batch.dataset_id, "offline_dataset_size": len(observations)},
            save_code=True,
            tags=["abhinav_task", "bc", args.env_id],
        )

    start_time = time.time()
    eval_mean, eval_std = evaluate(actor, args.env_id, args.seed + 10_000, device, args.num_eval_episodes)
    initial_metrics = {
        "algorithm": "bc",
        "env": args.env_id,
        "seed": args.seed,
        "env_steps": 0,
        "gradient_updates": 0,
        "offline_updates": 0,
        "wall_clock_sec": 0.0,
        "phase": require_known_phase("bc"),
        "switched": False,
        "switch_step": None,
        "switch_reason": None,
        "trigger_value": None,
        "dataset_id": demo_batch.dataset_id,
        "dataset_cache_path": str(cached_dataset_path),
        "offline_dataset_size": int(len(observations)),
        "eval_return_mean": eval_mean,
        "eval_return_std": eval_std,
    }
    write_metric(metrics_path, initial_metrics, wandb_run)

    last_eval_mean = eval_mean
    for update in range(1, args.total_updates + 1):
        batch_indices = np.random.randint(0, len(observations), size=min(args.batch_size, len(observations)))
        obs_batch = torch.as_tensor(observations[batch_indices], dtype=torch.float32, device=device)
        action_batch = torch.as_tensor(actions[batch_indices], dtype=torch.float32, device=device)

        pred_actions, _, _ = actor.get_action(obs_batch, deterministic=True)
        bc_loss = F.mse_loss(pred_actions, action_batch)
        optimizer.zero_grad()
        bc_loss.backward()
        optimizer.step()

        if update % args.eval_interval == 0 or update == args.total_updates:
            eval_mean, eval_std = evaluate(actor, args.env_id, args.seed + 10_000 + update, device, args.num_eval_episodes)
            last_eval_mean = eval_mean
            metrics = {
                "algorithm": "bc",
                "env": args.env_id,
                "seed": args.seed,
                "env_steps": 0,
                "gradient_updates": update,
                "offline_updates": update,
                "wall_clock_sec": time.time() - start_time,
                "phase": require_known_phase("bc"),
                "switched": False,
                "switch_step": None,
                "switch_reason": None,
                "trigger_value": None,
                "dataset_id": demo_batch.dataset_id,
                "offline_dataset_size": int(len(observations)),
                "bc_loss": float(bc_loss.item()),
                "eval_return_mean": eval_mean,
                "eval_return_std": eval_std,
            }
            write_metric(metrics_path, metrics, wandb_run)

        if update % args.save_interval == 0 or update == args.total_updates:
            save_checkpoint(
                save_dir / f"checkpoint_update_{update}.pt",
                args,
                actor,
                optimizer,
                update,
                demo_batch.dataset_id,
                int(len(observations)),
                last_eval_mean,
            )

    save_checkpoint(
        save_dir / "bc_policy.pt",
        args,
        actor,
        optimizer,
        args.total_updates,
        demo_batch.dataset_id,
        int(len(observations)),
        last_eval_mean,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
