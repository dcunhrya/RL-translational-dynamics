import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "rl-translational-dynamics-experiment-0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WANDB_SECRET_NAME = os.environ.get("WANDB_MODAL_SECRET", "wandb-api-key")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "gymnasium[mujoco]",
        "mujoco",
        "wandb",
        "numpy",
    )
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/root/project/src")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    timeout=60 * 60 * 6,
    cpu=4.0,
    memory=8192,
)
def run_training(
    algorithm: str,
    env_id: str,
    seed: int,
    total_timesteps: int = 100_000,
    eval_interval: int = 5_000,
    num_eval_episodes: int = 5,
    save_dir: str = "/root/results/raw/experiment_0",
    wandb_project: str = "rl-translational-dynamics",
    wandb_group: str = "experiment_0",
) -> None:
    if algorithm not in {"sac", "ppo"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    env = os.environ.copy()
    env["WANDB_PROJECT"] = wandb_project

    command = [
        "python",
        f"/root/project/src/train_{algorithm}.py",
        "--env-id",
        env_id,
        "--seed",
        str(seed),
        "--total-timesteps",
        str(total_timesteps),
        "--eval-interval",
        str(eval_interval),
        "--num-eval-episodes",
        str(num_eval_episodes),
        "--save-dir",
        save_dir,
        "--wandb-project",
        wandb_project,
        "--wandb-group",
        wandb_group,
        "--track",
    ]
    subprocess.run(command, check=True, env=env)


@app.local_entrypoint()
def main(
    total_timesteps: int = 100_000,
    eval_interval: int = 5_000,
    num_eval_episodes: int = 5,
    wandb_project: str = "rl-translational-dynamics",
) -> None:
    jobs = []
    for algorithm in ("sac", "ppo"):
        for env_id in ("Hopper-v4", "Walker2d-v4"):
            for seed in (0, 1):
                jobs.append(
                    run_training.spawn(
                        algorithm=algorithm,
                        env_id=env_id,
                        seed=seed,
                        total_timesteps=total_timesteps,
                        eval_interval=eval_interval,
                        num_eval_episodes=num_eval_episodes,
                        wandb_project=wandb_project,
                    )
                )

    for job in jobs:
        job.get()
