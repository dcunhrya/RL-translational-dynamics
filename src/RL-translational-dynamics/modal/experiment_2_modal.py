import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "rl-translational-dynamics-experiment-2"
WANDB_SECRET_NAME = os.environ.get("WANDB_MODAL_SECRET", "wandb-api-key")
TRAINING_REL_DIR = Path("src") / "RL-translational-dynamics" / "exp0"
REMOTE_TRAINING_DIR = Path("/root/project/exp0")


def find_training_source_dir() -> Path:
    override = os.environ.get("TRAINING_SOURCE_DIR") or os.environ.get("PROJECT_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
        if path.name == "exp0" and (path / "train_handoff.py").exists():
            return path
        if (path / TRAINING_REL_DIR / "train_handoff.py").exists():
            return path / TRAINING_REL_DIR
        raise FileNotFoundError(
            f"Override does not contain train_handoff.py or {TRAINING_REL_DIR / 'train_handoff.py'}: {path}"
        )

    if (REMOTE_TRAINING_DIR / "train_handoff.py").exists():
        return REMOTE_TRAINING_DIR

    candidates = [Path.cwd().resolve()]
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        training_dir = candidate / TRAINING_REL_DIR
        if (training_dir / "train_handoff.py").exists():
            return training_dir
    raise FileNotFoundError(f"Could not locate {TRAINING_REL_DIR / 'train_handoff.py'}")


TRAINING_SOURCE_DIR = find_training_source_dir()

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
    .add_local_dir(TRAINING_SOURCE_DIR, remote_path=str(REMOTE_TRAINING_DIR))
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    timeout=60 * 60 * 8,
    cpu=4.0,
    memory=16384,
    gpu="T4",
)
def run_training(
    env_id: str,
    seed: int,
    switch_fraction: float,
    total_timesteps: int = 100_000,
    eval_interval: int = 5_000,
    num_eval_episodes: int = 5,
    save_dir: str = "/root/results/raw/experiment_2_fixed_handoff",
    wandb_project: str = "rl-translational-dynamics",
    wandb_group_prefix: str = "experiment_2_fixed_handoff",
    value_warmup_updates: int = 2,
    distill_steps: int = 500,
) -> None:
    switch_pct = int(round(switch_fraction * 100))
    wandb_group = f"{wandb_group_prefix}__{env_id}__switch_{switch_pct}pct"

    env = os.environ.copy()
    env["WANDB_PROJECT"] = wandb_project

    command = [
        "python",
        "/root/project/exp0/train_handoff.py",
        "--env-id",
        env_id,
        "--seed",
        str(seed),
        "--total-timesteps",
        str(total_timesteps),
        "--switch-fraction",
        str(switch_fraction),
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
        "--value-warmup-updates",
        str(value_warmup_updates),
        "--distill-steps",
        str(distill_steps),
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
    for switch_fraction in (0.25, 0.5, 0.75):
        for env_id in ("Hopper-v4", "Walker2d-v4"):
            for seed in (0, 1, 2):
                jobs.append(
                    run_training.spawn(
                        env_id=env_id,
                        seed=seed,
                        switch_fraction=switch_fraction,
                        total_timesteps=total_timesteps,
                        eval_interval=eval_interval,
                        num_eval_episodes=num_eval_episodes,
                        wandb_project=wandb_project,
                    )
                )

    for job in jobs:
        job.get()
