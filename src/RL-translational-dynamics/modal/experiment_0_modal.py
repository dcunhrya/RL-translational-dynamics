import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "rl-translational-dynamics-experiment-0"
WANDB_SECRET_NAME = os.environ.get("WANDB_MODAL_SECRET", "wandb-api-key")
RESULTS_VOLUME_NAME = os.environ.get("MODAL_RESULTS_VOLUME", "herschethan")
TRAINING_REL_DIR = Path("src") / "RL-translational-dynamics" / "exp0"
REMOTE_TRAINING_DIR = Path("/root/project/exp0")
REMOTE_RESULTS_DIR = Path("/root/results")


def find_training_source_dir() -> Path:
    override = os.environ.get("TRAINING_SOURCE_DIR") or os.environ.get("PROJECT_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
        if path.name == "exp0" and (path / "train_sac.py").exists():
            return path
        if (path / TRAINING_REL_DIR / "train_sac.py").exists():
            return path / TRAINING_REL_DIR
        raise FileNotFoundError(f"Override does not contain train_sac.py or {TRAINING_REL_DIR / 'train_sac.py'}: {path}")

    if (REMOTE_TRAINING_DIR / "train_sac.py").exists():
        return REMOTE_TRAINING_DIR

    candidates = [Path.cwd().resolve()]
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        training_dir = candidate / TRAINING_REL_DIR
        if (training_dir / "train_sac.py").exists():
            return training_dir
    raise FileNotFoundError(f"Could not locate {TRAINING_REL_DIR / 'train_sac.py'}")


TRAINING_SOURCE_DIR = find_training_source_dir()

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

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
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
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
    save_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "experiment_0"),
    wandb_project: str = "rl-translational-dynamics",
    wandb_group: str = "experiment_0",
) -> None:
    if algorithm not in {"sac", "ppo"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    env = os.environ.copy()
    env["WANDB_PROJECT"] = wandb_project

    command = [
        "python",
        f"/root/project/exp0/train_{algorithm}.py",
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
