import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "rl-translational-dynamics-ethan"
WANDB_SECRET_NAME = os.environ.get("WANDB_MODAL_SECRET", "wandb-api-key")
RESULTS_VOLUME_NAME = os.environ.get("MODAL_RESULTS_VOLUME", "herschethan")
TRAINING_REL_DIR = Path("src") / "RL-translational-dynamics" / "exp0"
REMOTE_TRAINING_DIR = Path("/root/project/exp0")
REMOTE_RESULTS_DIR = Path("/root/results")


def find_training_source_dir() -> Path:
    override = os.environ.get("TRAINING_SOURCE_DIR") or os.environ.get("PROJECT_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
        if path.name == "exp0" and (path / "train_reverse_handoff.py").exists():
            return path
        if (path / TRAINING_REL_DIR / "train_reverse_handoff.py").exists():
            return path / TRAINING_REL_DIR
        raise FileNotFoundError(
            f"Override does not contain train_reverse_handoff.py or {TRAINING_REL_DIR / 'train_reverse_handoff.py'}: {path}"
        )

    if (REMOTE_TRAINING_DIR / "train_reverse_handoff.py").exists():
        return REMOTE_TRAINING_DIR

    candidates = [Path.cwd().resolve()]
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        training_dir = candidate / TRAINING_REL_DIR
        if (training_dir / "train_reverse_handoff.py").exists():
            return training_dir
    raise FileNotFoundError(f"Could not locate {TRAINING_REL_DIR / 'train_reverse_handoff.py'}")


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
        "matplotlib",
        "numpy<2",
    )
    .add_local_dir(TRAINING_SOURCE_DIR, remote_path=str(REMOTE_TRAINING_DIR))
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=os.environ.get("MODAL_GPU", "T4"),
)
def run_reverse_handoff(
    env_id: str,
    seed: int,
    total_timesteps: int,
    switch_fraction: float,
    policy_init: str,
    value_init: str,
    switch_trigger: str = "fixed_fraction",
    patience: int = 3,
    min_first_phase: int = 0,
    eval_interval: int = 5_000,
    num_eval_episodes: int = 5,
    save_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "ethan_task"),
    wandb_project: str = "rl-translational-dynamics",
    wandb_group_prefix: str = "ethan_task",
    distill_steps: int = 500,
    sac_critic_warmup_updates: int = 1_000,
) -> None:
    switch_pct = int(round(switch_fraction * 100))
    arm = f"ppo_sac__policy_{policy_init}__value_{value_init}__trigger_{switch_trigger}"
    wandb_group = f"{wandb_group_prefix}__{arm}__{env_id}__switch_{switch_pct}pct"
    env = os.environ.copy()
    env["WANDB_PROJECT"] = wandb_project

    command = [
        "python",
        "/root/project/exp0/train_reverse_handoff.py",
        "--env-id",
        env_id,
        "--seed",
        str(seed),
        "--total-timesteps",
        str(total_timesteps),
        "--switch-fraction",
        str(switch_fraction),
        "--policy-init",
        policy_init,
        "--value-init",
        value_init,
        "--switch-trigger",
        switch_trigger,
        "--patience",
        str(patience),
        "--min-first-phase",
        str(min_first_phase),
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
        "--distill-steps",
        str(distill_steps),
        "--sac-critic-warmup-updates",
        str(sac_critic_warmup_updates),
        "--track",
    ]
    subprocess.run(command, check=True, env=env)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 12,
    cpu=4.0,
    memory=16384,
    gpu=os.environ.get("MODAL_GPU", "T4"),
)
def run_ppo(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int = 5_000,
    num_eval_episodes: int = 5,
    save_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "ethan_task_long_horizon_ppo"),
    wandb_project: str = "rl-translational-dynamics",
    wandb_group_prefix: str = "ethan_task_long_horizon_ppo",
) -> None:
    wandb_group = f"{wandb_group_prefix}__{env_id}"
    env = os.environ.copy()
    env["WANDB_PROJECT"] = wandb_project
    command = [
        "python",
        "/root/project/exp0/train_ppo.py",
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


@app.function(
    image=image,
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60,
    cpu=2.0,
    memory=8192,
)
def summarize_results() -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        "python",
        "/root/project/exp0/summarize_ethan_task.py",
        "--results-dir",
        str(REMOTE_RESULTS_DIR / "raw" / "ethan_task"),
        "--extra-results-dir",
        str(REMOTE_RESULTS_DIR / "raw" / "ethan_task_long_horizon_reverse_handoff"),
        "--extra-results-dir",
        str(REMOTE_RESULTS_DIR / "raw" / "ethan_task_long_horizon_ppo"),
        "--output-dir",
        str(REMOTE_RESULTS_DIR / "processed" / "ethan_task"),
        "--notes-dir",
        str(REMOTE_RESULTS_DIR / "processed" / "ethan_task"),
    ]
    subprocess.run(command, check=True, env=env)
    results_volume.commit()


@app.local_entrypoint()
def main(
    total_timesteps: int = 500_000,
    long_timesteps: int = 1_000_000,
    eval_interval: int = 5_000,
    num_eval_episodes: int = 5,
    wandb_project: str = "rl-translational-dynamics",
    include_tier1: bool = True,
    include_timing: bool = True,
    include_adaptive: bool = True,
    include_long: bool = True,
    summarize: bool = False,
) -> None:
    if summarize:
        summarize_results.remote()
        return

    jobs = []

    if include_tier1:
        for value_init in ("random", "self-warmup", "source-aligned"):
            for env_id in ("Hopper-v4", "Walker2d-v4"):
                for seed in (0, 1, 2, 3, 4):
                    jobs.append(
                        run_reverse_handoff.spawn(
                            env_id=env_id,
                            seed=seed,
                            total_timesteps=total_timesteps,
                            switch_fraction=0.5,
                            policy_init="distill",
                            value_init=value_init,
                            eval_interval=eval_interval,
                            num_eval_episodes=num_eval_episodes,
                            wandb_project=wandb_project,
                        )
                    )

    if include_timing:
        for switch_fraction in (0.25, 0.75):
            for env_id in ("Hopper-v4", "Walker2d-v4"):
                for seed in (0, 1, 2):
                    jobs.append(
                        run_reverse_handoff.spawn(
                            env_id=env_id,
                            seed=seed,
                            total_timesteps=total_timesteps,
                            switch_fraction=switch_fraction,
                            policy_init="distill",
                            value_init="self-warmup",
                            eval_interval=eval_interval,
                            num_eval_episodes=num_eval_episodes,
                            wandb_project=wandb_project,
                        )
                    )

    if include_adaptive:
        min_first_phase = max(1, int(total_timesteps * 0.25))
        for env_id in ("Hopper-v4", "Walker2d-v4"):
            for seed in (0, 1, 2):
                jobs.append(
                    run_reverse_handoff.spawn(
                        env_id=env_id,
                        seed=seed,
                        total_timesteps=total_timesteps,
                        switch_fraction=0.75,
                        policy_init="distill",
                        value_init="self-warmup",
                        switch_trigger="no-improve",
                        patience=3,
                        min_first_phase=min_first_phase,
                        eval_interval=eval_interval,
                        num_eval_episodes=num_eval_episodes,
                        wandb_project=wandb_project,
                    )
                )

    if include_long:
        for seed in (0, 1, 2):
            jobs.append(
                run_ppo.spawn(
                    env_id="Hopper-v4",
                    seed=seed,
                    total_timesteps=long_timesteps,
                    eval_interval=eval_interval,
                    num_eval_episodes=num_eval_episodes,
                    wandb_project=wandb_project,
                )
            )
            jobs.append(
                run_reverse_handoff.spawn(
                    env_id="Hopper-v4",
                    seed=seed,
                    total_timesteps=long_timesteps,
                    switch_fraction=0.5,
                    policy_init="distill",
                    value_init="self-warmup",
                    eval_interval=eval_interval,
                    num_eval_episodes=num_eval_episodes,
                    save_dir=str(REMOTE_RESULTS_DIR / "raw" / "ethan_task_long_horizon_reverse_handoff"),
                    wandb_project=wandb_project,
                )
            )

    print(f"Spawned {len(jobs)} Ethan task Modal jobs.")
    print("This entrypoint intentionally does not wait on job.get(); spawned Modal calls continue remotely.")
