import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "rl-translational-dynamics-experiment-4"
WANDB_SECRET_NAME = os.environ.get("WANDB_MODAL_SECRET", "wandb-api-key")
RESULTS_VOLUME_NAME = os.environ.get("MODAL_RESULTS_VOLUME", "herschethan")
SRC_REL_DIR = Path("src") / "RL-translational-dynamics"
REMOTE_SRC_DIR = Path("/root/project/src/RL-translational-dynamics")
REMOTE_RESULTS_DIR = Path("/root/results")
DEFAULT_GPU = os.environ.get("MODAL_GPU", "L4")

ENVS = ("Hopper-v4", "Walker2d-v4")
EASY_ENVS = ("Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4")
SEEDS = (0, 1, 2, 3, 4)
STRETCH_SEEDS = (0, 1, 2)
INTERLEAVED_K = (25_000, 50_000, 100_000)


def find_source_dir() -> Path:
    try:
        if (REMOTE_SRC_DIR / "exp4" / "train_bc.py").exists():
            return REMOTE_SRC_DIR
    except OSError:
        pass

    override = os.environ.get("PROJECT_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
        if (path / SRC_REL_DIR / "exp4" / "train_bc.py").exists():
            return path / SRC_REL_DIR
        raise FileNotFoundError(f"PROJECT_ROOT does not contain {SRC_REL_DIR}: {path}")

    candidates = [Path.cwd().resolve()]
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        source_dir = candidate / SRC_REL_DIR
        if (source_dir / "exp4" / "train_bc.py").exists():
            return source_dir
    raise FileNotFoundError(f"Could not locate {SRC_REL_DIR / 'exp4' / 'train_bc.py'}")


SOURCE_DIR = find_source_dir()

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "gymnasium[mujoco]",
        "mujoco",
        "wandb",
        "numpy<2",
        "matplotlib",
        "just-d4rl>=0.2407.5",
    )
    .add_local_dir(SOURCE_DIR, remote_path=str(REMOTE_SRC_DIR))
)


def run_command(command: list[str], wandb_project: str | None = None) -> None:
    env = os.environ.copy()
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project
    if "WANDB_API_KEY" not in env:
        env["WANDB_MODE"] = "offline"
    subprocess.run(command, check=True, env=env)


def latest_policy(policy_root: Path, prefix: str, policy_filename: str) -> str:
    candidates = sorted(policy_root.glob(f"{prefix}*/{policy_filename}"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No {policy_filename} found under {policy_root} with prefix {prefix}")
    return str(candidates[-1])


def parse_env_ids(env_ids: str) -> tuple[str, ...]:
    parsed = tuple(env_id.strip() for env_id in env_ids.split(",") if env_id.strip())
    if not parsed:
        raise ValueError("At least one environment id is required.")
    return parsed


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 4,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_bc_pretrain(
    env_id: str,
    total_updates: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
    save_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "bc_pretrain"),
) -> None:
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp4" / "train_bc.py"),
            "--env-id",
            env_id,
            "--seed",
            "0",
            "--total-updates",
            str(total_updates),
            "--eval-interval",
            str(eval_interval),
            "--num-eval-episodes",
            str(num_eval_episodes),
            "--save-dir",
            save_dir,
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_bc_pretrain__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_bc_to_sac(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    bc_anchor_interval: int,
    save_dir: str,
    wandb_project: str,
) -> None:
    env_slug = env_id.replace("-v", "_v").replace("-", "_")
    bc_policy_path = latest_policy(
        REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "bc_pretrain",
        f"bc__{env_slug}__",
        "bc_policy.pt",
    )
    group_prefix = "abhinav_bc_anchor_sac" if bc_anchor_interval > 0 else "abhinav_bc_to_sac"
    command = [
        "python",
        str(REMOTE_SRC_DIR / "exp0" / "train_sac.py"),
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
        "--bc-policy-path",
        bc_policy_path,
        "--offline-policy-source",
        "bc",
        "--bc-distill-steps",
        "500",
        "--wandb-project",
        wandb_project,
        "--wandb-group",
        f"{group_prefix}__{env_id}",
        "--track",
    ]
    if bc_anchor_interval > 0:
        command.extend(
            [
                "--bc-anchor-interval",
                str(bc_anchor_interval),
                "--bc-anchor-steps",
                "25",
                "--bc-anchor-start",
                "5000",
            ]
        )
    run_command(command, wandb_project)
    results_volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_bc_to_ppo(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    save_dir: str,
    wandb_project: str,
) -> None:
    env_slug = env_id.replace("-v", "_v").replace("-", "_")
    bc_policy_path = latest_policy(
        REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "bc_pretrain",
        f"bc__{env_slug}__",
        "bc_policy.pt",
    )
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp0" / "train_ppo.py"),
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
            "--bc-policy-path",
            bc_policy_path,
            "--offline-policy-source",
            "bc",
            "--bc-distill-steps",
            "500",
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_bc_to_ppo__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_bc_sac_ppo(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    save_dir: str,
    wandb_project: str,
) -> None:
    env_slug = env_id.replace("-v", "_v").replace("-", "_")
    bc_policy_path = latest_policy(
        REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "bc_pretrain",
        f"bc__{env_slug}__",
        "bc_policy.pt",
    )
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp2" / "train_handoff.py"),
            "--env-id",
            env_id,
            "--seed",
            str(seed),
            "--total-timesteps",
            str(total_timesteps),
            "--switch-fraction",
            "0.5",
            "--policy-init",
            "distill",
            "--value-init",
            "self-warmup",
            "--policy-source",
            "sac",
            "--bc-policy-path",
            bc_policy_path,
            "--offline-policy-source",
            "bc",
            "--eval-interval",
            str(eval_interval),
            "--num-eval-episodes",
            str(num_eval_episodes),
            "--save-dir",
            save_dir,
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_bc_sac_ppo__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 6,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_awac_pretrain(
    env_id: str,
    total_updates: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
    save_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "awac_pretrain"),
) -> None:
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp4" / "train_awac.py"),
            "--env-id",
            env_id,
            "--seed",
            "0",
            "--total-updates",
            str(total_updates),
            "--eval-interval",
            str(eval_interval),
            "--num-eval-episodes",
            str(num_eval_episodes),
            "--save-dir",
            save_dir,
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_awac_pretrain__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_awac_to_sac(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    save_dir: str,
    wandb_project: str,
) -> None:
    env_slug = env_id.replace("-v", "_v").replace("-", "_")
    awac_policy_path = latest_policy(
        REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "awac_pretrain",
        f"awac__{env_slug}__",
        "awac_policy.pt",
    )
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp0" / "train_sac.py"),
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
            "--bc-policy-path",
            awac_policy_path,
            "--offline-policy-source",
            "awac",
            "--bc-distill-steps",
            "500",
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_awac_to_sac__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_awac_to_ppo(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    save_dir: str,
    wandb_project: str,
) -> None:
    env_slug = env_id.replace("-v", "_v").replace("-", "_")
    awac_policy_path = latest_policy(
        REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "awac_pretrain",
        f"awac__{env_slug}__",
        "awac_policy.pt",
    )
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp0" / "train_ppo.py"),
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
            "--bc-policy-path",
            awac_policy_path,
            "--offline-policy-source",
            "awac",
            "--bc-distill-steps",
            "500",
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_awac_to_ppo__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_sac_baseline(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
    save_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "sac_baseline_extended"),
) -> None:
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp0" / "train_sac.py"),
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
            f"abhinav_sac_baseline__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 14,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_bc_to_sac_chain(
    env_id: str,
    seed: int,
    total_timesteps: int,
    bc_updates: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
    pretrain_save_root: str = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "bc_pretrain_extended"),
    transfer_save_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "bc_to_sac_extended"),
) -> None:
    env_slug = env_id.replace("-v", "_v").replace("-", "_")
    pretrain_save_dir = str(Path(pretrain_save_root) / f"{env_slug}__seed_{seed}")
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp4" / "train_bc.py"),
            "--env-id",
            env_id,
            "--seed",
            str(seed),
            "--total-updates",
            str(bc_updates),
            "--eval-interval",
            str(eval_interval),
            "--num-eval-episodes",
            str(num_eval_episodes),
            "--save-dir",
            pretrain_save_dir,
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_bc_pretrain_extended__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()

    bc_policy_path = latest_policy(
        Path(pretrain_save_dir),
        f"bc__{env_slug}__seed_{seed}__",
        "bc_policy.pt",
    )
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp0" / "train_sac.py"),
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
            transfer_save_dir,
            "--bc-policy-path",
            bc_policy_path,
            "--offline-policy-source",
            "bc",
            "--bc-distill-steps",
            "500",
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_bc_to_sac_extended__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_easy_sac_pretrain(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
    save_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "easy_sac_pretrain"),
) -> None:
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp0" / "train_sac.py"),
            "--env-id",
            env_id,
            "--seed",
            str(seed),
            "--total-timesteps",
            str(total_timesteps),
            "--easy-env-mode",
            "ignore_termination",
            "--eval-interval",
            str(eval_interval),
            "--num-eval-episodes",
            str(num_eval_episodes),
            "--save-dir",
            save_dir,
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_easy_sac_pretrain__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_easy_sac_to_sac(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    save_dir: str,
    wandb_project: str,
) -> None:
    env_slug = env_id.replace("-v", "_v").replace("-", "_")
    source_policy_path = latest_policy(
        REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "easy_sac_pretrain",
        f"easy_sac__{env_slug}__seed_{seed}__",
        f"checkpoint_step_{total_timesteps}.pt",
    )
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp0" / "train_sac.py"),
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
            "--bc-policy-path",
            source_policy_path,
            "--offline-policy-source",
            "easy_sac",
            "--bc-distill-steps",
            "500",
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_easy_sac_to_sac__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 20,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_easy_sac_chain(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
    pretrain_save_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "easy_sac_pretrain"),
    transfer_save_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "easy_transfer_extended"),
) -> None:
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp0" / "train_sac.py"),
            "--env-id",
            env_id,
            "--seed",
            str(seed),
            "--total-timesteps",
            str(total_timesteps),
            "--easy-env-mode",
            "ignore_termination",
            "--eval-interval",
            str(eval_interval),
            "--num-eval-episodes",
            str(num_eval_episodes),
            "--save-dir",
            pretrain_save_dir,
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_easy_sac_pretrain__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()

    env_slug = env_id.replace("-v", "_v").replace("-", "_")
    source_policy_path = latest_policy(
        Path(pretrain_save_dir),
        f"easy_sac__{env_slug}__seed_{seed}__",
        f"checkpoint_step_{total_timesteps}.pt",
    )
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp0" / "train_sac.py"),
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
            transfer_save_dir,
            "--bc-policy-path",
            source_policy_path,
            "--offline-policy-source",
            "easy_sac",
            "--bc-distill-steps",
            "500",
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_easy_sac_to_sac__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60,
    cpu=2.0,
    memory=8192,
)
def summarize_results(
    results_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task"),
    output_dir: str = str(REMOTE_RESULTS_DIR / "processed" / "abhinav_task"),
    notes_path: str = str(REMOTE_RESULTS_DIR / "processed" / "abhinav_task" / "results.md"),
) -> None:
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp4" / "summarize_experiment_4.py"),
            "--results-dir",
            results_dir,
            "--output-dir",
            output_dir,
            "--notes-path",
            notes_path,
        ]
    )
    results_volume.commit()


def run_specs_batched(specs: list[tuple], max_parallel_gpu: int) -> None:
    if max_parallel_gpu < 1:
        raise ValueError("max_parallel_gpu must be >= 1.")
    for batch_start in range(0, len(specs), max_parallel_gpu):
        batch = specs[batch_start : batch_start + max_parallel_gpu]
        calls = [fn.spawn(**kwargs) for fn, kwargs in batch]
        print(f"Started batch {batch_start // max_parallel_gpu + 1}: {len(calls)} GPU jobs.")
        for call in calls:
            call.get()
        print(f"Completed batch {batch_start // max_parallel_gpu + 1}.")


def core_specs(total_timesteps: int, eval_interval: int, num_eval_episodes: int, wandb_project: str) -> list:
    specs = []
    save_dir = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "tier1")
    for env_id in ENVS:
        for seed in SEEDS:
            specs.append(
                (
                    run_bc_to_sac,
                    {
                        "env_id": env_id,
                        "seed": seed,
                        "total_timesteps": total_timesteps,
                        "eval_interval": eval_interval,
                        "num_eval_episodes": num_eval_episodes,
                        "bc_anchor_interval": 0,
                        "save_dir": save_dir,
                        "wandb_project": wandb_project,
                    },
                )
            )
            specs.append(
                (
                    run_bc_to_ppo,
                    {
                        "env_id": env_id,
                        "seed": seed,
                        "total_timesteps": total_timesteps,
                        "eval_interval": eval_interval,
                        "num_eval_episodes": num_eval_episodes,
                        "save_dir": save_dir,
                        "wandb_project": wandb_project,
                    },
                )
            )
            specs.append(
                (
                    run_bc_sac_ppo,
                    {
                        "env_id": env_id,
                        "seed": seed,
                        "total_timesteps": total_timesteps,
                        "eval_interval": eval_interval,
                        "num_eval_episodes": num_eval_episodes,
                        "save_dir": save_dir,
                        "wandb_project": wandb_project,
                    },
                )
            )
    return specs


def interleaved_specs(total_timesteps: int, eval_interval: int, num_eval_episodes: int, wandb_project: str) -> list:
    specs = []
    save_dir = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "interleaved_bc")
    for interval in INTERLEAVED_K:
        for seed in STRETCH_SEEDS:
            specs.append(
                (
                    run_bc_to_sac,
                    {
                        "env_id": "Hopper-v4",
                        "seed": seed,
                        "total_timesteps": total_timesteps,
                        "eval_interval": eval_interval,
                        "num_eval_episodes": num_eval_episodes,
                        "bc_anchor_interval": interval,
                        "save_dir": save_dir,
                        "wandb_project": wandb_project,
                    },
                )
            )
    return specs


def interleaved_walker_specs(
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
    bc_anchor_interval: int,
) -> list:
    specs = []
    save_dir = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "interleaved_bc_walker")
    for seed in STRETCH_SEEDS:
        specs.append(
            (
                run_bc_to_sac,
                {
                    "env_id": "Walker2d-v4",
                    "seed": seed,
                    "total_timesteps": total_timesteps,
                    "eval_interval": eval_interval,
                    "num_eval_episodes": num_eval_episodes,
                    "bc_anchor_interval": bc_anchor_interval,
                    "save_dir": save_dir,
                    "wandb_project": wandb_project,
                },
            )
        )
    return specs


def long_specs(long_timesteps: int, eval_interval: int, num_eval_episodes: int, wandb_project: str) -> list:
    specs = []
    save_dir = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "long_horizon")
    for seed in STRETCH_SEEDS:
        specs.append(
            (
                run_bc_to_sac,
                {
                    "env_id": "Hopper-v4",
                    "seed": seed,
                    "total_timesteps": long_timesteps,
                    "eval_interval": eval_interval,
                    "num_eval_episodes": num_eval_episodes,
                    "bc_anchor_interval": 0,
                    "save_dir": save_dir,
                    "wandb_project": wandb_project,
                },
            )
        )
        specs.append(
            (
                run_bc_to_ppo,
                {
                    "env_id": "Hopper-v4",
                    "seed": seed,
                    "total_timesteps": long_timesteps,
                    "eval_interval": eval_interval,
                    "num_eval_episodes": num_eval_episodes,
                    "save_dir": save_dir,
                    "wandb_project": wandb_project,
                },
            )
        )
    return specs


def easy_pretrain_specs(
    env_ids: tuple[str, ...],
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
) -> list:
    specs = []
    for env_id in env_ids:
        for seed in STRETCH_SEEDS:
            specs.append(
                (
                    run_easy_sac_pretrain,
                    {
                        "env_id": env_id,
                        "seed": seed,
                        "total_timesteps": total_timesteps,
                        "eval_interval": eval_interval,
                        "num_eval_episodes": num_eval_episodes,
                        "wandb_project": wandb_project,
                    },
                )
            )
    return specs


def easy_transfer_specs(
    env_ids: tuple[str, ...],
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
) -> list:
    specs = []
    save_dir = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "easy_transfer_extended")
    for env_id in env_ids:
        for seed in STRETCH_SEEDS:
            specs.append(
                (
                    run_easy_sac_to_sac,
                    {
                        "env_id": env_id,
                        "seed": seed,
                        "total_timesteps": total_timesteps,
                        "eval_interval": eval_interval,
                        "num_eval_episodes": num_eval_episodes,
                        "save_dir": save_dir,
                        "wandb_project": wandb_project,
                    },
                )
            )
    return specs


def easy_chain_specs(
    env_ids: tuple[str, ...],
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
) -> list:
    specs = []
    for env_id in env_ids:
        for seed in STRETCH_SEEDS:
            specs.append(
                (
                    run_easy_sac_chain,
                    {
                        "env_id": env_id,
                        "seed": seed,
                        "total_timesteps": total_timesteps,
                        "eval_interval": eval_interval,
                        "num_eval_episodes": num_eval_episodes,
                        "wandb_project": wandb_project,
                    },
                )
            )
    return specs


def sac_baseline_specs(
    env_ids: tuple[str, ...],
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
) -> list:
    specs = []
    for env_id in env_ids:
        for seed in STRETCH_SEEDS:
            specs.append(
                (
                    run_sac_baseline,
                    {
                        "env_id": env_id,
                        "seed": seed,
                        "total_timesteps": total_timesteps,
                        "eval_interval": eval_interval,
                        "num_eval_episodes": num_eval_episodes,
                        "wandb_project": wandb_project,
                    },
                )
            )
    return specs


def bc_to_sac_chain_specs(
    env_ids: tuple[str, ...],
    total_timesteps: int,
    bc_updates: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
) -> list:
    specs = []
    for env_id in env_ids:
        for seed in STRETCH_SEEDS:
            specs.append(
                (
                    run_bc_to_sac_chain,
                    {
                        "env_id": env_id,
                        "seed": seed,
                        "total_timesteps": total_timesteps,
                        "bc_updates": bc_updates,
                        "eval_interval": eval_interval,
                        "num_eval_episodes": num_eval_episodes,
                        "wandb_project": wandb_project,
                    },
                )
            )
    return specs


def spawn_specs(specs: list[tuple]) -> list[str]:
    calls = [fn.spawn(**kwargs) for fn, kwargs in specs]
    return [str(getattr(call, "object_id", None) or getattr(call, "id", None) or call) for call in calls]


@app.local_entrypoint()
def main(
    mode: str = "core",
    total_timesteps: int = 500_000,
    long_timesteps: int = 1_000_000,
    bc_updates: int = 50_000,
    awac_updates: int = 100_000,
    eval_interval: int = 5_000,
    num_eval_episodes: int = 5,
    wandb_project: str = "rl-translational-dynamics",
    skip_bc_pretrain: bool = False,
    max_parallel_gpu: int = 10,
    interleaved_walker_interval: int = 50_000,
    easy_env_ids: str = ",".join(EASY_ENVS),
    single_seed: int = 0,
) -> None:
    if mode == "summarize":
        summarize_results.remote()
        return
    if mode == "summarize-interleaved":
        summarize_results.remote(
            results_dir=str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "interleaved_bc"),
            output_dir=str(REMOTE_RESULTS_DIR / "processed" / "abhinav_task" / "interleaved_bc"),
            notes_path=str(REMOTE_RESULTS_DIR / "processed" / "abhinav_task" / "interleaved_bc" / "results.md"),
        )
        return

    easy_envs = parse_env_ids(easy_env_ids)

    if mode in {"easy-full", "easy-chain"}:
        specs = easy_chain_specs(easy_envs, total_timesteps, eval_interval, num_eval_episodes, wandb_project)
        call_ids = spawn_specs(specs)
        print(f"Spawned {len(call_ids)} detached easy SAC -> SAC chain jobs.")
        print("Each chain runs easy-env SAC pretraining and then real-env SAC transfer for one env/seed.")
        return

    if mode in {"baseline-sac", "sac-baseline"}:
        specs = sac_baseline_specs(easy_envs, total_timesteps, eval_interval, num_eval_episodes, wandb_project)
        call_ids = spawn_specs(specs)
        print(f"Spawned {len(call_ids)} detached SAC baseline jobs.")
        return

    if mode in {"bc-to-sac", "bc-sac"}:
        specs = bc_to_sac_chain_specs(easy_envs, total_timesteps, bc_updates, eval_interval, num_eval_episodes, wandb_project)
        call_ids = spawn_specs(specs)
        print(f"Spawned {len(call_ids)} detached BC -> SAC chain jobs.")
        return

    if mode in {"bc-to-sac-wait", "bc-sac-wait"}:
        specs = bc_to_sac_chain_specs(easy_envs, total_timesteps, bc_updates, eval_interval, num_eval_episodes, wandb_project)
        print(f"Starting {len(specs)} BC -> SAC chain jobs and waiting for completion.")
        run_specs_batched(specs, max_parallel_gpu)
        return

    if mode in {"bc-to-sac-one", "bc-sac-one"}:
        if len(easy_envs) != 1:
            raise ValueError("bc-to-sac-one expects exactly one --easy-env-ids value.")
        call = run_bc_to_sac_chain.spawn(
            env_id=easy_envs[0],
            seed=single_seed,
            total_timesteps=total_timesteps,
            bc_updates=bc_updates,
            eval_interval=eval_interval,
            num_eval_episodes=num_eval_episodes,
            wandb_project=wandb_project,
        )
        print(f"Spawned detached BC -> SAC chain for {easy_envs[0]} seed {single_seed}: {call}")
        return

    if mode == "easy-pretrain":
        specs = easy_pretrain_specs(easy_envs, total_timesteps, eval_interval, num_eval_episodes, wandb_project)
        run_specs_batched(specs, max_parallel_gpu)
        return

    if mode == "tier2-pretrain":
        specs = [
            (
                run_awac_pretrain,
                {
                    "env_id": env_id,
                    "total_updates": awac_updates,
                    "eval_interval": eval_interval,
                    "num_eval_episodes": num_eval_episodes,
                    "wandb_project": wandb_project,
                },
            )
            for env_id in ENVS
        ]
        run_specs_batched(specs, max_parallel_gpu)
        return

    if mode in {"core", "interleaved", "interleaved-walker", "long", "all"} and not skip_bc_pretrain:
        for env_id in ENVS:
            run_bc_pretrain.remote(
                env_id=env_id,
                total_updates=bc_updates,
                eval_interval=eval_interval,
                num_eval_episodes=num_eval_episodes,
                wandb_project=wandb_project,
            )

    specs = []
    if mode in {"core", "all"}:
        specs.extend(core_specs(total_timesteps, eval_interval, num_eval_episodes, wandb_project))
    if mode in {"interleaved", "all"}:
        specs.extend(interleaved_specs(total_timesteps, eval_interval, num_eval_episodes, wandb_project))
    if mode in {"interleaved-walker", "all"}:
        specs.extend(
            interleaved_walker_specs(
                total_timesteps,
                eval_interval,
                num_eval_episodes,
                wandb_project,
                interleaved_walker_interval,
            )
        )
    if mode in {"long", "all"}:
        specs.extend(long_specs(long_timesteps, eval_interval, num_eval_episodes, wandb_project))
    if mode in {"easy-full", "all"}:
        print(f"Running easy SAC pretraining for {len(easy_envs)} envs before transfer.")
        run_specs_batched(
            easy_pretrain_specs(easy_envs, total_timesteps, eval_interval, num_eval_episodes, wandb_project),
            max_parallel_gpu,
        )
    if mode in {"easy", "easy-transfer", "easy-full", "all"}:
        specs.extend(easy_transfer_specs(easy_envs, total_timesteps, eval_interval, num_eval_episodes, wandb_project))
    if mode == "all":
        for env_id in ENVS:
            run_awac_pretrain.remote(
                env_id=env_id,
                total_updates=awac_updates,
                eval_interval=eval_interval,
                num_eval_episodes=num_eval_episodes,
                wandb_project=wandb_project,
            )
    if mode in {"tier2", "tier2-transfer", "all"}:
        save_dir = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "tier2_awac")
        for env_id in ENVS:
            for seed in STRETCH_SEEDS:
                specs.append(
                    (
                        run_awac_to_sac,
                        {
                            "env_id": env_id,
                            "seed": seed,
                            "total_timesteps": total_timesteps,
                            "eval_interval": eval_interval,
                            "num_eval_episodes": num_eval_episodes,
                            "save_dir": save_dir,
                            "wandb_project": wandb_project,
                        },
                    )
                )
                specs.append(
                    (
                        run_awac_to_ppo,
                        {
                            "env_id": env_id,
                            "seed": seed,
                            "total_timesteps": total_timesteps,
                            "eval_interval": eval_interval,
                            "num_eval_episodes": num_eval_episodes,
                            "save_dir": save_dir,
                            "wandb_project": wandb_project,
                        },
                    )
                )

    print(f"Running {len(specs)} Abhinav experiment jobs in batches of at most {max_parallel_gpu}.")
    run_specs_batched(specs, max_parallel_gpu)
