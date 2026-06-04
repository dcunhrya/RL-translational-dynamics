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


@app.function(
    image=image,
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
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 6,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_iql_pretrain(
    env_id: str,
    total_updates: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
    save_dir: str = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "iql_pretrain"),
) -> None:
    run_command(
        [
            "python",
            str(REMOTE_SRC_DIR / "exp4" / "train_iql.py"),
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
            f"abhinav_iql_pretrain__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
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
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_iql_to_sac(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    save_dir: str,
    wandb_project: str,
) -> None:
    env_slug = env_id.replace("-v", "_v").replace("-", "_")
    iql_policy_path = latest_policy(
        REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "iql_pretrain",
        f"iql__{env_slug}__",
        "iql_policy.pt",
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
            iql_policy_path,
            "--offline-policy-source",
            "iql",
            "--bc-distill-steps",
            "500",
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_iql_to_sac__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_iql_to_ppo(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    save_dir: str,
    wandb_project: str,
) -> None:
    env_slug = env_id.replace("-v", "_v").replace("-", "_")
    iql_policy_path = latest_policy(
        REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "iql_pretrain",
        f"iql__{env_slug}__",
        "iql_policy.pt",
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
            iql_policy_path,
            "--offline-policy-source",
            "iql",
            "--bc-distill-steps",
            "500",
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_iql_to_ppo__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_iql_sac_ppo(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    save_dir: str,
    wandb_project: str,
) -> None:
    env_slug = env_id.replace("-v", "_v").replace("-", "_")
    iql_policy_path = latest_policy(
        REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "iql_pretrain",
        f"iql__{env_slug}__",
        "iql_policy.pt",
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
            iql_policy_path,
            "--offline-policy-source",
            "iql",
            "--eval-interval",
            str(eval_interval),
            "--num-eval-episodes",
            str(num_eval_episodes),
            "--save-dir",
            save_dir,
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            f"abhinav_iql_sac_ppo__{env_id}",
            "--track",
        ],
        wandb_project,
    )
    results_volume.commit()


@app.function(
    image=image,
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


def easy_transfer_specs(total_timesteps: int, eval_interval: int, num_eval_episodes: int, wandb_project: str) -> list:
    specs = []
    save_dir = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "easy_transfer")
    for seed in STRETCH_SEEDS:
        specs.append(
            (
                run_easy_sac_to_sac,
                {
                    "env_id": "Hopper-v4",
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


def iql_transfer_specs(total_timesteps: int, eval_interval: int, num_eval_episodes: int, wandb_project: str) -> list:
    specs = []
    save_dir = str(REMOTE_RESULTS_DIR / "raw" / "abhinav_task" / "tier2_iql")
    for env_id in ENVS:
        for seed in STRETCH_SEEDS:
            specs.append(
                (
                    run_iql_to_sac,
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
                    run_iql_to_ppo,
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
                    run_iql_sac_ppo,
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


@app.local_entrypoint()
def main(
    mode: str = "core",
    total_timesteps: int = 500_000,
    long_timesteps: int = 1_000_000,
    bc_updates: int = 50_000,
    awac_updates: int = 100_000,
    iql_updates: int = 100_000,
    eval_interval: int = 5_000,
    num_eval_episodes: int = 5,
    wandb_project: str = "rl-translational-dynamics",
    skip_bc_pretrain: bool = False,
    max_parallel_gpu: int = 10,
    interleaved_walker_interval: int = 50_000,
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

    if mode == "easy-pretrain":
        specs = [
            (
                run_easy_sac_pretrain,
                {
                    "env_id": "Hopper-v4",
                    "seed": seed,
                    "total_timesteps": total_timesteps,
                    "eval_interval": eval_interval,
                    "num_eval_episodes": num_eval_episodes,
                    "wandb_project": wandb_project,
                },
            )
            for seed in STRETCH_SEEDS
        ]
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

    if mode in {"iql-pretrain", "iql", "all"}:
        specs = [
            (
                run_iql_pretrain,
                {
                    "env_id": env_id,
                    "total_updates": iql_updates,
                    "eval_interval": eval_interval,
                    "num_eval_episodes": num_eval_episodes,
                    "wandb_project": wandb_project,
                },
            )
            for env_id in ENVS
        ]
        run_specs_batched(specs, max_parallel_gpu)
        if mode == "iql-pretrain":
            return

    if mode in {"core", "interleaved", "interleaved-walker", "long", "all"} and not skip_bc_pretrain:
        specs = [
            (
                run_bc_pretrain,
                {
                    "env_id": env_id,
                    "total_updates": bc_updates,
                    "eval_interval": eval_interval,
                    "num_eval_episodes": num_eval_episodes,
                    "wandb_project": wandb_project,
                },
            )
            for env_id in ENVS
        ]
        run_specs_batched(specs, max_parallel_gpu)

    if mode == "all":
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
    if mode == "all":
        for seed in STRETCH_SEEDS:
            run_easy_sac_pretrain.remote(
                env_id="Hopper-v4",
                seed=seed,
                total_timesteps=total_timesteps,
                eval_interval=eval_interval,
                num_eval_episodes=num_eval_episodes,
                wandb_project=wandb_project,
            )
    if mode in {"easy", "easy-transfer", "all"}:
        specs.extend(easy_transfer_specs(total_timesteps, eval_interval, num_eval_episodes, wandb_project))
    if mode in {"iql", "iql-transfer", "all"}:
        specs.extend(iql_transfer_specs(total_timesteps, eval_interval, num_eval_episodes, wandb_project))
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
