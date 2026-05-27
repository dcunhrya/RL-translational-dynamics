import json
import os
import subprocess
import time
from pathlib import Path

import modal


APP_NAME = "rl-translational-dynamics-ryan"
WANDB_SECRET_NAME = os.environ.get("WANDB_MODAL_SECRET", "wandb-api-key")
RESULTS_VOLUME_NAME = os.environ.get("MODAL_RESULTS_VOLUME", "herschethan")
SRC_REL_DIR = Path("src") / "RL-translational-dynamics"
REMOTE_SRC_DIR = Path("/root/project/src/RL-translational-dynamics")
REMOTE_RESULTS_DIR = Path("/root/results")
DEFAULT_GPU = os.environ.get("RYAN_MODAL_GPU", "L4")

ENVS = ("Hopper-v4", "Walker2d-v4")
SEEDS = (0, 1, 2, 3, 4)
VALUE_INITS = ("random", "self-warmup", "source-aligned")
SMOKE_STEPS = 50_000
FULL_STEPS = 500_000
LONG_HORIZON_STEPS = 1_000_000


def find_source_dir() -> Path:
    if (REMOTE_SRC_DIR / "exp2" / "train_handoff.py").exists():
        return REMOTE_SRC_DIR

    override = os.environ.get("PROJECT_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
        if (path / SRC_REL_DIR / "exp2" / "train_handoff.py").exists():
            return path / SRC_REL_DIR
        raise FileNotFoundError(f"PROJECT_ROOT does not contain {SRC_REL_DIR}: {path}")

    candidates = [Path.cwd().resolve()]
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        source_dir = candidate / SRC_REL_DIR
        if (source_dir / "exp2" / "train_handoff.py").exists():
            return source_dir
    raise FileNotFoundError(f"Could not locate {SRC_REL_DIR / 'exp2' / 'train_handoff.py'}")


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
    )
    .add_local_dir(SOURCE_DIR, remote_path=str(REMOTE_SRC_DIR))
)


def run_command(command: list[str], wandb_project: str) -> None:
    env = os.environ.copy()
    env["WANDB_PROJECT"] = wandb_project
    subprocess.run(command, check=True, env=env)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_sac(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    save_dir: str,
    wandb_project: str,
    wandb_group: str,
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
            wandb_group,
            "--track",
        ],
        wandb_project,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_ppo(
    env_id: str,
    seed: int,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    save_dir: str,
    wandb_project: str,
    wandb_group: str,
) -> None:
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
            "--wandb-project",
            wandb_project,
            "--wandb-group",
            wandb_group,
            "--track",
        ],
        wandb_project,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    volumes={str(REMOTE_RESULTS_DIR): results_volume},
    timeout=60 * 60 * 10,
    cpu=4.0,
    memory=16384,
    gpu=DEFAULT_GPU,
)
def run_sac_to_ppo(
    env_id: str,
    seed: int,
    value_init: str,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    save_dir: str,
    wandb_project: str,
    wandb_group: str,
) -> None:
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
            value_init,
            "--policy-source",
            "sac",
            "--value-source",
            "sac" if value_init == "source-aligned" else "none",
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
        ],
        wandb_project,
    )


def call_id(call) -> str:
    return str(getattr(call, "object_id", None) or getattr(call, "id", None) or call)


def spawn_specs(
    mode: str,
    total_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    wandb_project: str,
) -> list[dict]:
    save_dir = str(REMOTE_RESULTS_DIR / "raw" / "ryan_experiment")
    specs = []
    if mode == "smoke":
        smoke_envs = ENVS
        smoke_seed = 0
        for env_id in smoke_envs:
            specs.append(
                {
                    "kind": "sac",
                    "env_id": env_id,
                    "seed": smoke_seed,
                    "total_timesteps": SMOKE_STEPS,
                    "eval_interval": 5_000,
                    "num_eval_episodes": num_eval_episodes,
                    "save_dir": save_dir,
                    "wandb_project": wandb_project,
                    "wandb_group": f"ryan_smoke__sac__{env_id}",
                }
            )
            specs.append(
                {
                    "kind": "ppo",
                    "env_id": env_id,
                    "seed": smoke_seed,
                    "total_timesteps": SMOKE_STEPS,
                    "eval_interval": 5_000,
                    "num_eval_episodes": num_eval_episodes,
                    "save_dir": save_dir,
                    "wandb_project": wandb_project,
                    "wandb_group": f"ryan_smoke__ppo__{env_id}",
                }
            )
            for value_init in VALUE_INITS:
                specs.append(
                    {
                        "kind": "sac_to_ppo",
                        "env_id": env_id,
                        "seed": smoke_seed,
                        "value_init": value_init,
                        "total_timesteps": SMOKE_STEPS,
                        "eval_interval": 5_000,
                        "num_eval_episodes": num_eval_episodes,
                        "save_dir": save_dir,
                        "wandb_project": wandb_project,
                        "wandb_group": f"ryan_smoke__sac_to_ppo__{env_id}__value_{value_init}",
                    }
                )
        return specs

    for env_id in ENVS:
        for seed in SEEDS:
            specs.append(
                {
                    "kind": "sac",
                    "env_id": env_id,
                    "seed": seed,
                    "total_timesteps": total_timesteps,
                    "eval_interval": eval_interval,
                    "num_eval_episodes": num_eval_episodes,
                    "save_dir": save_dir,
                    "wandb_project": wandb_project,
                    "wandb_group": f"ryan_full__sac__{env_id}__500k",
                }
            )
            specs.append(
                {
                    "kind": "ppo",
                    "env_id": env_id,
                    "seed": seed,
                    "total_timesteps": total_timesteps,
                    "eval_interval": eval_interval,
                    "num_eval_episodes": num_eval_episodes,
                    "save_dir": save_dir,
                    "wandb_project": wandb_project,
                    "wandb_group": f"ryan_full__ppo__{env_id}__500k",
                }
            )
            for value_init in VALUE_INITS:
                specs.append(
                    {
                        "kind": "sac_to_ppo",
                        "env_id": env_id,
                        "seed": seed,
                        "value_init": value_init,
                        "total_timesteps": total_timesteps,
                        "eval_interval": eval_interval,
                        "num_eval_episodes": num_eval_episodes,
                        "save_dir": save_dir,
                        "wandb_project": wandb_project,
                        "wandb_group": f"ryan_full__sac_to_ppo__{env_id}__value_{value_init}__500k",
                    }
                )

    for seed in (0, 1, 2):
        specs.append(
            {
                "kind": "sac",
                "env_id": "Hopper-v4",
                "seed": seed,
                "total_timesteps": LONG_HORIZON_STEPS,
                "eval_interval": eval_interval,
                "num_eval_episodes": num_eval_episodes,
                "save_dir": save_dir,
                "wandb_project": wandb_project,
                "wandb_group": "ryan_long_horizon__sac__Hopper-v4__1m",
            }
        )
    return specs


def spawn_job(spec: dict):
    kwargs = {key: value for key, value in spec.items() if key != "kind"}
    if spec["kind"] == "sac":
        return run_sac.spawn(**kwargs)
    if spec["kind"] == "ppo":
        return run_ppo.spawn(**kwargs)
    if spec["kind"] == "sac_to_ppo":
        return run_sac_to_ppo.spawn(**kwargs)
    raise ValueError(f"Unknown job kind: {spec['kind']}")


def write_manifest(manifest_path: Path, mode: str, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "app": APP_NAME,
        "mode": mode,
        "created_at_unix": int(time.time()),
        "gpu": DEFAULT_GPU,
        "job_count": len(rows),
        "jobs": rows,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@app.local_entrypoint()
def main(
    mode: str = "full",
    total_timesteps: int = FULL_STEPS,
    eval_interval: int = 10_000,
    num_eval_episodes: int = 5,
    wandb_project: str = "rl-translational-dynamics",
    manifest_path: str = "experiments/ryan_modal_manifest.json",
) -> None:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")

    rows = []
    calls = []
    for spec in spawn_specs(mode, total_timesteps, eval_interval, num_eval_episodes, wandb_project):
        call = spawn_job(spec)
        rows.append({**spec, "modal_call_id": call_id(call)})
        calls.append(call)

    write_manifest(Path(manifest_path), mode, rows)
    print(f"Spawned {len(rows)} Ryan Modal jobs in {mode} mode.")
    print(f"Wrote manifest to {manifest_path}.")

    if mode == "smoke":
        for call in calls:
            call.get()
        print("Ryan Modal smoke jobs completed successfully.")
