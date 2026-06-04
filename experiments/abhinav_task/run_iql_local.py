#!/usr/bin/env python3
"""Run Abhinav IQL transfer experiments on local GPUs.

This mirrors the Modal IQL transfer grid while keeping jobs on local CUDA
devices. It skips any run that already reached the requested budget.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ENVS = ("Hopper-v4", "Walker2d-v4")
SEEDS = (0, 1, 2, 3, 4)
ALGOS = ("iql_to_sac", "iql_to_ppo", "iql_to_sac_to_ppo")


@dataclass(frozen=True)
class Job:
    algo: str
    env_id: str
    seed: int
    command: list[str]
    log_path: Path

    @property
    def name(self) -> str:
        env_slug = slug_env(self.env_id)
        return f"{self.algo}__{env_slug}__seed_{self.seed}"


@dataclass
class RunningJob:
    job: Job
    process: subprocess.Popen
    gpu: str
    log_file: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IQL transfer jobs on local CUDA devices.")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--num-eval-episodes", type=int, default=5)
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated CUDA device IDs to use.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/raw/abhinav_task"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/abhinav_task/iql_local"))
    parser.add_argument("--wandb-project", type=str, default="rl-translational-dynamics")
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def slug_env(env_id: str) -> str:
    return env_id.replace("-v", "_v").replace("-", "_")


def latest_iql_policy(pretrain_dir: Path, env_id: str) -> Path:
    env_slug = slug_env(env_id)
    candidates = sorted(
        pretrain_dir.glob(f"iql__{env_slug}__*/iql_policy.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No IQL policy found under {pretrain_dir} for {env_id}.")
    return candidates[-1]


def run_is_complete(save_dir: Path, algo: str, env_id: str, seed: int, total_timesteps: int) -> bool:
    for metrics_path in save_dir.rglob("metrics.jsonl"):
        max_env_steps = 0
        saw_matching_run = False
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("algorithm") != algo or row.get("env") != env_id or int(row.get("seed", -1)) != seed:
                        continue
                    if int(row.get("total_timesteps", total_timesteps)) != total_timesteps:
                        continue
                    saw_matching_run = True
                    max_env_steps = max(max_env_steps, int(row.get("env_steps", 0)))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if saw_matching_run and max_env_steps >= total_timesteps:
            return True
    return False


def build_jobs(args: argparse.Namespace, repo_root: Path) -> list[Job]:
    source_dir = repo_root / "src" / "RL-translational-dynamics"
    pretrain_dir = args.results_dir / "iql_pretrain"
    transfer_dir = args.results_dir / "tier2_iql"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[Job] = []
    for env_id in ENVS:
        iql_policy = latest_iql_policy(pretrain_dir, env_id)
        for seed in SEEDS:
            for algo in ALGOS:
                if run_is_complete(transfer_dir, algo, env_id, seed, args.total_timesteps):
                    print(f"skip complete: {algo} {env_id} seed={seed}")
                    continue

                common = [
                    "--env-id",
                    env_id,
                    "--seed",
                    str(seed),
                    "--total-timesteps",
                    str(args.total_timesteps),
                    "--eval-interval",
                    str(args.eval_interval),
                    "--num-eval-episodes",
                    str(args.num_eval_episodes),
                    "--save-dir",
                    str(transfer_dir),
                    "--bc-policy-path",
                    str(iql_policy),
                    "--offline-policy-source",
                    "iql",
                    "--wandb-project",
                    args.wandb_project,
                ]
                if args.track:
                    common.append("--track")

                if algo == "iql_to_sac":
                    command = [
                        sys.executable,
                        str(source_dir / "exp0" / "train_sac.py"),
                        *common,
                        "--bc-distill-steps",
                        "500",
                        "--wandb-group",
                        f"abhinav_iql_to_sac__{env_id}",
                    ]
                elif algo == "iql_to_ppo":
                    command = [
                        sys.executable,
                        str(source_dir / "exp0" / "train_ppo.py"),
                        *common,
                        "--bc-distill-steps",
                        "500",
                        "--wandb-group",
                        f"abhinav_iql_to_ppo__{env_id}",
                    ]
                elif algo == "iql_to_sac_to_ppo":
                    command = [
                        sys.executable,
                        str(source_dir / "exp2" / "train_handoff.py"),
                        "--env-id",
                        env_id,
                        "--seed",
                        str(seed),
                        "--total-timesteps",
                        str(args.total_timesteps),
                        "--switch-fraction",
                        "0.5",
                        "--policy-init",
                        "distill",
                        "--value-init",
                        "self-warmup",
                        "--policy-source",
                        "sac",
                        "--bc-policy-path",
                        str(iql_policy),
                        "--offline-policy-source",
                        "iql",
                        "--eval-interval",
                        str(args.eval_interval),
                        "--num-eval-episodes",
                        str(args.num_eval_episodes),
                        "--save-dir",
                        str(transfer_dir),
                        "--wandb-project",
                        args.wandb_project,
                        "--wandb-group",
                        f"abhinav_iql_sac_ppo__{env_id}",
                    ]
                    if args.track:
                        command.append("--track")
                else:
                    raise ValueError(f"Unknown algo: {algo}")

                log_path = args.log_dir / f"{algo}__{slug_env(env_id)}__seed_{seed}.log"
                jobs.append(Job(algo=algo, env_id=env_id, seed=seed, command=command, log_path=log_path))
    return jobs


def launch(job: Job, gpu: str, repo_root: Path) -> RunningJob:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_file = job.log_path.open("a", encoding="utf-8")
    log_file.write(f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} ===\n")
    log_file.write(" ".join(job.command) + "\n")
    log_file.flush()
    process = subprocess.Popen(
        job.command,
        cwd=repo_root,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"started gpu={gpu}: {job.name} pid={process.pid} log={job.log_path}")
    return RunningJob(job=job, process=process, gpu=gpu, log_file=log_file)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU must be provided.")

    jobs = build_jobs(args, repo_root)
    print(f"pending jobs: {len(jobs)}")
    if args.dry_run:
        for job in jobs:
            print(job.name)
            print("  " + " ".join(job.command))
        return 0

    running: list[RunningJob] = []
    free_gpus = list(gpus)
    failed: list[tuple[Job, int]] = []
    pending = list(jobs)

    try:
        while pending or running:
            while pending and free_gpus and not (args.fail_fast and failed):
                gpu = free_gpus.pop(0)
                running.append(launch(pending.pop(0), gpu, repo_root))

            time.sleep(10)
            still_running: list[RunningJob] = []
            for item in running:
                return_code = item.process.poll()
                if return_code is None:
                    still_running.append(item)
                    continue
                item.log_file.close()
                free_gpus.append(item.gpu)
                if return_code == 0:
                    print(f"finished gpu={item.gpu}: {item.job.name}")
                else:
                    failed.append((item.job, return_code))
                    print(f"failed rc={return_code} gpu={item.gpu}: {item.job.name} log={item.job.log_path}")
            running = still_running

            if args.fail_fast and failed and pending and not running:
                break
    finally:
        for item in running:
            item.log_file.flush()

    if failed:
        print("failed jobs:")
        for job, return_code in failed:
            print(f"  rc={return_code} {job.name} log={job.log_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
