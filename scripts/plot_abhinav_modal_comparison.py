#!/usr/bin/env python3
"""Pull W&B histories and plot SAC/easy-SAC/BC-SAC comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


ENTITY_PROJECT = "herschethan-stanford-university/rl-translational-dynamics"
DEFAULT_ENV_ORDER = ("Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4")
ENV_ORDER = DEFAULT_ENV_ORDER
METHOD_ORDER = ("sac", "easy_sac_to_sac", "bc_to_sac")
METHOD_LABELS = {
    "sac": "SAC",
    "easy_sac_to_sac": "Easy SAC -> SAC",
    "bc_to_sac": "BC -> SAC",
}
METHOD_COLORS = {
    "sac": "#1f77b4",
    "easy_sac_to_sac": "#2ca02c",
    "bc_to_sac": "#d62728",
}
HORIZON_STEPS = 500_000
STEP_GRID = np.arange(0, HORIZON_STEPS + 1, 5_000, dtype=float)
KNOWN_CURRENT_RUNS = (
    ("yl0u6j9q", "easy_sac_to_sac", "Hopper-v4", 2),
    ("36j25b59", "easy_sac_to_sac", "Hopper-v4", 0),
    ("niq5tv91", "easy_sac_to_sac", "Hopper-v4", 1),
    ("pf3nfms1", "easy_sac_to_sac", "Walker2d-v4", 2),
    ("grt59pjj", "easy_sac_to_sac", "Walker2d-v4", 0),
    ("zvtpb9si", "easy_sac_to_sac", "Walker2d-v4", 1),
    ("erq3snpb", "easy_sac_to_sac", "HalfCheetah-v4", 2),
    ("dwxzqiby", "easy_sac_to_sac", "HalfCheetah-v4", 0),
    ("17q4f639", "easy_sac_to_sac", "HalfCheetah-v4", 1),
    ("bvlmnueq", "easy_sac_to_sac", "Ant-v4", 0),
    ("swt89brj", "easy_sac_to_sac", "Ant-v4", 1),
    ("x9japip8", "easy_sac_to_sac", "Ant-v4", 2),
    ("u4gyp7tu", "sac", "Hopper-v4", 0),
    ("2xvqqrkg", "sac", "Hopper-v4", 1),
    ("k8bjar87", "sac", "Hopper-v4", 2),
    ("b73tlkvm", "sac", "Walker2d-v4", 0),
    ("pfqbddsn", "sac", "Walker2d-v4", 1),
    ("yzi9mx7h", "sac", "Walker2d-v4", 2),
    ("hjkv57oe", "sac", "HalfCheetah-v4", 0),
    ("ubrdc023", "sac", "HalfCheetah-v4", 1),
    ("mfux8kin", "sac", "HalfCheetah-v4", 2),
    ("sahxozyd", "sac", "Ant-v4", 0),
    ("x9h1f1xg", "sac", "Ant-v4", 1),
    ("azqyzs4c", "sac", "Ant-v4", 2),
    ("xr1z9xgt", "bc_to_sac", "HalfCheetah-v4", 0),
    ("xvboy0u5", "bc_to_sac", "HalfCheetah-v4", 1),
    ("mfok9kei", "bc_to_sac", "HalfCheetah-v4", 2),
)


@dataclass(frozen=True)
class RunRef:
    run_id: str
    name: str
    group: str
    state: str
    created_at: str
    method: str
    env: str
    seed: int
    url: str


@dataclass
class RunSeries:
    ref: RunRef
    steps: np.ndarray
    returns: np.ndarray

    @property
    def final_return(self) -> float:
        return float(self.returns[-1])

    @property
    def max_step(self) -> int:
        return int(self.steps[-1])

    @property
    def normalized_auc(self) -> float:
        if len(self.steps) < 2:
            return float("nan")
        trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(trapezoid(self.returns, self.steps) / HORIZON_STEPS)

    def interpolated(self) -> np.ndarray:
        return np.interp(STEP_GRID, self.steps, self.returns, left=self.returns[0], right=self.returns[-1])


def env_from_slug(slug: str) -> str | None:
    normalized = slug.replace("-", "_")
    mapping = {
        "Hopper_v4": "Hopper-v4",
        "Walker2d_v4": "Walker2d-v4",
        "HalfCheetah_v4": "HalfCheetah-v4",
        "Ant_v4": "Ant-v4",
    }
    return mapping.get(normalized)


def classify_run(name: str, group: str) -> tuple[str, str, int] | None:
    method = None
    if group.startswith("abhinav_easy_sac_to_sac__") or name.startswith("easy_sac_to_sac__"):
        method = "easy_sac_to_sac"
    elif group.startswith("abhinav_sac_baseline__") or name.startswith("sac__"):
        method = "sac"
    elif group.startswith("abhinav_bc_to_sac__") or group.startswith("abhinav_bc_to_sac_extended__") or name.startswith("bc_to_sac__"):
        method = "bc_to_sac"
    if method is None:
        return None

    if "bc_pretrain" in group or name.startswith("bc__"):
        return None

    env = None
    for env_id in ENV_ORDER:
        if env_id in group or env_id.replace("-", "_") in name:
            env = env_id
            break
    if env is None:
        env_match = re.search(r"__(Hopper|Walker2d|HalfCheetah|Ant)[_-]v4__", name)
        if env_match:
            env = env_from_slug(env_match.group(0).strip("_"))
    if env is None:
        return None

    seed_match = re.search(r"seed_(\d+)", name)
    if seed_match is None:
        return None
    seed = int(seed_match.group(1))

    horizon_match = re.search(r"__(\d+)k__", name)
    if horizon_match is not None and int(horizon_match.group(1)) != HORIZON_STEPS // 1000:
        return None

    return method, env, seed


def discover_runs(project: str, max_runs: int) -> list[RunRef]:
    api = wandb.Api(timeout=90)
    refs: list[RunRef] = []
    for idx, run in enumerate(api.runs(project, per_page=200)):
        if idx >= max_runs:
            break
        classified = classify_run(run.name or "", run.group or "")
        if classified is None:
            continue
        method, env, seed = classified
        if run.state not in {"finished", "failed", "crashed"}:
            continue
        refs.append(
            RunRef(
                run_id=run.id,
                name=run.name,
                group=run.group or "",
                state=run.state,
                created_at=str(run.created_at),
                method=method,
                env=env,
                seed=seed,
                url=run.url,
            )
        )
    return refs


def dedupe_refs(refs: list[RunRef]) -> list[RunRef]:
    grouped: dict[tuple[str, str, int], list[RunRef]] = defaultdict(list)
    for ref in refs:
        grouped[(ref.method, ref.env, ref.seed)].append(ref)
    latest = []
    for candidates in grouped.values():
        latest.append(sorted(candidates, key=lambda ref: ref.created_at)[-1])
    return sorted(latest, key=lambda ref: (ENV_ORDER.index(ref.env), METHOD_ORDER.index(ref.method), ref.seed))


def known_current_refs(project: str) -> list[RunRef]:
    api = wandb.Api(timeout=90)
    refs = []
    for run_id, method, env, seed in KNOWN_CURRENT_RUNS:
        run = api.run(f"{project}/{run_id}")
        refs.append(
            RunRef(
                run_id=run.id,
                name=run.name,
                group=run.group or "",
                state=run.state,
                created_at=str(run.created_at),
                method=method,
                env=env,
                seed=seed,
                url=run.url,
            )
        )
    return refs


def cache_path(cache_dir: Path, ref: RunRef) -> Path:
    env_slug = ref.env.replace("-", "_")
    return cache_dir / f"{ref.method}__{env_slug}__seed_{ref.seed}__{ref.run_id}.json"


def load_or_fetch_series(project: str, ref: RunRef, cache_dir: Path, refresh: bool) -> RunSeries | None:
    path = cache_path(cache_dir, ref)
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        api = wandb.Api(timeout=90)
        run = api.run(f"{project}/{ref.run_id}")
        rows = []
        for row in run.history(keys=["env_steps", "eval_return_mean", "eval_return_std"], pandas=False, samples=20_000):
            if row.get("env_steps") is None or row.get("eval_return_mean") is None:
                continue
            value = float(row["eval_return_mean"])
            if not math.isfinite(value):
                continue
            rows.append(
                {
                    "env_steps": int(row["env_steps"]),
                    "eval_return_mean": value,
                    "eval_return_std": None if row.get("eval_return_std") is None else float(row["eval_return_std"]),
                }
            )
        payload = {"run": ref.__dict__, "history": rows}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    history = payload.get("history", [])
    if len(history) < 2:
        return None

    by_step = {}
    for row in history:
        step = int(row["env_steps"])
        if 0 <= step <= HORIZON_STEPS:
            by_step[step] = float(row["eval_return_mean"])
    if len(by_step) < 2:
        return None
    steps = np.asarray(sorted(by_step), dtype=float)
    returns = np.asarray([by_step[int(step)] for step in steps], dtype=float)
    return RunSeries(ref=ref, steps=steps, returns=returns)


def grouped_series(series: list[RunSeries]) -> dict[tuple[str, str], list[RunSeries]]:
    grouped: dict[tuple[str, str], list[RunSeries]] = defaultdict(list)
    for run_series in series:
        grouped[(run_series.ref.env, run_series.ref.method)].append(run_series)
    return grouped


def mean_std(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.vstack(arrays)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
    sem = std / math.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros_like(mean)
    return mean, std, sem


def plot_training_curves(series: list[RunSeries], output_dir: Path) -> Path:
    grouped = grouped_series(series)
    ncols = 2 if len(ENV_ORDER) > 1 else 1
    nrows = math.ceil(len(ENV_ORDER) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 5 * nrows), sharex=True, squeeze=False)
    axes = axes.flatten()
    for ax, env in zip(axes, ENV_ORDER):
        plotted = False
        for method in METHOD_ORDER:
            runs = grouped.get((env, method), [])
            if not runs:
                continue
            arrays = [run.interpolated() for run in runs]
            mean, std, _ = mean_std(arrays)
            color = METHOD_COLORS[method]
            for arr in arrays:
                ax.plot(STEP_GRID / 1000, arr, color=color, alpha=0.18, linewidth=0.8)
            ax.plot(STEP_GRID / 1000, mean, color=color, linewidth=2.3, label=f"{METHOD_LABELS[method]} (n={len(runs)})")
            ax.fill_between(STEP_GRID / 1000, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
            plotted = True
        ax.set_title(env)
        ax.set_xlabel("Environment steps (k)")
        ax.set_ylabel("Evaluation return")
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(fontsize=8)
    for ax in axes[len(ENV_ORDER) :]:
        ax.axis("off")
    fig.suptitle("Training Curves: Easy SAC -> SAC vs SAC vs BC -> SAC", fontsize=16)
    fig.tight_layout()
    output_path = output_dir / "training_curves_by_env.png"
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path


def bar_values_by_env(
    rows: list[dict],
    metric: str,
    std_metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    ncols = 2 if len(ENV_ORDER) > 1 else 1
    nrows = math.ceil(len(ENV_ORDER) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.6 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax, env in zip(axes, ENV_ORDER):
        x = np.arange(len(METHOD_ORDER))
        means = []
        stds = []
        labels = []
        for method in METHOD_ORDER:
            row = next((r for r in rows if r["env"] == env and r["method"] == method), None)
            labels.append(METHOD_LABELS[method])
            means.append(np.nan if row is None else row[metric])
            stds.append(0.0 if row is None else row[std_metric])
        colors = [METHOD_COLORS[method] for method in METHOD_ORDER]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        ax.set_title(env)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for idx, mean in enumerate(means):
            if np.isnan(mean):
                ax.text(idx, 0.02, "missing", ha="center", va="bottom", rotation=90, transform=ax.get_xaxis_transform())
    for ax in axes[len(ENV_ORDER) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path


def plot_variance_over_training(series: list[RunSeries], output_dir: Path) -> Path:
    grouped = grouped_series(series)
    ncols = 2 if len(ENV_ORDER) > 1 else 1
    nrows = math.ceil(len(ENV_ORDER) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 5 * nrows), sharex=True, squeeze=False)
    axes = axes.flatten()
    for ax, env in zip(axes, ENV_ORDER):
        for method in METHOD_ORDER:
            runs = grouped.get((env, method), [])
            if len(runs) < 2:
                continue
            arrays = [run.interpolated() for run in runs]
            _, std, _ = mean_std(arrays)
            ax.plot(STEP_GRID / 1000, std, color=METHOD_COLORS[method], linewidth=2, label=f"{METHOD_LABELS[method]} (n={len(runs)})")
        ax.set_title(env)
        ax.set_xlabel("Environment steps (k)")
        ax.set_ylabel("Across-seed return std")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[len(ENV_ORDER) :]:
        ax.axis("off")
    fig.suptitle("Across-Seed Noise During Training", fontsize=16)
    fig.tight_layout()
    output_path = output_dir / "variance_over_training_by_env.png"
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path


def first_step_to_threshold(run_series: RunSeries, threshold: float) -> float:
    hits = run_series.steps[run_series.returns >= threshold]
    return float(hits[0]) if len(hits) else float("nan")


def summarize(series: list[RunSeries]) -> list[dict]:
    grouped = grouped_series(series)
    rows = []
    best_final_by_env = {}
    for env in ENV_ORDER:
        method_finals = []
        for method in METHOD_ORDER:
            runs = grouped.get((env, method), [])
            if runs:
                method_finals.append(np.mean([run.final_return for run in runs]))
        best_final_by_env[env] = max(method_finals) if method_finals else float("nan")

    for env in ENV_ORDER:
        threshold = 0.8 * best_final_by_env[env] if math.isfinite(best_final_by_env[env]) else float("nan")
        for method in METHOD_ORDER:
            runs = grouped.get((env, method), [])
            if not runs:
                continue
            finals = np.asarray([run.final_return for run in runs], dtype=float)
            aucs = np.asarray([run.normalized_auc for run in runs], dtype=float)
            convergence = np.asarray([first_step_to_threshold(run, threshold) for run in runs], dtype=float)
            reached = convergence[np.isfinite(convergence)]
            rows.append(
                {
                    "env": env,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "n": len(runs),
                    "seeds": " ".join(str(run.ref.seed) for run in sorted(runs, key=lambda s: s.ref.seed)),
                    "final_mean": float(np.mean(finals)),
                    "final_std": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
                    "final_sem": float(np.std(finals, ddof=1) / math.sqrt(len(finals))) if len(finals) > 1 else 0.0,
                    "final_min": float(np.min(finals)),
                    "final_max": float(np.max(finals)),
                    "normalized_auc_mean": float(np.mean(aucs)),
                    "normalized_auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                    "normalized_auc_sem": float(np.std(aucs, ddof=1) / math.sqrt(len(aucs))) if len(aucs) > 1 else 0.0,
                    "convergence_threshold": float(threshold),
                    "convergence_reached_n": int(len(reached)),
                    "convergence_step_mean": float(np.mean(reached)) if len(reached) else float("nan"),
                    "convergence_step_std": float(np.std(reached, ddof=1)) if len(reached) > 1 else 0.0,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(rows: list[dict], output_dir: Path) -> Path:
    ncols = 2 if len(ENV_ORDER) > 1 else 1
    nrows = math.ceil(len(ENV_ORDER) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.6 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax, env in zip(axes, ENV_ORDER):
        x = np.arange(len(METHOD_ORDER))
        vals = []
        errs = []
        labels = []
        for method in METHOD_ORDER:
            row = next((r for r in rows if r["env"] == env and r["method"] == method), None)
            labels.append(METHOD_LABELS[method])
            vals.append(np.nan if row is None else row["convergence_step_mean"] / 1000)
            errs.append(0.0 if row is None else row["convergence_step_std"] / 1000)
        ax.bar(x, vals, yerr=errs, capsize=4, color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=0.85)
        ax.set_title(env)
        ax.set_ylabel("Steps to 80% best final (k)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for idx, val in enumerate(vals):
            if np.isnan(val):
                ax.text(idx, 0.02, "not reached / missing", ha="center", va="bottom", rotation=90, transform=ax.get_xaxis_transform())
    for ax in axes[len(ENV_ORDER) :]:
        ax.axis("off")
    fig.suptitle("Convergence Dynamics: Time to Common Reward Threshold", fontsize=16)
    fig.tight_layout()
    output_path = output_dir / "convergence_steps_by_env.png"
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path


def write_run_manifest(path: Path, refs: list[RunRef], series: list[RunSeries]) -> None:
    available = {(s.ref.run_id, s.ref.method, s.ref.env, s.ref.seed): s for s in series}
    rows = []
    for ref in refs:
        run_series = available.get((ref.run_id, ref.method, ref.env, ref.seed))
        rows.append(
            {
                **ref.__dict__,
                "history_points": 0 if run_series is None else len(run_series.steps),
                "max_env_steps": None if run_series is None else run_series.max_step,
                "final_return": None if run_series is None else run_series.final_return,
                "normalized_auc": None if run_series is None else run_series.normalized_auc,
            }
        )
    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=ENTITY_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed/abhinav_task_modal_comparison"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/raw/abhinav_task_wandb_cache"))
    parser.add_argument("--max-runs", type=int, default=600)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--discover", action="store_true", help="Discover matching runs instead of using the known current run set.")
    parser.add_argument("--envs", default=",".join(DEFAULT_ENV_ORDER), help="Comma-separated env ids to include in plots.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    global ENV_ORDER
    ENV_ORDER = tuple(env.strip() for env in args.envs.split(",") if env.strip())

    refs = dedupe_refs(discover_runs(args.project, args.max_runs)) if args.discover else known_current_refs(args.project)
    series = []
    for ref in refs:
        run_series = load_or_fetch_series(args.project, ref, args.cache_dir, args.refresh)
        if run_series is None:
            print(f"Skipping {ref.name}: no usable eval history")
            continue
        if run_series.max_step < HORIZON_STEPS:
            print(f"Skipping {ref.name}: only reached {run_series.max_step} steps")
            continue
        series.append(run_series)

    rows = summarize(series)
    write_csv(args.output_dir / "run_manifest.csv", [ref.__dict__ for ref in refs])
    write_run_manifest(args.output_dir / "run_manifest_with_metrics.csv", refs, series)
    write_csv(args.output_dir / "summary_metrics.csv", rows)

    figures = [
        plot_training_curves(series, args.output_dir),
        bar_values_by_env(rows, "final_mean", "final_std", "Final evaluation return", "Final Reward Comparison", args.output_dir / "final_reward_by_env.png"),
        bar_values_by_env(rows, "normalized_auc_mean", "normalized_auc_std", "Normalized AUC", "AUC / Sample Efficiency Comparison", args.output_dir / "auc_by_env.png"),
        plot_convergence(rows, args.output_dir),
        bar_values_by_env(rows, "final_std", "final_sem", "Final-return std across seeds", "Final Reward Variance Across Seeds", args.output_dir / "final_reward_std_by_env.png"),
        bar_values_by_env(rows, "normalized_auc_std", "normalized_auc_sem", "AUC std across seeds", "AUC Variance Across Seeds", args.output_dir / "auc_std_by_env.png"),
        plot_variance_over_training(series, args.output_dir),
    ]
    print("Saved figures:")
    for figure in figures:
        print(f"  {figure}")
    print(f"Saved summary CSV: {args.output_dir / 'summary_metrics.csv'}")
    print(f"Saved run manifest: {args.output_dir / 'run_manifest_with_metrics.csv'}")


if __name__ == "__main__":
    main()
