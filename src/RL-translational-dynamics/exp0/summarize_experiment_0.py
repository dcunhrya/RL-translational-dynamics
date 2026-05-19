import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunSummary:
    algorithm: str
    env: str
    seed: int
    run_dir: Path
    initial_eval: float | None
    final_eval: float | None
    max_env_steps: int
    gradient_updates: int
    has_nan: bool
    has_checkpoint: bool
    metric_count: int

    @property
    def improved(self) -> bool:
        return self.initial_eval is not None and self.final_eval is not None and self.final_eval > self.initial_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Experiment 0 sanity runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/raw/experiment_0"))
    parser.add_argument("--envs", nargs="+", default=["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4"])
    parser.add_argument("--algorithms", nargs="+", default=["sac", "ppo"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--gate-env", default="Hopper-v4")
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed/experiment_0"))
    return parser.parse_args()


def is_bad_number(value) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def load_metrics(metrics_path: Path) -> list[dict]:
    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {metrics_path} line {line_number}: {exc}") from exc
    return rows


def summarize_run(run_dir: Path) -> RunSummary | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    metrics = load_metrics(metrics_path)
    if not metrics:
        return None

    algorithm = str(metrics[0].get("algorithm", "unknown"))
    env = str(metrics[0].get("env", "unknown"))
    seed = int(metrics[0].get("seed", -1))
    eval_rows = [row for row in metrics if "eval_return_mean" in row]
    initial_eval = float(eval_rows[0]["eval_return_mean"]) if eval_rows else None
    final_eval = float(eval_rows[-1]["eval_return_mean"]) if eval_rows else None
    max_env_steps = max(int(row.get("env_steps", 0)) for row in metrics)
    gradient_updates = max(int(row.get("gradient_updates", 0)) for row in metrics)
    has_nan = any(is_bad_number(value) for row in metrics for value in row.values())
    has_checkpoint = any(run_dir.glob("checkpoint_step_*.pt"))

    return RunSummary(
        algorithm=algorithm,
        env=env,
        seed=seed,
        run_dir=run_dir,
        initial_eval=initial_eval,
        final_eval=final_eval,
        max_env_steps=max_env_steps,
        gradient_updates=gradient_updates,
        has_nan=has_nan,
        has_checkpoint=has_checkpoint,
        metric_count=len(metrics),
    )


def latest_runs(results_dir: Path) -> dict[tuple[str, str, int], RunSummary]:
    grouped: dict[tuple[str, str, int], list[RunSummary]] = defaultdict(list)
    for metrics_path in results_dir.glob("*/metrics.jsonl"):
        summary = summarize_run(metrics_path.parent)
        if summary is not None:
            grouped[(summary.algorithm, summary.env, summary.seed)].append(summary)

    latest = {}
    for key, summaries in grouped.items():
        latest[key] = max(summaries, key=lambda summary: summary.run_dir.stat().st_mtime)
    return latest


def format_eval(value: float | None) -> str:
    return "missing" if value is None else f"{value:.2f}"


def eval_series(run_dir: Path) -> list[tuple[int, float]]:
    metrics = load_metrics(run_dir / "metrics.jsonl")
    series = []
    for row in metrics:
        if "eval_return_mean" not in row:
            continue
        series.append((int(row["env_steps"]), float(row["eval_return_mean"])))
    return series


def mean_curve_for_env(
    runs: dict[tuple[str, str, int], RunSummary],
    algorithm: str,
    env: str,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    per_seed = []
    for seed in seeds:
        summary = runs.get((algorithm, env, seed))
        if summary is None:
            return None
        series = eval_series(summary.run_dir)
        if not series:
            return None
        per_seed.append({step: value for step, value in series})

    common_steps = sorted(set.intersection(*(set(series.keys()) for series in per_seed)))
    if not common_steps:
        return None

    values = np.asarray([[series[step] for step in common_steps] for series in per_seed], dtype=float)
    steps = np.asarray(common_steps, dtype=int)
    return steps, values.mean(axis=0), values.std(axis=0)


def plot_learning_curves(
    runs: dict[tuple[str, str, int], RunSummary],
    envs: list[str],
    algorithms: list[str],
    seeds: list[int],
    output_dir: Path,
) -> Path:
    colors = {"sac": "#1f77b4", "ppo": "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, env in zip(axes, envs):
        for algorithm in algorithms:
            curve = mean_curve_for_env(runs, algorithm, env, seeds)
            if curve is None:
                continue
            steps, mean_values, std_values = curve
            color = colors.get(algorithm, None)
            ax.plot(steps, mean_values, label=algorithm.upper(), color=color, linewidth=2)
            ax.fill_between(steps, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)
        ax.set_title(env)
        ax.set_xlabel("Env steps")
        ax.set_ylabel("Eval return")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[len(envs) :]:
        ax.axis("off")

    fig.suptitle("Experiment 0: SAC vs PPO mean eval return across 2 seeds", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "sac_vs_ppo_learning_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_final_returns(
    runs: dict[tuple[str, str, int], RunSummary],
    envs: list[str],
    algorithms: list[str],
    seeds: list[int],
    output_dir: Path,
) -> Path:
    x = np.arange(len(envs))
    width = 0.35
    colors = {"sac": "#1f77b4", "ppo": "#d62728"}

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, algorithm in enumerate(algorithms):
        means = []
        stds = []
        for env in envs:
            finals = []
            for seed in seeds:
                summary = runs.get((algorithm, env, seed))
                if summary is None or summary.final_eval is None:
                    finals = []
                    break
                finals.append(summary.final_eval)
            means.append(float(np.mean(finals)) if finals else np.nan)
            stds.append(float(np.std(finals)) if finals else np.nan)
        offset = (idx - (len(algorithms) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=algorithm.upper(),
            color=colors.get(algorithm, None),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15)
    ax.set_ylabel("Final eval return")
    ax.set_title("Experiment 0: final eval return mean +/- std across 2 seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "sac_vs_ppo_final_returns.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    runs = latest_runs(args.results_dir)
    failures = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment 0 summary from {args.results_dir}")
    print()

    for algorithm in args.algorithms:
        for env in args.envs:
            for seed in args.seeds:
                key = (algorithm, env, seed)
                summary = runs.get(key)
                if summary is None:
                    failures.append(f"missing run: algorithm={algorithm} env={env} seed={seed}")
                    print(f"MISS {algorithm:>3} {env:>12} seed={seed}")
                    continue

                status = "PASS"
                reasons = []
                if summary.has_nan:
                    status = "FAIL"
                    reasons.append("non-finite metric")
                if not summary.has_checkpoint:
                    status = "FAIL"
                    reasons.append("missing checkpoint")
                if env == args.gate_env and not summary.improved:
                    status = "FAIL"
                    reasons.append("no eval improvement")

                if status == "FAIL":
                    failures.append(f"{algorithm} {env} seed={seed}: {', '.join(reasons)}")

                delta = None
                if summary.initial_eval is not None and summary.final_eval is not None:
                    delta = summary.final_eval - summary.initial_eval
                delta_text = "missing" if delta is None else f"{delta:+.2f}"
                reason_text = "" if not reasons else f" ({', '.join(reasons)})"
                print(
                    f"{status} {algorithm:>3} {env:>12} seed={seed} "
                    f"initial={format_eval(summary.initial_eval)} final={format_eval(summary.final_eval)} "
                    f"delta={delta_text} steps={summary.max_env_steps} updates={summary.gradient_updates} "
                    f"metrics={summary.metric_count}{reason_text}"
                )

    print()
    if failures:
        print("Experiment 0 gate: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Experiment 0 gate: PASS")
    learning_curves_path = plot_learning_curves(runs, args.envs, args.algorithms, args.seeds, args.output_dir)
    final_returns_path = plot_final_returns(runs, args.envs, args.algorithms, args.seeds, args.output_dir)
    print(f"Saved learning curves plot to {learning_curves_path}")
    print(f"Saved final returns plot to {final_returns_path}")


if __name__ == "__main__":
    main()
