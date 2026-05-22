import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunSummary:
    env: str
    seed: int
    switch_step: int
    switch_fraction: float
    run_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Experiment 3 reverse handoff runs.")
    parser.add_argument("--handoff-results-dir", type=Path, default=Path("results/raw/experiment_3_reverse_handoff"))
    parser.add_argument("--baseline-results-dir", type=Path, default=Path("results/raw/experiment_0"))
    parser.add_argument("--envs", nargs="+", default=["Hopper-v4", "Walker2d-v4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--switch-fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed/experiment_3_reverse_handoff"))
    return parser.parse_args()


def load_metrics(metrics_path: Path) -> list[dict]:
    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_reverse_runs(results_dir: Path) -> dict[tuple[str, int, int], RunSummary]:
    grouped: dict[tuple[str, int, int], list[RunSummary]] = defaultdict(list)
    for metrics_path in results_dir.glob("*/metrics.jsonl"):
        rows = load_metrics(metrics_path)
        if not rows:
            continue
        env = str(rows[0]["env"])
        seed = int(rows[0]["seed"])
        switch_step = int(rows[0]["planned_switch_step"])
        switch_fraction = float(rows[0]["handoff_fraction"])
        summary = RunSummary(env=env, seed=seed, switch_step=switch_step, switch_fraction=switch_fraction, run_dir=metrics_path.parent)
        grouped[(env, seed, switch_step)].append(summary)
    latest = {}
    for key, summaries in grouped.items():
        latest[key] = max(summaries, key=lambda summary: summary.run_dir.stat().st_mtime)
    return latest


def latest_baseline_runs(results_dir: Path) -> dict[tuple[str, str, int], Path]:
    grouped: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for metrics_path in results_dir.glob("*/metrics.jsonl"):
        rows = load_metrics(metrics_path)
        if not rows:
            continue
        algorithm = str(rows[0]["algorithm"])
        env = str(rows[0]["env"])
        seed = int(rows[0]["seed"])
        grouped[(algorithm, env, seed)].append(metrics_path.parent)
    latest = {}
    for key, run_dirs in grouped.items():
        latest[key] = max(run_dirs, key=lambda run_dir: run_dir.stat().st_mtime)
    return latest


def eval_series(run_dir: Path) -> list[tuple[int, float]]:
    rows = load_metrics(run_dir / "metrics.jsonl")
    series = []
    seen_steps = set()
    for row in rows:
        if "eval_return_mean" not in row:
            continue
        step = int(row["env_steps"])
        if step in seen_steps:
            continue
        seen_steps.add(step)
        series.append((step, float(row["eval_return_mean"])))
    return series


def mean_curve(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    per_run = []
    for run_dir in run_dirs:
        series = eval_series(run_dir)
        if not series:
            return None
        per_run.append({step: value for step, value in series})
    common_steps = sorted(set.intersection(*(set(run.keys()) for run in per_run)))
    if not common_steps:
        return None
    values = np.asarray([[run[step] for step in common_steps] for run in per_run], dtype=float)
    return np.asarray(common_steps), values.mean(axis=0), values.std(axis=0)


def final_eval(run_dir: Path) -> float | None:
    series = eval_series(run_dir)
    return None if not series else float(series[-1][1])


def plot_env_learning_curves(
    env: str,
    reverse_runs: dict[tuple[str, int, int], RunSummary],
    baseline_runs: dict[tuple[str, str, int], Path],
    seeds: list[int],
    switch_fractions: list[float],
    output_dir: Path,
) -> Path:
    colors = {0.25: "#1f77b4", 0.5: "#ff7f0e", 0.75: "#2ca02c"}
    fig, ax = plt.subplots(figsize=(11, 7))
    for switch_fraction in switch_fractions:
        switch_step = int(100_000 * switch_fraction)
        run_dirs = []
        for seed in seeds:
            summary = reverse_runs.get((env, seed, switch_step))
            if summary is not None:
                run_dirs.append(summary.run_dir)
        curve = mean_curve(run_dirs)
        if curve is None:
            continue
        steps, mean_values, std_values = curve
        color = colors.get(switch_fraction)
        label = f"PPO->{int(switch_fraction * 100)}% switch"
        ax.plot(steps, mean_values, linewidth=2.5, color=color, label=label)
        ax.fill_between(steps, mean_values - std_values, mean_values + std_values, color=color, alpha=0.18)
        ax.axvline(switch_step, color=color, linestyle="--", alpha=0.8)

    for algorithm, style, color in [("ppo", ":", "#444444"), ("sac", "-.", "#999999")]:
        run_dirs = []
        for seed in [0, 1]:
            run_dir = baseline_runs.get((algorithm, env, seed))
            if run_dir is not None:
                run_dirs.append(run_dir)
        curve = mean_curve(run_dirs)
        if curve is None:
            continue
        steps, mean_values, _ = curve
        ax.plot(steps, mean_values, linestyle=style, linewidth=2, color=color, label=f"{algorithm.upper()} baseline")

    ax.set_title(f"{env}: reverse handoff learning curves")
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Eval return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / f"{env.replace('-', '_')}_reverse_handoff_learning_curves.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_final_returns(
    envs: list[str],
    reverse_runs: dict[tuple[str, int, int], RunSummary],
    baseline_runs: dict[tuple[str, str, int], Path],
    seeds: list[int],
    switch_fractions: list[float],
    output_dir: Path,
) -> Path:
    labels = ["PPO", "25%", "50%", "75%", "SAC"]
    x = np.arange(len(labels))
    colors = {"ppo": "#444444", 0.25: "#1f77b4", 0.5: "#ff7f0e", 0.75: "#2ca02c", "sac": "#999999"}
    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5), sharey=False)
    if len(envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, envs):
        means = []
        stds = []
        ppo_vals = [final_eval(baseline_runs[("ppo", env, seed)]) for seed in [0, 1] if ("ppo", env, seed) in baseline_runs]
        means.append(float(np.mean(ppo_vals)))
        stds.append(float(np.std(ppo_vals)))
        for switch_fraction in switch_fractions:
            switch_step = int(100_000 * switch_fraction)
            vals = []
            for seed in seeds:
                summary = reverse_runs.get((env, seed, switch_step))
                if summary is not None:
                    val = final_eval(summary.run_dir)
                    if val is not None:
                        vals.append(val)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        sac_vals = [final_eval(baseline_runs[("sac", env, seed)]) for seed in [0, 1] if ("sac", env, seed) in baseline_runs]
        means.append(float(np.mean(sac_vals)))
        stds.append(float(np.std(sac_vals)))
        bar_colors = [colors["ppo"], colors[0.25], colors[0.5], colors[0.75], colors["sac"]]
        ax.bar(x, means, yerr=stds, capsize=4, color=bar_colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(env)
        ax.set_ylabel("Final eval return")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Experiment 3: reverse handoff final return comparison", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "reverse_handoff_final_return_comparison.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_markdown_summary(
    envs: list[str],
    reverse_runs: dict[tuple[str, int, int], RunSummary],
    baseline_runs: dict[tuple[str, str, int], Path],
    seeds: list[int],
    switch_fractions: list[float],
    output_dir: Path,
) -> Path:
    lines = ["# Experiment 3 Reverse Handoff Summary", "", "Final `eval_return_mean` averages across seeds.", ""]
    for env in envs:
        lines.append(f"## {env}")
        ppo_vals = [final_eval(baseline_runs[("ppo", env, seed)]) for seed in [0, 1] if ("ppo", env, seed) in baseline_runs]
        sac_vals = [final_eval(baseline_runs[("sac", env, seed)]) for seed in [0, 1] if ("sac", env, seed) in baseline_runs]
        lines.append(f"- PPO baseline: {np.mean(ppo_vals):.2f} +/- {np.std(ppo_vals):.2f}")
        for switch_fraction in switch_fractions:
            switch_step = int(100_000 * switch_fraction)
            vals = []
            for seed in seeds:
                summary = reverse_runs.get((env, seed, switch_step))
                if summary is not None:
                    val = final_eval(summary.run_dir)
                    if val is not None:
                        vals.append(val)
            lines.append(f"- Reverse handoff {int(switch_fraction * 100)}%: {np.mean(vals):.2f} +/- {np.std(vals):.2f}")
        lines.append(f"- SAC baseline: {np.mean(sac_vals):.2f} +/- {np.std(sac_vals):.2f}")
        lines.append("")
    output_path = output_dir / "experiment_3_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reverse_runs = latest_reverse_runs(args.handoff_results_dir)
    baseline_runs = latest_baseline_runs(args.baseline_results_dir)
    generated = []
    for env in args.envs:
        generated.append(plot_env_learning_curves(env, reverse_runs, baseline_runs, args.seeds, args.switch_fractions, args.output_dir))
    generated.append(plot_final_returns(args.envs, reverse_runs, baseline_runs, args.seeds, args.switch_fractions, args.output_dir))
    generated.append(write_markdown_summary(args.envs, reverse_runs, baseline_runs, args.seeds, args.switch_fractions, args.output_dir))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
