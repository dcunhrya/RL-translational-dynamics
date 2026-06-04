import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunKey:
    algorithm: str
    env: str
    seed: int
    horizon_steps: int | None = None
    policy_source: str | None = None
    value_init: str | None = None
    bc_anchor_interval: int | None = None


@dataclass
class RunSummary:
    key: RunKey
    run_dir: Path
    final_eval: float | None
    eval_auc: float | None
    normalized_auc: float | None
    max_env_steps: int
    gradient_updates: int
    switch_step: int | None
    metric_count: int
    phases: tuple[str, ...]
    has_nan: bool
    has_checkpoint: bool


SUMMARY_ALWAYS_LINE_MARKERS = (
    '"eval_return_mean"',
    '"switch_step"',
    '"bc_distill_loss"',
    '"bc_anchor_loss"',
    '"policy_retention_action_mse"',
    '"policy_retention_approx_kl"',
)
SUMMARY_SAMPLED_LINE_MARKERS = (
    '"ppo_explained_variance"',
    '"sac_qf1_mean"',
    '"sac_qf2_mean"',
    '"iql_q_loss"',
    '"iql_value_loss"',
    '"awac_critic_loss"',
)
SAMPLED_MARKER_STRIDE = 5_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Abhinav/Experiment 4 offline-assisted runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/raw/abhinav_task"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed/abhinav_task"))
    parser.add_argument("--notes-path", type=Path, default=Path("experiments/abhinav_task/results.md"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@lru_cache(maxsize=None)
def load_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if rows:
                has_always_marker = any(marker in line for marker in SUMMARY_ALWAYS_LINE_MARKERS)
                has_sampled_marker = any(marker in line for marker in SUMMARY_SAMPLED_LINE_MARKERS)
                if not has_always_marker and not (has_sampled_marker and line_number % SAMPLED_MARKER_STRIDE == 0):
                    continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    return rows


def is_bad_number(value) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def eval_series(rows: list[dict]) -> list[tuple[int, float]]:
    series = []
    for row in rows:
        if "eval_return_mean" in row and row["eval_return_mean"] is not None:
            series.append((int(row.get("env_steps", 0)), float(row["eval_return_mean"])))
    return series


def auc_from_series(series: list[tuple[int, float]]) -> tuple[float | None, float | None]:
    if len(series) < 2:
        return None, None
    series = sorted(series)
    steps = np.asarray([step for step, _ in series], dtype=float)
    values = np.asarray([value for _, value in series], dtype=float)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(trapezoid(values, steps))
    horizon = float(max(steps[-1] - steps[0], 1.0))
    return auc, auc / horizon


def horizon_from_run_dir(run_dir: Path) -> int | None:
    match = re.search(r"__(\d+)k__", run_dir.name)
    if match:
        return int(match.group(1)) * 1000
    return None


def run_key_from_rows(rows: list[dict], run_dir: Path) -> RunKey | None:
    if not rows:
        return None
    first = rows[0]
    env = str(first.get("env", "unknown"))
    seed = int(first.get("seed", -1))
    algorithm = str(first.get("algorithm", "unknown"))
    horizon_steps = first.get("total_timesteps") or horizon_from_run_dir(run_dir)
    if horizon_steps is not None:
        horizon_steps = int(horizon_steps)
    policy_source = first.get("policy_source") or first.get("starter_policy_source")
    value_init = first.get("value_init")
    bc_anchor_interval = first.get("bc_anchor_interval")
    if bc_anchor_interval is not None:
        bc_anchor_interval = int(bc_anchor_interval)
    return RunKey(
        algorithm=algorithm,
        env=env,
        seed=seed,
        horizon_steps=horizon_steps,
        policy_source=str(policy_source) if policy_source is not None else None,
        value_init=str(value_init) if value_init is not None else None,
        bc_anchor_interval=bc_anchor_interval,
    )


def summarize_run(run_dir: Path) -> RunSummary | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    rows = load_metrics(metrics_path)
    if not rows:
        return None
    key = run_key_from_rows(rows, run_dir)
    if key is None:
        return None
    series = eval_series(rows)
    final_eval = series[-1][1] if series else None
    eval_auc, normalized_auc = auc_from_series(series)
    max_env_steps = max(int(row.get("env_steps", 0)) for row in rows)
    gradient_updates = max(int(row.get("gradient_updates", 0)) for row in rows)
    switch_steps = [int(row["switch_step"]) for row in rows if row.get("switch_step") is not None]
    phases = tuple(sorted({str(row.get("phase")) for row in rows if row.get("phase") is not None}))
    return RunSummary(
        key=key,
        run_dir=run_dir,
        final_eval=final_eval,
        eval_auc=eval_auc,
        normalized_auc=normalized_auc,
        max_env_steps=max_env_steps,
        gradient_updates=gradient_updates,
        switch_step=switch_steps[0] if switch_steps else None,
        metric_count=len(rows),
        phases=phases,
        has_nan=any(is_bad_number(value) for row in rows for value in row.values()),
        has_checkpoint=any(run_dir.glob("*.pt")) or any(run_dir.glob("checkpoint_*.pt")),
    )


def collect_runs(results_dir: Path) -> dict[RunKey, RunSummary]:
    grouped: dict[RunKey, list[RunSummary]] = defaultdict(list)
    for metrics_path in results_dir.rglob("metrics.jsonl"):
        summary = summarize_run(metrics_path.parent)
        if summary is not None:
            grouped[summary.key].append(summary)
    latest = {}
    for key, summaries in grouped.items():
        latest[key] = max(summaries, key=lambda summary: (summary.max_env_steps, summary.run_dir.stat().st_mtime))
    return latest


def bootstrap_ci(values: list[float], samples: int, rng: np.random.Generator) -> tuple[float | None, float | None, float | None]:
    clean = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if len(clean) == 0:
        return None, None, None
    if len(clean) == 1:
        value = float(clean[0])
        return value, value, value
    draws = rng.choice(clean, size=(samples, len(clean)), replace=True).mean(axis=1)
    return float(clean.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def method_label(key: RunKey) -> str:
    if key.algorithm == "bc_anchor_sac":
        label = f"BC-anchor SAC K={key.bc_anchor_interval}"
    else:
        label = key.algorithm.replace("_", " -> ").upper()
    if key.horizon_steps and key.horizon_steps != 500_000:
        label = f"{label} ({key.horizon_steps // 1000}k)"
    return label


def aggregate(runs: dict[RunKey, RunSummary], bootstrap_samples: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, str, str | None, str | None, int | None], list[RunSummary]] = defaultdict(list)
    for summary in runs.values():
        group_key = (
            summary.key.env,
            summary.key.algorithm,
            summary.key.horizon_steps,
            summary.key.policy_source,
            summary.key.value_init,
            summary.key.bc_anchor_interval,
        )
        grouped[group_key].append(summary)

    rows = []
    for (env, algorithm, horizon_steps, policy_source, value_init, bc_anchor_interval), summaries in sorted(grouped.items()):
        final_values = [summary.final_eval for summary in summaries if summary.final_eval is not None]
        auc_values = [summary.normalized_auc for summary in summaries if summary.normalized_auc is not None]
        final_mean, final_lo, final_hi = bootstrap_ci(final_values, bootstrap_samples, rng)
        auc_mean, auc_lo, auc_hi = bootstrap_ci(auc_values, bootstrap_samples, rng)
        rows.append(
            {
                "env": env,
                "algorithm": algorithm,
                "horizon_steps": horizon_steps,
                "policy_source": policy_source,
                "value_init": value_init,
                "bc_anchor_interval": bc_anchor_interval,
                "num_seeds": len({summary.key.seed for summary in summaries}),
                "final_return_mean": final_mean,
                "final_return_ci_low": final_lo,
                "final_return_ci_high": final_hi,
                "normalized_auc_mean": auc_mean,
                "normalized_auc_ci_low": auc_lo,
                "normalized_auc_ci_high": auc_hi,
                "worst_seed_final_return": float(np.min(final_values)) if final_values else None,
                "collapse_count_final_lt_100": int(sum(value < 100.0 for value in final_values)),
                "max_env_steps": max(summary.max_env_steps for summary in summaries),
            }
        )
    return rows


def plot_learning_curves(runs: dict[RunKey, RunSummary], output_dir: Path) -> Path | None:
    if not runs:
        return None
    envs = sorted({key.env for key in runs})
    fig, axes = plt.subplots(1, len(envs), figsize=(7 * len(envs), 5), squeeze=False)
    axes = axes.flatten()
    for ax, env in zip(axes, envs):
        env_runs = [summary for summary in runs.values() if summary.key.env == env]
        methods = sorted(
            {
                (
                    summary.key.algorithm,
                    summary.key.horizon_steps,
                    summary.key.policy_source,
                    summary.key.value_init,
                    summary.key.bc_anchor_interval,
                )
                for summary in env_runs
            }
        )
        for algorithm, horizon_steps, policy_source, value_init, anchor_interval in methods:
            summaries = [
                summary
                for summary in env_runs
                if summary.key.algorithm == algorithm
                and summary.key.horizon_steps == horizon_steps
                and summary.key.policy_source == policy_source
                and summary.key.value_init == value_init
                and summary.key.bc_anchor_interval == anchor_interval
            ]
            per_seed = []
            switch_steps = []
            for summary in summaries:
                rows = load_metrics(summary.run_dir / "metrics.jsonl")
                series = eval_series(rows)
                if series:
                    per_seed.append({step: value for step, value in series})
                if summary.switch_step is not None:
                    switch_steps.append(summary.switch_step)
            if not per_seed:
                continue
            common_steps = sorted(set.intersection(*(set(series.keys()) for series in per_seed)))
            if not common_steps:
                continue
            values = np.asarray([[series[step] for step in common_steps] for series in per_seed], dtype=float)
            steps = np.asarray(common_steps, dtype=float)
            label_key = summaries[0].key
            ax.plot(steps, values.mean(axis=0), linewidth=2, label=method_label(label_key))
            ax.fill_between(steps, values.mean(axis=0) - values.std(axis=0), values.mean(axis=0) + values.std(axis=0), alpha=0.15)
            if switch_steps:
                ax.axvline(float(np.mean(switch_steps)), linestyle="--", linewidth=1, alpha=0.4)
        ax.set_title(env)
        ax.set_xlabel("Env steps")
        ax.set_ylabel("Eval return")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / "phase_marked_learning_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_family(runs: dict[RunKey, RunSummary], metric_names: list[str], output_dir: Path, filename: str, title: str) -> Path | None:
    records = []
    for summary in runs.values():
        rows = load_metrics(summary.run_dir / "metrics.jsonl")
        for row in rows:
            for metric in metric_names:
                if metric in row and row[metric] is not None:
                    records.append((summary.key.env, method_label(summary.key), int(row.get("env_steps", 0)), metric, float(row[metric])))
    if not records:
        return None
    envs = sorted({record[0] for record in records})
    fig, axes = plt.subplots(1, len(envs), figsize=(7 * len(envs), 5), squeeze=False)
    axes = axes.flatten()
    for ax, env in zip(axes, envs):
        env_records = [record for record in records if record[0] == env]
        labels = sorted({record[1] for record in env_records})
        for label in labels:
            label_records = [record for record in env_records if record[1] == label]
            by_step: dict[int, list[float]] = defaultdict(list)
            for _, _, step, _, value in label_records:
                by_step[step].append(value)
            steps = np.asarray(sorted(by_step), dtype=float)
            values = np.asarray([np.mean(by_step[int(step)]) for step in steps], dtype=float)
            ax.plot(steps, values, label=label, linewidth=2)
        ax.set_title(env)
        ax.set_xlabel("Env steps")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def handoff_transients(runs: dict[RunKey, RunSummary]) -> list[dict]:
    records = []
    for summary in runs.values():
        if summary.switch_step is None:
            continue
        series = sorted(eval_series(load_metrics(summary.run_dir / "metrics.jsonl")))
        before = [value for step, value in series if step <= summary.switch_step]
        after = [value for step, value in series if step >= summary.switch_step]
        if not before or len(after) < 2:
            continue
        records.append(
            {
                "env": summary.key.env,
                "algorithm": summary.key.algorithm,
                "seed": summary.key.seed,
                "switch_step": summary.switch_step,
                "pre_switch_eval": before[-1],
                "first_post_switch_eval": after[1],
                "handoff_delta": after[1] - before[-1],
            }
        )
    return records


def write_outputs(args: argparse.Namespace, runs: dict[RunKey, RunSummary]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.notes_path.parent.mkdir(parents=True, exist_ok=True)
    rows = aggregate(runs, args.bootstrap_samples, args.seed)
    transient_rows = handoff_transients(runs)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"runs": [row for row in rows], "handoff_transients": transient_rows}, f, indent=2)

    plot_paths = [
        plot_learning_curves(runs, args.output_dir),
        plot_metric_family(
            runs,
            ["policy_retention_action_mse", "policy_retention_approx_kl", "bc_distill_loss", "bc_anchor_loss"],
            args.output_dir,
            "policy_retention.png",
            "Policy retention / distillation metric",
        ),
        plot_metric_family(
            runs,
            ["ppo_explained_variance", "sac_qf1_mean", "sac_qf2_mean", "iql_q_loss", "iql_value_loss", "awac_critic_loss"],
            args.output_dir,
            "value_quality.png",
            "Value quality metric",
        ),
    ]

    lines = [
        "# Abhinav Task Results",
        "",
        "Offline-assisted runs report online env steps separately from offline dataset size and offline updates.",
        "",
        "## Summary",
        "",
        "| Env | Method | Seeds | Final return 95% CI | Normalized AUC 95% CI | Worst seed | Collapses |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        method = row["algorithm"]
        if row["bc_anchor_interval"]:
            method = f"{method} K={row['bc_anchor_interval']}"
        if row["horizon_steps"] and row["horizon_steps"] != 500_000:
            method = f"{method} ({row['horizon_steps'] // 1000}k)"
        final = "n/a" if row["final_return_mean"] is None else (
            f"{row['final_return_mean']:.2f} [{row['final_return_ci_low']:.2f}, {row['final_return_ci_high']:.2f}]"
        )
        auc = "n/a" if row["normalized_auc_mean"] is None else (
            f"{row['normalized_auc_mean']:.2f} [{row['normalized_auc_ci_low']:.2f}, {row['normalized_auc_ci_high']:.2f}]"
        )
        worst = "n/a" if row["worst_seed_final_return"] is None else f"{row['worst_seed_final_return']:.2f}"
        lines.append(
            f"| {row['env']} | {method} | {row['num_seeds']} | {final} | {auc} | {worst} | "
            f"{row['collapse_count_final_lt_100']} |"
        )

    if transient_rows:
        lines.extend(["", "## Handoff Transients", "", "| Env | Method | Seed | Switch step | Delta |", "| --- | --- | ---: | ---: | ---: |"])
        for row in transient_rows:
            lines.append(
                f"| {row['env']} | {row['algorithm']} | {row['seed']} | {row['switch_step']} | "
                f"{row['handoff_delta']:.2f} |"
            )

    lines.extend(["", "## Figures", ""])
    for path in plot_paths:
        if path is not None:
            lines.append(f"- `{path}`")

    args.notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = collect_runs(args.results_dir)
    print(f"Found {len(runs)} latest runs under {args.results_dir}")
    write_outputs(args, runs)
    print(f"Wrote summary to {args.output_dir / 'summary.json'}")
    print(f"Wrote notes to {args.notes_path}")


if __name__ == "__main__":
    main()
