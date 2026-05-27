import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunKey:
    algorithm: str
    env: str
    seed: int
    handoff_fraction: float | None = None
    policy_init: str | None = None
    value_init: str | None = None


@dataclass
class RunSummary:
    key: RunKey
    run_dir: Path
    initial_eval: float | None
    final_eval: float | None
    max_env_steps: int
    gradient_updates: int
    switch_step: int | None
    has_nan: bool
    has_checkpoint: bool
    has_handoff_phase: bool
    metric_count: int

    @property
    def eval_auc(self) -> float | None:
        series = eval_series(self.run_dir)
        if len(series) < 2:
            return None
        steps = np.asarray([step for step, _ in series], dtype=float)
        values = np.asarray([value for _, value in series], dtype=float)
        trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(trapezoid(values, steps))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Experiment 2 fixed handoff runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/raw/experiment_2"))
    parser.add_argument("--envs", nargs="+", default=["Hopper-v4", "Walker2d-v4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--switch-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
    )
    parser.add_argument(
        "--value-inits",
        nargs="+",
        default=[None],
        help="Optional SAC->PPO value-init arms to require, e.g. random self-warmup source-aligned.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed/experiment_2"))
    parser.add_argument(
        "--skip-checkpoint-gate",
        action="store_true",
        help="Do not fail the gate when checkpoint_step_*.pt files are missing (e.g. metrics-only fetch).",
    )
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


def run_key_from_metrics(metrics: list[dict]) -> RunKey | None:
    if not metrics:
        return None
    first = metrics[0]
    algorithm = str(first.get("algorithm", "unknown"))
    env = str(first.get("env", "unknown"))
    seed = int(first.get("seed", -1))
    handoff_fraction = None
    if algorithm == "sac_to_ppo":
        raw_fraction = first.get("handoff_fraction")
        if raw_fraction is None:
            return None
        handoff_fraction = float(raw_fraction)
        policy_init = str(first["policy_init"]) if first.get("policy_init") is not None else None
        value_init = str(first["value_init"]) if first.get("value_init") is not None else None
    else:
        policy_init = None
        value_init = None
    return RunKey(
        algorithm=algorithm,
        env=env,
        seed=seed,
        handoff_fraction=handoff_fraction,
        policy_init=policy_init,
        value_init=value_init,
    )


def summarize_run(run_dir: Path) -> RunSummary | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    metrics = load_metrics(metrics_path)
    if not metrics:
        return None

    key = run_key_from_metrics(metrics)
    if key is None:
        return None

    eval_rows = [row for row in metrics if "eval_return_mean" in row]
    initial_eval = float(eval_rows[0]["eval_return_mean"]) if eval_rows else None
    final_eval = float(eval_rows[-1]["eval_return_mean"]) if eval_rows else None
    max_env_steps = max(int(row.get("env_steps", 0)) for row in metrics)
    gradient_updates = max(int(row.get("gradient_updates", 0)) for row in metrics)
    switch_steps = [int(row["switch_step"]) for row in metrics if row.get("switch_step") is not None]
    switch_step = switch_steps[0] if switch_steps else None
    has_nan = any(is_bad_number(value) for row in metrics for value in row.values())
    has_checkpoint = any(run_dir.glob("checkpoint_step_*.pt"))
    has_handoff_phase = any(row.get("phase") == "handoff" and row.get("switched") is True for row in metrics)

    return RunSummary(
        key=key,
        run_dir=run_dir,
        initial_eval=initial_eval,
        final_eval=final_eval,
        max_env_steps=max_env_steps,
        gradient_updates=gradient_updates,
        switch_step=switch_step,
        has_nan=has_nan,
        has_checkpoint=has_checkpoint,
        has_handoff_phase=has_handoff_phase,
        metric_count=len(metrics),
    )


def latest_runs(results_dir: Path) -> dict[RunKey, RunSummary]:
    grouped: dict[RunKey, list[RunSummary]] = defaultdict(list)
    for metrics_path in results_dir.glob("*/metrics.jsonl"):
        summary = summarize_run(metrics_path.parent)
        if summary is not None:
            grouped[summary.key].append(summary)

    latest = {}
    for key, summaries in grouped.items():
        latest[key] = max(summaries, key=lambda summary: summary.run_dir.stat().st_mtime)
    return latest


def eval_series(run_dir: Path) -> list[tuple[int, float]]:
    metrics = load_metrics(run_dir / "metrics.jsonl")
    series = []
    for row in metrics:
        if "eval_return_mean" not in row:
            continue
        series.append((int(row["env_steps"]), float(row["eval_return_mean"])))
    return series


def mean_curve_for_key(
    runs: dict[RunKey, RunSummary],
    key_template: RunKey,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None] | None:
    per_seed = []
    switch_steps = []
    for seed in seeds:
        key = RunKey(
            algorithm=key_template.algorithm,
            env=key_template.env,
            seed=seed,
            handoff_fraction=key_template.handoff_fraction,
            policy_init=key_template.policy_init,
            value_init=key_template.value_init,
        )
        summary = runs.get(key)
        if summary is None:
            return None
        series = eval_series(summary.run_dir)
        if not series:
            return None
        per_seed.append({step: value for step, value in series})
        if summary.switch_step is not None:
            switch_steps.append(summary.switch_step)

    common_steps = sorted(set.intersection(*(set(series.keys()) for series in per_seed)))
    if not common_steps:
        return None

    values = np.asarray([[series[step] for step in common_steps] for series in per_seed], dtype=float)
    steps = np.asarray(common_steps, dtype=int)
    mean_switch = float(np.mean(switch_steps)) if switch_steps else None
    return steps, values.mean(axis=0), values.std(axis=0), mean_switch


def method_label(key_template: RunKey) -> str:
    if key_template.algorithm == "sac_to_ppo" and key_template.handoff_fraction is not None:
        if key_template.value_init is not None:
            return f"handoff@{key_template.handoff_fraction:.2f}/{key_template.value_init}"
        return f"handoff@{key_template.handoff_fraction:.2f}"
    return key_template.algorithm.upper()


def plot_learning_curves(
    runs: dict[RunKey, RunSummary],
    envs: list[str],
    seeds: list[int],
    switch_fractions: list[float],
    output_dir: Path,
    num_seeds: int,
) -> Path:
    colors = {
        "sac": "#1f77b4",
        "ppo": "#d62728",
        0.25: "#2ca02c",
        0.5: "#ff7f0e",
        0.75: "#9467bd",
    }
    fig, axes = plt.subplots(1, len(envs), figsize=(7 * len(envs), 6), squeeze=False)
    axes = axes.flatten()

    method_keys = [
        RunKey(algorithm="sac", env="", seed=0),
        RunKey(algorithm="ppo", env="", seed=0),
    ]
    for fraction in switch_fractions:
        method_keys.append(RunKey(algorithm="sac_to_ppo", env="", seed=0, handoff_fraction=fraction))

    for ax, env in zip(axes, envs):
        for method_key in method_keys:
            template = RunKey(
                algorithm=method_key.algorithm,
                env=env,
                seed=0,
                handoff_fraction=method_key.handoff_fraction,
            )
            curve = mean_curve_for_key(runs, template, seeds)
            if curve is None:
                continue
            steps, mean_values, std_values, mean_switch = curve
            if method_key.algorithm == "sac_to_ppo":
                color = colors.get(method_key.handoff_fraction, None)
            else:
                color = colors.get(method_key.algorithm, None)
            label = method_label(template)
            ax.plot(steps, mean_values, label=label, color=color, linewidth=2)
            ax.fill_between(steps, mean_values - std_values, mean_values + std_values, color=color, alpha=0.15)
            if mean_switch is not None and method_key.algorithm == "sac_to_ppo":
                ax.axvline(mean_switch, color=color, linestyle="--", alpha=0.5, linewidth=1)

        ax.set_title(env)
        ax.set_xlabel("Env steps")
        ax.set_ylabel("Eval return")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Experiment 2: handoff vs baselines (mean eval return, {num_seeds} seeds)", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "handoff_vs_baselines_learning_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_final_returns(
    runs: dict[RunKey, RunSummary],
    envs: list[str],
    seeds: list[int],
    switch_fractions: list[float],
    output_dir: Path,
    num_seeds: int,
) -> Path:
    method_specs = [("sac", None), ("ppo", None)]
    for fraction in switch_fractions:
        method_specs.append(("sac_to_ppo", fraction))

    x = np.arange(len(envs))
    width = 0.8 / len(method_specs)
    colors = {
        ("sac", None): "#1f77b4",
        ("ppo", None): "#d62728",
        ("sac_to_ppo", 0.25): "#2ca02c",
        ("sac_to_ppo", 0.5): "#ff7f0e",
        ("sac_to_ppo", 0.75): "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, (algorithm, fraction) in enumerate(method_specs):
        means = []
        stds = []
        for env in envs:
            finals = []
            for seed in seeds:
                key = RunKey(algorithm=algorithm, env=env, seed=seed, handoff_fraction=fraction)
                summary = runs.get(key)
                if summary is None or summary.final_eval is None:
                    finals = []
                    break
                finals.append(summary.final_eval)
            means.append(float(np.mean(finals)) if finals else np.nan)
            stds.append(float(np.std(finals)) if finals else np.nan)
        offset = (idx - (len(method_specs) - 1) / 2.0) * width
        label = "SAC" if algorithm == "sac" else "PPO" if algorithm == "ppo" else f"handoff@{fraction:.2f}"
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=label,
            color=colors.get((algorithm, fraction), None),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15)
    ax.set_ylabel("Final eval return")
    ax.set_title(f"Experiment 2: final eval return mean +/- std across {num_seeds} seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path = output_dir / "handoff_vs_baselines_final_returns.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_auc_ranking(
    runs: dict[RunKey, RunSummary],
    envs: list[str],
    seeds: list[int],
    switch_fractions: list[float],
    output_dir: Path,
) -> Path:
    method_specs = [("sac", None), ("ppo", None)]
    for fraction in switch_fractions:
        method_specs.append(("sac_to_ppo", fraction))

    fig, axes = plt.subplots(1, len(envs), figsize=(7 * len(envs), 5), squeeze=False)
    axes = axes.flatten()

    for ax, env in zip(axes, envs):
        labels = []
        auc_means = []
        auc_stds = []
        for algorithm, fraction in method_specs:
            aucs = []
            for seed in seeds:
                key = RunKey(algorithm=algorithm, env=env, seed=seed, handoff_fraction=fraction)
                summary = runs.get(key)
                if summary is None or summary.eval_auc is None:
                    aucs = []
                    break
                aucs.append(summary.eval_auc)
            if not aucs:
                continue
            labels.append("SAC" if algorithm == "sac" else "PPO" if algorithm == "ppo" else f"handoff@{fraction:.2f}")
            auc_means.append(float(np.mean(aucs)))
            auc_stds.append(float(np.std(aucs)))

        x_pos = np.arange(len(labels))
        ax.bar(x_pos, auc_means, yerr=auc_stds, capsize=4, alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(env)
        ax.set_ylabel("Eval return AUC")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Experiment 2: eval return AUC (trapezoidal over env steps)", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "handoff_auc_ranking.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def aggregate_method_stats(
    runs: dict[RunKey, RunSummary],
    envs: list[str],
    seeds: list[int],
    switch_fractions: list[float],
) -> dict:
    method_specs = [("sac", None), ("ppo", None)]
    for fraction in switch_fractions:
        method_specs.append(("sac_to_ppo", fraction))

    per_env = {}
    success = []
    for env in envs:
        env_stats = {}
        sac_final = None
        ppo_final = None
        sac_auc = None
        ppo_auc = None

        for algorithm, fraction in method_specs:
            finals = []
            aucs = []
            for seed in seeds:
                key = RunKey(algorithm=algorithm, env=env, seed=seed, handoff_fraction=fraction)
                summary = runs.get(key)
                if summary is None:
                    continue
                if summary.final_eval is not None:
                    finals.append(summary.final_eval)
                if summary.eval_auc is not None:
                    aucs.append(summary.eval_auc)
            label = "sac" if algorithm == "sac" else "ppo" if algorithm == "ppo" else f"handoff_{fraction:.2f}"
            env_stats[label] = {
                "final_return_mean": float(np.mean(finals)) if finals else None,
                "final_return_std": float(np.std(finals)) if finals else None,
                "eval_auc_mean": float(np.mean(aucs)) if aucs else None,
                "eval_auc_std": float(np.std(aucs)) if aucs else None,
            }
            if algorithm == "sac":
                sac_final = env_stats[label]["final_return_mean"]
                sac_auc = env_stats[label]["eval_auc_mean"]
            elif algorithm == "ppo":
                ppo_final = env_stats[label]["final_return_mean"]
                ppo_auc = env_stats[label]["eval_auc_mean"]

        best_fraction = None
        for fraction in switch_fractions:
            label = f"handoff_{fraction:.2f}"
            handoff_stats = env_stats.get(label)
            if handoff_stats is None:
                continue
            hf = handoff_stats["final_return_mean"]
            ha = handoff_stats["eval_auc_mean"]
            beats_final = (
                hf is not None
                and sac_final is not None
                and ppo_final is not None
                and hf > sac_final
                and hf > ppo_final
            )
            beats_auc = (
                ha is not None
                and sac_auc is not None
                and ppo_auc is not None
                and ha > sac_auc
                and ha > ppo_auc
            )
            if beats_final or beats_auc:
                success.append(
                    {
                        "env": env,
                        "handoff_fraction": fraction,
                        "beats_final_return": beats_final,
                        "beats_eval_auc": beats_auc,
                    }
                )
                if best_fraction is None or (hf is not None and hf > env_stats.get(f"handoff_{best_fraction:.2f}", {}).get("final_return_mean", -math.inf)):
                    best_fraction = fraction

        per_env[env] = {"methods": env_stats, "best_handoff_fraction": best_fraction}
    return {"per_env": per_env, "success_cases": success}


def expected_keys(
    envs: list[str],
    seeds: list[int],
    switch_fractions: list[float],
    value_inits: list[str] | None = None,
) -> list[RunKey]:
    value_inits = value_inits or [None]
    keys = []
    for env in envs:
        for seed in seeds:
            keys.append(RunKey(algorithm="sac", env=env, seed=seed))
            keys.append(RunKey(algorithm="ppo", env=env, seed=seed))
            for fraction in switch_fractions:
                for value_init in value_inits:
                    keys.append(
                        RunKey(
                            algorithm="sac_to_ppo",
                            env=env,
                            seed=seed,
                            handoff_fraction=fraction,
                            policy_init="distill" if value_init is not None else None,
                            value_init=value_init,
                        )
                    )
    return keys


def main() -> None:
    args = parse_args()
    runs = latest_runs(args.results_dir)
    failures = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment 2 summary from {args.results_dir}")
    print()

    value_inits = [None if value == "None" else value for value in args.value_inits]
    for key in expected_keys(args.envs, args.seeds, args.switch_fractions, value_inits):
        summary = runs.get(key)
        label = method_label(key)
        if summary is None:
            failures.append(f"missing run: {label} env={key.env} seed={key.seed}")
            print(f"MISS {label:>14} {key.env:>12} seed={key.seed}")
            continue

        status = "PASS"
        reasons = []
        if summary.has_nan:
            status = "FAIL"
            reasons.append("non-finite metric")
        if not args.skip_checkpoint_gate and not summary.has_checkpoint:
            status = "FAIL"
            reasons.append("missing checkpoint")
        if key.algorithm == "sac_to_ppo" and not summary.has_handoff_phase:
            status = "FAIL"
            reasons.append("missing handoff phase")

        if status == "FAIL":
            failures.append(f"{label} {key.env} seed={key.seed}: {', '.join(reasons)}")

        reason_text = "" if not reasons else f" ({', '.join(reasons)})"
        print(
            f"{status} {label:>14} {key.env:>12} seed={key.seed} "
            f"final={summary.final_eval if summary.final_eval is not None else 'missing'} "
            f"auc={summary.eval_auc if summary.eval_auc is not None else 'missing'} "
            f"switch={summary.switch_step}{reason_text}"
        )

    print()
    if failures:
        print("Experiment 2 gate: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Experiment 2 gate: PASS")
    summary_stats = aggregate_method_stats(runs, args.envs, args.seeds, args.switch_fractions)
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    num_seeds = len(args.seeds)
    learning_curves_path = plot_learning_curves(
        runs, args.envs, args.seeds, args.switch_fractions, args.output_dir, num_seeds
    )
    final_returns_path = plot_final_returns(
        runs, args.envs, args.seeds, args.switch_fractions, args.output_dir, num_seeds
    )
    auc_path = plot_auc_ranking(runs, args.envs, args.seeds, args.switch_fractions, args.output_dir)

    print(f"Saved learning curves plot to {learning_curves_path}")
    print(f"Saved final returns plot to {final_returns_path}")
    print(f"Saved AUC ranking plot to {auc_path}")
    print(f"Saved summary JSON to {summary_path}")

    if summary_stats["success_cases"]:
        print("Success criteria met for:")
        for case in summary_stats["success_cases"]:
            print(
                f"- {case['env']} handoff@{case['handoff_fraction']:.2f} "
                f"(final={case['beats_final_return']}, auc={case['beats_eval_auc']})"
            )
    else:
        print("Success criteria: no handoff setting beat both baselines on final return or AUC.")


if __name__ == "__main__":
    main()
