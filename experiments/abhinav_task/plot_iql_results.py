"""Generate presentation figures for the Abhinav IQL transfer sweep.

The raw metric files are large because online training logs dense diagnostics.
This script selects the latest complete run for each (env, algorithm, seed)
using cheap first/last-line reads, then streams only evaluation rows for the
learning-curve figures.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ENVS = ("Hopper-v4", "Walker2d-v4")
ALGORITHMS = ("iql_to_ppo", "iql_to_sac", "iql_to_sac_to_ppo")
SEEDS = tuple(range(5))

METHOD_LABELS = {
    "iql_to_ppo": "IQL -> PPO",
    "iql_to_sac": "IQL -> SAC",
    "iql_to_sac_to_ppo": "IQL -> SAC -> PPO",
}

METHOD_COLORS = {
    "iql_to_ppo": "#4C78A8",
    "iql_to_sac": "#F58518",
    "iql_to_sac_to_ppo": "#54A24B",
}

OFFLINE_IQL_EVAL = {
    "Hopper-v4": 3044.3794803381297,
    "Walker2d-v4": 4928.855749405985,
}


@dataclass(frozen=True)
class RunRef:
    env: str
    algorithm: str
    seed: int
    steps: int
    path: Path
    mtime: float


def read_first_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"No JSON rows found in {path}")


def read_last_line(path: Path, chunk_size: int = 8192) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
        buffer = b""
        offset = 0
        while file_size - offset > 0:
            offset = min(file_size, offset + chunk_size)
            handle.seek(file_size - offset)
            buffer = handle.read(min(chunk_size, file_size - offset + chunk_size)) + buffer
            lines = [line for line in buffer.splitlines() if line.strip()]
            if len(lines) >= 1:
                return lines[-1].decode("utf-8")
    raise ValueError(f"No last line found in {path}")


def discover_latest_runs(results_dir: Path) -> dict[tuple[str, str, int], RunRef]:
    latest: dict[tuple[str, str, int], RunRef] = {}
    for metrics_path in results_dir.rglob("metrics.jsonl"):
        try:
            first = read_first_json(metrics_path)
            last = json.loads(read_last_line(metrics_path))
        except Exception as exc:
            print(f"Skipping unreadable metrics file {metrics_path}: {exc}")
            continue

        env = first.get("env")
        algorithm = first.get("algorithm")
        seed = first.get("seed")
        if env not in ENVS or algorithm not in ALGORITHMS or seed not in SEEDS:
            continue

        steps = int(last.get("env_steps") or 0)
        ref = RunRef(
            env=env,
            algorithm=algorithm,
            seed=int(seed),
            steps=steps,
            path=metrics_path,
            mtime=metrics_path.stat().st_mtime,
        )
        key = (ref.env, ref.algorithm, ref.seed)
        old = latest.get(key)
        if old is None or (ref.steps, ref.mtime) > (old.steps, old.mtime):
            latest[key] = ref
    return latest


def verify_complete_runs(runs: dict[tuple[str, str, int], RunRef]) -> list[RunRef]:
    selected: list[RunRef] = []
    missing: list[tuple[str, str, int]] = []
    incomplete: list[RunRef] = []
    for env in ENVS:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                key = (env, algorithm, seed)
                ref = runs.get(key)
                if ref is None:
                    missing.append(key)
                    continue
                if ref.steps < 500_000:
                    incomplete.append(ref)
                selected.append(ref)
    if missing or incomplete:
        msg = []
        if missing:
            msg.append(f"missing={missing}")
        if incomplete:
            msg.append(
                "incomplete="
                + repr([(r.env, r.algorithm, r.seed, r.steps, str(r.path)) for r in incomplete])
            )
        raise RuntimeError("IQL sweep is not complete: " + "; ".join(msg))
    return selected


def iter_eval_rows(refs: Iterable[RunRef]) -> Iterable[dict]:
    for ref in refs:
        with ref.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                value = row.get("eval_return_mean")
                if value is None:
                    continue
                yield {
                    "env": ref.env,
                    "algorithm": ref.algorithm,
                    "seed": ref.seed,
                    "env_steps": int(row.get("env_steps") or 0),
                    "eval_return_mean": float(value),
                    "phase": row.get("phase"),
                }


def load_summary(summary_path: Path) -> pd.DataFrame:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for row in data.get("runs", []):
        rows.append(
            {
                "env": row["env"],
                "algorithm": row["algorithm"],
                "method": METHOD_LABELS[row["algorithm"]],
                "final_return_mean": row["final_return_mean"],
                "final_return_ci_low": row["final_return_ci_low"],
                "final_return_ci_high": row["final_return_ci_high"],
                "normalized_auc_mean": row["normalized_auc_mean"],
                "normalized_auc_ci_low": row["normalized_auc_ci_low"],
                "normalized_auc_ci_high": row["normalized_auc_ci_high"],
                "worst_seed_final_return": row["worst_seed_final_return"],
                "collapse_count_final_lt_100": row["collapse_count_final_lt_100"],
            }
        )
    return pd.DataFrame(rows)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E6E8EB", linewidth=0.8)
    ax.grid(True, axis="x", color="#F1F3F5", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=9)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(eval_df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        eval_df.groupby(["env", "algorithm", "env_steps"], as_index=False)
        .agg(mean_return=("eval_return_mean", "mean"), sem_return=("eval_return_mean", "sem"))
        .sort_values(["env", "algorithm", "env_steps"])
    )
    grouped["sem_return"] = grouped["sem_return"].fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True)
    for ax, env in zip(axes, ENVS, strict=True):
        env_df = grouped[grouped["env"] == env]
        for algorithm in ALGORITHMS:
            curve = env_df[env_df["algorithm"] == algorithm]
            x = curve["env_steps"].to_numpy(dtype=float) / 1000.0
            y = curve["mean_return"].to_numpy(dtype=float)
            sem = curve["sem_return"].to_numpy(dtype=float)
            color = METHOD_COLORS[algorithm]
            ax.plot(x, y, color=color, linewidth=2.4, label=METHOD_LABELS[algorithm])
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.16, linewidth=0)

        ax.axvline(250, color="#6B7280", linestyle="--", linewidth=1.3, alpha=0.8)
        ax.axhline(
            OFFLINE_IQL_EVAL[env],
            color="#111827",
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
        )
        ax.text(
            252,
            ax.get_ylim()[1] * 0.93,
            "SAC -> PPO switch",
            color="#4B5563",
            fontsize=8,
            rotation=90,
            va="top",
        )
        ax.set_title(env, fontsize=13, weight="bold")
        ax.set_xlabel("Online environment steps (thousands)", fontsize=10)
        ax.set_ylabel("Evaluation return", fontsize=10)
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.8,
    )
    fig.suptitle("IQL Warm-Start Online Transfer Learning Curves", fontsize=15, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.035,
        "Lines show mean over 5 seeds; shaded bands show standard error. Dotted horizontal line is the 100k-update offline IQL policy evaluation.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.84))
    save_figure(fig, output_dir, "iql_learning_curves")


def plot_summary_bars(summary_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("final_return_mean", "final_return_ci_low", "final_return_ci_high", "Final Return"),
        ("normalized_auc_mean", "normalized_auc_ci_low", "normalized_auc_ci_high", "Normalized AUC"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    x = np.arange(len(ENVS))
    width = 0.24
    offsets = {
        "iql_to_ppo": -width,
        "iql_to_sac": 0.0,
        "iql_to_sac_to_ppo": width,
    }
    for ax, (mean_col, low_col, high_col, title) in zip(axes, metrics, strict=True):
        metric_high = float(summary_df[high_col].max())
        label_pad = metric_high * 0.018
        for algorithm in ALGORITHMS:
            rows = summary_df[summary_df["algorithm"] == algorithm].set_index("env").loc[list(ENVS)]
            means = rows[mean_col].to_numpy(dtype=float)
            lows = rows[low_col].to_numpy(dtype=float)
            highs = rows[high_col].to_numpy(dtype=float)
            yerr = np.vstack([means - lows, highs - means])
            bars = ax.bar(
                x + offsets[algorithm],
                means,
                width=width,
                label=METHOD_LABELS[algorithm],
                color=METHOD_COLORS[algorithm],
                edgecolor="white",
                linewidth=1.0,
            )
            ax.errorbar(
                x + offsets[algorithm],
                means,
                yerr=yerr,
                fmt="none",
                ecolor="#111827",
                elinewidth=1.1,
                capsize=3,
                capthick=1.1,
            )
            for bar, mean, high in zip(bars, means, highs, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    high + label_pad,
                    f"{mean:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#111827",
                )
        ax.set_xticks(x, ENVS)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_ylabel("Return", fontsize=10)
        ax.set_ylim(0, metric_high * 1.18)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.8,
    )
    fig.suptitle("IQL Transfer Aggregate Performance", fontsize=15, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.03, 1, 0.82))
    save_figure(fig, output_dir, "iql_final_auc_summary")


def plot_handoff_retention(eval_df: pd.DataFrame, output_dir: Path) -> None:
    handoff = eval_df[eval_df["algorithm"] == "iql_to_sac_to_ppo"].copy()
    rows = []
    for (env, seed), seed_df in handoff.groupby(["env", "seed"]):
        seed_df = seed_df.sort_values("env_steps")
        pre = seed_df[seed_df["env_steps"] <= 250_000].iloc[-1]["eval_return_mean"]
        final = seed_df.iloc[-1]["eval_return_mean"]
        rows.append({"env": env, "seed": seed, "pre_switch": pre, "final": final})
    paired = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=False)
    for ax, env in zip(axes, ENVS, strict=True):
        env_df = paired[paired["env"] == env].sort_values("seed")
        xs = np.array([0, 1], dtype=float)
        for _, row in env_df.iterrows():
            ax.plot(
                xs,
                [row["pre_switch"], row["final"]],
                color="#9CA3AF",
                linewidth=1.2,
                alpha=0.85,
                zorder=1,
            )
            ax.scatter(
                xs,
                [row["pre_switch"], row["final"]],
                s=38,
                color=["#F58518", "#54A24B"],
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
        mean_pre = env_df["pre_switch"].mean()
        mean_final = env_df["final"].mean()
        ax.plot(xs, [mean_pre, mean_final], color="#111827", linewidth=3.0, zorder=3)
        ax.set_xticks(xs, ["Before PPO", "Final"])
        ax.set_title(env, fontsize=13, weight="bold")
        ax.set_ylabel("Evaluation return", fontsize=10)
        ax.text(
            0.5,
            max(mean_pre, mean_final),
            f"mean {mean_pre:.0f} -> {mean_final:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )
        style_axes(ax)
    fig.suptitle("IQL -> SAC -> PPO Retention Across the PPO Phase", fontsize=15, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.035,
        "Each gray line is one seed. The thick black line shows the seed mean before the 250k-step switch and at the end of budget.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.88))
    save_figure(fig, output_dir, "iql_sac_ppo_retention")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/raw/abhinav_task/tier2_iql"))
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/processed/abhinav_task_iql_local/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/abhinav_task/figures"))
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    runs = verify_complete_runs(discover_latest_runs(args.results_dir))
    eval_df = pd.DataFrame(iter_eval_rows(runs))
    summary_df = load_summary(args.summary_json)

    plot_learning_curves(eval_df, args.output_dir)
    plot_summary_bars(summary_df, args.output_dir)
    plot_handoff_retention(eval_df, args.output_dir)
    print(f"Wrote IQL figures to {args.output_dir}")


if __name__ == "__main__":
    main()
