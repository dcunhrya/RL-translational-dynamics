import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Ethan PPO->SAC task runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/raw/ethan_task"))
    parser.add_argument("--extra-results-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed/ethan_task"))
    parser.add_argument("--notes-dir", type=Path, default=Path("experiments/ethan_task"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_compact_metrics(path: Path) -> list[dict]:
    rows = []
    first = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if first is None:
                first = row
                rows.append(row)
                continue
            if row.get("eval_return_mean") is not None or row.get("switched"):
                rows.append(row)
    return rows


def run_key(rows: list[dict], run_dir: Path) -> tuple:
    first = rows[0]
    env = first.get("env", "unknown")
    seed = int(first.get("seed", -1))
    policy_init = first.get("policy_init", "unknown")
    value_init = first.get("value_init", "unknown")
    switch_trigger = first.get("switch_trigger", "fixed_fraction")
    handoff_fraction = float(first.get("handoff_fraction", 0.0) or 0.0)
    algorithm = first.get("algorithm", run_dir.name.split("__")[0])
    return algorithm, env, seed, policy_init, value_init, switch_trigger, handoff_fraction


def load_runs(results_dirs: list[Path]) -> dict[tuple, list[dict]]:
    latest = {}
    for results_dir in results_dirs:
        if not results_dir.exists():
            print(f"Skipping missing results dir: {results_dir}", flush=True)
            continue
        metrics_paths = list(results_dir.rglob("metrics.jsonl"))
        print(f"Found {len(metrics_paths)} metric files under {results_dir}", flush=True)
        for index, metrics_path in enumerate(metrics_paths, start=1):
            if index == 1 or index % 10 == 0 or index == len(metrics_paths):
                print(f"Reading {index}/{len(metrics_paths)}: {metrics_path.parent.name}", flush=True)
            rows = read_compact_metrics(metrics_path)
            if not rows:
                continue
            key = run_key(rows, metrics_path.parent)
            stat = metrics_path.stat()
            previous = latest.get(key)
            if previous is None or stat.st_mtime >= previous[0]:
                latest[key] = (stat.st_mtime, rows)
    return {key: rows for key, (_, rows) in latest.items()}


def eval_points(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    points = [
        (float(row["env_steps"]), float(row["eval_return_mean"]))
        for row in rows
        if row.get("eval_return_mean") is not None and row.get("env_steps") is not None
    ]
    if not points:
        return np.array([], dtype=float), np.array([], dtype=float)
    points = sorted(set(points))
    return np.asarray([p[0] for p in points], dtype=float), np.asarray([p[1] for p in points], dtype=float)


def auc(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return float("nan")
    return float(np.trapz(ys, xs) / max(xs[-1] - xs[0], 1.0))


def final_return(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) == 0:
        return float("nan")
    return float(ys[-1])


def ci(values: list[float], samples: int) -> tuple[float, float, float]:
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    if clean.size == 1:
        return float(clean[0]), float(clean[0]), float(clean[0])
    rng = np.random.default_rng(0)
    means = []
    for _ in range(samples):
        means.append(float(rng.choice(clean, size=clean.size, replace=True).mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(clean.mean()), float(lo), float(hi)


def arm_name(key: tuple) -> str:
    _, _, _, policy_init, value_init, switch_trigger, handoff_fraction = key
    if switch_trigger == "no-improve":
        return f"adaptive {policy_init}/{value_init}"
    return f"{int(round(handoff_fraction * 100))}% {policy_init}/{value_init}"


def summarize_runs(runs: dict[tuple, list[dict]], bootstrap_samples: int) -> list[dict]:
    grouped = defaultdict(list)
    for key, rows in runs.items():
        algorithm, env, seed, policy_init, value_init, switch_trigger, handoff_fraction = key
        xs, ys = eval_points(rows)
        grouped[(algorithm, env, policy_init, value_init, switch_trigger, handoff_fraction)].append(
            {
                "seed": seed,
                "auc": auc(xs, ys),
                "final_return": final_return(xs, ys),
                "switch_step": next((r.get("switch_step") for r in rows if r.get("switched")), None),
            }
        )

    summaries = []
    for key, seed_rows in sorted(grouped.items()):
        algorithm, env, policy_init, value_init, switch_trigger, handoff_fraction = key
        auc_mean, auc_lo, auc_hi = ci([r["auc"] for r in seed_rows], bootstrap_samples)
        final_mean, final_lo, final_hi = ci([r["final_return"] for r in seed_rows], bootstrap_samples)
        switch_steps = [r["switch_step"] for r in seed_rows if r["switch_step"] is not None]
        summaries.append(
            {
                "algorithm": algorithm,
                "env": env,
                "policy_init": policy_init,
                "value_init": value_init,
                "switch_trigger": switch_trigger,
                "handoff_fraction": handoff_fraction,
                "seeds": len(seed_rows),
                "auc_mean": auc_mean,
                "auc_ci_low": auc_lo,
                "auc_ci_high": auc_hi,
                "final_mean": final_mean,
                "final_ci_low": final_lo,
                "final_ci_high": final_hi,
                "mean_switch_step": float(np.mean(switch_steps)) if switch_steps else float("nan"),
            }
        )
    return summaries


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(key, "")) for key in keys))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_learning_curves(runs: dict[tuple, list[dict]], output_dir: Path) -> list[Path]:
    generated = []
    by_env = defaultdict(list)
    for key, rows in runs.items():
        by_env[key[1]].append((key, rows))

    for env, env_runs in by_env.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for key, rows in sorted(env_runs, key=lambda item: arm_name(item[0])):
            xs, ys = eval_points(rows)
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, alpha=0.55, linewidth=1.4, label=f"{arm_name(key)} seed {key[2]}")
        ax.set_title(f"{env}: Ethan PPO->SAC learning curves")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Evaluation return")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{env.replace('-', '_')}_learning_curves.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        generated.append(output_path)
    return generated


def plot_summary_bars(summaries: list[dict], output_dir: Path) -> list[Path]:
    generated = []
    by_env = defaultdict(list)
    for row in summaries:
        by_env[row["env"]].append(row)
    for env, rows in by_env.items():
        rows = sorted(rows, key=lambda r: (r["switch_trigger"], r["handoff_fraction"], r["value_init"]))
        labels = [
            f"{int(round(r['handoff_fraction'] * 100))}%\n{r['value_init']}"
            if r["switch_trigger"] != "no-improve"
            else f"adaptive\n{r['value_init']}"
            for r in rows
        ]
        values = [r["auc_mean"] for r in rows]
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.8), 4.8))
        ax.bar(np.arange(len(rows)), values)
        ax.set_title(f"{env}: AUC summary")
        ax.set_ylabel("Mean eval-return AUC")
        ax.set_xticks(np.arange(len(rows)), labels, rotation=0)
        ax.grid(axis="y", alpha=0.25)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{env.replace('-', '_')}_auc_summary.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        generated.append(output_path)
    return generated


def write_markdown(summaries: list[dict], generated: list[Path], notes_dir: Path, output_dir: Path) -> Path:
    notes_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ethan Task Results",
        "",
        "## Scope",
        "",
        "- PPO -> SAC value ablation: `random`, `self-warmup`, `source-aligned`.",
        "- Timing sweep: fixed 25% and 75% switches, with 50% supplied by the ablation arm.",
        "- Adaptive trigger: no-improvement after a minimum first-phase budget.",
        "- Long-horizon checks: pure PPO and one PPO -> SAC arm on Hopper when included in the Modal launch.",
        "",
        "## Summary Table",
        "",
        "| Env | Trigger | Switch | Policy | Value | Seeds | AUC mean [95% CI] | Final mean [95% CI] | Mean switch step |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {env} | {trigger} | {switch:.2f} | {policy} | {value} | {seeds} | "
            "{auc:.2f} [{auc_lo:.2f}, {auc_hi:.2f}] | {final:.2f} [{final_lo:.2f}, {final_hi:.2f}] | {switch_step:.1f} |".format(
                env=row["env"],
                trigger=row["switch_trigger"],
                switch=row["handoff_fraction"],
                policy=row["policy_init"],
                value=row["value_init"],
                seeds=row["seeds"],
                auc=row["auc_mean"],
                auc_lo=row["auc_ci_low"],
                auc_hi=row["auc_ci_high"],
                final=row["final_mean"],
                final_lo=row["final_ci_low"],
                final_hi=row["final_ci_high"],
                switch_step=row["mean_switch_step"],
            )
        )
    lines.extend(["", "## Generated Artifacts", ""])
    for path in generated:
        lines.append(f"- `{path}`")
    lines.extend(["", f"CSV summary: `{output_dir / 'summary.csv'}`", ""])
    notes_path = notes_dir / "results.md"
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    return notes_path


def main() -> None:
    args = parse_args()
    results_dirs = [args.results_dir] + args.extra_results_dir
    runs = load_runs(results_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_runs(runs, args.bootstrap_samples)
    write_csv(summaries, args.output_dir / "summary.csv")
    generated = []
    generated.extend(plot_learning_curves(runs, args.output_dir))
    generated.extend(plot_summary_bars(summaries, args.output_dir))
    notes_path = write_markdown(summaries, generated, args.notes_dir, args.output_dir)
    print(f"Loaded {len(runs)} runs")
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {notes_path}")


if __name__ == "__main__":
    main()
