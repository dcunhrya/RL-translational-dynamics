import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


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
    parser.add_argument("--envs", nargs="+", default=["Hopper-v4", "Walker2d-v4"])
    parser.add_argument("--algorithms", nargs="+", default=["sac", "ppo"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--gate-env", default="Hopper-v4")
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


def main() -> None:
    args = parse_args()
    runs = latest_runs(args.results_dir)
    failures = []

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


if __name__ == "__main__":
    main()
