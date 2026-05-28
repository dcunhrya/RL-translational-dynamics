#!/usr/bin/env python3
"""Build the static Ryan results bundle from audited W&B runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ENTITY = "rdcunha-stanford-university"
PROJECT = "rl-translational-dynamics"
METHOD_ORDER = [
    ("sac", None),
    ("ppo", None),
    ("sac_to_ppo", "random"),
    ("sac_to_ppo", "self-warmup"),
    ("sac_to_ppo", "source-aligned"),
]
METHOD_COLORS = {
    "SAC": "#1f77b4",
    "PPO": "#d62728",
    "SAC->PPO random V": "#ff7f0e",
    "SAC->PPO self-warmup V": "#2ca02c",
    "SAC->PPO source-aligned V": "#9467bd",
}
DIAGNOSTIC_KEYS = [
    "handoff_action_mse",
    "handoff_action_kl_proxy",
    "source_value_warmup_loss_initial",
    "source_value_warmup_loss_final",
    "source_value_warmup_steps",
    "source_value_warmup_converged",
    "value_warmup_fallback_to_self",
    "ppo_explained_variance",
    "ppo_value_loss",
    "ppo_policy_loss",
    "ppo_entropy",
    "ppo_approx_kl",
    "advantages_mean",
    "advantages_std",
    "value_loss_at_handoff",
]
METRIC_EXTRACT_PATTERN = "|".join(["eval_return_mean", *DIAGNOSTIC_KEYS])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Ryan's static results bundle.")
    parser.add_argument("--audit-path", type=Path, default=Path("experiments/ryan_requeue_audit_latest.json"))
    parser.add_argument("--manifest-path", type=Path, default=Path("experiments/ryan_modal_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ryan_task"))
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--source", choices=["modal", "wandb"], default="modal")
    parser.add_argument("--modal-volume", default="herschethan")
    parser.add_argument("--remote-prefix", default="raw/ryan_experiment")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--refresh-wandb", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=224)
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def clean_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for key, value in row.items():
        if key.startswith("_") and key not in {"_step", "_timestamp"}:
            continue
        if isinstance(value, (dict, list, tuple)):
            continue
        clean[key] = clean_scalar(value)
    return clean


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row)
        fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def method_label(algorithm: str, value_init: str | None) -> str:
    if algorithm == "sac":
        return "SAC"
    if algorithm == "ppo":
        return "PPO"
    if value_init == "random":
        return "SAC->PPO random V"
    if value_init == "self-warmup":
        return "SAC->PPO self-warmup V"
    if value_init == "source-aligned":
        return "SAC->PPO source-aligned V"
    return "SAC->PPO"


def spec_key(spec: dict[str, Any]) -> tuple[str, str, int, int, str | None]:
    return (
        str(spec["kind"]),
        str(spec["env_id"]),
        int(spec["seed"]),
        int(spec["total_timesteps"]),
        spec.get("value_init"),
    )


def spec_slug(spec: dict[str, Any]) -> str:
    value = spec.get("value_init") or "na"
    return slugify(f"{spec['kind']}__{spec['env_id']}__seed_{spec['seed']}__{spec['total_timesteps']}__{value}")


def selected_audit_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in audit["audit"]:
        if not row.get("complete"):
            continue
        if not row.get("best_run"):
            continue
        rows.append(row)
    rows.sort(key=lambda row: spec_key(row["spec"]))
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def export_wandb_histories(
    audit_rows: list[dict[str, Any]],
    raw_dir: Path,
    entity: str,
    project: str,
    refresh: bool,
) -> list[dict[str, Any]]:
    import wandb

    runs_dir = raw_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=90)
    manifest_rows = []
    for idx, audit_row in enumerate(audit_rows, start=1):
        spec = audit_row["spec"]
        best = audit_row["best_run"]
        slug = spec_slug(spec)
        history_path = runs_dir / f"{slug}.jsonl"
        meta_path = runs_dir / f"{slug}.meta.json"
        if refresh or not history_path.exists():
            run = api.run(f"{entity}/{project}/{best['id']}")
            history = [clean_row(row) for row in run.scan_history(page_size=1000)]
            write_jsonl(history_path, history)
            metadata = {
                "audit": audit_row,
                "run": {
                    "id": run.id,
                    "name": run.name,
                    "group": run.group,
                    "state": run.state,
                    "config": dict(run.config or {}),
                    "summary": clean_row(dict(run.summary or {})),
                },
            }
            write_json(meta_path, metadata)
        history_bytes = history_path.stat().st_size if history_path.exists() else 0
        manifest_rows.append(
            {
                **spec,
                "run_id": best["id"],
                "run_name": best["name"],
                "run_group": best["group"],
                "run_state": best["state"],
                "best_env_steps": best.get("env_steps"),
                "attempt_count": audit_row.get("attempt_count", 0),
                "history_path": str(history_path),
                "metadata_path": str(meta_path),
                "history_bytes": history_bytes,
                "export_index": idx,
            }
        )
        print(f"exported/cache {idx:02d}/{len(audit_rows)} {slug} bytes={history_bytes}")
    write_csv(raw_dir / "run_manifest.csv", manifest_rows)
    write_json(raw_dir / "run_manifest.json", manifest_rows)
    return manifest_rows


def export_modal_histories(
    audit_rows: list[dict[str, Any]],
    raw_dir: Path,
    volume: str,
    remote_prefix: str,
    refresh: bool,
    workers: int,
) -> list[dict[str, Any]]:
    runs_dir = raw_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    prepared_rows = []
    download_jobs = []
    for idx, audit_row in enumerate(audit_rows, start=1):
        spec = audit_row["spec"]
        best = audit_row["best_run"]
        slug = spec_slug(spec)
        history_path = runs_dir / f"{slug}.jsonl"
        meta_path = runs_dir / f"{slug}.meta.json"
        remote_metrics = f"{remote_prefix.rstrip('/')}/{best['name']}/metrics.jsonl"
        prepared_rows.append((idx, spec, best, audit_row, history_path, meta_path, remote_metrics))
        if refresh or not history_path.exists():
            download_jobs.append((idx, remote_metrics, history_path, meta_path, audit_row, best))

    def download_one(job: tuple[int, str, Path, Path, dict[str, Any], dict[str, Any]]) -> int:
        idx, remote_metrics, history_path, meta_path, audit_row, best = job
        command = ["modal", "volume", "get", volume, remote_metrics, str(history_path), "--force"]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        write_json(
            meta_path,
            {
                "audit": audit_row,
                "run": best,
                "source": {
                    "kind": "modal_volume",
                    "volume": volume,
                    "remote_metrics": remote_metrics,
                },
            },
        )
        return idx

    if download_jobs:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(download_one, job): job for job in download_jobs}
            for future in as_completed(futures):
                idx = future.result()
                print(f"downloaded {idx:02d}/{len(audit_rows)}", flush=True)

    manifest_rows = []
    for idx, spec, best, audit_row, history_path, meta_path, remote_metrics in prepared_rows:
        slug = spec_slug(spec)
        history_bytes = history_path.stat().st_size if history_path.exists() else 0
        manifest_rows.append(
            {
                **spec,
                "run_id": best["id"],
                "run_name": best["name"],
                "run_group": best["group"],
                "run_state": best["state"],
                "best_env_steps": best.get("env_steps"),
                "attempt_count": audit_row.get("attempt_count", 0),
                "history_path": str(history_path),
                "metadata_path": str(meta_path),
                "history_bytes": history_bytes,
                "export_index": idx,
                "source": "modal_volume",
                "remote_metrics": remote_metrics,
            }
        )
        print(f"fetched/cache {idx:02d}/{len(audit_rows)} {slug} bytes={history_bytes}", flush=True)
    write_csv(raw_dir / "run_manifest.csv", manifest_rows)
    write_json(raw_dir / "run_manifest.json", manifest_rows)
    return manifest_rows


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def int_value(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def eval_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for row in history:
        value = finite_float(row.get("eval_return_mean"))
        if value is None:
            continue
        step = int_value(row.get("env_steps", row.get("_step", 0)))
        key = (step, value)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "env_steps": step,
                "eval_return_mean": value,
                "eval_return_std": finite_float(row.get("eval_return_std")),
                "phase": row.get("phase"),
            }
        )
    rows.sort(key=lambda row: row["env_steps"])
    return rows


def extracted_metric_rows(path: Path) -> list[dict[str, Any]]:
    command = [
        "rg",
        "--no-heading",
        "--no-filename",
        METRIC_EXTRACT_PATTERN,
        str(path),
    ]
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    if proc.returncode not in {0, 1}:
        raise RuntimeError(f"rg failed for {path}: {proc.stderr}")
    rows = []
    for line_number, line in enumerate(proc.stdout.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid extracted JSON in {path} extracted line {line_number}: {exc}") from exc
    return rows


def trapezoid_auc(series: list[dict[str, Any]], max_step: int) -> tuple[float | None, float | None]:
    if len(series) < 2:
        return None, None
    steps = np.asarray([row["env_steps"] for row in series if row["env_steps"] <= max_step], dtype=float)
    values = np.asarray([row["eval_return_mean"] for row in series if row["env_steps"] <= max_step], dtype=float)
    if len(steps) < 2:
        return None, None
    auc = float(np.trapz(values, steps))
    return auc, auc / float(max_step)


def summarize_run(manifest_row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    spec = manifest_row
    total_steps = int(spec["total_timesteps"])
    evals = []
    diagnostic_rows = []
    history_count = ""
    max_env_steps = int_value(manifest_row.get("best_env_steps"), total_steps)
    gradient_updates = 0
    seen_evals = set()
    for row in extracted_metric_rows(Path(manifest_row["history_path"])):
        step = int_value(row.get("env_steps", row.get("_step", 0)))
        max_env_steps = max(max_env_steps, step)
        gradient_updates = max(gradient_updates, int_value(row.get("gradient_updates", 0)))

        eval_value = finite_float(row.get("eval_return_mean"))
        if eval_value is not None:
            eval_key = (step, eval_value)
            if eval_key not in seen_evals:
                seen_evals.add(eval_key)
                evals.append(
                    {
                        "env_steps": step,
                        "eval_return_mean": eval_value,
                        "eval_return_std": finite_float(row.get("eval_return_std")),
                        "phase": row.get("phase"),
                    }
                )

        diag = {key: row.get(key) for key in DIAGNOSTIC_KEYS if row.get(key) is not None}
        if not diag:
            continue
        is_handoff_marker = row.get("phase") in {"handoff", "source_value_warmup", "ppo_value_warmup"}
        has_transfer_scalar = any(
            key in diag
            for key in (
                "handoff_action_mse",
                "handoff_action_kl_proxy",
                "source_value_warmup_loss_initial",
                "source_value_warmup_loss_final",
                "value_loss_at_handoff",
            )
        )
        if not (is_handoff_marker or has_transfer_scalar or step % 10_000 == 0 or step == total_steps):
            continue
        diagnostic_rows.append(
            {
                "algorithm": spec["kind"],
                "method": method_label(spec["kind"], spec.get("value_init")),
                "env": spec["env_id"],
                "seed": int(spec["seed"]),
                "total_timesteps": total_steps,
                "value_init": spec.get("value_init") or "",
                "env_steps": step,
                "phase": row.get("phase") or "",
                **diag,
            }
        )

    evals.sort(key=lambda row: row["env_steps"])
    auc, normalized_auc = trapezoid_auc(evals, total_steps)
    final_eval = evals[-1]["eval_return_mean"] if evals else None
    initial_eval = evals[0]["eval_return_mean"] if evals else None
    best_eval = max((row["eval_return_mean"] for row in evals), default=None)
    late_cutoff = int(total_steps * 0.8)
    late_values = [row["eval_return_mean"] for row in evals if row["env_steps"] >= late_cutoff]
    worst_late = min(late_values) if late_values else None
    label = method_label(spec["kind"], spec.get("value_init"))
    run_summary = {
        "algorithm": spec["kind"],
        "method": label,
        "env": spec["env_id"],
        "seed": int(spec["seed"]),
        "total_timesteps": total_steps,
        "value_init": spec.get("value_init") or "",
        "run_id": manifest_row["run_id"],
        "run_name": manifest_row["run_name"],
        "run_group": manifest_row["run_group"],
        "run_state": manifest_row["run_state"],
        "attempt_count": manifest_row.get("attempt_count", 0),
        "history_rows": history_count,
        "history_bytes": manifest_row.get("history_bytes", 0),
        "eval_points": len(evals),
        "initial_return": initial_eval,
        "final_return": final_eval,
        "best_return": best_eval,
        "worst_late_return": worst_late,
        "auc": auc,
        "normalized_auc": normalized_auc,
        "max_env_steps": max_env_steps,
        "gradient_updates": gradient_updates,
        "reached_budget": max_env_steps >= total_steps,
    }
    eval_table = [
        {
            "algorithm": spec["kind"],
            "method": label,
            "env": spec["env_id"],
            "seed": int(spec["seed"]),
            "total_timesteps": total_steps,
            "value_init": spec.get("value_init") or "",
            **row,
        }
        for row in evals
    ]
    return run_summary, eval_table, diagnostic_rows


def bootstrap_ci(values: list[float], rng: np.random.Generator, samples: int) -> tuple[float | None, float | None]:
    clean = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if clean.size == 0:
        return None, None
    if clean.size == 1:
        return float(clean[0]), float(clean[0])
    draws = rng.choice(clean, size=(samples, clean.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def aggregate_metrics(
    run_rows: list[dict[str, Any]],
    rng: np.random.Generator,
    samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        if row["total_timesteps"] != 500000:
            continue
        groups[(row["env"], row["method"], row["total_timesteps"], row["value_init"])].append(row)

    summary_rows = []
    for (env, method, total_steps, value_init), rows in sorted(groups.items()):
        finals = [finite_float(row["final_return"]) for row in rows]
        aucs = [finite_float(row["auc"]) for row in rows]
        norm_aucs = [finite_float(row["normalized_auc"]) for row in rows]
        final_clean = [value for value in finals if value is not None]
        auc_clean = [value for value in aucs if value is not None]
        norm_auc_clean = [value for value in norm_aucs if value is not None]
        final_low, final_high = bootstrap_ci(final_clean, rng, samples)
        auc_low, auc_high = bootstrap_ci(auc_clean, rng, samples)
        norm_low, norm_high = bootstrap_ci(norm_auc_clean, rng, samples)
        peak_by_seed = {row["seed"]: finite_float(row["best_return"]) for row in rows}
        collapse_count = 0
        for row in rows:
            final = finite_float(row["final_return"])
            peak = peak_by_seed.get(row["seed"])
            if final is not None and peak is not None and peak > 0 and final < 0.7 * peak:
                collapse_count += 1
        summary_rows.append(
            {
                "env": env,
                "method": method,
                "total_timesteps": total_steps,
                "value_init": value_init,
                "seed_count": len(rows),
                "final_mean": float(np.mean(final_clean)) if final_clean else None,
                "final_std": float(np.std(final_clean, ddof=1)) if len(final_clean) > 1 else 0.0,
                "final_sem": float(np.std(final_clean, ddof=1) / math.sqrt(len(final_clean))) if len(final_clean) > 1 else 0.0,
                "final_ci_low": final_low,
                "final_ci_high": final_high,
                "auc_mean": float(np.mean(auc_clean)) if auc_clean else None,
                "auc_std": float(np.std(auc_clean, ddof=1)) if len(auc_clean) > 1 else 0.0,
                "auc_sem": float(np.std(auc_clean, ddof=1) / math.sqrt(len(auc_clean))) if len(auc_clean) > 1 else 0.0,
                "auc_ci_low": auc_low,
                "auc_ci_high": auc_high,
                "normalized_auc_mean": float(np.mean(norm_auc_clean)) if norm_auc_clean else None,
                "normalized_auc_ci_low": norm_low,
                "normalized_auc_ci_high": norm_high,
                "worst_seed_final": min(final_clean) if final_clean else None,
                "collapse_count": collapse_count,
            }
        )

    rank_rows = []
    for metric in ("auc", "final_return"):
        by_env_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in run_rows:
            if row["total_timesteps"] == 500000 and finite_float(row.get(metric)) is not None:
                by_env_seed[(row["env"], row["seed"])].append(row)
        rank_accumulator: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (env, _seed), rows in by_env_seed.items():
            ordered = sorted(rows, key=lambda row: finite_float(row[metric]) or -np.inf, reverse=True)
            for rank, row in enumerate(ordered, start=1):
                rank_accumulator[(env, row["method"])].append(float(rank))
        for (env, method), ranks in sorted(rank_accumulator.items()):
            rank_rows.append(
                {
                    "env": env,
                    "method": method,
                    "metric": metric,
                    "average_rank": float(np.mean(ranks)),
                    "seed_count": len(ranks),
                }
            )

    delta_rows = []
    for env in sorted({row["env"] for row in run_rows if row["total_timesteps"] == 500000}):
        random_rows = {
            row["seed"]: row
            for row in run_rows
            if row["env"] == env
            and row["total_timesteps"] == 500000
            and row["method"] == "SAC->PPO random V"
        }
        for method in ("SAC->PPO self-warmup V", "SAC->PPO source-aligned V"):
            diffs_auc = []
            diffs_norm_auc = []
            diffs_final = []
            for row in run_rows:
                if row["env"] != env or row["total_timesteps"] != 500000 or row["method"] != method:
                    continue
                baseline = random_rows.get(row["seed"])
                if not baseline:
                    continue
                auc = finite_float(row["auc"])
                base_auc = finite_float(baseline["auc"])
                final = finite_float(row["final_return"])
                base_final = finite_float(baseline["final_return"])
                if auc is not None and base_auc is not None:
                    diffs_auc.append(auc - base_auc)
                    diffs_norm_auc.append((auc - base_auc) / float(row["total_timesteps"]))
                if final is not None and base_final is not None:
                    diffs_final.append(final - base_final)
            auc_low, auc_high = bootstrap_ci(diffs_auc, rng, samples)
            norm_auc_low, norm_auc_high = bootstrap_ci(diffs_norm_auc, rng, samples)
            final_low, final_high = bootstrap_ci(diffs_final, rng, samples)
            delta_rows.append(
                {
                    "env": env,
                    "method": method,
                    "baseline": "SAC->PPO random V",
                    "auc_delta_mean": float(np.mean(diffs_auc)) if diffs_auc else None,
                    "auc_delta_ci_low": auc_low,
                    "auc_delta_ci_high": auc_high,
                    "normalized_auc_delta_mean": float(np.mean(diffs_norm_auc)) if diffs_norm_auc else None,
                    "normalized_auc_delta_ci_low": norm_auc_low,
                    "normalized_auc_delta_ci_high": norm_auc_high,
                    "final_delta_mean": float(np.mean(diffs_final)) if diffs_final else None,
                    "final_delta_ci_low": final_low,
                    "final_delta_ci_high": final_high,
                    "paired_seed_count": len(diffs_auc),
                }
            )
    return summary_rows, rank_rows, delta_rows


def method_sort_key(method: str) -> int:
    labels = [method_label(algorithm, value_init) for algorithm, value_init in METHOD_ORDER]
    return labels.index(method) if method in labels else len(labels)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(eval_rows_table: list[dict[str, Any]], output_dir: Path) -> None:
    headline = output_dir / "figures" / "headline"
    envs = ["Hopper-v4", "Walker2d-v4"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for ax, env in zip(axes, envs):
        for algorithm, value_init in METHOD_ORDER:
            method = method_label(algorithm, value_init)
            rows = [
                row
                for row in eval_rows_table
                if row["env"] == env and row["total_timesteps"] == 500000 and row["method"] == method
            ]
            if not rows:
                continue
            by_seed: dict[int, dict[int, float]] = defaultdict(dict)
            for row in rows:
                by_seed[int(row["seed"])][int(row["env_steps"])] = float(row["eval_return_mean"])
            common_steps = sorted(set.intersection(*(set(series) for series in by_seed.values())))
            if not common_steps:
                continue
            values = np.asarray([[series[step] for step in common_steps] for series in by_seed.values()], dtype=float)
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0]) if values.shape[0] > 1 else np.zeros_like(mean)
            color = METHOD_COLORS.get(method)
            ax.plot(common_steps, mean, label=method, linewidth=2.0, color=color)
            ax.fill_between(common_steps, mean - sem, mean + sem, color=color, alpha=0.18)
        ax.axvline(250000, color="black", linestyle="--", linewidth=1, alpha=0.45)
        ax.set_title(env)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Evaluation return")
        ax.grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("500k runs: mean evaluation return with standard error")
    fig.tight_layout()
    save_figure(fig, headline / "learning_curves_500k")


def plot_bar_summary(summary_rows: list[dict[str, Any]], output_dir: Path, metric: str, title: str, filename: str) -> None:
    headline = output_dir / "figures" / "headline"
    envs = ["Hopper-v4", "Walker2d-v4"]
    methods = [method_label(algorithm, value_init) for algorithm, value_init in METHOD_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for ax, env in zip(axes, envs):
        rows = {row["method"]: row for row in summary_rows if row["env"] == env}
        x = np.arange(len(methods))
        means = []
        yerr_low = []
        yerr_high = []
        for method in methods:
            row = rows.get(method, {})
            mean = finite_float(row.get(f"{metric}_mean"))
            low = finite_float(row.get(f"{metric}_ci_low"))
            high = finite_float(row.get(f"{metric}_ci_high"))
            means.append(mean if mean is not None else np.nan)
            yerr_low.append((mean - low) if mean is not None and low is not None else 0.0)
            yerr_high.append((high - mean) if mean is not None and high is not None else 0.0)
        colors = [METHOD_COLORS.get(method) for method in methods]
        ax.bar(x, means, color=colors, alpha=0.85, yerr=np.asarray([yerr_low, yerr_high]), capsize=4)
        ax.set_title(env)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, headline / filename)


def plot_value_deltas(delta_rows: list[dict[str, Any]], output_dir: Path) -> None:
    headline = output_dir / "figures" / "headline"
    envs = ["Hopper-v4", "Walker2d-v4"]
    methods = ["SAC->PPO self-warmup V", "SAC->PPO source-aligned V"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, env in zip(axes, envs):
        rows = {(row["method"]): row for row in delta_rows if row["env"] == env}
        x = np.arange(len(methods))
        means = []
        yerr_low = []
        yerr_high = []
        for method in methods:
            row = rows.get(method, {})
            mean = finite_float(row.get("auc_delta_mean"))
            low = finite_float(row.get("auc_delta_ci_low"))
            high = finite_float(row.get("auc_delta_ci_high"))
            means.append(mean if mean is not None else np.nan)
            yerr_low.append((mean - low) if mean is not None and low is not None else 0.0)
            yerr_high.append((high - mean) if mean is not None and high is not None else 0.0)
        colors = [METHOD_COLORS.get(method) for method in methods]
        ax.axhline(0, color="black", linewidth=1, alpha=0.6)
        ax.bar(x, means, color=colors, alpha=0.85, yerr=np.asarray([yerr_low, yerr_high]), capsize=4)
        ax.set_title(env)
        ax.set_xticks(x)
        ax.set_xticklabels(["self-warmup", "source-aligned"], rotation=20, ha="right")
        ax.set_ylabel("AUC delta vs random value init")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("C1 value-initialization effect with paired-seed bootstrap CIs")
    fig.tight_layout()
    save_figure(fig, headline / "value_init_auc_deltas")


def plot_long_horizon(eval_rows_table: list[dict[str, Any]], output_dir: Path) -> None:
    headline = output_dir / "figures" / "headline"
    rows = [
        row
        for row in eval_rows_table
        if row["env"] == "Hopper-v4" and row["method"] == "SAC" and int(row["seed"]) in {0, 1, 2}
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for total_steps, label in [(500000, "SAC 500k seeds 0-2"), (1000000, "SAC 1M seeds 0-2")]:
        subset = [row for row in rows if row["total_timesteps"] == total_steps]
        by_seed: dict[int, dict[int, float]] = defaultdict(dict)
        for row in subset:
            by_seed[int(row["seed"])][int(row["env_steps"])] = float(row["eval_return_mean"])
        if not by_seed:
            continue
        common_steps = sorted(set.intersection(*(set(series) for series in by_seed.values())))
        values = np.asarray([[series[step] for step in common_steps] for series in by_seed.values()], dtype=float)
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0]) if values.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(common_steps, mean, label=label, linewidth=2)
        ax.fill_between(common_steps, mean - sem, mean + sem, alpha=0.18)
    ax.axvline(500000, color="black", linestyle="--", linewidth=1, alpha=0.45)
    ax.set_title("Hopper-v4 SAC long-horizon check")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation return")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, headline / "hopper_sac_long_horizon")


def plot_per_seed(eval_rows_table: list[dict[str, Any]], output_dir: Path) -> None:
    diag = output_dir / "figures" / "diagnostics"
    for env in ["Hopper-v4", "Walker2d-v4"]:
        fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
        axes_flat = axes.flatten()
        methods = [method_label(algorithm, value_init) for algorithm, value_init in METHOD_ORDER]
        for ax, method in zip(axes_flat, methods):
            rows = [
                row
                for row in eval_rows_table
                if row["env"] == env and row["method"] == method and row["total_timesteps"] == 500000
            ]
            for seed in sorted({int(row["seed"]) for row in rows}):
                seed_rows = sorted([row for row in rows if int(row["seed"]) == seed], key=lambda row: row["env_steps"])
                ax.plot(
                    [row["env_steps"] for row in seed_rows],
                    [row["eval_return_mean"] for row in seed_rows],
                    linewidth=1.4,
                    label=f"seed {seed}",
                )
            ax.axvline(250000, color="black", linestyle="--", linewidth=0.8, alpha=0.35)
            ax.set_title(method)
            ax.grid(alpha=0.2)
        axes_flat[-1].axis("off")
        axes_flat[0].legend(loc="upper left", frameon=False, fontsize=8)
        fig.suptitle(f"{env}: per-seed learning curves")
        fig.tight_layout()
        save_figure(fig, diag / f"{slugify(env)}_per_seed_curves")


def plot_handoff_transient(eval_rows_table: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    diag = output_dir / "figures" / "diagnostics"
    transient_rows = []
    for row_group_key, rows_iter in group_eval_by_run(eval_rows_table).items():
        algorithm, method, env, seed, total_steps, value_init = row_group_key
        if algorithm != "sac_to_ppo" or total_steps != 500000:
            continue
        rows = sorted(rows_iter, key=lambda row: row["env_steps"])
        before = [row for row in rows if row["env_steps"] < 250000]
        at_or_after = [row for row in rows if row["env_steps"] >= 250000]
        if not before or not at_or_after:
            continue
        pre = before[-1]["eval_return_mean"]
        post = at_or_after[0]["eval_return_mean"]
        transient_rows.append(
            {
                "method": method,
                "env": env,
                "seed": seed,
                "value_init": value_init,
                "pre_handoff_return": pre,
                "post_handoff_return": post,
                "post_minus_pre": post - pre,
            }
        )
    if transient_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
        for ax, env in zip(axes, ["Hopper-v4", "Walker2d-v4"]):
            rows = [row for row in transient_rows if row["env"] == env]
            methods = ["SAC->PPO random V", "SAC->PPO self-warmup V", "SAC->PPO source-aligned V"]
            data = [[row["post_minus_pre"] for row in rows if row["method"] == method] for method in methods]
            ax.axhline(0, color="black", linewidth=1, alpha=0.6)
            ax.boxplot(data, tick_labels=["random", "self", "source"], showmeans=True)
            ax.set_title(env)
            ax.set_ylabel("Post-handoff eval minus pre-handoff eval")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("SAC->PPO handoff transient at 250k steps")
        fig.tight_layout()
        save_figure(fig, diag / "handoff_transient")
    return transient_rows


def group_eval_by_run(eval_rows_table: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows_table:
        key = (
            row["algorithm"],
            row["method"],
            row["env"],
            int(row["seed"]),
            int(row["total_timesteps"]),
            row["value_init"],
        )
        groups[key].append(row)
    return groups


def plot_mechanism_diagnostics(diagnostic_rows: list[dict[str, Any]], output_dir: Path) -> None:
    diag = output_dir / "figures" / "diagnostics"
    if not diagnostic_rows:
        return
    numeric_keys = []
    for key in DIAGNOSTIC_KEYS:
        if any(finite_float(row.get(key)) is not None for row in diagnostic_rows):
            numeric_keys.append(key)
    selected = [key for key in numeric_keys if key in {"handoff_action_mse", "handoff_action_kl_proxy", "source_value_warmup_loss_final", "ppo_explained_variance", "ppo_value_loss", "advantages_std"}]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 2, figsize=(13, 3.2 * len(selected)), squeeze=False)
    for row_idx, key in enumerate(selected):
        for col_idx, env in enumerate(["Hopper-v4", "Walker2d-v4"]):
            ax = axes[row_idx][col_idx]
            data = []
            labels = []
            for method in ["SAC->PPO random V", "SAC->PPO self-warmup V", "SAC->PPO source-aligned V"]:
                values = [
                    finite_float(row.get(key))
                    for row in diagnostic_rows
                    if row["env"] == env and row["method"] == method and int(row["total_timesteps"]) == 500000
                ]
                values = [value for value in values if value is not None]
                if values:
                    data.append(values)
                    labels.append(method.replace("SAC->PPO ", "").replace(" V", ""))
            if data:
                ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.set_title(f"{env}: {key}")
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, diag / "mechanism_diagnostics")


def report_number(value: Any, digits: int = 1) -> str:
    value = finite_float(value)
    if value is None:
        return "missing"
    return f"{value:.{digits}f}"


def best_by_metric(summary_rows: list[dict[str, Any]], env: str, metric: str) -> dict[str, Any] | None:
    rows = [row for row in summary_rows if row["env"] == env and finite_float(row.get(f"{metric}_mean")) is not None]
    if not rows:
        return None
    return max(rows, key=lambda row: finite_float(row[f"{metric}_mean"]) or -np.inf)


def write_report(
    output_dir: Path,
    audit: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
) -> None:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ryan Results",
        "",
        f"Generated at Unix time `{int(time.time())}` from `{audit.get('source_manifest', 'experiments/ryan_modal_manifest.json')}` and the W&B completion audit.",
        "",
        "## Completion",
        "",
        f"All `{audit['complete_count']}/{audit['expected_count']}` Ryan jobs were completed and included. This covers 500k-step SAC/PPO baselines, 500k-step SAC->PPO value-initialization ablations on Hopper-v4 and Walker2d-v4, and the Hopper-v4 SAC 1M long-horizon check.",
        "",
        "## Headline Figures",
        "",
        "- [Learning curves](../figures/headline/learning_curves_500k.png)",
        "- [AUC summary](../figures/headline/auc_summary.png)",
        "- [Final return summary](../figures/headline/final_return_summary.png)",
        "- [Value-init AUC deltas](../figures/headline/value_init_auc_deltas.png)",
        "- [Hopper SAC long-horizon check](../figures/headline/hopper_sac_long_horizon.png)",
        "",
        "## Main Results",
        "",
    ]
    for env in ["Hopper-v4", "Walker2d-v4"]:
        best_auc = best_by_metric(summary_rows, env, "auc")
        best_final = best_by_metric(summary_rows, env, "final")
        if best_auc and best_final:
            lines.append(
                f"- `{env}`: best mean AUC is `{best_auc['method']}` (`{report_number(best_auc['normalized_auc_mean'], 3)}` normalized AUC); best mean final return is `{best_final['method']}` (`{report_number(best_final['final_mean'])}`)."
            )
    lines.extend(["", "## C1 Value-Ablation Test", ""])
    for row in delta_rows:
        lines.append(
            f"- `{row['env']}` `{row['method']}` vs random value init: mean normalized AUC delta `{report_number(row['normalized_auc_delta_mean'], 2)}` with 95% CI [`{report_number(row['normalized_auc_delta_ci_low'], 2)}`, `{report_number(row['normalized_auc_delta_ci_high'], 2)}`] over `{row['paired_seed_count']}` paired seeds."
        )
    lines.extend(
        [
            "",
            "Interpretation should focus on whether these paired CIs separate from zero. A null or mixed result is still informative: with policy distillation fixed, value initialization may not be the limiting transfer mechanism under this matched 500k-step budget.",
            "",
            "## Summary Table",
            "",
            "| Env | Method | Seeds | Final mean | Final 95% CI | Norm. AUC mean | Norm. AUC 95% CI | Collapse count |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(summary_rows, key=lambda item: (item["env"], method_sort_key(item["method"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["env"],
                    row["method"],
                    str(row["seed_count"]),
                    report_number(row["final_mean"]),
                    f"[{report_number(row['final_ci_low'])}, {report_number(row['final_ci_high'])}]",
                    report_number(row["normalized_auc_mean"], 3),
                    f"[{report_number(row['normalized_auc_ci_low'], 3)}, {report_number(row['normalized_auc_ci_high'], 3)}]",
                    str(row["collapse_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "- [Hopper per-seed curves](../figures/diagnostics/Hopper-v4_per_seed_curves.png)",
            "- [Walker2d per-seed curves](../figures/diagnostics/Walker2d-v4_per_seed_curves.png)",
            "- [Handoff transient](../figures/diagnostics/handoff_transient.png)",
            "- [Mechanism diagnostics](../figures/diagnostics/mechanism_diagnostics.png)",
            "",
            "Processed CSVs live in `../processed/`, including per-run summaries, per-arm summaries, rank summaries, paired value-init deltas, handoff transients, and diagnostic rows.",
            "",
            "## Limitations",
            "",
            "These results cover Hopper-v4 and Walker2d-v4 with five 500k-step seeds per arm, plus a three-seed Hopper SAC 1M check. They should support Ryan's SAC->PPO mechanism claim, not a universal claim that handoffs beat strong standalone SAC across all environments.",
            "",
        ]
    )
    (report_dir / "ryan_results.md").write_text("\n".join(lines), encoding="utf-8")


def copy_audit_inputs(output_dir: Path, audit_path: Path, manifest_path: Path) -> None:
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audit_path, audit_dir / audit_path.name)
    shutil.copy2(manifest_path, audit_dir / manifest_path.name)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    figure_dir = output_dir / "figures"
    for path in [raw_dir, processed_dir, figure_dir / "headline", figure_dir / "diagnostics", output_dir / "reports"]:
        path.mkdir(parents=True, exist_ok=True)

    audit = load_json(args.audit_path)
    if audit.get("missing_count") != 0:
        raise ValueError(f"Audit is incomplete: missing_count={audit.get('missing_count')}")
    audit_rows = selected_audit_rows(audit)
    if args.source == "modal":
        manifest_rows = export_modal_histories(
            audit_rows,
            raw_dir,
            args.modal_volume,
            args.remote_prefix,
            args.refresh_wandb,
            args.download_workers,
        )
    else:
        manifest_rows = export_wandb_histories(audit_rows, raw_dir, args.entity, args.project, args.refresh_wandb)

    run_rows = []
    eval_table = []
    diagnostic_rows = []
    for manifest_row in manifest_rows:
        run_summary, run_evals, run_diagnostics = summarize_run(manifest_row)
        run_rows.append(run_summary)
        eval_table.extend(run_evals)
        diagnostic_rows.extend(run_diagnostics)

    rng = np.random.default_rng(args.seed)
    summary_rows, rank_rows, delta_rows = aggregate_metrics(run_rows, rng, args.bootstrap_samples)
    transient_rows = plot_handoff_transient(eval_table, output_dir)

    write_csv(processed_dir / "per_run_summary.csv", run_rows)
    write_csv(processed_dir / "eval_curves.csv", eval_table)
    write_csv(processed_dir / "arm_summary.csv", summary_rows)
    write_csv(processed_dir / "rank_summary.csv", rank_rows)
    write_csv(processed_dir / "value_init_deltas.csv", delta_rows)
    write_csv(processed_dir / "handoff_transients.csv", transient_rows)
    write_csv(processed_dir / "diagnostics.csv", diagnostic_rows)
    write_json(
        processed_dir / "bundle_summary.json",
        {
            "created_at_unix": int(time.time()),
            "expected_count": audit["expected_count"],
            "complete_count": audit["complete_count"],
            "included_run_count": len(run_rows),
            "eval_row_count": len(eval_table),
            "diagnostic_row_count": len(diagnostic_rows),
            "headline_figures": [
                "figures/headline/learning_curves_500k.png",
                "figures/headline/auc_summary.png",
                "figures/headline/final_return_summary.png",
                "figures/headline/value_init_auc_deltas.png",
                "figures/headline/hopper_sac_long_horizon.png",
            ],
        },
    )

    plot_learning_curves(eval_table, output_dir)
    plot_bar_summary(summary_rows, output_dir, "auc", "500k AUC by environment", "auc_summary")
    plot_bar_summary(summary_rows, output_dir, "final", "500k final return by environment", "final_return_summary")
    plot_value_deltas(delta_rows, output_dir)
    plot_long_horizon(eval_table, output_dir)
    plot_per_seed(eval_table, output_dir)
    plot_mechanism_diagnostics(diagnostic_rows, output_dir)
    copy_audit_inputs(output_dir, args.audit_path, args.manifest_path)
    write_report(output_dir, audit, summary_rows, delta_rows, rank_rows)
    print(f"Wrote Ryan results bundle to {output_dir}")


if __name__ == "__main__":
    main()
