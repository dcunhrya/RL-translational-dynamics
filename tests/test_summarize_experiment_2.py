import sys
from pathlib import Path

import pytest

from conftest import write_metrics_jsonl

EXP2_DIR = Path(__file__).resolve().parents[1] / "src" / "RL-translational-dynamics" / "exp2"
sys.path.insert(0, str(EXP2_DIR))

from summarize_experiment_2 import (
    RunKey,
    aggregate_method_stats,
    eval_series,
    expected_keys,
    latest_runs,
    run_key_from_metrics,
    summarize_run,
)


def test_run_key_from_handoff_metrics(sample_handoff_metrics_rows):
    key = run_key_from_metrics(sample_handoff_metrics_rows)
    assert key == RunKey(algorithm="sac_to_ppo", env="Hopper-v4", seed=0, handoff_fraction=0.5)


def test_run_key_from_sac_metrics(sample_sac_metrics_rows):
    key = run_key_from_metrics(sample_sac_metrics_rows)
    assert key == RunKey(algorithm="sac", env="Hopper-v4", seed=0, handoff_fraction=None)


def test_summarize_run_detects_handoff_phase(tmp_path, sample_handoff_metrics_rows):
    run_dir = tmp_path / "handoff_run"
    write_metrics_jsonl(run_dir, sample_handoff_metrics_rows)
    summary = summarize_run(run_dir)
    assert summary is not None
    assert summary.has_handoff_phase is True
    assert summary.final_eval == 800.0
    assert summary.switch_step == 50000


def test_eval_series_and_auc(tmp_path, sample_handoff_metrics_rows):
    run_dir = tmp_path / "handoff_run"
    write_metrics_jsonl(run_dir, sample_handoff_metrics_rows)
    series = eval_series(run_dir)
    assert series == [(0, 20.0), (100000, 800.0)]
    summary = summarize_run(run_dir)
    assert summary.eval_auc is not None
    assert summary.eval_auc > 0


def test_latest_runs_groups_by_key(minimal_results_dir):
    runs = latest_runs(minimal_results_dir)
    assert RunKey(algorithm="sac", env="Hopper-v4", seed=0) in runs
    assert RunKey(algorithm="sac_to_ppo", env="Hopper-v4", seed=0, handoff_fraction=0.5) in runs


def test_expected_keys_count():
    keys = expected_keys(["Hopper-v4"], [0, 1], [0.25, 0.5])
    # per seed: sac + ppo + 2 handoff fractions
    assert len(keys) == 2 * (2 + 2)


def test_expected_keys_can_expand_value_init_arms():
    keys = expected_keys(["Hopper-v4"], [0], [0.5], ["random", "self-warmup", "source-aligned"])
    handoff_keys = [key for key in keys if key.algorithm == "sac_to_ppo"]
    assert {key.value_init for key in handoff_keys} == {"random", "self-warmup", "source-aligned"}
    assert all(key.policy_init == "distill" for key in handoff_keys)


def test_latest_runs_keeps_value_init_arms_separate(tmp_path, sample_handoff_metrics_rows):
    random_rows = [dict(row, policy_init="distill", value_init="random") for row in sample_handoff_metrics_rows]
    source_rows = [dict(row, policy_init="distill", value_init="source-aligned") for row in sample_handoff_metrics_rows]
    random_rows[-1]["eval_return_mean"] = 700.0
    source_rows[-1]["eval_return_mean"] = 900.0

    write_metrics_jsonl(tmp_path / "handoff_random", random_rows)
    write_metrics_jsonl(tmp_path / "handoff_source_aligned", source_rows)

    runs = latest_runs(tmp_path)
    assert RunKey(
        algorithm="sac_to_ppo",
        env="Hopper-v4",
        seed=0,
        handoff_fraction=0.5,
        policy_init="distill",
        value_init="random",
    ) in runs
    assert RunKey(
        algorithm="sac_to_ppo",
        env="Hopper-v4",
        seed=0,
        handoff_fraction=0.5,
        policy_init="distill",
        value_init="source-aligned",
    ) in runs


def test_aggregate_method_stats_detects_success(tmp_path, sample_sac_metrics_rows, sample_handoff_metrics_rows):
    sac_rows = [
        dict(row, eval_return_mean=100.0 if row["env_steps"] == 100000 else row["eval_return_mean"])
        for row in sample_sac_metrics_rows
    ]
    ppo_rows = [
        dict(row, algorithm="ppo", eval_return_mean=100.0 if row["env_steps"] == 100000 else row["eval_return_mean"])
        for row in sample_sac_metrics_rows
    ]
    handoff_rows = list(sample_handoff_metrics_rows)
    handoff_rows[-1]["eval_return_mean"] = 2000.0

    write_metrics_jsonl(tmp_path / "sac", sac_rows)
    write_metrics_jsonl(tmp_path / "ppo", ppo_rows)
    write_metrics_jsonl(tmp_path / "handoff", handoff_rows)

    runs = latest_runs(tmp_path)
    stats = aggregate_method_stats(runs, ["Hopper-v4"], [0], [0.5])
    assert stats["success_cases"]
    assert stats["success_cases"][0]["beats_final_return"] is True


def test_gate_fails_on_missing_handoff_phase(tmp_path, sample_handoff_metrics_rows):
    bad_rows = [row for row in sample_handoff_metrics_rows if row.get("phase") != "handoff"]
    write_metrics_jsonl(tmp_path / "bad_handoff", bad_rows)
    summary = summarize_run(tmp_path / "bad_handoff")
    assert summary is not None
    assert summary.has_handoff_phase is False
