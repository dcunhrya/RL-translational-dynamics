import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_sac_metrics_rows() -> list[dict]:
    return [
        {
            "algorithm": "sac",
            "env": "Hopper-v4",
            "seed": 0,
            "env_steps": 0,
            "gradient_updates": 0,
            "eval_return_mean": 10.0,
        },
        {
            "algorithm": "sac",
            "env": "Hopper-v4",
            "seed": 0,
            "env_steps": 100000,
            "gradient_updates": 500,
            "eval_return_mean": 500.0,
        },
    ]


@pytest.fixture
def sample_handoff_metrics_rows() -> list[dict]:
    rows = [
        {
            "algorithm": "sac_to_ppo",
            "env": "Hopper-v4",
            "seed": 0,
            "handoff_fraction": 0.5,
            "phase": "sac",
            "switched": False,
            "planned_switch_step": 50000,
            "env_steps": 0,
            "gradient_updates": 0,
            "eval_return_mean": 20.0,
        },
        {
            "algorithm": "sac_to_ppo",
            "env": "Hopper-v4",
            "seed": 0,
            "handoff_fraction": 0.5,
            "phase": "handoff",
            "switched": True,
            "switch_step": 50000,
            "switch_reason": "fixed_fraction",
            "env_steps": 50000,
            "gradient_updates": 200,
            "handoff_distill_loss": 0.01,
        },
        {
            "algorithm": "sac_to_ppo",
            "env": "Hopper-v4",
            "seed": 0,
            "handoff_fraction": 0.5,
            "phase": "ppo",
            "switched": True,
            "switch_step": 50000,
            "env_steps": 100000,
            "gradient_updates": 400,
            "eval_return_mean": 800.0,
        },
    ]
    return rows


def write_metrics_jsonl(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (run_dir / "checkpoint_step_100000.pt").write_bytes(b"stub")


@pytest.fixture
def minimal_results_dir(tmp_path, sample_sac_metrics_rows, sample_handoff_metrics_rows) -> Path:
    write_metrics_jsonl(tmp_path / "sac__Hopper-v4__seed_0__1", sample_sac_metrics_rows)
    write_metrics_jsonl(tmp_path / "ppo__Hopper-v4__seed_0__2", sample_sac_metrics_rows)
    write_metrics_jsonl(tmp_path / "handoff__Hopper-v4__seed_0__frac_0.50__1", sample_handoff_metrics_rows)
    return tmp_path
