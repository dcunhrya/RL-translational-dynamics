import sys
from pathlib import Path

import pytest

EXP2_DIR = Path(__file__).resolve().parents[1] / "src" / "RL-translational-dynamics" / "exp2"
sys.path.insert(0, str(EXP2_DIR))

from handoff_utils import build_metadata, compute_switch_step, transfer_arm_name, validate_switch_fraction, validate_transfer_config


def test_compute_switch_step_midpoint():
    assert compute_switch_step(100_000, 0.5) == 50_000


def test_compute_switch_step_clamps_low_fraction():
    assert compute_switch_step(100, 0.01) == 1


def test_compute_switch_step_clamps_high_fraction():
    assert compute_switch_step(100, 0.99) == 99


def test_validate_switch_fraction_rejects_edges():
    with pytest.raises(ValueError):
        validate_switch_fraction(0.0)
    with pytest.raises(ValueError):
        validate_switch_fraction(1.0)


def test_compute_switch_step_rejects_short_horizon():
    with pytest.raises(ValueError):
        compute_switch_step(1, 0.5)


def test_build_metadata_handoff_phase():
    metadata = build_metadata(0.25, "handoff", switch_step=25_000, switched=True)
    assert metadata["algorithm"] == "sac_to_ppo"
    assert metadata["phase"] == "handoff"
    assert metadata["switched"] is True
    assert metadata["switch_reason"] == "fixed_fraction"
    assert metadata["handoff_fraction"] == 0.25
    assert metadata["trigger_value"] == 0.25
    assert metadata["policy_init"] == "distill"
    assert metadata["value_init"] == "self-warmup"


def test_build_metadata_sac_phase_not_switched():
    metadata = build_metadata(0.75, "sac", switch_step=75_000, switched=False)
    assert metadata["switched"] is False
    assert metadata["switch_step"] is None
    assert metadata["switch_reason"] is None


def test_transfer_arm_name_is_deterministic():
    assert (
        transfer_arm_name("distill", "source-aligned", policy_source="sac", value_source="sac")
        == "policy_distill_from_sac__value_source-aligned_from_sac"
    )


def test_validate_transfer_config_rejects_unknown_modes():
    with pytest.raises(ValueError):
        validate_transfer_config("copy", "self-warmup")
    with pytest.raises(ValueError):
        validate_transfer_config("distill", "critic-copy")


def test_build_metadata_preserves_transfer_flags():
    metadata = build_metadata(
        0.5,
        "source_value_warmup",
        switch_step=250_000,
        switched=True,
        policy_init="distill",
        value_init="source-aligned",
        policy_source="sac",
        value_source="sac",
    )
    assert metadata["policy_init"] == "distill"
    assert metadata["value_init"] == "source-aligned"
    assert metadata["policy_source"] == "sac"
    assert metadata["value_source"] == "sac"
    assert metadata["arm_name"] == "policy_distill_from_sac__value_source-aligned_from_sac"
