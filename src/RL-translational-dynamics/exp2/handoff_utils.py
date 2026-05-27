"""Pure helpers for SAC -> PPO handoff (no torch / gym imports)."""

POLICY_INITS = ("random", "distill")
VALUE_INITS = ("random", "self-warmup", "source-aligned")


def validate_switch_fraction(switch_fraction: float) -> None:
    if not 0.0 < switch_fraction < 1.0:
        raise ValueError(f"switch_fraction must be in (0, 1), got {switch_fraction}.")


def compute_switch_step(total_timesteps: int, switch_fraction: float) -> int:
    if total_timesteps < 2:
        raise ValueError("total_timesteps must be at least 2 for a SAC -> PPO handoff.")
    validate_switch_fraction(switch_fraction)
    switch_step = int(total_timesteps * switch_fraction)
    return min(max(1, switch_step), total_timesteps - 1)


def validate_transfer_config(
    policy_init: str,
    value_init: str,
    policy_source: str | None = None,
    value_source: str | None = None,
) -> None:
    if policy_init not in POLICY_INITS:
        raise ValueError(f"policy_init must be one of {POLICY_INITS}, got {policy_init}.")
    if value_init not in VALUE_INITS:
        raise ValueError(f"value_init must be one of {VALUE_INITS}, got {value_init}.")
    if policy_init == "distill" and policy_source not in (None, "sac"):
        raise ValueError("SAC -> PPO policy distillation currently supports only policy_source='sac'.")
    if value_init == "source-aligned" and value_source not in (None, "sac"):
        raise ValueError("SAC -> PPO source-aligned value warm-up currently supports only value_source='sac'.")


def transfer_arm_name(
    policy_init: str,
    value_init: str,
    policy_source: str | None = None,
    value_source: str | None = None,
) -> str:
    validate_transfer_config(policy_init, value_init, policy_source, value_source)
    policy_source = policy_source or ("sac" if policy_init == "distill" else "none")
    value_source = value_source or ("sac" if value_init == "source-aligned" else "none")
    return f"policy_{policy_init}_from_{policy_source}__value_{value_init}_from_{value_source}"


def build_metadata(
    handoff_fraction: float,
    phase: str,
    switch_step: int,
    switched: bool,
    policy_init: str = "distill",
    value_init: str = "self-warmup",
    policy_source: str | None = None,
    value_source: str | None = None,
) -> dict:
    validate_transfer_config(policy_init, value_init, policy_source, value_source)
    policy_source = policy_source or ("sac" if policy_init == "distill" else "none")
    value_source = value_source or ("sac" if value_init == "source-aligned" else "none")
    return {
        "algorithm": "sac_to_ppo",
        "phase": phase,
        "switched": switched,
        "switch_step": switch_step if switched else None,
        "switch_reason": "fixed_fraction" if switched else None,
        "trigger_value": handoff_fraction if switched else None,
        "planned_switch_step": switch_step,
        "handoff_fraction": handoff_fraction,
        "policy_init": policy_init,
        "value_init": value_init,
        "policy_source": policy_source,
        "value_source": value_source,
        "arm_name": transfer_arm_name(policy_init, value_init, policy_source, value_source),
    }
