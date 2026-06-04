PHASES = (
    "bc",
    "iql",
    "awac",
    "sac",
    "distill",
    "value_warmup",
    "source_value_warmup",
    "ppo",
    "ppo_value_warmup",
    "sac_critic_warmup",
    "bc_anchor",
    "adaptive_switch",
)

IDENTITY_KEYS = (
    "algorithm",
    "env",
    "seed",
    "env_steps",
    "gradient_updates",
    "wall_clock_sec",
    "phase",
    "policy_init",
    "policy_source",
    "value_init",
    "value_source",
    "switch_step",
    "switch_reason",
    "trigger_value",
)

DIAGNOSTIC_KEYS = (
    "eval_return_mean",
    "eval_return_std",
    "bc_loss",
    "bc_distill_loss",
    "bc_anchor_loss",
    "policy_retention_action_mse",
    "policy_retention_approx_kl",
    "ppo_explained_variance",
    "ppo_approx_kl",
    "ppo_clip_fraction",
    "sac_policy_entropy",
    "sac_qf1_mean",
    "sac_qf2_mean",
    "iql_actor_loss",
    "iql_q_loss",
    "iql_value_loss",
    "awac_actor_loss",
    "awac_critic_loss",
)


def require_known_phase(phase: str) -> str:
    if phase not in PHASES:
        raise ValueError(f"Unknown diagnostic phase {phase!r}. Expected one of {PHASES}.")
    return phase
