# RL Translational Dynamics

Study of **algorithm sequencing in deep RL**, with a focus on transitioning from **SAC (off-policy, sample efficient)** to **PPO (on-policy, stable)** during training.

## Project Question

Can we improve convergence and final performance under fixed budgets by sequencing RL algorithms (`SAC -> PPO`) instead of training with only one algorithm end-to-end?

## Current Repository Contents

- `CS_224R_Proposal.pdf`: project proposal
- `FIRST_EXPERIMENTS.md`: recommended first experiment sequence

## Proposed Experimental Scope

Primary environments:
- `Hopper-v4`
- `Walker2d-v4`

Planned experiment tracks:
1. Single-algorithm baselines (`SAC`, `PPO`)
2. Fixed schedule handoff (`SAC -> PPO`)
3. Adaptive handoff trigger (internal metric based)
4. Compute-constrained comparisons (equalized by updates/FLOPs proxy)

## Suggested Project Layout

As code is added, use this layout:

```text
.
├── README.md
├── CS_224R_Proposal.pdf
├── FIRST_EXPERIMENTS.md
├── configs/
│   ├── sac/
│   ├── ppo/
│   └── handoff/
├── src/
│   ├── train_sac.py
│   ├── train_ppo.py
│   ├── train_handoff.py
│   ├── switching/
│   └── utils/
├── scripts/
│   ├── run_baselines.sh
│   ├── run_handoff_sweep.sh
│   └── aggregate_results.py
└── results/
    ├── raw/
    ├── processed/
    └── figures/
```

## Reproducibility Conventions

- Run every configuration with multiple seeds (target: >=5 for core comparisons).
- Log at least:
  - `algorithm`, `env`, `seed`
  - `env_steps`, `gradient_updates`, `wall_clock_sec`
  - `eval_return_mean`, `eval_return_std`
  - handoff fields: `switch_step`, `switch_reason`, `trigger_value`
- Report mean and variance, not single-seed results.

## Immediate Next Steps

1. Implement and validate pure SAC/PPO training pipelines.
2. Produce baseline curves on Hopper/Walker2d.
3. Implement fixed handoff training (`SAC -> PPO`) and run first schedule sweep.
4. Add adaptive switch trigger once fixed handoff is stable.

## Team Alignment (from proposal)

- Ryan: literature scoping, baselines, fixed-ordering experiments
- Ethan: adaptive switching and benchmark expansion
- Abhinav: compute-constrained scheduling and budget strategy
