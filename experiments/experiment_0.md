# Experiment 0

Sanity and reproducibility sweep for short-horizon MuJoCo baselines.

## Setup

- Date: `2026-05-18`
- Algorithms: `SAC`, `PPO`
- Environments: `Hopper-v4`, `Walker2d-v4`, `HalfCheetah-v4`, `Ant-v4`
- Seeds: `0`, `1`
- Horizon: `100000` env steps per run
- Logging: W&B enabled for all runs
- W&B project: `herschethan-stanford-university/rl-translational-dynamics`

## Status

- Experiment gate: `PASS`
- No recurring crashes observed
- No NaN or inf failures observed
- Both `SAC` and `PPO` improved on `Hopper-v4` across both seeds

## Final Results

Final `eval_return_mean` averaged across 2 seeds, reported as mean +/- std:

- `Hopper-v4`: `SAC 1204.24 +/- 795.57`, `PPO 346.23 +/- 15.37`
- `Walker2d-v4`: `SAC 326.17 +/- 44.50`, `PPO 339.01 +/- 33.65`
- `HalfCheetah-v4`: `SAC 4830.85 +/- 1075.63`, `PPO 275.69 +/- 26.16`
- `Ant-v4`: `SAC 1152.57 +/- 307.84`, `PPO 448.39 +/- 92.06`

## Interpretation

- At this short horizon, `SAC` is clearly stronger on `Hopper-v4`, `HalfCheetah-v4`, and `Ant-v4`.
- `Walker2d-v4` is roughly tied, with a small edge to `PPO`.
- These results support treating `SAC` as the stronger early-training baseline before testing `SAC -> PPO` handoff schedules.

## Artifacts

- Learning curves: `results/processed/experiment_0/sac_vs_ppo_learning_curves.png`
- Final return comparison: `results/processed/experiment_0/sac_vs_ppo_final_returns.png`
- Raw runs: `results/raw/experiment_0/`
- Summary script: `src/RL-translational-dynamics/exp0/summarize_experiment_0.py`
