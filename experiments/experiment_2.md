# Experiment 2

Fixed `SAC -> PPO` handoff pilot for short-horizon MuJoCo training.

## Setup

- Algorithms: fixed-order `SAC -> PPO`
- Environments: `Hopper-v4`, `Walker2d-v4`
- Seeds: `0`, `1`, `2`
- Total budget: `100000` env steps
- Switch schedules:
  - `25%` handoff at `25000` steps
  - `50%` handoff at `50000` steps
  - `75%` handoff at `75000` steps

## Main plots

- `results/processed/experiment_2_fixed_handoff/Hopper_v4_handoff_learning_curves.png`
- `results/processed/experiment_2_fixed_handoff/Walker2d_v4_handoff_learning_curves.png`
- `results/processed/experiment_2_fixed_handoff/handoff_final_return_comparison.png`

These learning-curve plots show the seed-mean evaluation return over training for each handoff schedule, with vertical switch markers at the transition step and pure `SAC` / pure `PPO` baselines overlaid for reference.

## Final return summary

Final `eval_return_mean` averages across seeds:

- `Hopper-v4`
  - `SAC`: `1204.24 +/- 795.57`
  - `25%` handoff: `362.91 +/- 16.55`
  - `50%` handoff: `421.44 +/- 243.20`
  - `75%` handoff: `735.27 +/- 251.71`
  - `PPO`: `346.23 +/- 15.37`
- `Walker2d-v4`
  - `SAC`: `326.17 +/- 44.50`
  - `25%` handoff: `365.50 +/- 52.86`
  - `50%` handoff: `513.48 +/- 86.63`
  - `75%` handoff: `476.30 +/- 38.86`
  - `PPO`: `339.01 +/- 33.65`

## Takeaway

- `Hopper-v4`: later handoff is clearly better than earlier handoff, but pure `SAC` is still strongest at this budget.
- `Walker2d-v4`: handoff helps, and the `50%` switch is the strongest of the tested schedules.
- The curves suggest ordering does matter, but the best switch point is environment-dependent.
