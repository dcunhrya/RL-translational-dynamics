# Experiment 3

Fixed `PPO -> SAC` reverse-handoff pilot for short-horizon MuJoCo training.

## Setup

- Algorithms: fixed-order `PPO -> SAC`
- Environments: `Hopper-v4`, `Walker2d-v4`
- Seeds: `0`, `1`, `2`
- Total budget: `100000` env steps
- Switch schedules:
  - `25%` handoff at `25000` steps
  - `50%` handoff at `50000` steps
  - `75%` handoff at `75000` steps

## Main plots

- `results/processed/experiment_3_reverse_handoff/Hopper_v4_reverse_handoff_learning_curves.png`
- `results/processed/experiment_3_reverse_handoff/Walker2d_v4_reverse_handoff_learning_curves.png`
- `results/processed/experiment_3_reverse_handoff/reverse_handoff_final_return_comparison.png`

These plots show seed-mean evaluation return over training for each reverse-handoff schedule, with vertical switch markers and pure `PPO` / pure `SAC` baselines overlaid.

## Final return summary

Final `eval_return_mean` averages across seeds:

- `Hopper-v4`
  - `PPO`: `346.23 +/- 15.37`
  - `25%` reverse handoff: `754.55 +/- 374.10`
  - `50%` reverse handoff: `564.02 +/- 124.98`
  - `75%` reverse handoff: `358.61 +/- 31.52`
  - `SAC`: `1204.24 +/- 795.57`
- `Walker2d-v4`
  - `PPO`: `339.01 +/- 33.65`
  - `25%` reverse handoff: `566.35 +/- 126.00`
  - `50%` reverse handoff: `528.83 +/- 280.14`
  - `75%` reverse handoff: `357.77 +/- 43.38`
  - `SAC`: `326.17 +/- 44.50`

## Takeaway

- `Hopper-v4`: early `PPO -> SAC` handoff helps substantially over pure `PPO`, but pure `SAC` remains strongest at this budget.
- `Walker2d-v4`: reverse handoff helps clearly, with `25%` and `50%` both stronger than the pure baselines.
- Relative to `SAC -> PPO`, reverse handoff looks better for earlier switches on both environments, while later switches are less attractive.
- The direction of the switch matters, not just the fact that a switch happens.
