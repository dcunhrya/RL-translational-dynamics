# Ethan Task Results

## Scope

- PPO -> SAC value ablation: `random`, `self-warmup`, `source-aligned`.
- Timing sweep: fixed 25% and 75% switches, with 50% supplied by the ablation arm.
- Adaptive trigger: no-improvement after a minimum first-phase budget.
- Long-horizon checks: pure PPO and one PPO -> SAC arm on Hopper when included in the Modal launch.

## Summary Table

| Env | Trigger | Switch | Policy | Value | Seeds | AUC mean [95% CI] | Final mean [95% CI] | Mean switch step |
|---|---|---:|---|---|---:|---:|---:|---:|
| Hopper-v4 | fixed_fraction | 0.50 | distill | random | 1 | 110.28 [110.28, 110.28] | 52.04 [52.04, 52.04] | 1000.0 |
| Hopper-v4 | fixed_fraction | 0.50 | distill | self-warmup | 1 | 105.13 [105.13, 105.13] | 59.54 [59.54, 59.54] | 1000.0 |
| Hopper-v4 | no-improve | 0.75 | distill | self-warmup | 1 | 97.20 [97.20, 97.20] | 184.27 [184.27, 184.27] | 1000.0 |
| Hopper-v4 | fixed_fraction | 0.50 | distill | source-aligned | 1 | 139.15 [139.15, 139.15] | 77.45 [77.45, 77.45] | 1000.0 |

## Generated Artifacts

- `results/processed/ethan_smoke/Hopper_v4_learning_curves.png`
- `results/processed/ethan_smoke/Hopper_v4_auc_summary.png`

CSV summary: `results/processed/ethan_smoke/summary.csv`
