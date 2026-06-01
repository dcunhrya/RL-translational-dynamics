# Ethan Task Results

## Scope

- PPO -> SAC value ablation: `random`, `self-warmup`, `source-aligned`.
- Timing sweep: fixed 25% and 75% switches, with 50% supplied by the ablation arm.
- Adaptive trigger: no-improvement after a minimum first-phase budget.
- Long-horizon checks: pure PPO and one PPO -> SAC arm on Hopper when included in the Modal launch.

## Summary Table

| Env | Trigger | Switch | Policy | Value | Seeds | AUC mean [95% CI] | Final mean [95% CI] | Mean switch step |
|---|---|---:|---|---|---:|---:|---:|---:|
| Hopper-v4 | fixed_fraction | 0.00 | unknown | unknown | 3 | 416.59 [404.76, 436.05] | 639.17 [499.98, 901.48] | nan |
| Hopper-v4 | fixed_fraction | 0.50 | distill | random | 5 | 640.80 [603.39, 678.39] | 1488.63 [930.90, 2270.49] | 250000.0 |
| Hopper-v4 | fixed_fraction | 0.25 | distill | self-warmup | 3 | 1411.49 [1261.01, 1698.47] | 2451.67 [999.54, 3192.30] | 125000.0 |
| Hopper-v4 | fixed_fraction | 0.50 | distill | self-warmup | 5 | 729.14 [633.81, 824.46] | 1760.63 [1155.67, 2559.62] | 400000.0 |
| Hopper-v4 | fixed_fraction | 0.75 | distill | self-warmup | 3 | 392.75 [385.23, 403.28] | 956.56 [558.04, 1238.25] | 375000.0 |
| Hopper-v4 | no-improve | 0.75 | distill | self-warmup | 3 | 1130.97 [954.10, 1379.00] | 1986.97 [1022.16, 3224.60] | 150000.0 |
| Hopper-v4 | fixed_fraction | 0.50 | distill | source-aligned | 5 | 688.58 [663.67, 727.75] | 2274.69 [1497.58, 2967.20] | 250000.0 |
| Walker2d-v4 | fixed_fraction | 0.50 | distill | random | 5 | 586.37 [505.99, 682.50] | 2227.23 [1486.85, 2788.95] | 250000.0 |
| Walker2d-v4 | fixed_fraction | 0.25 | distill | self-warmup | 3 | 1280.69 [831.55, 1652.36] | 2380.18 [912.36, 3238.02] | 125000.0 |
| Walker2d-v4 | fixed_fraction | 0.50 | distill | self-warmup | 5 | 597.87 [509.37, 684.43] | 2090.43 [1363.21, 2817.64] | 250000.0 |
| Walker2d-v4 | fixed_fraction | 0.75 | distill | self-warmup | 3 | 349.03 [322.65, 362.91] | 697.36 [346.61, 1071.27] | 375000.0 |
| Walker2d-v4 | no-improve | 0.75 | distill | self-warmup | 3 | 1498.34 [1315.43, 1713.89] | 3777.64 [3253.18, 4086.07] | 146666.7 |
| Walker2d-v4 | fixed_fraction | 0.50 | distill | source-aligned | 5 | 486.88 [441.96, 542.60] | 1545.67 [1142.63, 2266.68] | 250000.0 |

## Generated Artifacts

- `/root/results/processed/ethan_task/Hopper_v4_learning_curves.png`
- `/root/results/processed/ethan_task/Walker2d_v4_learning_curves.png`
- `/root/results/processed/ethan_task/Hopper_v4_auc_summary.png`
- `/root/results/processed/ethan_task/Walker2d_v4_auc_summary.png`

CSV summary: `/root/results/processed/ethan_task/summary.csv`
