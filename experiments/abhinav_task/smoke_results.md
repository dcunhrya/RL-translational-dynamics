# Abhinav Task Results

Offline-assisted runs report online env steps separately from offline dataset size and offline updates.

## Summary

| Env | Method | Seeds | Final return 95% CI | Normalized AUC 95% CI | Worst seed | Collapses |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Hopper-v4 | bc | 1 | 87.90 [87.90, 87.90] | 0.00 [0.00, 0.00] | 87.90 | 1 |
| Hopper-v4 | bc_anchor_sac K=10 | 1 | 70.76 [70.76, 70.76] | 68.12 [68.12, 68.12] | 70.76 | 1 |
| Hopper-v4 | bc_to_ppo | 1 | 83.93 [83.93, 83.93] | 85.67 [85.67, 85.67] | 83.93 | 1 |
| Hopper-v4 | bc_to_sac_to_ppo | 1 | 27.09 [27.09, 27.09] | 55.80 [55.80, 55.80] | 27.09 | 1 |

## Handoff Transients

| Env | Method | Seed | Switch step | Delta |
| --- | --- | ---: | ---: | ---: |
| Hopper-v4 | bc_to_sac_to_ppo | 0 | 10 | 0.00 |

## Figures

- `results/processed/abhinav_task_smoke/phase_marked_learning_curves.png`
- `results/processed/abhinav_task_smoke/policy_retention.png`
- `results/processed/abhinav_task_smoke/value_quality.png`
