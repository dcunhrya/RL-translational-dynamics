# Abhinav Task Results

Offline-assisted runs report online env steps separately from offline dataset size and offline updates.

## Summary

| Env | Method | Seeds | Final return 95% CI | Normalized AUC 95% CI | Worst seed | Collapses |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Hopper-v4 | awac | 1 | 842.47 [842.47, 842.47] | 0.00 [0.00, 0.00] | 842.47 | 0 |
| Hopper-v4 | awac_to_ppo | 3 | 518.06 [429.24, 603.29] | 526.53 [462.14, 616.97] | 429.24 | 0 |
| Hopper-v4 | awac_to_sac | 3 | 2609.22 [1198.67, 3381.81] | 2224.63 [2156.95, 2330.06] | 1198.67 | 0 |
| Hopper-v4 | bc | 1 | 3048.87 [3048.87, 3048.87] | 0.00 [0.00, 0.00] | 3048.87 | 0 |
| Hopper-v4 | bc_anchor_sac K=25000 | 3 | 1167.39 [770.45, 1501.80] | 1436.41 [1243.08, 1773.83] | 770.45 | 0 |
| Hopper-v4 | bc_anchor_sac K=50000 | 3 | 1994.57 [920.96, 3285.69] | 1975.20 [1660.32, 2135.89] | 920.96 | 0 |
| Hopper-v4 | bc_anchor_sac K=100000 | 3 | 1812.34 [850.67, 3216.29] | 1792.60 [684.23, 2422.23] | 850.67 | 0 |
| Hopper-v4 | bc_to_ppo | 5 | 503.05 [416.87, 589.23] | 553.33 [449.09, 662.40] | 365.33 | 0 |
| Hopper-v4 | bc_to_ppo (1000k) | 3 | 555.03 [524.51, 576.01] | 501.33 [487.71, 514.52] | 524.51 | 0 |
| Hopper-v4 | bc_to_sac | 5 | 3047.11 [2646.98, 3299.83] | 1897.68 [1461.20, 2315.44] | 2280.32 | 0 |
| Hopper-v4 | bc_to_sac (1000k) | 3 | 2763.85 [1528.48, 3435.66] | 2301.51 [1998.65, 2824.64] | 1528.48 | 0 |
| Hopper-v4 | bc_to_sac_to_ppo | 5 | 472.41 [410.66, 529.48] | 852.49 [599.49, 1078.81] | 374.30 | 0 |
| Hopper-v4 | easy_sac | 3 | 4.48 [4.25, 4.69] | 22.82 [18.38, 27.71] | 4.25 | 3 |
| Hopper-v4 | easy_sac_to_sac | 3 | 3333.09 [3175.39, 3437.63] | 2133.92 [1999.69, 2398.93] | 3175.39 | 0 |
| Walker2d-v4 | awac | 1 | 4934.75 [4934.75, 4934.75] | 0.00 [0.00, 0.00] | 4934.75 | 0 |
| Walker2d-v4 | awac_to_ppo | 3 | 793.52 [421.33, 1056.62] | 459.16 [309.91, 587.44] | 421.33 | 0 |
| Walker2d-v4 | awac_to_sac | 3 | 3545.32 [2314.41, 4208.81] | 2235.97 [1940.49, 2411.20] | 2314.41 | 0 |
| Walker2d-v4 | bc | 1 | 4898.39 [4898.39, 4898.39] | 0.00 [0.00, 0.00] | 4898.39 | 0 |
| Walker2d-v4 | bc_anchor_sac K=50000 | 3 | 2750.64 [1749.22, 4485.52] | 2492.97 [2045.04, 2820.54] | 1749.22 | 0 |
| Walker2d-v4 | bc_to_ppo | 5 | 625.78 [371.42, 994.58] | 543.75 [425.44, 708.44] | 258.38 | 0 |
| Walker2d-v4 | bc_to_sac | 5 | 3965.27 [3582.83, 4353.23] | 2350.95 [2193.65, 2520.11] | 3485.34 | 0 |
| Walker2d-v4 | bc_to_sac_to_ppo | 5 | 371.66 [186.42, 576.87] | 770.13 [664.51, 879.37] | 119.87 | 0 |

## Handoff Transients

| Env | Method | Seed | Switch step | Delta |
| --- | --- | ---: | ---: | ---: |
| Hopper-v4 | bc_to_sac_to_ppo | 0 | 250000 | 0.00 |
| Hopper-v4 | bc_to_sac_to_ppo | 1 | 250000 | 0.00 |
| Hopper-v4 | bc_to_sac_to_ppo | 2 | 250000 | 0.00 |
| Hopper-v4 | bc_to_sac_to_ppo | 3 | 250000 | 0.00 |
| Hopper-v4 | bc_to_sac_to_ppo | 4 | 250000 | 0.00 |
| Walker2d-v4 | bc_to_sac_to_ppo | 0 | 250000 | 0.00 |
| Walker2d-v4 | bc_to_sac_to_ppo | 1 | 250000 | 0.00 |
| Walker2d-v4 | bc_to_sac_to_ppo | 2 | 250000 | 0.00 |
| Walker2d-v4 | bc_to_sac_to_ppo | 3 | 250000 | 0.00 |
| Walker2d-v4 | bc_to_sac_to_ppo | 4 | 250000 | 0.00 |

## Figures

- `/root/results/processed/abhinav_task/phase_marked_learning_curves.png`
- `/root/results/processed/abhinav_task/policy_retention.png`
- `/root/results/processed/abhinav_task/value_quality.png`
