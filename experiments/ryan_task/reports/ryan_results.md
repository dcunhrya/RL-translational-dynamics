# Ryan Results

Generated at Unix time `1779994610` from `experiments/ryan_modal_manifest.json` and the W&B completion audit.

## Completion

All `53/53` Ryan jobs were completed and included. This covers 500k-step SAC/PPO baselines, 500k-step SAC->PPO value-initialization ablations on Hopper-v4 and Walker2d-v4, and the Hopper-v4 SAC 1M long-horizon check.

## Headline Figures

- [Learning curves](../figures/headline/learning_curves_500k.png)
- [AUC summary](../figures/headline/auc_summary.png)
- [Final return summary](../figures/headline/final_return_summary.png)
- [Value-init AUC deltas](../figures/headline/value_init_auc_deltas.png)
- [Hopper SAC long-horizon check](../figures/headline/hopper_sac_long_horizon.png)

## Main Results

- `Hopper-v4`: best mean AUC is `SAC` (`2260.239` normalized AUC); best mean final return is `SAC` (`2895.6`).
- `Walker2d-v4`: best mean AUC is `SAC` (`1642.230` normalized AUC); best mean final return is `SAC` (`3310.4`).

## C1 Value-Ablation Test

- `Hopper-v4` `SAC->PPO self-warmup V` vs random value init: mean normalized AUC delta `-73.54` with 95% CI [`-165.26`, `18.18`] over `5` paired seeds.
- `Hopper-v4` `SAC->PPO source-aligned V` vs random value init: mean normalized AUC delta `-88.16` with 95% CI [`-188.38`, `12.05`] over `5` paired seeds.
- `Walker2d-v4` `SAC->PPO self-warmup V` vs random value init: mean normalized AUC delta `21.22` with 95% CI [`-7.26`, `54.91`] over `5` paired seeds.
- `Walker2d-v4` `SAC->PPO source-aligned V` vs random value init: mean normalized AUC delta `115.85` with 95% CI [`-25.31`, `278.73`] over `5` paired seeds.

Interpretation should focus on whether these paired CIs separate from zero. A null or mixed result is still informative: with policy distillation fixed, value initialization may not be the limiting transfer mechanism under this matched 500k-step budget.

## Summary Table

| Env | Method | Seeds | Final mean | Final 95% CI | Norm. AUC mean | Norm. AUC 95% CI | Collapse count |
|---|---|---:|---:|---:|---:|---:|---:|
| Hopper-v4 | SAC | 5 | 2895.6 | [2125.4, 3332.0] | 2260.239 | [1959.151, 2561.327] | 1 |
| Hopper-v4 | PPO | 5 | 384.3 | [366.8, 401.8] | 324.941 | [295.808, 349.616] | 0 |
| Hopper-v4 | SAC->PPO random V | 5 | 403.5 | [219.6, 528.0] | 1219.303 | [1041.024, 1387.314] | 5 |
| Hopper-v4 | SAC->PPO self-warmup V | 5 | 439.2 | [391.1, 487.4] | 1145.765 | [889.789, 1382.994] | 5 |
| Hopper-v4 | SAC->PPO source-aligned V | 5 | 384.9 | [370.9, 398.1] | 1131.139 | [933.887, 1328.390] | 5 |
| Walker2d-v4 | SAC | 5 | 3310.4 | [2932.4, 3688.5] | 1642.230 | [1483.623, 1776.310] | 0 |
| Walker2d-v4 | PPO | 5 | 441.5 | [408.1, 467.4] | 381.714 | [345.874, 434.004] | 1 |
| Walker2d-v4 | SAC->PPO random V | 5 | 442.5 | [330.2, 525.6] | 602.467 | [543.601, 661.333] | 5 |
| Walker2d-v4 | SAC->PPO self-warmup V | 5 | 464.6 | [359.5, 545.4] | 623.689 | [571.644, 683.761] | 5 |
| Walker2d-v4 | SAC->PPO source-aligned V | 5 | 646.6 | [446.9, 899.0] | 718.320 | [579.152, 859.397] | 5 |

## Diagnostics

- [Hopper per-seed curves](../figures/diagnostics/Hopper-v4_per_seed_curves.png)
- [Walker2d per-seed curves](../figures/diagnostics/Walker2d-v4_per_seed_curves.png)
- [Handoff transient](../figures/diagnostics/handoff_transient.png)
- [Mechanism diagnostics](../figures/diagnostics/mechanism_diagnostics.png)

Processed CSVs live in `../processed/`, including per-run summaries, per-arm summaries, rank summaries, paired value-init deltas, handoff transients, and diagnostic rows.

## Limitations

These results cover Hopper-v4 and Walker2d-v4 with five 500k-step seeds per arm, plus a three-seed Hopper SAC 1M check. They should support Ryan's SAC->PPO mechanism claim, not a universal claim that handoffs beat strong standalone SAC across all environments.
