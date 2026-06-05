# IQL Warm-Start Transfer Results

This report covers the local IQL transfer sweep completed after the Modal run was stopped to conserve credits. The transfer jobs were run on the two local RTX 3090s with:

```bash
uv run python experiments/abhinav_task/run_iql_local.py --gpus 0,0,0,1,1,1
```

Modal was used only before the credit warning to produce the two IQL pretrain artifacts that were downloaded locally. No additional Modal GPU work was launched after switching to the local GPUs.

## Experiment Setup

- Environments: `Hopper-v4`, `Walker2d-v4`
- Seeds: 5 per transfer method and environment
- Offline pretraining: IQL, 100k offline updates from D4RL expert datasets
- Online budget: 500k environment steps per transfer run
- Evaluation: every 5k environment steps, 5 evaluation episodes
- Transfer mechanism: actor distillation from the IQL actor into the online policy initialization
- Important limitation: IQL value functions are not directly transplanted into SAC/PPO; the online algorithms initialize or self-warm value estimates in their own format.

Offline IQL pretrain checkpoints:

| Env | Offline updates | Final offline-policy eval |
| --- | ---: | ---: |
| Hopper-v4 | 100,000 | 3044.38 |
| Walker2d-v4 | 100,000 | 4928.86 |

## Summary

| Env | Method | Seeds | Final return 95% CI | Normalized AUC 95% CI | Worst seed | Collapses |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Hopper-v4 | IQL -> PPO | 5 | 503.17 [439.01, 560.96] | 491.09 [437.72, 544.45] | 376.13 | 0 |
| Hopper-v4 | IQL -> SAC | 5 | 2617.85 [1812.18, 3202.80] | 2160.22 [1895.49, 2409.08] | 1129.75 | 0 |
| Hopper-v4 | IQL -> SAC -> PPO | 5 | 433.94 [410.05, 457.83] | 1049.17 [943.63, 1154.71] | 401.28 | 0 |
| Walker2d-v4 | IQL -> PPO | 5 | 410.80 [256.74, 504.76] | 570.49 [515.88, 625.43] | 104.25 | 0 |
| Walker2d-v4 | IQL -> SAC | 5 | 4384.95 [3940.94, 4921.42] | 2125.53 [1876.07, 2338.78] | 3701.90 | 0 |
| Walker2d-v4 | IQL -> SAC -> PPO | 5 | 405.48 [330.52, 483.76] | 624.28 [541.48, 756.48] | 275.24 | 0 |

## High-Quality Figures

The new tracked figure set is in `experiments/abhinav_task/figures/`, with both PNG and PDF versions for report/poster use.

- Learning curves: [figures/iql_learning_curves.png](figures/iql_learning_curves.png)
- Final return and AUC summary: [figures/iql_final_auc_summary.png](figures/iql_final_auc_summary.png)
- PPO-phase retention diagnostic: [figures/iql_sac_ppo_retention.png](figures/iql_sac_ppo_retention.png)

The learning curves make the phase effect clear. `IQL -> SAC` uses the online budget most effectively after the offline warm start, reaching final returns of 2617.85 on Hopper-v4 and 4384.95 on Walker2d-v4. `IQL -> PPO` is stable but low-return, ending near 503.17 on Hopper-v4 and 410.80 on Walker2d-v4.

The retention diagnostic shows why the three-phase `IQL -> SAC -> PPO` schedule is not yet a strong final method. Before the PPO handoff, the SAC phase reaches useful policies, especially on Hopper-v4. After the PPO phase, mean return drops from 2812 to 434 on Hopper-v4 and from 2441 to 405 on Walker2d-v4. This supports a phase-based interpretation: the SAC phase is productive, but the current PPO refinement mechanism does not preserve the SAC policy.

## Interpretation

The strongest IQL transfer schedule in this sweep is `IQL -> SAC`. It reaches high final return on both environments and has strong AUC, especially on Walker2d-v4. This is consistent with the phase-based hypothesis: IQL gives SAC a competent non-random actor initialization, then SAC uses the online budget effectively.

`IQL -> PPO` behaves similarly to the earlier BC-to-PPO results: it is stable and avoids final collapses, but it does not turn the strong offline policy into high online returns. PPO refinement from the distilled offline actor remains conservative and low-return in these two environments.

`IQL -> SAC -> PPO` is the clearest negative result. Its AUC is higher than `IQL -> PPO` on Hopper-v4, which indicates useful SAC-phase learning before the switch. But final returns collapse back toward PPO-level performance after the PPO phase. That means the current PPO refinement stage is not preserving SAC's high-return policy under this transfer setup.

Compared with the earlier BC/AWAC warm-start runs, IQL is competitive with other offline warm starts but does not change the main conclusion: the online SAC phase is doing most of the useful post-warm-start improvement, while the final PPO phase needs a better preservation/refinement mechanism before it can be defended as a stabilizing phase.

## Takeaways

- IQL warm starts are useful when followed by SAC.
- The result supports an algorithm-sequencing interpretation, not a universal claim that every additional phase helps.
- The PPO final phase is currently destructive for `IQL -> SAC -> PPO` on these two environments.
- The cleanest next diagnostic is to test a more conservative PPO refinement stage: lower PPO learning rate, smaller clip range, shorter PPO phase, KL early stopping, or explicit behavior regularization to the incoming SAC policy.

## Artifacts

- Tracked plotting script: `experiments/abhinav_task/plot_iql_results.py`
- Tracked learning curves: `experiments/abhinav_task/figures/iql_learning_curves.{png,pdf}`
- Tracked aggregate summary: `experiments/abhinav_task/figures/iql_final_auc_summary.{png,pdf}`
- Tracked retention diagnostic: `experiments/abhinav_task/figures/iql_sac_ppo_retention.{png,pdf}`
- Summary JSON: `results/processed/abhinav_task_iql_local/summary.json`
- Learning curves: `results/processed/abhinav_task_iql_local/phase_marked_learning_curves.png`
- Policy retention plot: `results/processed/abhinav_task_iql_local/policy_retention.png`
- Value quality plot: `results/processed/abhinav_task_iql_local/value_quality.png`
