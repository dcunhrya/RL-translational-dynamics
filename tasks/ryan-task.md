# Ryan — Task Assignment

**Spec:** [`docs/superpowers/specs/2026-05-26-rl-sequencing-transfer-mechanism-design.md`](../docs/superpowers/specs/2026-05-26-rl-sequencing-transfer-mechanism-design.md)
**Role:** baselines + the SAC→PPO explanatory spine (Tier 0) + its long-horizon check.
**Modal-compute target:** ~56 units (~112 GPU-hr, ~$112 of the shared $500).

---

## Compute model (shared across all three files)

- **1 unit** = one 500k-step MuJoCo run ≈ ~2 GPU-hr ≈ ~$2 Modal (L4/A10G class).
- 1M-step run = **2 units**. Offline pretrain (BC/IQL/AWAC, no env stepping) ≈ **0.3 unit**.
- Core arms: **5 seeds × 2 envs** (Hopper-v4, Walker2d-v4). Stretch arms: **3 seeds**. Long-horizon: **3 seeds × Hopper × 1M**.
- Project total ≈ **173 units** (~346 GPU-hr, ~$346 Modal). The 2× local 3090s absorb smoke-tests + overflow, keeping Modal spend under the $500 cap.
- Per-member target ≈ **~58 units**.

### Global split summary

| Member | Owns | Units |
|---|---|---|
| **Ryan** | baselines + SAC→PPO value ablation + long-horizon (baselines) | ~56 |
| Ethan | PPO→SAC value ablation + timing sweep + adaptive trigger + long-horizon (1 T0 arm + PPO) | ~60 |
| Abhinav | offline pipeline (BC, IQL/AWAC, interleaved-BC) + infra (Modal/logging/analysis) + long-horizon (2 T0 arms) | ~57 |

---

## Implementation tasks

1. **Factored transfer flags on `train_handoff.py` (SAC→PPO direction).** Add `--policy-init {random,distill}`, `--value-init {random,self-warmup,source-aligned}`, `--policy-source`, `--value-source`. Each Tier-0 arm = an explicit flag combo. Keep optimizer reset (RULES.md). The existing 500-step distillation stays as the fixed `policy=distill` operation.
2. **`source-aligned` value warm-up for SAC→PPO.** Implement `V(s) ← E_{a∼π_distilled}[min(Q1,Q2)(s,a)]` regressed over replay states; log `warmup_loss`. Fall back to `self-warmup` if it doesn't converge — document, don't hide.
3. **Baseline runners** for pure SAC / pure PPO at 500k (and 1M for long-horizon). Coordinate with Abhinav's Modal wrapper + diagnostic-logging harness (upstream dependency).

Depends on: Abhinav's diagnostic-logging harness + Modal wrapper landing Day-1 morning.

## Experiments owned

| Experiment | arms×envs×seeds×steps | runs | units |
|---|---|---|---|
| Pure SAC baseline | 1×2×5×500k | 10 | 10 |
| Pure PPO baseline | 1×2×5×500k | 10 | 10 |
| **Tier 0** SAC→PPO value ablation `{random, self-warmup, source-aligned}` | 3×2×5×500k | 30 | 30 |
| Long-horizon: pure SAC @1M Hopper | 1×1×3×1M | 3 | 6 |
| **Total** | | **53** | **56** |

## Acceptance criteria

- Each new arm passes the smoke gate (1 seed × 50k: no NaN, beats random floor, logs all diagnostics + correct arm metadata) before its 5-seed sweep.
- Tier-0 arms report AUC with bootstrap CIs; the `value` factor's effect is judged by CI separation against the `value=random` arm (the C1 test).
- Diagnostic suite logged on every arm: policy-retention (action-MSE/KL from SAC), PPO explained variance, advantage stats, `value_loss_at_handoff`, pre/post-handoff eval transient.

## Notes

- You own the **C1 result**. The headline question for your slice: *given fixed policy distillation, does the value PPO inherits change the outcome?* A null is a valid, reportable result — pair it with the negative-result diagnostics in the spec.
- Reuse existing 100k baseline data only for sanity context; the comparison budget is 500k, so baselines are fresh.
