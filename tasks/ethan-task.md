# Ethan — Task Assignment

**Spec:** [`docs/superpowers/specs/2026-05-26-rl-sequencing-transfer-mechanism-design.md`](../docs/superpowers/specs/2026-05-26-rl-sequencing-transfer-mechanism-design.md)
**Role:** PPO→SAC value ablation (Tier 1) + switching-dynamics studies (timing sweep, adaptive trigger; Tier 4) + part of long-horizon.
**Modal-compute target:** ~60 units (~120 GPU-hr, ~$120 of the shared $500).

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
| Ryan | baselines + SAC→PPO value ablation + long-horizon (baselines) | ~56 |
| **Ethan** | PPO→SAC value ablation + timing sweep + adaptive trigger + long-horizon (1 T0 arm + PPO) | ~60 |
| Abhinav | offline pipeline (BC, IQL/AWAC, interleaved-BC) + infra (Modal/logging/analysis) + long-horizon (2 T0 arms) | ~57 |

---

## Implementation tasks

1. **Factored transfer flags on `train_reverse_handoff.py` (PPO→SAC direction).** Same flag surface as Ryan's SAC→PPO version (`--policy-init`, `--value-init {random,self-warmup,source-aligned}`, sources). The existing 1000-step Bellman warm-up becomes the `value=self-warmup` operation; `value=random` skips it.
2. **`source-aligned` value target for PPO→SAC.** Implement `Q(s,a) ← reward-to-go R_t` from stored PPO trajectories (Monte-Carlo return — **not** the misspecified `V(s)+r`). Open question per spec: optionally compare against a short Bellman seed; log both warm-up losses if cheap.
3. **Minimal adaptive trigger** (`--switch-trigger no-improve --patience N --min-first-phase B`): switch after `N` non-improving eval checkpoints past a minimum first-phase budget; log chosen `switch_step` + `switch_reason`. Framed as a "timing isn't magic" baseline, not a headline method.
4. **Timing-sweep harness**: reuse `--switch-fraction`; aggregate 25/50/75 with existing Exp 2/3 data so we don't re-run 50%.

Depends on: Abhinav's diagnostic-logging harness + Modal wrapper (Day-1 morning). Timing sweep + adaptive are **Day-2 gated** (need "best schedule" identified from Tier 0/1).

## Experiments owned

| Experiment | arms×envs×seeds×steps | runs | units |
|---|---|---|---|
| **Tier 1** PPO→SAC value ablation `{random, self-warmup, source-aligned}` | 3×2×5×500k | 30 | 30 |
| **Tier 4** timing sweep 25%/75% on best schedule (50% reused) | 2×2×3×500k | 12 | 12 |
| **Tier 4** adaptive trigger | 1×2×3×500k | 6 | 6 |
| Long-horizon: pure PPO @1M Hopper | 1×1×3×1M | 3 | 6 |
| Long-horizon: 1× Tier-0 arm @1M Hopper (coordinate w/ Ryan/Abhinav which arm) | 1×1×3×1M | 3 | 6 |
| **Total** | | **54** | **60** |

## Acceptance criteria

- Smoke gate (1 seed × 50k) before each sweep.
- PPO→SAC ablation reports AUC + bootstrap CIs; value factor judged by CI separation vs `value=random` — shows the C1 decomposition is not PPO-target-specific.
- Adaptive trigger logs its switch decisions; report whether it lands near the best fixed fraction (diagnostic: if it picks extreme points, fixed fractions were miscalibrated).
- Full diagnostic suite on the PPO→SAC arms (SAC-side: Q-scale/overestimation, entropy, α; policy-retention from PPO).

## Notes

- You own the **generalization-of-C1 result** (decomposition holds for the off-policy target too) plus the **timing-robustness** story. Both feed the "understanding" thesis, not a leaderboard.
- The timing sweep exists so the scheduler claim isn't pinned to a single 50% split — keep it lightweight.
