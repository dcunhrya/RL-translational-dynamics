# Abhinav — Task Assignment

**Spec:** [`docs/superpowers/specs/2026-05-26-rl-sequencing-transfer-mechanism-design.md`](../docs/superpowers/specs/2026-05-26-rl-sequencing-transfer-mechanism-design.md)
**Role:** offline-assisted pipeline (BC, IQL/AWAC, interleaved-BC) + shared infrastructure (Modal wrappers, diagnostic-logging harness, analysis) + part of long-horizon.
**Modal-compute target:** ~57 units (~114 GPU-hr, ~$114 of the shared $500).

> Note: you carry extra **non-Modal** infra labor (Modal/logging/analysis) on top of an equal compute share — consistent with the milestone role (you set up Modal + W&B). Compute is balanced; orchestration is the asymmetric part you already own.

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
| Ethan | PPO→SAC value ablation + timing sweep + adaptive trigger + long-horizon (1 T0 arm + PPO) | ~60 |
| **Abhinav** | offline pipeline (BC, IQL/AWAC, interleaved-BC) + infra (Modal/logging/analysis) + long-horizon (2 T0 arms) | ~57 |

---

## Implementation tasks

**Infra (upstream — must land Day-1 morning so Ryan/Ethan can run):**
1. **Diagnostic-logging harness** — the shared wandb scalar set + expanded phase markers (`sac | distill | value_warmup | ppo | sac_critic_warmup | bc_anchor | adaptive_switch`) used by every arm.
2. **Modal wrappers** — `experiment_4_modal.py` for Tier 0–1, plus wrappers for stretch tiers. Modal code stays out of core RL files (RULES.md Rule 1).
3. **Analysis** — `summarize_experiment_4.py`: per-arm AUC w/ bootstrap CIs, phase-marked learning curves, policy-retention, value-quality, handoff-transient plots.

**Offline pipeline (your experiments):**
4. **`train_bc.py`** — BC → standalone Gaussian policy from cached D4RL expert data (`demos.py` already loads/caches it). Transfer into a target reuses the existing 500-step distillation loop (keeps `policy=distill` uniform; sidesteps the SAC/PPO actor-arch mismatch). ~150 LOC.
5. **`train_iql.py` or `train_awac.py`** — one, chosen at the Tier-2 gate by whichever drops in cleaner. Offline on D4RL; saves policy + value for transfer.
6. **Interleaved-BC mode** — `--bc-anchor-interval K` in the SAC loop: short distillation toward the expert policy every K steps; log `bc_anchor` events. Distinguishes one-time init effect from ongoing distributional-anchoring.

**Stretch transfer warm-starts (after the core offline pipeline is stable):**
7. **Easy-environment pretraining** — train a starter policy on a simplified or more forgiving version of the target MuJoCo environment, then distill/init it into SAC or PPO on the original benchmark. Candidate simplifications: denser or more lenient reward shaping, easier termination conditions, shorter horizons, narrower initial-state perturbations, or adjusted action penalties that reward smoother behavior. Report pretraining updates separately and keep real-environment fine-tuning steps matched against BC→SAC / BC→PPO and pure SAC/PPO.
8. **General starter policy diagnostic** — exploratory only. Test whether a broadly pretrained starter policy can help multiple target environments if a clean transfer interface exists. Because MuJoCo tasks can have incompatible observation/action dimensions, prefer a narrow pilot or per-target distillation over a full sweep.

## Experiments owned

| Experiment | arms×envs×seeds×steps | runs | units |
|---|---|---|---|
| BC pretrain (offline) | 2 envs | 2 | ~0.5 |
| **Tier 1** BC→PPO, BC→SAC | 2×2×5×500k | 20 | 20 |
| **Tier 2** IQL/AWAC offline train | 2 envs | 2 | ~0.6 |
| **Tier 2** IQL/AWAC → PPO, → SAC | 2×2×3×500k | 12 | 12 |
| **Tier 3** interleaved-BC K-sweep (Hopper) | 3×1×3×500k | 9 | 9 |
| **Tier 3** interleaved-BC best-K (Walker2d) | 1×1×3×500k | 3 | 3 |
| Long-horizon: 2× Tier-0 arms @1M Hopper | 2×1×3×1M | 6 | 12 |
| **Total** | | **54** | **~57** |

## Acceptance criteria

- BC pretrain validated (eval return before fine-tuning logged — distinguishes a failed warm-start from a transfer-mechanism result).
- Tier-1 BC arms framed as the **interaction hypothesis** (C2): do NOT pre-assume BC→PPO fails. Log policy-retention from the BC policy under both PPO and SAC.
- Tier-2 (C3) is gated: build only if Tier 0/1 show a value effect (separated CIs). If built, isolate whether IQL/AWAC's transferred value explains any BC gap.
- Interleaved-BC: report whether mid-training anchoring helps/hurts vs init-only, with the anchor events marked on curves.
- Easy-environment pretraining: evaluate as the real stretch transfer experiment. Main metrics are early AUC, convergence speed, final return, stability, and policy retention after transfer. Claims should focus on whether curriculum-style warm starts improve real-environment learning, not on universal dominance over SAC.
- General starter policy: treat as exploratory unless the architecture/space mismatch is solved cleanly; do not let it displace the core BC/IQL/AWAC runs.

## Coordination / dependency notes

- **Your infra (items 1–3) is upstream for everyone.** Ship the logging harness + Modal wrapper first; Ryan/Ethan's sweeps block on it.
- BC pretrain must complete before any BC transfer arm (yours) — run it Day-1 morning alongside infra.
- Offline-assisted results live in the **separate budget category** (report offline dataset size + pretraining updates; no env-step-parity claims vs pure online).
- Tiers 2–4 and transfer warm-starts are stretch: if Modal burn or time runs short, cut stretch before touching any core 5-seed sweep.
