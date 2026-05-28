# RL Translational Dynamics — Progress Update

**Project:** CS 224R — Algorithm sequencing in deep reinforcement learning  
**Date:** May 21, 2026  
**Focus:** SAC → PPO handoff for MuJoCo continuous control under fixed interaction budgets

---

## 1. Research Question

Can we improve sample efficiency or final performance under a fixed environment-step budget by **sequencing** two RL algorithms—starting with off-policy **SAC** (sample-efficient exploration) and transitioning to on-policy **PPO** (stable policy refinement)—instead of training with a single algorithm end-to-end?

This project tests that hypothesis through:

1. Reproducible single-algorithm baselines (SAC, PPO)
2. Fixed-fraction handoff schedules (`SAC(f) → PPO(1−f)`)
3. *(Planned)* Adaptive handoff triggers based on internal optimization metrics
4. *(Planned)* Compute-normalized comparisons (gradient updates / FLOPs proxy)

**Primary environments:** `Hopper-v4`, `Walker2d-v4`  
**Secondary environments (sanity sweep):** `HalfCheetah-v4`, `Ant-v4`

---

## 2. Summary of Progress

| Milestone | Status | Notes |
|-----------|--------|-------|
| Training stack (CleanRL-style SAC/PPO) | **Done** | Single-file scripts, explicit losses, W&B logging |
| Modal cloud orchestration | **Done** | Separate modal wrappers; core logic runs locally |
| Experiment 0: sanity + reproducibility | **Done** | 100k steps, 2 seeds, 4 envs; gate passed |
| Experiment 2: handoff pipeline | **Done (code)** | Distillation, value warm-up, checkpointing, summarize script |
| Experiment 2: 100k pilot (3 seeds) | **Done (results)** | 32 runs fetched; handoff does not yet beat pure SAC |
| Experiment 2: full 1M grid (5 seeds) | **In progress** | Launcher and Modal infra ready; awaiting full run completion |
| Experiment 1: 1M baselines (5 seeds) | **Pending** | Will be covered by Experiment 2 baseline arm |
| Experiment 3: adaptive trigger | **Not started** | KL-drift trigger scoped in proposal |
| Experiment 4: compute-constrained comparison | **Not started** | Depends on best fixed/adaptive handoff |

**Bottom line so far:** The implementation stack is stable and reproducible. SAC is the stronger early-training algorithm on most tasks at short horizons. Fixed SAC→PPO handoff improves over pure PPO in the pilot but **does not yet beat pure SAC** at equal 100k-step budget. The critical open question is whether a longer budget (1M steps) and/or better switch timing unlocks gains—that is what the full Experiment 2 grid is designed to answer.

---

## 3. Infrastructure Built

### 3.1 Code organization

The repository follows a CleanRL-inspired layout: exposed training loops, minimal abstraction, and strict separation between RL logic and Modal deployment.

```
src/RL-translational-dynamics/
├── exp0/
│   ├── train_sac.py          # Pure SAC baseline
│   ├── train_ppo.py          # Pure PPO baseline
│   └── summarize_experiment_0.py
├── exp2/
│   ├── train_handoff.py      # SAC → PPO fixed-fraction handoff
│   ├── handoff_utils.py      # Switch-step math + logging metadata
│   └── summarize_experiment_2.py
└── modal/
    ├── experiment_0_modal.py
    └── experiment_2_modal.py
```

### 3.2 Handoff protocol (Experiment 2)

When the environment-step counter reaches `switch_step = int(total_timesteps × switch_fraction)`:

1. **SAC phase ends** — actor and twin Q-networks checkpointed at the switch boundary.
2. **Actor transfer** — PPO actor initialized from SAC actor weights.
3. **Distillation** — 500 gradient steps aligning PPO actor actions to SAC deterministic actions on replay-buffer observations (MSE loss, lr = 1e−3, batch size 1024).
4. **Value warm-up** — 2 PPO updates training **critic only** with a fresh Adam optimizer (lr = 3e−4).
5. **PPO phase** — full actor+critic updates with a **new** Adam optimizer (no momentum carry-over from SAC or distillation).
6. **Logging** — explicit phase labels (`sac`, `handoff`, `ppo_value_warmup`, `ppo`), switch metadata (`switch_step`, `switch_reason`, `handoff_fraction`), and internal PPO metrics (KL, clip fraction, explained variance, etc.).

Design choices follow project rules: no optimizer-state leakage across phases, handoff events visible in metrics, and locally runnable scripts without Modal.

### 3.3 Experiment orchestration

- **Local:** `scripts/run_experiment_0.sh`, `scripts/run_experiment_2.sh` (`LAUNCHER=local`)
- **Modal:** detached 50-job grid (30 handoff + 20 baselines) via `experiment_2_modal.py`
- **Fetch + summarize:** `scripts/fetch_experiment_2_results.sh` → `summarize_experiment_2.py`
- **Tests:** `tests/test_handoff_utils.py`, `tests/test_summarize_experiment_2.py`

### 3.4 Logging schema

Every run logs to `metrics.jsonl` (and optionally W&B):

- Run identity: `algorithm`, `env`, `seed`
- Progress: `env_steps`, `gradient_updates`, `wall_clock_sec`
- Evaluation: `eval_return_mean`, `eval_return_std` (every 5k steps)
- Handoff: `phase`, `switched`, `switch_step`, `switch_reason`, `handoff_fraction`, `planned_switch_step`
- PPO internals: policy/value loss, entropy, approximate KL, clip fraction, explained variance, advantage stats

W&B project: `rl-translational-dynamics` (Experiment 0 also logged under entity `herschethan-stanford-university`).

---

## 4. Experiment 0 — Sanity & Reproducibility

**Purpose:** Validate training stack stability before expensive sweeps.

| Setting | Value |
|---------|-------|
| Date completed | 2026-05-18 |
| Algorithms | SAC, PPO |
| Environments | Hopper-v4, Walker2d-v4, HalfCheetah-v4, Ant-v4 |
| Seeds | 0, 1 |
| Budget | 100,000 env steps |
| Eval interval | 5,000 steps, 5 episodes |

### 4.1 Gate criteria — PASS

- Both SAC and PPO improved reward on Hopper-v4 across both seeds
- No recurring crashes, NaNs, or divergence
- Logging outputs complete and consistent

### 4.2 Final eval return (mean ± std across 2 seeds)

| Environment | SAC | PPO |
|-------------|-----|-----|
| **Hopper-v4** | 1204.24 ± 795.57 | 346.23 ± 15.37 |
| **Walker2d-v4** | 326.17 ± 44.50 | 339.01 ± 33.65 |
| **HalfCheetah-v4** | 4830.85 ± 1075.63 | 275.69 ± 26.16 |
| **Ant-v4** | 1152.57 ± 307.84 | 448.39 ± 92.06 |

### 4.3 Interpretation

At 100k steps, **SAC dominates** on Hopper, HalfCheetah, and Ant. Walker2d is roughly tied with a slight PPO edge. High SAC variance on Hopper (795 std with mean 1204) reflects seed sensitivity at this short horizon—one seed learns much faster than the other.

**Implication for handoff:** SAC is the natural first-phase algorithm. The handoff hypothesis is not "PPO is better early" but "PPO refinement after SAC pre-training may outperform either algorithm alone at matched total budget"—especially once budgets extend beyond 100k steps where PPO may catch up.

### 4.4 Artifacts

- Learning curves: `results/processed/experiment_0/sac_vs_ppo_learning_curves.png`
- Final returns: `results/processed/experiment_0/sac_vs_ppo_final_returns.png`
- Full write-up: `experiments/experiment_0.md`

---

## 5. Experiment 2 — Fixed Handoff Pilot (100k steps, 3 seeds)

**Purpose:** Verify handoff mechanics and obtain an early signal on whether fixed SAC→PPO schedules help before committing to the full 1M-step grid.

### 5.1 Pilot configuration

| Setting | Pilot (completed) | Full grid (planned) |
|---------|-------------------|---------------------|
| Budget | 100,000 env steps | 1,000,000 env steps |
| Seeds | 0, 1, 2 | 0, 1, 2, 3, 4 |
| Environments | Hopper-v4, Walker2d-v4 | Hopper-v4, Walker2d-v4 |
| Handoff fractions | 0.25, 0.50, 0.75 | 0.25, 0.50, 0.75 |
| Switch steps | 25k / 50k / 75k | 250k / 500k / 750k |
| Runs | 32 fetched locally | 50 total (30 handoff + 20 baselines) |

Handoff fractions map to switch points via `switch_step = int(total_timesteps × fraction)`, clamped to `[1, total_timesteps − 1]`.

### 5.2 Pilot results — final eval return (mean ± std, 3 seeds)

#### Hopper-v4

| Method | Final return | vs SAC | vs PPO |
|--------|-------------|--------|--------|
| **SAC** | 2191.9 ± 1135.2 | — | +517% |
| **PPO** | 356.1 ± 33.1 | −84% | — |
| handoff@0.25 | 328.8 ± 84.6 | −85% | −8% |
| handoff@0.50 | 414.7 ± 59.7 | −81% | +16% |
| handoff@0.75 | 532.5 ± 12.7 | −76% | +50% |

#### Walker2d-v4

| Method | Final return | vs SAC | vs PPO |
|--------|-------------|--------|--------|
| **SAC** | 621.6 ± 264.7 | — | +107% |
| **PPO** | 301.0 ± 24.9 | −52% | — |
| handoff@0.25 | 535.9 ± 294.4 | −14% | +78% |
| handoff@0.50 | 517.7 ± 166.0 | −17% | +72% |
| handoff@0.75 | 485.9 ± 78.1 | −22% | +61% |

*Percent deltas are approximate, computed from pilot means.*

### 5.3 Pilot results — eval return AUC (trapezoidal over env steps)

AUC captures sample efficiency across the full training horizon, not just the endpoint.

| Environment | SAC | PPO | handoff@0.25 | handoff@0.50 | handoff@0.75 |
|-------------|-----|-----|--------------|--------------|--------------|
| Hopper-v4 | 79.8M ± 43.3M | 27.6M ± 2.8M | 32.5M ± 6.5M | 41.8M ± 5.9M | 47.1M ± 11.9M |
| Walker2d-v4 | 37.4M ± 1.5M | 27.4M ± 2.1M | 42.0M ± 7.8M | 39.3M ± 4.1M | 35.8M ± 1.0M |

On Walker2d, **handoff@0.25 achieves the highest AUC** among all methods in the pilot, suggesting early switching preserves SAC's sample efficiency while still benefiting from some PPO refinement—but final return still trails pure SAC.

### 5.4 Success criteria — NOT MET (pilot)

Experiment 2 success requires at least one handoff fraction to beat **both** pure SAC and pure PPO on final return **or** eval AUC at equal budget. In the 100k pilot:

- **No handoff schedule beats both baselines** on either environment.
- All handoff variants **beat pure PPO** on both environments (largest gap on Hopper@0.75: +50%).
- **Pure SAC wins on final return** in both environments; handoff trails by 14–85% depending on env and fraction.
- On Walker2d AUC, handoff@0.25 exceeds SAC, but SAC still wins on Walker2d final return.

### 5.5 Interpretation

**What worked:**

- Handoff pipeline runs end-to-end on Modal without crashes or NaN failures.
- Phase transitions (SAC → distillation → value warm-up → PPO) are logged correctly.
- Transfer + distillation produces a PPO policy that substantially outperforms a cold-start PPO at 100k steps.
- Later switch fractions (0.75) help on Hopper—more SAC pre-training before switching—but still fall far short of pure SAC's final performance at this budget.

**What did not work (yet):**

- Switching to PPO **hurts** compared to continuing SAC for the full 100k steps. The PPO phase at 25k–75k remaining steps is too short to recover SAC's advantage and may introduce distribution-shift instability.
- High variance on SAC (especially Hopper: ±1135) means pilot conclusions should be treated as directional, not definitive.

**Hypotheses for the 1M-step grid:**

1. PPO may need more post-switch steps to surpass SAC's asymptotic performance—100k total budget leaves only 25k–75k PPO steps.
2. SAC's lead may shrink at 1M steps as PPO catches up, changing the relative ranking (Experiment 0 at 100k already showed PPO competitive on Walker2d).
3. Optimal switch fraction may be environment-dependent (early switch best for Walker2d AUC; late switch best for Hopper final return in pilot).

---

## 6. Planned Experiments (Not Yet Run)

### Experiment 1 — 1M-step baseline curves

Core reference curves for pure SAC and pure PPO at 1M steps with ≥5 seeds. This is structurally identical to the baseline arm of Experiment 2 (`experiment_2_baselines` W&B group). Deliverables: learning curves with confidence bands, time-to-threshold analysis, return vs gradient updates.

### Experiment 3 — Adaptive handoff trigger

Replace fixed switch fractions with an internal metric trigger:

- Monitor KL drift between consecutive SAC policy snapshots on a fixed state buffer
- Switch when moving-average KL < τ for k consecutive checks
- Initial sweep: τ ∈ {0.01, 0.02, 0.05}, k ∈ {2, 3}
- Compare against best fixed handoff from Experiment 2

### Experiment 4 — Compute-constrained comparison

Equalize methods by gradient updates or estimated FLOPs. Compare pure SAC, pure PPO, best fixed handoff, and best adaptive handoff. Output: return vs compute curves at matched budget points.

---

## 7. Risks & Open Questions

| Risk | Status | Mitigation |
|------|--------|------------|
| PPO instability after SAC weight transfer | Observed indirectly (handoff > PPO but << SAC) | Distillation + value warm-up; log KL/clip fraction across switch |
| Unfair compute comparisons | Not yet tested | Experiment 4 will normalize by gradient updates |
| Overfitting to one environment | Partially addressed | Both Hopper and Walker2d in all core experiments |
| Short-horizon pilot misleading | Active concern | Full 1M grid in flight; pilot explicitly labeled as early signal |
| SAC seed variance | Observed on Hopper | 5 seeds in full grid; report confidence bands |

**Open questions:**

1. At 1M steps, does any fixed handoff fraction beat both baselines?
2. Is there a PPO "catch-up" regime where late switching (high fraction) wins on final return but early switching wins on AUC?
3. Does adaptive switching (Experiment 3) outperform the best fixed schedule?
4. Under matched gradient updates, does handoff still help?

---

## 8. Next Steps

1. **Complete Experiment 2 full grid** — 1M steps, 5 seeds, 50 Modal jobs; fetch results and run `summarize_experiment_2.py`.
2. **Analyze 1M results** — update success criteria, produce learning-curve and AUC plots, identify best handoff fraction per environment.
3. **Experiment 3 scoping** — implement KL-drift trigger on top of stable handoff pipeline; reuse Modal orchestration pattern.
4. **Experiment 4 planning** — choose compute proxy (gradient updates vs FLOPs estimate) and matched budget points.
5. **Optional:** extend sanity sweep envs (HalfCheetah, Ant) to handoff if primary envs show promise at 1M.

---

## 9. Reproducibility

```bash
# Experiment 0 (local sanity sweep)
./scripts/run_experiment_0.sh
uv run python src/RL-translational-dynamics/exp0/summarize_experiment_0.py

# Experiment 2 full grid (Modal)
./scripts/run_experiment_2.sh
./scripts/fetch_experiment_2_results.sh
uv run python src/RL-translational-dynamics/exp2/summarize_experiment_2.py --skip-checkpoint-gate

# Unit tests
uv run pytest tests/
```

Environment: Python ≥3.10, PyTorch, Gymnasium+MuJoCo, W&B, Modal. Dependencies managed via `uv` (`pyproject.toml`).

---

## 10. Team Contributions (from proposal)

| Member | Focus area |
|--------|------------|
| Ryan | Literature scoping, baselines, fixed-ordering experiments |
| Ethan | Adaptive switching, benchmark expansion |
| Abhinav | Compute-constrained scheduling, budget strategy |

---

## Appendix: Key Figures

| Figure | Path |
|--------|------|
| Exp 0 learning curves | `results/processed/experiment_0/sac_vs_ppo_learning_curves.png` |
| Exp 0 final returns | `results/processed/experiment_0/sac_vs_ppo_final_returns.png` |
| Exp 2 learning curves | `results/processed/experiment_2/handoff_vs_baselines_learning_curves.png` *(after summarize)* |
| Exp 2 final returns | `results/processed/experiment_2/handoff_vs_baselines_final_returns.png` *(after summarize)* |
| Exp 2 AUC ranking | `results/processed/experiment_2/handoff_auc_ranking.png` *(after summarize)* |

---

## Appendix: Definition of Success

A convincing result for this project (from `FIRST_EXPERIMENTS.md`):

> One SAC → PPO schedule that beats both pure baselines on at least one environment at equal budget, while maintaining similar or lower variance across seeds.

The 100k pilot has **not** achieved this bar. The full 1M-step Experiment 2 is the next decisive test.
