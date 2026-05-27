# Design — What Transfers In Algorithm Sequencing

**Date:** 2026-05-26
**Status:** Draft (pending user review)
**Project:** RL Translational Dynamics (CS 224R)

## Context

The CS 224R project has so far benchmarked pure `SAC`, pure `PPO`, and fixed-fraction `SAC -> PPO` / `PPO -> SAC` handoffs at 100k env steps on Hopper-v4, Walker2d-v4, HalfCheetah-v4, and Ant-v4. The milestone-level takeaway is descriptive: at 100k steps `SAC` dominates `PPO` on 3/4 envs, and any schedule that allocates more budget to `SAC` does well. That is not a mechanism — it is the relative quality of the two algorithms at this horizon plus a budget tautology.

The remaining time budget is ~2 days of execution, with parallel subagent implementation, 2x RTX 3090s, and ~$500 of Modal credit.

## Headline Claim (Narrative Spine)

> In algorithm sequencing, the *transferable substrate* — policy weights, value function, replay buffer — determines the magnitude and direction of benefit. The same decomposition explains both online-online (SAC <-> PPO) and offline-online (BC / IQL -> SAC / PPO) sequencing.

This is a "mechanism + generalization" claim: a mechanistic explanation of when and why sequencing helps, validated across multiple algorithm families.

## Falsifiable Sub-Claims

- **C1 — policy is necessary but not sufficient.** For `SAC <-> PPO`, transferring the policy alone (with random-init target value) underperforms transferring policy + value. If `policy` and `policy + value` ablations are statistically indistinguishable, C1 is rejected.
- **C2 — BC -> PPO underperforms BC -> SAC, because BC carries no value.** PPO depends on advantage estimates from a warm value; BC provides only a policy prior. Therefore `BC -> PPO` should rapidly drift away from the BC policy as GAE drives updates, while `BC -> SAC` benefits from the policy prior even with a random Q init (SAC bootstraps Q from scratch anyway).
- **C3 — IQL closes the value-transfer gap.** IQL produces an offline policy and value function. `IQL -> PPO` should outperform `BC -> PPO` at matched online compute, and the gap should be attributable to the transferred value function (cf. AWAC / IQL prior work).

**Pre-committed falsification clause.** If all transfer variants are statistically indistinguishable at the experimental horizons, we report a clean negative result: at fixed budgets in this regime, transfer substrate does not dominate seed variance, and we characterize *why* (e.g., variance washes the signal, or warm-ups are reabsorbed within the first 10% of online updates).

## Experimental Matrix

10 variants. Each at 500k env steps on Hopper-v4 and Walker2d-v4, 5 seeds. Switch fraction held at **50%** for all online-online variants to isolate transfer substrate as the only variable. Total: **100 runs** in the main matrix.

| #  | Source | Target | What's transferred | Tests |
|----|--------|--------|--------------------|-------|
| 1  | —      | SAC    | —                                                 | baseline |
| 2  | —      | PPO    | —                                                 | baseline |
| 3  | SAC    | PPO    | policy only (V_phi random init)                   | **C1**   |
| 4  | SAC    | PPO    | policy + value (V_phi warm-up from Q_SAC)         | **C1**   |
| 5  | PPO    | SAC    | policy only (Q_theta random init)                 | **C1**   |
| 6  | PPO    | SAC    | policy + value (Q_theta warm-up from V_PPO)       | **C1**   |
| 7  | BC     | SAC    | policy only (no value available)                  | **C2**   |
| 8  | BC     | PPO    | policy only (no value available)                  | **C2**   |
| 9  | IQL    | SAC    | policy + value (Q_theta <- Q_IQL)                 | **C3**   |
| 10 | IQL    | PPO    | policy + value (V_phi <- V_IQL)                   | **C3**   |

### Headline 1M-step comparison (Hopper-v4)

5 representative variants × 5 seeds = **25 long runs**. The variants are: pure SAC, pure PPO, best SAC <-> PPO from C1, BC -> SAC, IQL -> SAC (or BC -> SAC if C3 is dropped). This is the "does PPO catch up asymptotically?" headline answer.

### Compute accounting

Online compute is reported in env steps and gradient updates. BC and IQL pretraining is treated as offline pre-processing on the D4RL `*-expert-v2` datasets (standard treatment in AWAC / IQL literature). The report will be transparent about this convention.

### Switch direction and target representations

Value transfer between SAC (Q-function) and PPO (V-function) is non-trivial because their value representations differ. Cleanest approach: at the handoff, run a short supervised warm-up phase (~5k gradient updates) that aligns the target value function to the source's. For SAC -> PPO, train V_phi on `E_{a ~ pi}[Q_SAC(s, a)]` over states drawn from SAC's replay. For PPO -> SAC, train Q_theta on `V_PPO(s) + r(s, a)` over PPO's rollout buffer. The warm-up phase is logged explicitly (`ppo_value_warmup` per RULES.md) and counted toward the online gradient-update budget.

## Implementation Scope

### MUST (required for C1 + C2)

1. **`train_bc.py`** — single-file BC on D4RL `hopper-expert-v2` and `walker2d-expert-v2`. Gaussian policy matching the existing `MLPGaussian` architecture so checkpoints load directly into SAC / PPO actors. ~150 LOC.
2. **`demos.py`** — D4RL loader with runtime sanity check that observation / action dims match Gymnasium v4 envs (they do for Hopper and Walker2d). Cached locally to avoid re-downloads per run.
3. **Transfer-ablation harness** — `--transfer policy | value | policy+value | policy+value+replay` flag added to `train_handoff.py` and `train_reverse_handoff.py`. Selective checkpoint loading + optimizer reset (per RULES.md, no stale Adam moments across the handoff).
4. **Value-warm-up phase** — short supervised phase aligning target's value head to source's, as described above. Capped at 5k updates. Logged warmup loss curve.
5. **Logging extensions** — additional wandb scalars: `transfer_components`, `warmup_loss`, `value_loss_at_handoff`, `policy_kl_at_handoff`. Phase markers unchanged: `sac | handoff | ppo_value_warmup | ppo`.
6. **Modal wrappers** — `experiment_4_modal.py` for the 100-run matrix; `experiment_5_modal.py` for the 25-run 1M-step headline. All Modal code stays out of core RL logic files per RULES.md Rule 1.
7. **Analysis scripts** — `summarize_experiment_4.py` and `summarize_experiment_5.py` producing: (a) transfer-component bar chart per env, (b) learning curves with handoff + warm-up markers, (c) AUC table, (d) value-loss-at-handoff plot.

### CONDITIONAL (gated on day-2 morning review; required for C3)

8. **`train_iql.py`** — offline IQL on D4RL. Expectile V regression + advantage-weighted policy update. ~300 LOC. Reference: official IQL repo or CleanRL IQL.
9. **IQL transfer wiring** — extend the harness to accept IQL checkpoints (compatible serialization for policy, V, and Q).

### Smoke-test gate before main sweep

Each new script must pass three checks at 1 seed × 50k steps before any 5-seed sweep is launched:
1. trains without NaN / crash,
2. produces a learning curve above the random-policy floor,
3. logs all required wandb fields and the correct `transfer_components` metadata.

## Execution Plan

### Day 1 — implementation + main sweep

- **Morning (~4 hr).** Spawn 3 parallel subagents:
  - Agent 1: `train_bc.py` + `demos.py`.
  - Agent 2: transfer-ablation harness + value-warm-up phase.
  - Agent 3: Modal wrappers + analysis-script skeletons.
- **Midday gate.** Run smoke tests for all three components. Block sweep launch until all gates green.
- **Afternoon (~4 hr).** Launch variants 1–8 (80 runs at 500k env steps × 5 seeds × 2 envs). Run in parallel across Modal + 2x 3090s.
- **Overnight.** Main matrix completes.

### Day 2 — IQL gate + headline + analysis

- **Morning (~4 hr).** Review Day-1 results. Make C3 go / no-go decision:
  - If C1 evidence is clean (policy vs policy+value ablation produces a statistically meaningful gap): green-light IQL. Spawn Agent 4 to implement `train_iql.py`.
  - If C1 is null or noisy: skip IQL; deepen analysis instead.
- **Afternoon (~4 hr).** If green-lit, smoke-test IQL and launch variants 9–10 (20 runs at 500k). Launch the 25-run 1M-step headline on Hopper.
- **Evening.** Finalize plots, AUC tables, value-loss-at-handoff figures. Draft `RESULTS.md` headline + mechanism section.

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| D4RL v2 ↔ Gym v4 obs / action mismatch | Low | Runtime sanity check at load; Hopper / Walker2d stable across versions |
| BC checkpoint arch mismatch with SAC / PPO actors | Medium | Share `MLPGaussian` from existing code; BC trained as Gaussian policy |
| Value warm-up doesn't converge | Medium | Cap at 5k updates; log warmup loss; fall back to random-init value and document |
| Modal credit burn faster than expected | Medium | Overflow to 3090s; can drop to 3 seeds for non-headline variants |
| IQL implementation buggy | Medium-High | Use reference impl; if not green by Day-2 afternoon, drop C3 and report C1 + C2 |
| All ablations statistically indistinguishable | Low-Medium | Pre-committed falsification clause; report as clean negative + characterize |
| One env shows a result, the other doesn't | Medium | Already a finding; report per-env breakdown; investigate via internal metrics |

## Deliverables (end of Day 2)

- All raw runs in W&B project `herschethan-stanford-university/rl-translational-dynamics`.
- Processed plots:
  - transfer-component bar chart per env
  - learning curves with phase + warm-up markers per env
  - AUC table across variants
  - value-loss-at-handoff comparison
  - 1M-step headline figure
- `experiments/experiment_4.md` and `experiments/experiment_5.md` with full configs + per-env takeaways.
- `RESULTS.md` writing up C1 / C2 / (C3 if available) findings with concrete numbers.

## Out-of-Scope (Explicitly Future Work)

- Cyclic / multi-switch sequences (e.g., SAC -> PPO -> SAC). Cheap to add later but not on the mechanistic critical path.
- Cross-environment transfer (policy learned on one env, fine-tuned on another).
- Discrete-action env variants (e.g., DQN on gridworld). Generalization to discrete control is a separate paper.
- GRPO-family sequencing. Mostly LLM-oriented; weak motivation for MuJoCo.
- Model-based -> model-free sequencing (e.g., MBPO -> SAC). Significant new infrastructure.
- Adaptive switching trigger (KL-drift based). Originally Experiment 3 in the proposal; defer to a follow-up because trigger validation is high-execution-risk in the remaining time budget.

## Open Questions to Resolve During Execution

1. **Value warm-up duration.** Is 5k updates the right cap? May need to be env- or algorithm-pair-specific. Resolution: log warmup loss curves; pick the elbow.
2. **BC policy entropy regularization.** Should BC train with action-noise regularization to better match SAC's stochasticity? Default: no, train deterministic-NLL on Gaussian; reassess if BC -> SAC collapses early.
3. **Replay buffer pre-fill (variant 6 + replay).** Not in the main matrix but cheap to add. Decision: add iff variant 6 results are interesting enough to justify the micro-ablation.
4. **Seed count for headline 1M runs.** 5 seeds × 5 variants = 25 runs. If compute is tight, drop to 3 seeds × 5 variants = 15. Resolution: actual Modal burn rate by end of Day-1.
