# Design — What Transfers In Algorithm Sequencing

**Date:** 2026-05-26 (revised after Ryan + Ethan critiques)
**Status:** Draft v2 (pending user review)
**Project:** RL Translational Dynamics (CS 224R)

## Context

The CS 224R project has benchmarked pure `SAC`, pure `PPO`, and fixed-fraction `SAC -> PPO` / `PPO -> SAC` handoffs at 100k env steps on Hopper-v4, Walker2d-v4, HalfCheetah-v4, Ant-v4. The milestone-level takeaway is descriptive: at 100k steps `SAC` dominates `PPO` on 3/4 envs, and any schedule that allocates more budget to `SAC` does well. That is not a mechanism — it is the relative quality of the two algorithms at this horizon plus a budget tautology.

The remaining time budget is ~2 days of execution, with parallel subagent implementation, 2x RTX 3090s, and ~$500 of Modal credit.

**Contribution posture (decided 2026-05-26):** the deliverable is *understanding*, not a leaderboard. We are not trying to show "our schedule beats the baselines." We are trying to *explain why* sequencing helps when it helps and fails when it fails. This posture narrows scope: every variant we add must earn its place by sharpening the explanation, because each variant needs its own diagnostic suite, and diagnostics are where the time goes. Depth wins ties over breadth.

## Headline Claim (Narrative Spine)

**Framing wrapper (accessible):** RL algorithms can be viewed as *training phases* with complementary strengths — exploration, stabilization, refinement. Sequencing chains them.

**Primary claim (the thesis — mechanistic):**

> What carries across a phase boundary — the policy, the value function, or the data distribution — determines whether sequencing helps, by how much, and in which direction. We decompose the boundary into transferable substrates and show the *same* decomposition explains online↔online (SAC↔PPO) and offline→online (BC / IQL|AWAC → SAC / PPO) sequencing.

Performance curves are *evidence for the mechanism*, not the headline. This framing is deliberately robust to the likely result that pure SAC often wins on final return: "SAC wins" becomes a data point the mechanism explains, not a refutation of the thesis.

**Why mechanism-first over a performance headline (recorded rationale):** (1) the existing data won't support "sequencing wins" — SAC beats most schedules at 100k; a performance headline is fragile against our own results. (2) An explanation-first result lives on diagnostic quality, which we control regardless of the leaderboard.

## Ground Truth — What the Handoff Code Actually Does

*(This section exists because the v1 spec mis-described the pipeline. Verified against `train_handoff.py` / `train_reverse_handoff.py` on 2026-05-26.)*

**SAC→PPO (`train_handoff.py`):**
1. Train SAC to `switch_step`.
2. PPO actor is **fresh random init** — never weight-copied (architectures differ; see below).
3. **500 steps of behavioral distillation**: MSE between PPO action-mean and SAC *deterministic* actions on replay states (`distill_steps=500`, `distill_lr=1e-3`).
4. **Value warm-up = 2 PPO rollout-iterations, critic-only** (`value_warmup_updates=2`). Critically, V is fit to **PPO's own fresh GAE returns**, *not* to SAC's Q.
5. Full PPO with a fresh optimizer (optimizer reset logged).

**PPO→SAC (`train_reverse_handoff.py`):**
1. Train PPO to `switch_step`, adding every transition to a replay buffer as a side effect.
2. SAC actor fresh random init → **500-step distillation** toward PPO deterministic actions.
3. **1000 Bellman critic-warm-up steps** on the replay buffer (`sac_critic_warmup_updates=1000`).
4. Full SAC.

**Architectures (incompatible — this is why the code distills):**
- SAC actor: 256×256 ReLU, **state-dependent** `fc_mean` + `fc_logstd` heads.
- PPO actor: 64×64 Tanh, **state-independent** `actor_logstd` `nn.Parameter`.
- There is no shared `MLPGaussian` class. Direct weight loading between actors is impossible.

**Consequence for the design:** "policy transfer" in this codebase *is behavioral distillation*, and the existing value warm-up is *self-supervised from the target's own returns*. The v1 C1 ("policy-only vs policy+value with value aligned to the source") did not map onto the code. v2 reframes C1 as a factored ablation that holds distillation fixed and varies the value substrate — testable on the existing pipeline.

## Transfer-Substrate Vocabulary (name every arm by its operation)

Each experimental arm is a point in a factor space. No arm is labeled with an ambiguous term like "policy+value."

- **Policy init:** `random` | `distill` (500-step action-MSE toward a source policy) | `weight-load` (only when architectures match).
- **Value init:** `random` | `self-warmup` (target fits its own returns: 2 PPO critic iters, or 1000 SAC Bellman steps — the *current* protocol) | `source-aligned` (target value regressed toward a transformed source value; precise target defined per direction below).
- **Data init:** `none` | `replay-prefill` (target SAC replay seeded with source rollouts; already implicit in PPO→SAC).
- **Policy source / value source:** `{none, sac, ppo, bc, iql/awac}` checkpoints.

**`source-aligned` value targets (precisely specified):**
- SAC→PPO: `V(s) ← E_{a∼π_distilled}[min(Q1,Q2)(s,a)]` regressed over replay states (well-defined; π is the just-distilled PPO actor).
- PPO→SAC: `Q(s,a) ← reward-to-go R_t` from stored PPO trajectories (Monte-Carlo return, *not* the misspecified `V(s)+r`). Flagged open: may instead seed from a short Bellman regression; resolve at implementation.

## Falsifiable Sub-Claims (v2)

- **C1 — given fixed policy distillation, the inherited value substrate changes outcomes (necessary/sufficient test).** On SAC→PPO, hold `policy=distill` fixed and vary `value ∈ {random, self-warmup, source-aligned}`. If all three are statistically indistinguishable (overlapping bootstrap CIs on the primary metric), the value substrate is *not* the lever — C1 rejected, and we say so. Decoupled from C3.
- **C2 — BC-only policy transfer interacts *differently* with on-policy vs off-policy learners (interaction hypothesis, direction empirical).** BC carries a policy but no value. We do *not* pre-commit to "BC→PPO fails." We test how a value-less policy prior is retained or overwritten under PPO (advantage-driven) vs SAC (Q-bootstrapped), and let the diagnostics explain whichever direction appears.
- **C3 — a value-carrying offline source (IQL or AWAC) shifts the offline→online outcome relative to BC, and the shift is attributable to the transferred value.** Tests whether the C1 value lever generalizes to the offline→online setting. Gated at Tier 2 (not on C1's significance).

**Negative-result diagnostics (per-claim, not one generic clause).** A null is interpreted, not shrugged off:
- BC eval low *before* online fine-tuning → warm-start itself failed (not a transfer-mechanism result).
- value warm-up loss drops but return doesn't move → inherited value is not behaviorally useful.
- policy-KL-from-source spikes immediately post-handoff → target overwrites the transferred policy (explains why substrate "doesn't matter").
- seed variance dominates the CI → underpowered, *not* evidence of no effect.
- adaptive trigger picks extreme switch points → fixed fractions were miscalibrated.

## Pre-Registered Metrics

Declared before seeing results; this order decides the headline when metrics disagree:
1. **AUC** of eval return over the matched *online* budget (primary — sample efficiency).
2. Final eval return.
3. Worst-seed return / collapse frequency (robustness).
4. Seed standard error.
5. Average rank across environments.

Diagnostics *explain* outcomes; they never select the headline post hoc. Effect-size rule: a substrate "matters" only if its primary-metric bootstrap CI separates from the comparison arm's.

## Budget Categories (never mix)

- **Online-only, env-step matched:** pure SAC, pure PPO, SAC→PPO arms, PPO→SAC arms. Directly comparable on env steps + gradient updates.
- **Offline-assisted:** BC→*, IQL/AWAC→*, BC→SAC→PPO, interleaved-BC. Report offline dataset size + offline pretraining updates *separately*. No unqualified "more sample efficient than pure online" claims.

## Experimental Tiers (depth-protected; replaces the v1 flat matrix)

Switch fraction fixed at **50%** inside the factored ablation to isolate substrate. Hopper-v4 + Walker2d-v4. ≥5 seeds for spine arms (high SAC variance demands it); bootstrap CIs reported.

### Tier 0 — explanatory spine (GUARANTEED)
SAC→PPO, `policy=distill` fixed, **value ∈ {random, self-warmup (current), source-aligned}**. Reference arms: pure PPO (= random policy + random value) and pure SAC. Throughline question: *does the value PPO inherits explain the handoff outcome?* Full diagnostic suite (below) on every arm.

### Tier 1 — generalization of the decomposition (GUARANTEED)
- **BC→PPO** and **BC→SAC** (policy=distill from offline expert, value per target's native bootstrap) — the C2 interaction test. Reuses `demos.py`.
- **PPO→SAC** factored value ablation (`value ∈ {random, self-warmup=1000 Bellman, source-aligned}`) — shows the decomposition is not PPO-target-specific.

### Tier 2 — value-carrying offline source (STRETCH; gate on Tier 0/1 showing a value effect)
IQL *or* AWAC (one, chosen at the gate) → PPO and → SAC. The offline source carries both policy and value; isolates whether value transfer explains any BC gap (C3).

### Tier 3 — novel probe (STRETCH; high understanding-value)
**Interleaved BC during SAC**: periodic short BC re-anchoring every K steps throughout SAC training, not just at init. Distinguishes a one-time *initialization* effect from an ongoing *distributional-anchoring* effect — a genuine "why" question. Diagnostic-rich, small compute.

### Tier 4 — timing / robustness (STRETCH; cheap)
- 25/50/75 switch-fraction sweep on the best schedule (reuses `switch_fraction`; partly reuses existing Exp 2/3 data so the scheduler claim isn't pinned to 50%).
- Minimal adaptive trigger: switch after `N` non-improving eval checkpoints past a minimum first-phase budget; log chosen switch step + reason. Framed as a "timing isn't magic" baseline, not a headline method.

### Long-horizon check
One long-horizon (≥500k, up to 1M on Hopper) comparison of the Tier-0 arms, to test whether the inherited-value effect *persists or washes out* — a mechanism question, not a "does PPO catch up" benchmark. Reuse existing baseline runs where available rather than re-running pure SAC/PPO.

## Diagnostic Suite (this is where the contribution lives)

Logged on every spine/generalization arm, per RULES.md high-fidelity logging:
- **Policy retention:** action-MSE and approx-KL between current target policy and the transferred source policy, over post-handoff steps.
- **Value quality:** explained variance (PPO), Q-scale / overestimation gap (SAC), `value_loss_at_handoff` and its trajectory.
- **Advantage health (PPO target):** advantage mean/std, clip fraction, approx-KL.
- **Exploration (SAC target):** policy entropy, action std, α.
- **Handoff transient:** eval return immediately pre- vs post-handoff (does the switch cause a dip, and does it recover?).
- **Phase markers:** `sac | distill | value_warmup | ppo | sac_critic_warmup | bc_anchor | adaptive_switch`.

## Implementation Scope

### Already done (by teammates — do not rebuild)
- **`demos.py`** — D4RL expert loader (`just-d4rl` preferred, legacy `d4rl`+`gym` fallback) with the exact obs/action shape-check the v1 spec asked for; caches `.npz`; gates Ant behind `--include-ant`. `just-d4rl>=0.2407.5` is in `pyproject.toml`.

### MUST (Tier 0–1)
1. **`train_bc.py`** — BC producing a standalone Gaussian policy from cached D4RL expert data. Transfer into a target uses the *existing 500-step distillation loop* (so `policy=distill` is uniform across online and offline arms, and the actor-architecture mismatch is sidestepped). ~150 LOC.
2. **Factored transfer flags** on `train_handoff.py` / `train_reverse_handoff.py`: `--policy-init`, `--value-init`, `--replay-init`, `--policy-source`, `--value-source`. Each matrix arm = an explicit flag combination. Optimizer reset preserved per RULES.md.
3. **`source-aligned` value warm-up** implementing the precise targets above; logged `warmup_loss`. Falls back to `self-warmup` if it doesn't converge (documented, not hidden).
4. **Diagnostic logging** — the suite above as wandb scalars + the expanded phase markers.
5. **Modal wrappers** — `experiment_4_modal.py` (Tier 0–1). Modal code stays out of core RL files (RULES.md Rule 1).
6. **Analysis** — `summarize_experiment_4.py`: per-arm AUC table w/ bootstrap CIs, learning curves with phase markers, policy-retention plot, value-quality plot, handoff-transient plot.

### STRETCH (Tier 2–4; build at the gate)
7. **`train_iql.py` or `train_awac.py`** — one, chosen at the Tier-2 gate by whichever drops in cleaner. Offline on D4RL. Saves policy + value for transfer.
8. **Interleaved-BC mode** — `--bc-anchor-interval K` in the SAC loop; short distillation toward the expert policy every K steps; log anchor events.
9. **Adaptive trigger** — `--switch-trigger no-improve --patience N --min-first-phase B`.
10. **Timing sweep** — reuse `switch_fraction`; aggregate with existing Exp 2/3 data.

### Smoke-test gate before any sweep
Each new script at 1 seed × 50k steps must: (a) run without NaN/crash, (b) beat the random-policy floor, (c) log all required diagnostics + correct arm metadata.

**Dataset mapping note.** Default BC expert datasets:
- `Hopper-v4` → `hopper-expert-v2`
- `Walker2d-v4` → `walker2d-expert-v2`
- `HalfCheetah-v4` → `halfcheetah-expert-v2`
- `Ant-v4` → `ant-expert-v2`

Ant stays gated behind strict obs/action compatibility checks; version differences can change observation features across Gym/Gymnasium/D4RL stacks.

## Execution Plan

### Day 1 — implement + run Tier 0–1
- **Morning.** Parallel subagents: (A) `train_bc.py`; (B) factored transfer flags + `source-aligned` warm-up; (C) diagnostic logging + Modal wrapper + analysis skeleton.
- **Midday gate.** Smoke-test all components; block sweep until green.
- **Afternoon.** Launch Tier 0 (SAC→PPO value ablation) + Tier 1 (BC→PPO, BC→SAC, PPO→SAC value ablation), ≥5 seeds × 2 envs, across Modal + 3090s.
- **Overnight.** Tier 0–1 completes.

### Day 2 — gate, stretch, analysis
- **Morning.** Review Tier 0–1 with full diagnostics. **Tier-2 gate:** if a value effect is present (separated CIs), build IQL *or* AWAC. In parallel: start Tier-0 figures + draft the mechanism narrative.
- **Afternoon.** Tier 2 runs if green-lit; Tier 3 interleaved-BC; Tier 4 timing + adaptive (cheap). Long-horizon Tier-0 check.
- **Evening.** Finalize diagnostic figures, AUC tables w/ CIs; draft `RESULTS.md` organized by claim + diagnostic interpretation.

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Existing distillation confounds the substrate ablation | High (now understood) | Hold `policy=distill` fixed across value arms; name arms by operation; include cold reference (pure PPO) |
| `source-aligned` value target misspecified (esp. PPO→SAC Q) | Medium | Use reward-to-go target; cap warm-up; fall back to `self-warmup` and document |
| High seed variance (Hopper σ≈±795) swamps effects | Medium-High | ≥5 seeds on spine; bootstrap CIs; pre-declared effect-size rule; treat overlap as underpowered, not null |
| Over-scoping (Tiers 2–4) eats the depth | High | Tier 0–1 guaranteed; 2–4 strictly gated and ordered; IQL XOR AWAC |
| BC distillation collapses target policy | Medium | Log policy-retention; reuse validated distill loop; tune distill steps if needed |
| Modal credit burn | Medium | Overflow to 3090s; drop stretch tiers before touching spine seeds |

## Deliverables (end of Day 2)
- All raw runs in W&B `herschethan-stanford-university/rl-translational-dynamics`.
- Diagnostic figures: per-arm AUC w/ CIs, phase-marked learning curves, policy-retention, value-quality, handoff-transient, long-horizon Tier-0 check.
- `experiments/experiment_4.md` with full per-arm configs + per-env takeaways.
- `RESULTS.md` organized by C1/C2/(C3) with explicit diagnostic interpretation of each outcome (including nulls).

## Out-of-Scope (Future Work)
- Both IQL and AWAC (pick one).
- Cross-environment transfer.
- Discrete-action variants / DQN on gridworld.
- GRPO-family sequencing (LLM-oriented).
- Model-based → model-free (e.g., MBPO → SAC).
- A *learned* adaptive scheduler beyond the minimal no-improve trigger.

## Open Questions (resolve during execution)
1. `source-aligned` PPO→SAC Q target: reward-to-go vs short Bellman seed. Resolve at implementation; log both warm-up losses if cheap.
2. Interleaved-BC interval K and anchor strength. Start with a coarse sweep on Hopper.
3. Most-diagnostic spine pair confirmation: spine is SAC→PPO (PPO is the value-hungry target, unifying with C2/C3). Revisit only if Tier-0 diagnostics are uninformative.
4. Long-horizon seed count under Modal burn rate.

## Critique Integration Log (2026-05-26)
- **Ryan (verified in code):** C1/distillation confound → reframed C1 as factored value ablation + added "Ground Truth" section + operation-named arms. Architecture mismatch → BC via distillation, dropped invented `MLPGaussian`. Missing value-only arm → `value=random` arm added. Value recipe mismatch → precise `source-aligned` targets specified. C3 gate logic → decoupled from C1. Stat power → ≥5 seeds + bootstrap CIs + effect-size rule. D4RL dep → confirmed already resolved (`just-d4rl` + `demos.py`).
- **Ethan:** thesis over-centered on transfer → mechanism kept as thesis (per project goal) with phase-vocabulary wrapper; performance demoted. 50/50 too narrow → Tier-4 timing sweep. C2 determinism → reframed as interaction hypothesis. Offline budget language → separate budget categories. Three-phase scheduler → BC→SAC→PPO retained (Tier 1/3 family). Adaptive switching → minimal trigger (Tier 4). Metric pre-registration → added. Negative-result framing → per-claim diagnostics added.
