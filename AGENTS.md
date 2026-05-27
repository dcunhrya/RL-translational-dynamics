## Project: Algorithm Sequencing in Deep Reinforcement Learning

This repo studies whether deep RL algorithms should be treated as **training phases** rather than fixed end-to-end choices. The central question is:

> Under a fixed environment-step or compute budget, can sequencing RL algorithms improve sample efficiency, stability, or final return compared with using a single algorithm throughout?

The project currently focuses on MuJoCo continuous-control environments and algorithm handoffs among PPO, SAC, and offline/imitation-style warm starts such as BC, IQL, or AWAC.

---

## Core Narrative

Do **not** frame the project as:

> "SAC is better than PPO."

That is too shallow and misses the research contribution.

Instead, frame the project as:

> Different RL algorithms have complementary strengths at different stages of training. Algorithm sequencing asks whether we can exploit those phase-specific strengths under a matched budget.

The main hypothesis is:

> Sequencing helps when the first phase produces a useful initialization, state distribution, or representation for the second phase, while leaving enough budget for the second phase to improve the policy.

A strong final takeaway does **not** need to show that the proposed method beats SAC everywhere. A defensible result is:

> Algorithm sequencing is not a universal replacement for strong single algorithms like SAC, but phase-based schedules can improve early learning, robustness, stability, or average performance across environments.

---

## Proposed Method: Three-Phase Scheduler

The repo should organize experiments around a three-phase view of RL training.

### Phase 1: Warm Start

Purpose: get a non-random or partially competent policy quickly.

Candidate methods:

- Behavior Cloning (BC) from expert demonstrations
- IQL from offline data
- AWAC from offline data
- A2C/PPO warmup if offline demonstrations are unavailable

Expected benefit:

- Better initial policy quality
- Faster early reward improvement
- Better state distribution for later online learning

### Phase 2: Exploration / Online Improvement

Purpose: aggressively improve the policy through online interaction.

Primary method:

- Soft Actor-Critic (SAC)

Why SAC:

- Strong continuous-control baseline
- Off-policy and sample-efficient
- Entropy-regularized exploration
- Often high final return in MuJoCo

Expected benefit:

- Fast convergence
- Strong final returns
- Effective use of post-warm-start budget

### Phase 3: Stabilization / Refinement

Purpose: conservatively refine a learned policy and reduce destructive updates.

Primary method:

- PPO

Why PPO:

- Stable policy updates
- Conservative trust-region-like behavior
- Potentially useful after SAC or BC has already produced a competent policy

Expected benefit:

- Lower variance
- Smoother final learning
- Less policy collapse late in training

---

## Current Baseline Story

Initial experiments have evaluated:

- PPO
- SAC
- SAC → PPO
- PPO → SAC

across MuJoCo environments including Hopper-v4, Walker2d-v4, HalfCheetah-v4, and Ant-v4.

Observed so far:

- SAC is often the strongest standalone baseline.
- Handoffs often outperform pure PPO.
- Pure SAC can still beat handoff methods on some environments.
- PPO → SAC can be stronger than expected.
- Switch timing and direction matter substantially.

Interpretation:

> The value of sequencing depends on ordering, switch timing, transfer mechanism, and environment. Handoffs are not automatically better than the best single algorithm, but they reveal meaningful phase effects.

---

## Experimental Priorities

Prioritize a narrow, coherent experiment set over trying every possible algorithm.

### Tier 1: Must Run

These are the core experiments for the final report.

1. **Single-algorithm baselines**

   Run each baseline under the same environment-step budget:
   - PPO
   - SAC
   - BC only, if expert data is available
   - IQL/AWAC only, if offline implementations are available

2. **Online handoff baselines**

   Evaluate fixed switch fractions:
   - PPO → SAC
   - SAC → PPO

   Suggested switch fractions:
   - 25%
   - 50%
   - 75%

3. **Offline/imitation warm-start schedules**

   Main proposed schedules:
   - BC → SAC
   - BC → PPO
   - BC → SAC → PPO

   Optional, if IQL/AWAC is available:
   - IQL → SAC
   - AWAC → SAC
   - IQL → SAC → PPO
   - AWAC → SAC → PPO

4. **Adaptive switching**

   Compare fixed schedules against a simple adaptive trigger.

   Candidate triggers:
   - Return plateau over last K evaluations
   - Entropy drops below threshold
   - Critic loss stabilizes
   - KL/update magnitude stabilizes
   - No improvement in evaluation return for N checkpoints

   The adaptive scheduler should be simple, explicit, and reproducible.

### Tier 2: Optional

Only run these if the core pipeline is stable.

- Cross-environment transfer
- Warm-starting from easier or clipped environment versions
- SAC warmup → BC → SAC diagnostic
- Model-based warmup → SAC
- Additional offline RL baselines

### Tier 3: Deprioritize

Avoid unless there is a very specific reason.

- GRPO on MuJoCo
- DQN on continuous-action MuJoCo
- Large model-based RL implementations from scratch

Reason:

> These broaden the project but weaken the narrative and increase implementation risk.

---

## Evaluation Metrics

Every experiment should report more than final return.

### Primary Metrics

1. **Final return**

   Measures end-of-budget performance.

2. **AUC / reward integral**

   Measures sample efficiency over the full training curve.

3. **Stability**

   Suggested measurements:
   - Standard deviation across seeds
   - Standard error bands
   - Frequency of collapse
   - Worst-seed performance
   - Average rank across seeds/environments

4. **Compute-normalized performance**

   If feasible, include:
   - Return vs environment steps
   - Return vs gradient updates
   - Return vs wall-clock proxy

### Recommended Claim Structure

Prefer claims like:

- "BC → SAC improves early learning."
- "PPO → SAC often improves over PPO by giving SAC most of the post-switch budget."
- "SAC remains a strong standalone baseline, especially for final return."
- "Adaptive switching reduces sensitivity to fixed handoff timing."
- "The proposed scheduler improves average rank or AUC across environments."

Avoid claims like:

- "Our method universally beats SAC."
- "Switching is always better than single-algorithm training."
- "PPO stabilizes SAC in all environments."

---

## Plotting Requirements

Final plots should emphasize convergence and stability.

For each environment, produce learning curves with:

- x-axis: environment steps
- y-axis: evaluation return
- line: mean over seeds
- shaded region: standard error or standard deviation

Recommended environments:

- Hopper-v4
- Walker2d-v4
- HalfCheetah-v4
- Ant-v4

Recommended plotted methods:

- PPO
- SAC
- PPO → SAC
- SAC → PPO
- BC → SAC
- BC → PPO
- BC → SAC → PPO
- Adaptive scheduler

If there are too many curves, split into two plots:

1. Baselines and online handoffs
2. Warm-start and adaptive methods

Also include a summary table with:

- Final return
- AUC
- Stability metric
- Average rank

---

## Implementation Guidelines for Coding Agents

When modifying this repo, preserve experimental rigor and reproducibility.

### General Rules

- Do not silently change budgets, seeds, or environment settings.
- Keep all comparisons budget-matched.
- Log all hyperparameters.
- Save all configs used for runs.
- Use deterministic seeds where possible.
- Never overwrite old experiment outputs unless explicitly requested.
- Prefer small pilot runs before launching full sweeps.
- Keep method names consistent across scripts, logs, and plots.

### Experiment Naming

Use names that encode:

- environment
- seed
- algorithm sequence
- switch rule
- budget
- timestamp or run id

Example:

```text
hopper_v4_seed3_bc_sac_ppo_fixed_25_50_25_100k
walker2d_v4_seed1_ppo_sac_fixed_25_75_100k
ant_v4_seed5_adaptive_plateau_bc_sac_ppo_100k
```
