# Project Summary and Research Paper Scaffold

This document is a self-contained summary of the RL Translational Dynamics project. It is intended to be detailed enough to give to a large language model as source material for writing the full research paper. It consolidates the project motivation, research questions, method design, task assignments, completed experiments, current results, expected outcomes, deviations from those expectations, limitations, and a proposed paper structure.

## One-Sentence Project Summary

This project studies whether deep reinforcement learning algorithms should be treated as training phases rather than fixed end-to-end choices, asking whether sequencing algorithms such as PPO and SAC under matched environment-step budgets can improve sample efficiency, stability, or final return by exploiting phase-specific strengths.

## Core Research Motivation

Most empirical RL comparisons treat algorithms as complete training procedures: train PPO from scratch, train SAC from scratch, compare their learning curves, and declare a winner. That framing is too coarse for this project. PPO and SAC do not merely differ in final performance; they differ in how they use data, how stable their updates are, how they explore, and how they respond to initialization.

The central motivation is that RL training may naturally decompose into phases:

- An early phase where the agent needs a non-random policy, useful state coverage, or a stable initialization.
- A middle phase where the agent needs aggressive online improvement and sample-efficient use of experience.
- A late phase where the agent may benefit from conservative refinement and reduced destructive updates.

Under this view, asking "Is SAC better than PPO?" is the wrong question. SAC may be stronger as a standalone baseline, especially in MuJoCo continuous-control tasks, but a project about algorithm sequencing asks a more mechanistic question:

> Can the strengths of one algorithm at one stage of training become useful input to another algorithm at a later stage, under a fixed interaction or compute budget?

The working hypothesis is:

> Sequencing helps when the first phase produces a useful initialization, state distribution, or representation for the second phase, while leaving enough budget for the second phase to improve the policy.

The project is therefore not primarily a leaderboard paper. It is an explanation paper. A valid and useful result is not necessarily "our scheduler beats SAC everywhere." A more defensible result is:

> Strong single algorithms such as SAC remain difficult to beat, but phase-based schedules reveal meaningful timing, transfer, and stability effects that explain when sequencing helps and when it fails.

## High-Level Paper Thesis

The strongest thesis supported by the repository so far is:

> Deep RL algorithm sequencing should be understood as phase allocation under a matched budget. In MuJoCo control, SAC is a strong standalone baseline, but PPO/SAC handoffs reveal real phase effects: switch direction, switch timing, and transfer mechanism substantially affect sample efficiency and final return. Early PPO -> SAC handoffs and adaptive switching can perform well when SAC receives enough post-switch budget, while SAC -> PPO handoffs show that switching away from a productive SAC learner can be costly. Value initialization at the handoff is environment-dependent and does not provide a universal improvement.

This thesis deliberately avoids claiming universal dominance over SAC. Instead, it frames SAC's strength as part of the explanation: if SAC continues improving after the nominal handoff point, replacing it with PPO can reduce performance even when the PPO policy is initialized from SAC behavior.

## Paper Abstract Scaffold

Use this as the basis for the final abstract:

```text
Deep reinforcement learning algorithms are usually evaluated as fixed end-to-end training procedures, but different algorithms may be better suited to different phases of learning. We study algorithm sequencing for continuous-control reinforcement learning, asking whether policies trained under a fixed environment-step budget can benefit from switching between algorithms such as PPO and SAC. We frame sequencing as a transfer problem across phase boundaries, where the transferred substrate may include policy behavior, value estimates, replay data, or training dynamics.

We evaluate single-algorithm SAC and PPO baselines, fixed SAC -> PPO and PPO -> SAC handoffs, value-initialization ablations, switch-timing sweeps, adaptive switching, and planned offline warm-start schedules on MuJoCo environments including Hopper-v4 and Walker2d-v4. Across completed 500k-step experiments, SAC is the strongest standalone baseline in Ryan's SAC -> PPO slice, achieving the best AUC and final return on both Hopper-v4 and Walker2d-v4. SAC -> PPO schedules improve over pure PPO but fail to match continued SAC, suggesting that switching away from a still-improving SAC learner can be costly. The value-initialization ablation is mixed: source-aligned and self-warmup value transfer help on Walker2d but not Hopper, and all paired confidence intervals include zero.

In the PPO -> SAC direction, switch timing is the clearest signal. Early handoffs substantially outperform late handoffs, and a simple no-improvement adaptive trigger switches near 150k steps and performs strongly, especially on Walker2d. These results support a phase-allocation view of sequencing: the first algorithm can be useful as a warm start, but only if the second algorithm receives enough budget to exploit it. Overall, algorithm sequencing is not a universal replacement for strong single algorithms, but it exposes meaningful phase effects and provides a framework for designing adaptive training schedules.
```

## Main Research Questions

The project can be organized around five research questions.

### RQ1: Do fixed algorithm handoffs improve over single-algorithm baselines?

Compare pure SAC and pure PPO against fixed handoffs:

- SAC -> PPO
- PPO -> SAC

Metrics:

- Final return
- AUC / reward integral over training
- Stability across seeds
- Collapse frequency

Current answer:

Fixed handoffs often improve over pure PPO, but completed results do not show robust dominance over pure SAC. This is especially clear in Ryan's 500k SAC -> PPO experiments, where SAC dominates both AUC and final return.

### RQ2: Does switch direction matter?

Compare SAC -> PPO against PPO -> SAC.

Current answer:

Yes. Direction matters substantially. SAC -> PPO often benefits from SAC's early phase but loses ground after switching to PPO. PPO -> SAC is often stronger when the switch is early, because PPO can provide a non-random warm start while SAC still receives enough budget to perform the main online improvement.

### RQ3: Does switch timing matter?

Compare 25%, 50%, and 75% switch fractions.

Current answer:

Yes. Ethan's PPO -> SAC timing results show the clearest signal in the repository: early 25% handoffs are much stronger than late 75% handoffs on both Hopper-v4 and Walker2d-v4. Late handoff leaves SAC too little budget to exploit its sample efficiency.

### RQ4: What actually transfers across the phase boundary?

The design spec decomposes transfer into:

- Policy initialization: random, behavioral distillation, or weight loading where possible.
- Value initialization: random, self-warmup, or source-aligned.
- Data initialization: none or replay prefill.

Current answer:

Policy transfer through behavioral distillation is the common handoff mechanism because SAC and PPO actor architectures are incompatible. Value transfer is not a universal lever. Ryan's SAC -> PPO value-initialization ablation is mixed and statistically inconclusive; Ethan's PPO -> SAC value ablation is also environment-dependent.

### RQ5: Can adaptive switching reduce sensitivity to fixed handoff timing?

Current answer:

Preliminary Ethan results are promising. A simple no-improvement trigger switched near 150k steps in 500k-step runs and performed strongly, especially on Walker2d-v4. This supports the claim that adaptive switching can avoid badly mistimed handoffs, though it should not yet be framed as universally optimal.

## Algorithms and Phase Roles

### SAC

SAC is the strongest standalone baseline in the completed 500k-step Ryan results.

Why SAC is useful:

- Off-policy and sample-efficient.
- Uses replay data effectively.
- Entropy-regularized exploration.
- Strong continuous-control performance in MuJoCo.
- Often continues improving after the nominal handoff point.

Role in the phase story:

SAC is best interpreted as the main online improvement algorithm. In PPO -> SAC schedules, it should receive enough post-switch budget to exploit the warm start. In SAC -> PPO schedules, switching away from SAC too early can be costly if SAC would have continued improving.

### PPO

PPO is weaker than SAC as a standalone baseline in the completed 500k Ryan results, but it is still useful for the sequencing story.

Why PPO is useful:

- Stable policy-gradient updates.
- Conservative trust-region-like behavior.
- Can produce a non-random policy or state distribution before handoff.
- May act as a warm-start phase in PPO -> SAC schedules.

Role in the phase story:

PPO appears more useful as an early warm-start phase before SAC than as a late replacement for SAC in the current experiments. The idea that PPO stabilizes or refines SAC after handoff is not supported robustly by the current SAC -> PPO results.

### Offline / Imitation Warm Starts

The planned three-phase scheduler includes offline warm starts such as BC, IQL, or AWAC.

Expected role:

- BC gives a policy prior but no value function.
- IQL or AWAC can provide both policy and value information.
- Offline warm starts test whether non-random initialization improves early online learning.

Current status:

Abhinav's task file defines BC, IQL/AWAC, and interleaved-BC experiments, but no completed `experiments/abhinav_task/` result directory was found in the repo. These should be described as planned or pending, not completed.

## Transfer Mechanism Vocabulary

The design spec reframes handoffs in terms of explicit transfer substrates.

### Policy Initialization

- `random`: target policy starts normally.
- `distill`: target policy is trained by behavioral distillation to imitate the source policy.
- `weight-load`: direct weight loading, only possible when architectures match.

In this codebase, SAC and PPO actor architectures differ, so policy transfer is behavioral distillation rather than direct weight loading.

### Value Initialization

- `random`: target critic/value starts normally.
- `self-warmup`: target algorithm fits its own return or Bellman targets before full training.
- `source-aligned`: target value is regressed toward a source-derived value target.

The central C1 question is:

> With policy distillation fixed, does the inherited value substrate change the outcome?

### Data Initialization

- `none`: target starts without source data.
- `replay-prefill`: target receives source rollouts or replay data.

PPO -> SAC naturally involves replay data because SAC can train off-policy from stored transitions.

## Experimental Design

## Environments

Completed and planned work focuses primarily on:

- Hopper-v4
- Walker2d-v4

Early sanity experiments also included:

- HalfCheetah-v4
- Ant-v4

The final paper should focus on Hopper-v4 and Walker2d-v4 because they have the most complete sequencing results.

## Budgets and Seeds

The shared task files define:

- One 500k-step MuJoCo run as one compute unit.
- One 1M-step run as two compute units.
- Core arms: 5 seeds across 2 environments.
- Stretch arms: 3 seeds.
- Long-horizon checks: 3 seeds on Hopper-v4 at 1M steps.

For the completed Ryan slice:

- 53/53 planned jobs completed.
- Includes 500k SAC/PPO baselines.
- Includes 500k SAC -> PPO value-initialization ablations.
- Includes Hopper-v4 SAC 1M long-horizon check.

For the completed Ethan slice:

- PPO -> SAC value ablation.
- Timing sweep.
- Adaptive no-improvement trigger.
- Long-horizon entries may be mixed into one summary row, which requires a caveat.

## Pre-Registered Metrics

The design spec prioritizes metrics in this order:

1. AUC of evaluation return over matched online budget.
2. Final evaluation return.
3. Worst-seed return and collapse frequency.
4. Seed standard error.
5. Average rank across environments.

AUC is the primary metric because it captures sample efficiency over the full training curve rather than only end-of-budget performance.

## Task Assignments

### Ryan

Ryan owns the SAC -> PPO explanatory spine.

Scope:

- Pure SAC baseline at 500k.
- Pure PPO baseline at 500k.
- SAC -> PPO value ablation with fixed policy distillation.
- Value variants: random, self-warmup, source-aligned.
- Hopper SAC 1M long-horizon check.

Research contribution:

Ryan owns the C1 result:

> Given fixed policy distillation, does the value PPO inherits change the outcome?

Current status:

Completed. All 53/53 Ryan jobs are included in the generated report.

### Ethan

Ethan owns PPO -> SAC generalization and timing dynamics.

Scope:

- PPO -> SAC value ablation.
- Timing sweep at 25%, 50%, and 75%.
- Adaptive no-improvement trigger.
- Long-horizon PPO and one handoff arm.

Research contribution:

Ethan owns:

- Whether C1-style value decomposition generalizes to the off-policy SAC target.
- Whether timing and adaptive switching explain phase effects.

Current status:

Results exist in `experiments/ethan_task/results.md` and `experiments/ethan_task/ethan-experiment-takeaways.md`. One aggregation caveat must be addressed before quoting every exact number.

### Abhinav

Abhinav owns offline-assisted scheduling and shared infrastructure.

Scope:

- Diagnostic logging harness.
- Modal wrappers.
- Analysis scripts.
- BC pretraining.
- BC -> PPO and BC -> SAC.
- Optional IQL or AWAC.
- Interleaved BC.
- Long-horizon Tier-0 arms.

Research contribution:

Abhinav's planned work tests whether offline policy or value sources improve online sequencing.

Current status:

Task file exists and is detailed, but no completed Abhinav experiment result directory was found. The paper should describe these as planned, pending, or future work unless additional results are generated.

## Completed Results So Far

## Experiment 0: Short-Horizon Baselines

Purpose:

Validate that the training stack is stable and establish early SAC/PPO behavior.

Setup:

- Algorithms: SAC, PPO.
- Environments: Hopper-v4, Walker2d-v4, HalfCheetah-v4, Ant-v4.
- Seeds: 0 and 1.
- Budget: 100k environment steps.
- Logging: W&B.

Final returns:

| Environment | SAC | PPO | Interpretation |
|---|---:|---:|---|
| Hopper-v4 | 1204.24 +/- 795.57 | 346.23 +/- 15.37 | SAC much stronger, high variance. |
| Walker2d-v4 | 326.17 +/- 44.50 | 339.01 +/- 33.65 | Roughly tied, slight PPO edge. |
| HalfCheetah-v4 | 4830.85 +/- 1075.63 | 275.69 +/- 26.16 | SAC much stronger. |
| Ant-v4 | 1152.57 +/- 307.84 | 448.39 +/- 92.06 | SAC stronger. |

Takeaway:

The stack is stable and SAC is generally the stronger early-training baseline. This justified focusing later experiments on whether handoffs can improve over SAC or explain useful phase effects.

## Experiment 2: 100k SAC -> PPO Handoff Pilot

Purpose:

Test fixed SAC -> PPO handoff mechanics before larger sweeps.

Setup:

- Environments: Hopper-v4, Walker2d-v4.
- Seeds: 0, 1, 2.
- Budget: 100k environment steps.
- Switch fractions: 25%, 50%, 75%.

Final returns:

| Environment | SAC | PPO | 25% SAC -> PPO | 50% SAC -> PPO | 75% SAC -> PPO |
|---|---:|---:|---:|---:|---:|
| Hopper-v4 | 1204.24 +/- 795.57 | 346.23 +/- 15.37 | 362.91 +/- 16.55 | 421.44 +/- 243.20 | 735.27 +/- 251.71 |
| Walker2d-v4 | 326.17 +/- 44.50 | 339.01 +/- 33.65 | 365.50 +/- 52.86 | 513.48 +/- 86.63 | 476.30 +/- 38.86 |

Takeaway:

SAC -> PPO improved over PPO in several settings but did not robustly beat SAC. On Hopper, later handoff was better, likely because more SAC training was beneficial. On Walker2d, the 50% switch was strongest among handoff arms. The best switch point was environment-dependent.

## Experiment 3: 100k PPO -> SAC Reverse-Handoff Pilot

Purpose:

Test whether the reverse direction behaves differently.

Setup:

- Environments: Hopper-v4, Walker2d-v4.
- Seeds: 0, 1, 2.
- Budget: 100k environment steps.
- Switch fractions: 25%, 50%, 75%.

Final returns:

| Environment | PPO | SAC | 25% PPO -> SAC | 50% PPO -> SAC | 75% PPO -> SAC |
|---|---:|---:|---:|---:|---:|
| Hopper-v4 | 346.23 +/- 15.37 | 1204.24 +/- 795.57 | 754.55 +/- 374.10 | 564.02 +/- 124.98 | 358.61 +/- 31.52 |
| Walker2d-v4 | 339.01 +/- 33.65 | 326.17 +/- 44.50 | 566.35 +/- 126.00 | 528.83 +/- 280.14 | 357.77 +/- 43.38 |

Takeaway:

Direction matters. Early PPO -> SAC looked better than late PPO -> SAC, especially on Walker2d. This suggested that PPO may be useful as a warm-start phase, while SAC needs enough remaining budget to perform online improvement.

## Ryan 500k Results: SAC -> PPO Value Ablation

Ryan's result bundle is the most complete completed experiment set in the repository.

Scope:

- 500k-step SAC baseline.
- 500k-step PPO baseline.
- 500k-step SAC -> PPO handoffs at 50% switch.
- SAC -> PPO policy transfer fixed via distillation.
- Value initialization varied across random, self-warmup, and source-aligned.
- Hopper SAC 1M long-horizon check.
- 5 seeds per 500k arm.

Completion:

- 53/53 Ryan jobs completed.

### Ryan Main Results

| Environment | Method | Seeds | Final Mean | Final 95% CI | Normalized AUC Mean | Normalized AUC 95% CI | Collapse Count |
|---|---|---:|---:|---|---:|---|---:|
| Hopper-v4 | SAC | 5 | 2895.6 | [2125.4, 3332.0] | 2260.239 | [1959.151, 2561.327] | 1 |
| Hopper-v4 | PPO | 5 | 384.3 | [366.8, 401.8] | 324.941 | [295.808, 349.616] | 0 |
| Hopper-v4 | SAC -> PPO random V | 5 | 403.5 | [219.6, 528.0] | 1219.303 | [1041.024, 1387.314] | 5 |
| Hopper-v4 | SAC -> PPO self-warmup V | 5 | 439.2 | [391.1, 487.4] | 1145.765 | [889.789, 1382.994] | 5 |
| Hopper-v4 | SAC -> PPO source-aligned V | 5 | 384.9 | [370.9, 398.1] | 1131.139 | [933.887, 1328.390] | 5 |
| Walker2d-v4 | SAC | 5 | 3310.4 | [2932.4, 3688.5] | 1642.230 | [1483.623, 1776.310] | 0 |
| Walker2d-v4 | PPO | 5 | 441.5 | [408.1, 467.4] | 381.714 | [345.874, 434.004] | 1 |
| Walker2d-v4 | SAC -> PPO random V | 5 | 442.5 | [330.2, 525.6] | 602.467 | [543.601, 661.333] | 5 |
| Walker2d-v4 | SAC -> PPO self-warmup V | 5 | 464.6 | [359.5, 545.4] | 623.689 | [571.644, 683.761] | 5 |
| Walker2d-v4 | SAC -> PPO source-aligned V | 5 | 646.6 | [446.9, 899.0] | 718.320 | [579.152, 859.397] | 5 |

### Ryan Interpretation

SAC is the best method in Ryan's completed 500k slice on both environments by both AUC and final return.

On Hopper-v4:

- SAC normalized AUC: 2260.239.
- Best SAC -> PPO normalized AUC: 1219.303 with random value initialization.
- SAC final return: 2895.6.
- Best SAC -> PPO final return: 439.2 with self-warmup value initialization.

On Walker2d-v4:

- SAC normalized AUC: 1642.230.
- Best SAC -> PPO normalized AUC: 718.320 with source-aligned value initialization.
- SAC final return: 3310.4.
- Best SAC -> PPO final return: 646.6 with source-aligned value initialization.

The handoff arms improve over PPO in AUC, especially on Hopper, but they do not approach SAC. This supports a nuanced claim:

> SAC -> PPO transfers useful first-phase behavior relative to cold PPO, but the PPO phase does not preserve or improve SAC's trajectory enough to beat continued SAC.

### Ryan C1 Value-Ablation Result

The C1 test asks whether value initialization changes outcomes when policy distillation is fixed.

Paired normalized AUC deltas relative to SAC -> PPO random value initialization:

| Environment | Comparison | Mean Delta | 95% CI | Paired Seeds |
|---|---|---:|---|---:|
| Hopper-v4 | self-warmup V vs random V | -73.54 | [-165.26, 18.18] | 5 |
| Hopper-v4 | source-aligned V vs random V | -88.16 | [-188.38, 12.05] | 5 |
| Walker2d-v4 | self-warmup V vs random V | 21.22 | [-7.26, 54.91] | 5 |
| Walker2d-v4 | source-aligned V vs random V | 115.85 | [-25.31, 278.73] | 5 |

Interpretation:

- On Hopper-v4, both value-transfer variants trend worse than random value initialization.
- On Walker2d-v4, both value-transfer variants trend better than random value initialization, especially source-aligned value initialization.
- All confidence intervals include zero.

The safest claim:

> Value initialization changes SAC -> PPO behavior, but the effect is environment-dependent and not statistically decisive in this five-seed run. The results do not support a universal source-aligned value-transfer win.

What this deviates from:

The initial hope was that better value initialization would make the PPO phase preserve or exploit the SAC warm start. Instead, the value effect is mixed, and the dominant observation is that continuing SAC is much stronger than switching to PPO under the 500k budget.

## Ethan 500k Results: PPO -> SAC Timing, Value, and Adaptive Switching

Ethan's results are important because they show the clearest switch-timing signal.

Scope:

- PPO -> SAC value ablation.
- Fixed timing sweep at 25%, 50%, and 75%.
- Adaptive no-improvement trigger.
- Hopper-v4 and Walker2d-v4.

Important caveat:

The current Ethan summary has a known aggregation issue for the Hopper 50% self-warmup row. The row reports `mean_switch_step = 400000`, which is inconsistent with a clean 500k run at 50% switch because that should be 250000. This likely means 500k and 1M runs were mixed in that aggregate row. Therefore, exact numbers from that row should not be quoted as a clean 500k result until the table is regenerated with total budget included in the grouping key.

### Ethan Summary Table

| Environment | Trigger | Switch | Policy | Value | Seeds | AUC Mean [95% CI] | Final Mean [95% CI] | Mean Switch Step |
|---|---|---:|---|---|---:|---:|---:|---:|
| Hopper-v4 | fixed_fraction | 0.00 | unknown | unknown | 3 | 416.59 [404.76, 436.05] | 639.17 [499.98, 901.48] | nan |
| Hopper-v4 | fixed_fraction | 0.50 | distill | random | 5 | 640.80 [603.39, 678.39] | 1488.63 [930.90, 2270.49] | 250000.0 |
| Hopper-v4 | fixed_fraction | 0.25 | distill | self-warmup | 3 | 1411.49 [1261.01, 1698.47] | 2451.67 [999.54, 3192.30] | 125000.0 |
| Hopper-v4 | fixed_fraction | 0.50 | distill | self-warmup | 5 | 729.14 [633.81, 824.46] | 1760.63 [1155.67, 2559.62] | 400000.0 |
| Hopper-v4 | fixed_fraction | 0.75 | distill | self-warmup | 3 | 392.75 [385.23, 403.28] | 956.56 [558.04, 1238.25] | 375000.0 |
| Hopper-v4 | no-improve | 0.75 | distill | self-warmup | 3 | 1130.97 [954.10, 1379.00] | 1986.97 [1022.16, 3224.60] | 150000.0 |
| Hopper-v4 | fixed_fraction | 0.50 | distill | source-aligned | 5 | 688.58 [663.67, 727.75] | 2274.69 [1497.58, 2967.20] | 250000.0 |
| Walker2d-v4 | fixed_fraction | 0.50 | distill | random | 5 | 586.37 [505.99, 682.50] | 2227.23 [1486.85, 2788.95] | 250000.0 |
| Walker2d-v4 | fixed_fraction | 0.25 | distill | self-warmup | 3 | 1280.69 [831.55, 1652.36] | 2380.18 [912.36, 3238.02] | 125000.0 |
| Walker2d-v4 | fixed_fraction | 0.50 | distill | self-warmup | 5 | 597.87 [509.37, 684.43] | 2090.43 [1363.21, 2817.64] | 250000.0 |
| Walker2d-v4 | fixed_fraction | 0.75 | distill | self-warmup | 3 | 349.03 [322.65, 362.91] | 697.36 [346.61, 1071.27] | 375000.0 |
| Walker2d-v4 | no-improve | 0.75 | distill | self-warmup | 3 | 1498.34 [1315.43, 1713.89] | 3777.64 [3253.18, 4086.07] | 146666.7 |
| Walker2d-v4 | fixed_fraction | 0.50 | distill | source-aligned | 5 | 486.88 [441.96, 542.60] | 1545.67 [1142.63, 2266.68] | 250000.0 |

### Ethan Timing Result

The strongest signal is switch timing.

For self-warmup PPO -> SAC:

| Environment | 25% AUC | 50% AUC | 75% AUC |
|---|---:|---:|---:|
| Hopper-v4 | 1411.49 | 729.14 | 392.75 |
| Walker2d-v4 | 1280.69 | 597.87 | 349.03 |

Caveat:

The Hopper 50% self-warmup number is the row affected by the aggregation bug, so it should not be used as a final exact statistic. Even with that caveat, the broad pattern is clear: 25% is much stronger than 75% in both environments.

Interpretation:

PPO appears useful as a warm-start phase, not as the main learner. SAC needs sufficient post-switch budget. A 75% PPO phase leaves SAC only the final quarter of the budget, which is too little for SAC to exploit its sample efficiency.

This supports the central hypothesis:

> Sequencing helps when the first phase does enough useful work to initialize the second phase, but not so much that it starves the second phase of training budget.

### Ethan Adaptive Result

Adaptive trigger:

- Start with PPO.
- Evaluate periodically.
- After a minimum first-phase budget, switch to SAC if evaluation return has not improved for a patience window.
- Current settings: no-improve trigger, patience 3, minimum first phase approximately 25%.

Observed adaptive switch steps:

- Hopper-v4: mean switch step 150000.
- Walker2d-v4: mean switch step 146666.7.

Adaptive results:

| Environment | Adaptive AUC | Adaptive Final Return |
|---|---:|---:|
| Hopper-v4 | 1130.97 | 1986.97 |
| Walker2d-v4 | 1498.34 | 3777.64 |

Interpretation:

The adaptive rule avoided waiting until a late 75% fallback and instead switched soon after PPO stopped improving. On Walker2d-v4, the adaptive row is the best row in the current Ethan table by both AUC and final return.

Defensible claim:

> Adaptive switching can reduce sensitivity to fixed switch fractions by moving budget to SAC when PPO has plateaued.

Avoid claiming:

> Adaptive switching is always optimal.

### Ethan Value-Initialization Result

At the 50% switch:

Hopper-v4:

- random value AUC: 640.80, final return: 1488.63.
- source-aligned value AUC: 688.58, final return: 2274.69.
- self-warmup value AUC: 729.14, final return: 1760.63, but this row has the 500k/1M grouping caveat.

Walker2d-v4:

- random value AUC: 586.37, final return: 2227.23.
- self-warmup value AUC: 597.87, final return: 2090.43.
- source-aligned value AUC: 486.88, final return: 1545.67.

Interpretation:

Value initialization is environment-dependent. Source-aligned value initialization appears to help Hopper final return relative to random but hurts Walker2d. This aligns with Ryan's result that value transfer is not a universal fix.

## What Was Expected vs What Deviated

## Expected: SAC -> PPO might combine SAC sample efficiency with PPO stability

Original intuition:

- SAC learns quickly and explores well.
- PPO is stable and conservative.
- Therefore SAC -> PPO might use SAC for fast early learning and PPO for stable late refinement.

Observed:

- SAC -> PPO improves over pure PPO but does not match pure SAC in Ryan's 500k results.
- The PPO phase often fails to preserve SAC's gains.
- All SAC -> PPO arms have high collapse counts in the Ryan summary.

Interpretation:

PPO did not function as a reliable late-stage stabilizer in the tested setup. Switching away from SAC appears costly when SAC remains productive.

## Expected: Source-aligned value transfer might improve handoff quality

Original intuition:

- A target algorithm initialized with a better value estimate should make better updates after handoff.
- Source-aligned value warm-up should help more than random value initialization.

Observed:

- Ryan SAC -> PPO: source-aligned value hurts Hopper AUC relative to random but helps Walker2d AUC relative to random; all CIs include zero.
- Ethan PPO -> SAC: source-aligned value improves Hopper final return relative to random but hurts Walker2d.

Interpretation:

Value transfer is not uniformly beneficial. It may be sensitive to environment, target algorithm, target mismatch, source-value calibration, and post-handoff dynamics.

## Expected: Handoff timing would matter

Original intuition:

- Switching too early may leave the first algorithm insufficient time to provide a useful initialization.
- Switching too late may leave the second algorithm insufficient time to improve.

Observed:

- Strongly supported, especially by Ethan's PPO -> SAC timing sweep.
- Early PPO -> SAC handoff is much better than late PPO -> SAC handoff.
- Adaptive switching tends to switch early, near 150k steps, and performs well.

Interpretation:

Timing is one of the strongest validated findings. The phase-boundary location is not a minor implementation detail; it determines whether sequencing has enough budget to work.

## Expected: Sequencing might beat SAC everywhere

This expectation should be rejected or avoided.

Observed:

- SAC remains the strongest standalone baseline in Ryan's completed 500k slice.
- PPO -> SAC can be strong, especially with early or adaptive switching, but the final paper should compare directly against budget-matched SAC before making any dominance claim.

Interpretation:

The paper should not claim universal performance superiority. The contribution is a phase-based analysis of when sequencing helps, not a universal replacement for SAC.

## Best Current Claims

The final paper should emphasize these claims.

### Claim 1: SAC is a strong standalone baseline

Supported by:

- Experiment 0 short-horizon baselines.
- Ryan 500k SAC dominance in Hopper-v4 and Walker2d-v4.

Suggested wording:

> SAC remains a strong standalone baseline in MuJoCo continuous control, often achieving the best AUC and final return under matched budgets.

### Claim 2: Handoffs reveal real phase effects

Supported by:

- SAC -> PPO improves over PPO but not SAC.
- PPO -> SAC early handoff substantially outperforms late handoff.
- Direction and timing change outcomes substantially.

Suggested wording:

> Even when sequencing does not beat the strongest single algorithm, handoff results expose meaningful phase effects: the order and timing of algorithms strongly influence sample efficiency and final return.

### Claim 3: PPO is more useful as an early warm start than as a late SAC replacement

Supported by:

- SAC -> PPO struggles after switch.
- PPO -> SAC performs well when SAC receives enough remaining budget.

Suggested wording:

> In the completed experiments, PPO appears more useful as a warm-start phase before SAC than as a late-stage replacement for SAC.

### Claim 4: Value initialization is environment-dependent

Supported by:

- Ryan C1 paired AUC deltas.
- Ethan value-init comparisons.

Suggested wording:

> Value initialization affects handoff behavior, but the direction of the effect is environment-dependent and does not support a universal source-aligned transfer rule.

### Claim 5: Adaptive switching is promising

Supported by:

- Ethan no-improve trigger.
- Mean switch step near 150k.
- Strong Walker2d AUC and final return.

Suggested wording:

> A simple adaptive no-improvement trigger can avoid badly mistimed handoffs and preserve more budget for the algorithm that benefits from it.

## Claims to Avoid

Do not claim:

- SAC -> PPO beats SAC.
- PPO stabilizes SAC in all environments.
- Source-aligned value warm-up is universally better.
- Adaptive switching is always optimal.
- Sequencing is always better than single-algorithm training.
- Offline warm starts worked, unless Abhinav results are added later.

## Recommended Paper Structure

## 1. Introduction

Goals:

- Motivate algorithm sequencing as phase allocation.
- Explain why fixed end-to-end algorithm comparisons are incomplete.
- Introduce SAC, PPO, and the idea of handoff schedules.
- State that the paper focuses on matched-budget MuJoCo experiments.
- Make the contribution mechanism-first rather than leaderboard-first.

Suggested introduction flow:

1. Deep RL algorithms have different strengths at different stages of learning.
2. Standard benchmarking treats algorithms as monolithic choices.
3. In practice, one might want sample-efficient exploration early and stable refinement later.
4. This motivates algorithm sequencing.
5. The main question is not whether SAC or PPO is universally better, but whether phase schedules can exploit complementary strengths under fixed budgets.
6. Contributions:
   - Evaluate SAC/PPO baselines and handoffs.
   - Decompose handoff transfer into policy, value, and data substrates.
   - Show strong effects of switch direction and timing.
   - Show mixed value-transfer results.
   - Demonstrate a simple adaptive trigger as a promising scheduler.

## 2. Background

Cover:

- Markov decision processes and continuous-control RL.
- PPO:
  - On-policy.
  - Clipped surrogate objective.
  - Conservative updates.
  - Stable but less sample-efficient.
- SAC:
  - Off-policy.
  - Entropy-regularized objective.
  - Replay buffer.
  - Sample-efficient and strong in MuJoCo.
- Behavior cloning / offline RL, if discussing planned BC/IQL/AWAC work.
- AUC as a sample-efficiency metric.

## 3. Problem Formulation

Define algorithm sequencing:

Let total budget be `T` environment steps. A schedule divides training into phases:

```text
A1 for T1 steps, A2 for T2 steps, ..., Ak for Tk steps
sum_i Ti = T
```

For this project:

- `A1`, `A2` are usually PPO and SAC.
- Switch fraction `f` determines `T1 = fT`.
- Handoff may transfer policy, value, and/or data.

Define metrics:

- Final return at budget `T`.
- AUC over eval returns.
- Stability across seeds.
- Collapse frequency.
- Average rank.

## 4. Methods

### 4.1 Baselines

Describe:

- Pure SAC.
- Pure PPO.
- Same environments, seeds, evaluation protocol, and online budget.

### 4.2 Fixed Handoffs

Describe:

- SAC -> PPO.
- PPO -> SAC.
- Switch fractions: 25%, 50%, 75% where applicable.
- 50% switch for value ablations.

### 4.3 Transfer Mechanisms

Describe:

- Policy distillation:
  - Target actor is trained to imitate deterministic source actions.
  - Used because actor architectures are incompatible.
- Value initialization:
  - random.
  - self-warmup.
  - source-aligned.
- Replay/data transfer:
  - especially relevant for PPO -> SAC.

### 4.4 Adaptive Switching

Describe:

- no-improvement trigger.
- minimum first-phase budget.
- patience parameter.
- switch when eval return plateaus.

### 4.5 Offline Warm Starts

If included in final paper, describe:

- BC -> SAC.
- BC -> PPO.
- BC -> SAC -> PPO.
- IQL/AWAC planned or optional.

Current status:

These are planned in the task files but not backed by completed result summaries found in the repo.

## 5. Experiments

### 5.1 Environments

Main:

- Hopper-v4.
- Walker2d-v4.

Early sanity:

- HalfCheetah-v4.
- Ant-v4.

### 5.2 Evaluation Protocol

Include:

- Seeds.
- Environment steps.
- Evaluation frequency, if available.
- Mean and confidence intervals.
- AUC computation.
- Collapse count.

### 5.3 Experiment Groups

Group A: short-horizon baselines and pilots.

- Experiment 0: SAC/PPO baselines at 100k.
- Experiment 2: SAC -> PPO 100k pilot.
- Experiment 3: PPO -> SAC 100k pilot.

Group B: Ryan 500k SAC -> PPO mechanism study.

- SAC and PPO baselines.
- SAC -> PPO value ablation.
- Hopper SAC 1M long-horizon.

Group C: Ethan 500k PPO -> SAC timing and adaptive study.

- PPO -> SAC value ablation.
- 25/50/75 timing sweep.
- adaptive no-improvement trigger.

Group D: planned offline warm starts.

- BC, IQL/AWAC, interleaved BC.

## 6. Results

Organize results by claim rather than by file.

### 6.1 SAC is a strong baseline

Use:

- Experiment 0 table.
- Ryan 500k table.

Key statement:

SAC is strongest on Hopper and Walker2d in Ryan's 500k results by AUC and final return.

### 6.2 SAC -> PPO improves over PPO but not SAC

Use:

- Ryan 500k table.
- 100k SAC -> PPO pilot.

Key statement:

SAC -> PPO handoffs inherit useful early behavior from SAC, but PPO does not preserve or improve enough to beat continued SAC.

### 6.3 Value initialization is mixed

Use:

- Ryan paired deltas.
- Ethan value-init comparisons.

Key statement:

The value substrate matters in the sense that results change across variants, but current evidence is environment-dependent and statistically inconclusive.

### 6.4 PPO -> SAC is highly timing-sensitive

Use:

- Ethan timing sweep.
- 100k PPO -> SAC pilot.

Key statement:

Early PPO -> SAC is much stronger than late PPO -> SAC. This directly supports the phase-budget hypothesis.

### 6.5 Adaptive switching reduces timing sensitivity

Use:

- Ethan no-improve rows.

Key statement:

A simple adaptive trigger switches near 150k steps and performs strongly, especially on Walker2d.

## 7. Discussion

Themes:

### SAC's strength is not a failure of the project

SAC being strong does not invalidate algorithm sequencing. It clarifies the mechanism: if SAC remains productive, switching away from it is costly. This explains why SAC -> PPO struggles.

### Sequencing is conditional, not automatic

Bad schedules are bad. Late PPO -> SAC fails because SAC has too little budget. SAC -> PPO fails to beat SAC because PPO is not a stronger late-phase learner in these runs.

### Handoff mechanisms are fragile

Behavioral distillation can transfer policy behavior, but value and critic transfer may not align cleanly across algorithms. Source-aligned value transfer may introduce mismatch if source values are poorly calibrated for the target update rule.

### Adaptive scheduling is a natural next step

Fixed fractions are blunt. Adaptive triggers can respond to plateau behavior and allocate budget dynamically.

## 8. Limitations

Important limitations:

- Main completed results cover Hopper-v4 and Walker2d-v4, not a broad benchmark suite.
- Ryan's core mechanism results use 5 seeds, but some Ethan timing/adaptive rows use 3 seeds.
- Ethan summary contains a known aggregation bug for Hopper 50% self-warmup.
- Offline warm-start experiments are planned but not completed in the available result files.
- SAC is a very strong baseline, making it difficult to show performance dominance.
- Collapse counts in SAC -> PPO suggest instability that needs deeper diagnostics.
- Compute-normalized comparisons remain incomplete unless later results are added.

## 9. Future Work

Potential future directions:

- Regenerate Ethan results with total budget included in grouping keys.
- Add budget-matched SAC/PPO baselines to the same Ethan tables.
- Complete BC -> SAC and BC -> PPO experiments.
- Test BC -> SAC -> PPO as the full three-phase scheduler.
- Add IQL or AWAC as value-carrying offline sources.
- Expand to HalfCheetah and Ant for stronger generalization.
- Compare adaptive triggers beyond no-improvement:
  - entropy threshold.
  - critic loss stabilization.
  - KL/update magnitude stabilization.
  - return plateau.
- Run compute-normalized comparisons by gradient updates or wall-clock proxy.

## Suggested Final Paper Title Options

- Algorithm Sequencing in Deep Reinforcement Learning
- Phase-Based Scheduling for Deep Reinforcement Learning
- When Should Reinforcement Learning Algorithms Switch?
- Understanding Algorithm Handoffs in Continuous-Control RL
- From Baselines to Phases: Diagnosing SAC/PPO Sequencing in MuJoCo

## Suggested Final Paper Contributions

Use a concise contribution list like:

1. We frame deep RL algorithms as phase-specific training procedures rather than monolithic end-to-end choices.
2. We decompose algorithm handoffs into policy, value, and data transfer mechanisms.
3. We evaluate SAC/PPO sequencing under matched environment-step budgets on MuJoCo control tasks.
4. We show that SAC remains a strong standalone baseline, while handoffs expose meaningful direction and timing effects.
5. We find that value initialization is environment-dependent and not a universal solution to handoff instability.
6. We show that simple adaptive switching can avoid poorly timed handoffs and improve performance in some settings.

## Best Single-Paragraph Conclusion

The completed experiments suggest that algorithm sequencing in deep RL is best understood as a phase-allocation problem rather than a universal performance trick. SAC remains a powerful standalone baseline in MuJoCo, and SAC -> PPO handoffs do not beat continued SAC under the tested 500k budget. However, the handoff experiments reveal important structure: switching direction and timing matter substantially, PPO can be useful as a warm-start phase before SAC, and adaptive no-improvement switching can avoid late handoffs that starve SAC of training budget. Value initialization affects handoff behavior but is environment-dependent and statistically inconclusive in the current runs. Overall, the project supports a cautious but meaningful claim: sequencing is not automatically better than strong single algorithms, but phase-based schedules reveal mechanisms that can guide more robust RL training strategies.

## Source Files Used

Task files:

- `tasks/ryan-task.md`
- `tasks/ethan-task.md`
- `tasks/abhinav-task.md`

Design and project docs:

- `README.md`
- `FIRST_EXPERIMENTS.md`
- `PROGRESS_UPDATE.md`
- `docs/superpowers/specs/2026-05-26-rl-sequencing-transfer-mechanism-design.md`

Result summaries:

- `experiments/experiment_0.md`
- `experiments/experiment_2.md`
- `experiments/experiment_3.md`
- `experiments/ryan_task/reports/ryan_results.md`
- `experiments/ryan_task/reports/ryan_explanation.md`
- `experiments/ethan_task/results.md`
- `experiments/ethan_task/ethan-experiment-takeaways.md`

Ryan processed metrics:

- `experiments/ryan_task/processed/arm_summary.csv`
- `experiments/ryan_task/processed/value_init_deltas.csv`
- `experiments/ryan_task/processed/rank_summary.csv`

Useful figures and referenced artifacts:

- `experiments/ryan_task/figures/headline/learning_curves_500k.png`
- `experiments/ryan_task/figures/headline/auc_summary.png`
- `experiments/ryan_task/figures/headline/final_return_summary.png`
- `experiments/ryan_task/figures/headline/value_init_auc_deltas.png`
- `experiments/ryan_task/figures/headline/hopper_sac_long_horizon.png`
- `experiments/ethan_task/results.md` references `results/processed/ethan_task/Hopper_v4_learning_curves.png`
- `experiments/ethan_task/results.md` references `results/processed/ethan_task/Walker2d_v4_learning_curves.png`
- `experiments/ethan_task/results.md` references `results/processed/ethan_task/Hopper_v4_auc_summary.png`
- `experiments/ethan_task/results.md` references `results/processed/ethan_task/Walker2d_v4_auc_summary.png`
