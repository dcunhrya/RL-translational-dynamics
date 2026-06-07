# Project Summary: Algorithm Sequencing in Deep Reinforcement Learning

This document summarizes the project goals, approach, experiments, and results from the repository, with emphasis on `docs/`, `experiments/`, `tasks/`, the poster, milestone report, and final-report materials. The central interpretation is that this project is not a claim that one algorithm, such as SAC, is universally better than another. It studies whether deep RL algorithms should be treated as phase-specific training components whose order, switch timing, and transfer mechanisms affect learning under matched online budgets.

## Source and Document Notes

- The repository does not currently contain a top-level `results/` directory, although several documents reference one. The checked-in result summaries live mainly under `experiments/`.
- The proposal PDF referenced by `README.md` was not found in the repository. This summary reconstructs the proposal intent from `README.md`, `FIRST_EXPERIMENTS.md`, the milestone report, the design spec, task files, poster, and final report.
- The final project PDF begins with the RL sequencing paper, but later pages switch to an unrelated GPT-2/NLP report. Only the RL-relevant portions of that PDF are used here.
- Offline-assisted results use additional offline data or modified-environment pretraining. They should be interpreted separately from strictly online, environment-step-matched PPO/SAC comparisons.

## Abstract

Deep reinforcement learning algorithms are usually evaluated as fixed end-to-end training procedures, but training may be better understood as a sequence of phases. Early learning may need a non-random initialization or useful state coverage, middle learning may need sample-efficient online improvement, and late learning may need stable refinement. This project studies algorithm sequencing in MuJoCo continuous-control tasks, asking whether switching among PPO, SAC, and offline or curriculum warm starts can improve sample efficiency, stability, or final return under fixed online budgets.

The project evaluates SAC and PPO baselines, fixed SAC -> PPO and PPO -> SAC handoffs, value-initialization ablations, adaptive switching, behavior-cloning and AWAC warm starts, interleaved BC anchoring, and easy-environment SAC pretraining. The strongest online-only result is not that sequencing universally beats SAC. SAC remains the strongest standalone online baseline in the completed 500k-step SAC/PPO experiments on Hopper-v4 and Walker2d-v4. However, handoffs reveal meaningful phase effects: PPO -> SAC is much stronger when the switch is early or adaptive, while SAC -> PPO often loses SAC's gains after switching into PPO. Offline and curriculum warm starts are most effective when they hand into SAC, suggesting that SAC is the target learner most able to preserve and improve useful initial policies.

Overall, the project supports a phase-allocation view of RL training. Sequencing can help when a first phase produces behavior, data, or representations that the next learner can exploit, and when the switch leaves enough budget for that learner to improve. It does not support naive claims that any handoff is better than a strong single algorithm, that PPO reliably stabilizes SAC, or that value transfer alone solves handoff instability.

## Introduction

The project asks:

> Can explicitly sequencing RL algorithms improve convergence, sample efficiency, stability, or final performance under a fixed interaction budget?

The initial intuition was that different algorithms have complementary strengths. SAC is off-policy, replay-based, entropy-regularized, and often sample-efficient in continuous-control tasks. PPO is on-policy, uses conservative clipped updates, and is often valued for stable policy improvement. Offline and imitation methods such as BC, AWAC, or IQL can provide non-random policies before online training begins.

The early project framing emphasized SAC -> PPO: use SAC for fast exploration and online improvement, then switch to PPO for conservative late-stage refinement. The experiments changed that story in an important way. SAC proved difficult to outperform as a standalone baseline, and PPO did not reliably preserve strong SAC policies after handoff. The refined thesis is therefore more mechanistic:

> Algorithm sequencing is useful when the first phase creates a transferable policy, value estimate, replay distribution, or curriculum state that the next phase can preserve and improve. The value of a sequence depends on direction, timing, target learner, and transfer compatibility.

This reframing is visible across the milestone report, design spec, poster, and final report. The project becomes a study of phase allocation rather than a leaderboard attempt to beat SAC everywhere.

## Approach

The project models a training run as a schedule over a fixed online budget `B`:

```text
A1 for T1 steps, A2 for T2 steps, ..., Ak for Tk steps
sum_i Ti = B
```

Each phase may transfer information to the next phase. The main transfer substrates are:

- **Policy behavior:** transferred through behavioral distillation, because the SAC and PPO actor architectures are incompatible. SAC uses a 256x256 ReLU actor with state-dependent mean and log-std heads; PPO uses a 64x64 Tanh actor with a state-independent log-std parameter.
- **Value estimates:** initialized as random, self-warmup, or source-aligned. For SAC -> PPO, source alignment regresses PPO's value function toward SAC-derived Q estimates. For PPO -> SAC, source alignment uses PPO rollout returns to seed SAC critics.
- **Data/replay:** especially relevant for PPO -> SAC, where SAC can use stored transitions in an off-policy replay buffer.
- **Offline or pretrained policy priors:** BC and AWAC from D4RL expert data, plus easy-environment SAC pretraining.

The conceptual phase roles are:

- **Warm start:** BC, AWAC, PPO warmup, or easy-environment SAC can produce non-random behavior.
- **Online improvement:** SAC is the strongest candidate because it is sample-efficient and replay-based.
- **Refinement:** PPO was hypothesized to stabilize late-stage policies, but current results show this role is not yet reliable.

The project pre-registers several metrics:

- **Final return:** end-of-budget policy quality.
- **Normalized AUC:** sample efficiency over the full learning curve.
- **Stability:** seed variance, worst-seed return, and collapse count.
- **Handoff diagnostics:** policy retention, value quality, handoff transient, phase markers, and switch step.
- **Budget accounting:** online environment steps are matched for online-only comparisons; offline data and modified-environment pretraining are reported separately.

## Experiments

### Experiment 0: Short-Horizon Baselines

Experiment 0 validated the training stack with SAC and PPO at 100k environment steps on Hopper-v4, Walker2d-v4, HalfCheetah-v4, and Ant-v4 using two seeds. The gate passed: no recurring crashes or NaNs, complete logging, and expected learning behavior.

Final returns:

| Environment | SAC | PPO | Main takeaway |
| --- | ---: | ---: | --- |
| Hopper-v4 | 1204.24 +/- 795.57 | 346.23 +/- 15.37 | SAC stronger, high variance. |
| Walker2d-v4 | 326.17 +/- 44.50 | 339.01 +/- 33.65 | Roughly tied, slight PPO edge. |
| HalfCheetah-v4 | 4830.85 +/- 1075.63 | 275.69 +/- 26.16 | SAC much stronger. |
| Ant-v4 | 1152.57 +/- 307.84 | 448.39 +/- 92.06 | SAC stronger. |

This established SAC as a strong early baseline and motivated testing whether handoffs could do better than either pure algorithm.

### Experiments 2 and 3: 100k Online Handoff Pilots

Experiment 2 tested SAC -> PPO at switch fractions 25%, 50%, and 75% on Hopper-v4 and Walker2d-v4 with three seeds. Experiment 3 tested the reverse direction, PPO -> SAC, with the same environments, budget, and switch fractions.

Key 100k pilot results:

| Environment | Best SAC -> PPO | Best PPO -> SAC | Baseline context |
| --- | ---: | ---: | --- |
| Hopper-v4 | 735.27 +/- 251.71 at 75% SAC | 754.55 +/- 374.10 at 25% PPO | SAC baseline was 1204.24 +/- 795.57. |
| Walker2d-v4 | 513.48 +/- 86.63 at 50% SAC | 566.35 +/- 126.00 at 25% PPO | SAC was 326.17 +/- 44.50 and PPO was 339.01 +/- 33.65. |

The pilots showed that handoffs often outperform PPO and that direction matters. PPO -> SAC was stronger than expected, especially with early switches. This suggested that PPO may be more useful as a warm-start phase before SAC than as a final algorithm after SAC.

### Ryan Slice: 500k SAC -> PPO Value Ablation

Ryan's slice is the most complete online-only experiment bundle. It includes 53/53 completed runs:

- 500k SAC baselines.
- 500k PPO baselines.
- 500k SAC -> PPO with a 50% switch.
- Three PPO value-initialization variants after fixed policy distillation: random, self-warmup, source-aligned.
- Hopper-v4 SAC 1M long-horizon check.
- Five seeds per 500k arm, with three seeds for the Hopper SAC 1M check.

Main Ryan results:

| Environment | Method | Final mean | Normalized AUC mean | Collapse count |
| --- | --- | ---: | ---: | ---: |
| Hopper-v4 | SAC | 2895.6 | 2260.239 | 1 |
| Hopper-v4 | PPO | 384.3 | 324.941 | 0 |
| Hopper-v4 | SAC -> PPO random V | 403.5 | 1219.303 | 5 |
| Hopper-v4 | SAC -> PPO self-warmup V | 439.2 | 1145.765 | 5 |
| Hopper-v4 | SAC -> PPO source-aligned V | 384.9 | 1131.139 | 5 |
| Walker2d-v4 | SAC | 3310.4 | 1642.230 | 0 |
| Walker2d-v4 | PPO | 441.5 | 381.714 | 1 |
| Walker2d-v4 | SAC -> PPO random V | 442.5 | 602.467 | 5 |
| Walker2d-v4 | SAC -> PPO self-warmup V | 464.6 | 623.689 | 5 |
| Walker2d-v4 | SAC -> PPO source-aligned V | 646.6 | 718.320 | 5 |

SAC is the strongest method in this slice by both AUC and final return. SAC -> PPO improves substantially over PPO in AUC but does not preserve enough of SAC's trajectory to beat continued SAC. The high collapse counts for SAC -> PPO arms indicate that switching into PPO is fragile under the tested protocol.

The value-initialization ablation is mixed:

| Environment | Comparison vs random value init | Mean AUC delta | 95% CI |
| --- | --- | ---: | --- |
| Hopper-v4 | self-warmup V | -73.54 | [-165.26, 18.18] |
| Hopper-v4 | source-aligned V | -88.16 | [-188.38, 12.05] |
| Walker2d-v4 | self-warmup V | 21.22 | [-7.26, 54.91] |
| Walker2d-v4 | source-aligned V | 115.85 | [-25.31, 278.73] |

All confidence intervals include zero. The safest conclusion is that value initialization affects behavior but does not provide a universal improvement. Hopper trends negative for value transfer, while Walker2d trends positive, especially for source-aligned values.

The Hopper 1M SAC check further supports the interpretation that switching away from SAC can be costly. SAC continues improving with more budget, reaching final returns around 3307-3327 for seeds 0-2 at 1M steps.

### Ethan Slice: PPO -> SAC Timing, Value, and Adaptive Switching

Ethan's experiments study the reverse direction, PPO -> SAC. The slice includes:

- PPO -> SAC value ablation at the 50% switch.
- Fixed switch timing sweep at 25%, 50%, and 75%.
- Adaptive no-improvement trigger.
- Hopper-v4 and Walker2d-v4.

Important caveat: the current Ethan summary has a known aggregation issue for the Hopper 50% self-warmup row. Its mean switch step is 400000 rather than the expected 250000 for a clean 500k run, indicating that 500k and 1M rows were likely mixed. Exact values from that row should be regenerated before final paper use.

The clearest signal is switch timing. For self-warmup PPO -> SAC:

| Environment | 25% switch AUC | 50% switch AUC | 75% switch AUC |
| --- | ---: | ---: | ---: |
| Hopper-v4 | 1411.49 | 729.14 | 392.75 |
| Walker2d-v4 | 1280.69 | 597.87 | 349.03 |

Even with the Hopper 50% caveat, the broad pattern is clear: early PPO -> SAC handoff is much better than late handoff. SAC needs enough remaining budget to exploit the warm start.

The adaptive rule starts with PPO, requires a minimum first phase of about 25%, and switches to SAC after three evaluation checkpoints without improvement. It switched near 150k steps in 500k-step runs:

| Environment | Mean adaptive switch step | Adaptive AUC | Adaptive final return |
| --- | ---: | ---: | ---: |
| Hopper-v4 | 150000 | 1130.97 | 1986.97 |
| Walker2d-v4 | 146666.7 | 1498.34 | 3777.64 |

On Walker2d-v4, the adaptive row is the strongest row in the current Ethan table by both AUC and final return. This supports the claim that adaptive switching can reduce sensitivity to fixed switch fractions, though it should not be presented as universally optimal.

Value initialization in PPO -> SAC is also environment-dependent. Source-aligned values improve Hopper final return relative to random at 50%, but hurt Walker2d. The stronger and more robust finding is timing, not a universal value-transfer rule.

### Abhinav Slice: Offline-Assisted and Curriculum Warm Starts

Abhinav's slice extends the project beyond online-only handoffs. It includes:

- BC from D4RL expert demonstrations.
- BC -> SAC and BC -> PPO.
- BC -> SAC -> PPO as the full three-phase scheduler.
- Interleaved BC anchoring during SAC.
- AWAC and AWAC -> SAC/PPO.
- Easy SAC pretraining in a forgiving Hopper variant, then Easy SAC -> real SAC.

These experiments report online steps separately from offline dataset size and offline updates, so they should not be described as pure environment-step-matched comparisons against SAC/PPO.

Key offline-assisted results:

| Environment | Method | Seeds | Final return | Normalized AUC | Main interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Hopper-v4 | BC | 1 | 3048.87 | 0.00 | BC policy itself is strong, but offline-only AUC is not comparable. |
| Hopper-v4 | BC -> SAC | 5 | 3047.11 | 1897.68 | Strong SAC target use of BC warm start. |
| Hopper-v4 | BC -> PPO | 5 | 503.05 | 553.33 | PPO target remains weak. |
| Hopper-v4 | BC -> SAC -> PPO | 5 | 472.41 | 852.49 | Late PPO destroys much of the SAC trajectory. |
| Hopper-v4 | AWAC -> SAC | 3 | 2609.22 | 2224.63 | Viable into SAC, not clearly better than BC -> SAC on final return. |
| Hopper-v4 | AWAC -> PPO | 3 | 518.06 | 526.53 | PPO target remains weak. |
| Hopper-v4 | Easy SAC -> SAC | 3 | 3333.09 | 2133.92 | Strong curriculum-style warm start into SAC. |
| Walker2d-v4 | BC | 1 | 4898.39 | 0.00 | Strong offline policy evaluation. |
| Walker2d-v4 | BC -> SAC | 5 | 3965.27 | 2350.95 | Strongest BC transfer family result. |
| Walker2d-v4 | BC -> PPO | 5 | 625.78 | 543.75 | PPO target remains weak. |
| Walker2d-v4 | BC -> SAC -> PPO | 5 | 371.66 | 770.13 | Late PPO again performs poorly. |
| Walker2d-v4 | AWAC -> SAC | 3 | 3545.32 | 2235.97 | Viable into SAC. |
| Walker2d-v4 | AWAC -> PPO | 3 | 793.52 | 459.16 | PPO target remains weak. |

Interleaved BC anchoring produced nuanced results. On Hopper, BC-anchor SAC with K=50k achieved AUC 1975.20, slightly above BC -> SAC AUC 1897.68, but final return was lower. On Walker2d, BC-anchor SAC with K=50k achieved AUC 2492.97, above BC -> SAC AUC 2350.95, but final return was again lower. This suggests anchoring can improve early sample efficiency in some cases but may constrain final improvement.

The strongest offline/curriculum pattern is that warm starts work best when the target learner is SAC. The same initial policy transferred into PPO performs poorly, and schedules that end in PPO underperform schedules that end in SAC.

## Results

### Result 1: SAC Is the Strongest Online Baseline

Across completed 500k-step Ryan experiments, SAC has the best AUC and final return on both Hopper-v4 and Walker2d-v4. The Hopper 1M check shows SAC continues to improve beyond the 500k budget. This means a sequencing method must justify switching away from SAC; it cannot assume SAC has saturated at the handoff point.

### Result 2: SAC -> PPO Improves Over PPO but Not SAC

SAC -> PPO inherits useful early behavior from SAC and improves over PPO in AUC, especially on Hopper. However, the PPO phase does not preserve or improve the SAC trajectory enough to beat continued SAC. This weakens the original idea that PPO is a dependable final stabilization phase after SAC.

### Result 3: PPO -> SAC Shows Strong Timing Effects

Early PPO -> SAC handoffs substantially outperform late PPO -> SAC handoffs. PPO can act as a warm-start phase, but SAC needs enough remaining online budget to exploit the initialization. Late switches starve SAC of its main advantage: sample-efficient improvement through replay-based off-policy updates.

### Result 4: Adaptive Switching Is Promising

The no-improvement adaptive trigger switches near the strong early-switch regime and avoids waiting until a late fallback point. It performs strongly, especially on Walker2d-v4. The appropriate claim is that adaptive switching can reduce sensitivity to fixed switch fractions, not that it is always optimal.

### Result 5: Value Initialization Is Environment-Dependent

Value transfer does not show a universal benefit. In SAC -> PPO, source-aligned and self-warmup value initialization trend worse than random on Hopper but better on Walker2d, with all paired confidence intervals including zero. In PPO -> SAC, source-aligned values help Hopper final return relative to random but hurt Walker2d. This suggests value initialization is a fragile mechanism rather than a general solution.

### Result 6: Offline and Curriculum Warm Starts Work Best Into SAC

BC -> SAC, AWAC -> SAC, and Easy SAC -> SAC produce strong results. BC -> PPO and AWAC -> PPO remain weak, and BC -> SAC -> PPO underperforms BC -> SAC alone. This is one of the most consistent findings across the poster, final report, and experiment results: compatible schedules usually end in SAC.

### Result 7: The Full Three-Phase BC -> SAC -> PPO Scheduler Underperforms

The proposed three-phase structure is conceptually important, but the tested BC -> SAC -> PPO schedule does not outperform BC -> SAC. This mirrors the online SAC -> PPO failure mode and indicates that the current PPO refinement phase needs stronger policy-retention or state-distribution transfer mechanisms before it can serve as a dependable final phase.

## Discussion

The project's most important insight is that sequencing is conditional. A handoff is not useful simply because it combines two algorithms. It is useful only if the first phase creates something the second phase can exploit and if the second phase receives enough budget to improve.

SAC's strength should not be interpreted as a failure of the project. Instead, it explains why some schedules fail. If SAC is still improving, switching away from it is costly. This is exactly what Ryan's SAC -> PPO results and the Hopper long-horizon check show.

The direction of transfer matters. PPO -> SAC works better than SAC -> PPO in several settings because it assigns SAC the role of main online improver. PPO appears more useful as a short warm-start phase than as a late-stage replacement for SAC. Offline and curriculum experiments reinforce the same pattern: BC, AWAC, and Easy SAC starts are useful primarily when SAC is the target learner.

The transfer mechanism matters, but not always in the expected way. Behavioral distillation is necessary because PPO and SAC architectures differ. Value initialization was expected to improve handoff quality, but results are mixed and environment-specific. In some cases, the target algorithm may overwrite the transferred behavior or receive value targets that are not well calibrated for its update rule.

Adaptive switching is a natural extension because fixed switch fractions are blunt. The no-improvement trigger is simple, reproducible, and already shows that switching near the plateau of the first phase can avoid wasting budget. Richer triggers based on entropy, critic stability, KL drift, or update magnitudes are logical future work.

Important limitations:

- The strongest online-only evidence is mainly on Hopper-v4 and Walker2d-v4.
- Ryan's core runs use five seeds, but several Ethan and Abhinav stretch rows use three seeds, and standalone BC/AWAC evaluations sometimes use one seed.
- The Ethan Hopper 50% self-warmup row has a known aggregation bug and should be regenerated before exact final reporting.
- Offline-assisted methods use extra data or pretraining and must be reported separately from pure online matched-budget comparisons.
- The final report PDF contains unrelated NLP content after the RL-relevant sections, so the source document needs cleanup.
- PPO refinement is not yet solved; current handoffs into PPO often degrade transferred performance.

## Conclusion

The completed experiments support a cautious but meaningful conclusion: algorithm sequencing in deep RL is best understood as phase allocation under a fixed budget, not as a universal performance trick. SAC remains a strong standalone online baseline, and switching away from SAC into PPO is often harmful under the current transfer protocol. However, sequencing reveals real structure that single-algorithm comparisons hide: switch direction, switch timing, target learner, and transfer substrate substantially shape outcomes.

The most successful schedules either hand into SAC or keep SAC as the final online learner. PPO -> SAC benefits from early or adaptive switching because PPO can provide a short warm start while SAC receives enough budget to improve. BC -> SAC, AWAC -> SAC, and Easy SAC -> SAC show that offline and curriculum starts can produce useful policies when the target learner can preserve and improve them. In contrast, BC -> PPO, AWAC -> PPO, SAC -> PPO, and BC -> SAC -> PPO show that PPO is not yet a reliable final refinement phase in this implementation.

The best final project claim is therefore:

> Algorithm sequencing is not a universal replacement for strong single algorithms like SAC, but phase-based schedules expose meaningful timing and compatibility effects. Sequencing helps when a warm start, first algorithm, or curriculum phase produces information that the next learner can exploit while retaining enough budget to improve.

Future work should focus on stronger adaptive switching rules, better PPO policy-retention mechanisms, cleaner value-transfer targets, broader environment coverage, and compute-normalized comparisons. The immediate reporting priority is to keep online-only and offline-assisted budgets separate, fix the Ethan aggregation issue, and clean the final report source so only the RL sequencing paper remains.

## Instructions for GPT-5.5 to Write the Final Paper

Use this summary as the authoritative source for writing the final CS 224R paper. The goal is to produce a polished, rigorous, mechanism-first paper about algorithm sequencing in deep reinforcement learning. Do not write a generic SAC-vs-PPO comparison. The paper should argue that RL algorithms can be understood as phase-specific training components, and that successful sequencing depends on phase allocation, switch timing, target learner compatibility, and what information transfers across phase boundaries.

### Core Paper Thesis

The central thesis should be:

> Algorithm sequencing is not a universal replacement for strong single algorithms such as SAC, but it reveals meaningful phase effects in deep RL. Schedules help when an early phase produces a useful policy, state distribution, representation, replay buffer, or curriculum initialization that a later learner can preserve and improve with enough remaining budget.

The paper must not claim:

- "SAC is better than PPO" as the main contribution.
- "Sequencing always beats SAC."
- "PPO reliably stabilizes SAC."
- "Source-aligned value transfer is universally beneficial."
- "Offline-assisted schedules are directly environment-step-matched with pure online SAC/PPO."

### Recommended Title

Use a title close to:

> When Should Reinforcement Learning Algorithms Switch? A Study of Phase-Based Training in Continuous Control

Alternative acceptable titles:

- Algorithm Sequencing in Deep Reinforcement Learning
- Phase-Based Scheduling for Deep Reinforcement Learning
- Understanding Algorithm Handoffs in Continuous-Control RL

### Paper Structure

Write the paper with the following structure:

1. **Abstract**
   - State that RL algorithms are usually treated as monolithic training procedures.
   - Introduce algorithm sequencing as phase allocation under a fixed interaction budget.
   - Mention PPO, SAC, BC, AWAC, adaptive switching, and curriculum/easy-environment pretraining.
   - Summarize the main findings: SAC is the strongest online baseline; schedules that end in SAC work best; SAC -> PPO and BC -> SAC -> PPO are fragile; early/adaptive PPO -> SAC works better than late switching; value initialization is mixed; offline/curriculum warm starts must be reported separately.
   - End with the balanced conclusion that sequencing is useful as a framework for understanding phase compatibility, not as a universal performance trick.

2. **Introduction**
   - Motivate why fixed algorithm comparisons are incomplete.
   - Explain that early, middle, and late training place different demands on the learner.
   - Frame PPO as stable/on-policy, SAC as sample-efficient/off-policy, and BC/AWAC/Easy SAC as warm-start mechanisms.
   - Present the research question: when should RL algorithms switch under a matched budget?
   - List contributions:
     - Phase-allocation framing.
     - Transfer-substrate view of handoffs.
     - Online SAC/PPO handoff experiments.
     - Adaptive switch timing analysis.
     - Offline and curriculum warm-start experiments.
     - Negative/mixed value-transfer finding.

3. **Related Work**
   - Briefly cover PPO, SAC, behavior cloning, D4RL, AWAC, IQL, curriculum learning, and transfer learning in RL.
   - Keep this section concise. Do not let it dominate the paper.
   - Position this work as studying explicit handoffs and phase compatibility rather than standalone algorithm benchmarking.

4. **Approach**
   - Define a schedule under budget `B` as phases `A1, ..., Ak` with `sum_i Ti = B`.
   - Describe the three-phase scheduler:
     - Warm start: PPO warmup, BC, AWAC, Easy SAC.
     - Online improvement: SAC.
     - Refinement: PPO, noting that the experiments test whether this actually works.
   - Explain handoff mechanisms:
     - Behavioral policy distillation.
     - Value initialization: random, self-warmup, source-aligned.
     - Replay/data transfer.
     - Offline/curriculum initialization.
   - Explicitly state that SAC and PPO architectures are incompatible, so policy transfer uses behavioral distillation rather than weight loading.
   - Describe adaptive switching as a simple no-improvement trigger with a 25% minimum first phase and patience of three evaluation checkpoints.

5. **Experimental Setup**
   - Separate experiments into two budget categories:
     - Online-only, environment-step-matched: SAC, PPO, SAC -> PPO, PPO -> SAC, adaptive PPO -> SAC.
     - Offline-assisted or pretraining-assisted: BC -> SAC, BC -> PPO, BC -> SAC -> PPO, AWAC -> SAC/PPO, interleaved BC, Easy SAC -> SAC.
   - Main environments: Hopper-v4 and Walker2d-v4.
   - Mention HalfCheetah-v4 and Ant-v4 only as early 100k sanity environments.
   - State budgets and seeds:
     - 100k pilots with 2-3 seeds.
     - 500k core runs with 5 seeds where available.
     - 1M Hopper long-horizon checks with 3 seeds.
     - Some stretch/offline rows use 3 seeds or 1 seed for offline-only evaluation.
   - Metrics:
     - Final return.
     - Normalized AUC.
     - Seed variance / confidence intervals.
     - Worst seed and collapse count.
     - Handoff transients and diagnostics when available.

6. **Results**
   - Organize results by claim, not by teammate or file.
   - Recommended result subsections:
     - **SAC is a strong standalone online baseline.**
       Use Ryan 500k SAC/PPO table and Hopper 1M SAC check.
     - **SAC -> PPO improves over PPO but fails to match SAC.**
       Emphasize PPO does not preserve SAC's gains and collapse counts are high.
     - **PPO -> SAC is highly timing-sensitive.**
       Compare 25%, 50%, 75%; early switches beat late switches.
     - **Adaptive switching reduces timing sensitivity.**
       Report mean switch near 150k and strong Walker2d performance.
     - **Value initialization is mixed and environment-dependent.**
       Present Ryan paired AUC deltas and mention all CIs include zero.
     - **Offline and curriculum warm starts work best into SAC.**
       Compare BC -> SAC vs BC -> PPO, AWAC -> SAC vs AWAC -> PPO, Easy SAC -> SAC, and BC -> SAC -> PPO.
     - **The full BC -> SAC -> PPO three-phase schedule underperforms BC -> SAC.**
       Explain that this rejects the current PPO-as-final-refinement implementation, not the entire phase-allocation idea.

7. **Discussion**
   - Emphasize that SAC being strong is not a failure; it explains why switching away from SAC is costly.
   - Explain that compatible schedules usually end in SAC because SAC can exploit warm starts with replay-based off-policy updates.
   - Discuss why PPO may be useful as a short warm-start phase but not yet as a reliable final refinement phase.
   - Discuss value-transfer fragility: a value target can be miscalibrated for the target learner, and the target algorithm may overwrite transferred behavior.
   - Discuss adaptive switching as a natural next step because fixed fractions are brittle.
   - Be explicit that offline/pretraining schedules answer a different question than online-only matched-budget comparisons.

8. **Limitations**
   - Main full-scale evidence is on Hopper-v4 and Walker2d-v4.
   - Some rows use only three seeds or one seed.
   - Ethan's Hopper 50% self-warmup row has a known aggregation issue and should be treated cautiously.
   - Offline data and easy-environment pretraining introduce extra data/interaction outside the target online budget.
   - PPO refinement remains unsolved in the current implementation.
   - Compute-normalized comparisons are incomplete.
   - The final report source document appears corrupted with unrelated NLP material and should be cleaned before submission.

9. **Conclusion**
   - Restate the balanced claim:
     - Sequencing is conditional and mechanism-dependent.
     - SAC remains a strong online baseline.
     - Early/adaptive handoffs into SAC and offline/curriculum warm starts into SAC are promising.
     - Handoffs into PPO are currently fragile.
     - Value initialization alone is not a universal solution.
   - End with a forward-looking point: better adaptive triggers and stronger policy-retention mechanisms may make phase-based RL schedules more robust.

### Numerical Results to Prioritize

Use these numbers prominently:

- Ryan 500k SAC:
  - Hopper-v4 final 2895.6, AUC 2260.239.
  - Walker2d-v4 final 3310.4, AUC 1642.230.
- Ryan SAC -> PPO:
  - Hopper best handoff final 439.2, best AUC 1219.303, far below SAC.
  - Walker2d best handoff final 646.6, AUC 718.320, far below SAC.
  - SAC -> PPO collapse count 5/5 for all handoff arms in both environments.
- Ryan value deltas:
  - Hopper self-warmup vs random: -73.54, CI [-165.26, 18.18].
  - Hopper source-aligned vs random: -88.16, CI [-188.38, 12.05].
  - Walker2d self-warmup vs random: +21.22, CI [-7.26, 54.91].
  - Walker2d source-aligned vs random: +115.85, CI [-25.31, 278.73].
- Ethan PPO -> SAC timing:
  - Hopper 25% AUC 1411.49 vs 75% AUC 392.75.
  - Walker2d 25% AUC 1280.69 vs 75% AUC 349.03.
- Ethan adaptive:
  - Hopper switch near 150k, AUC 1130.97, final 1986.97.
  - Walker2d switch near 146667, AUC 1498.34, final 3777.64.
- Abhinav offline/curriculum:
  - Hopper BC -> SAC final 3047.11, AUC 1897.68.
  - Walker2d BC -> SAC final 3965.27, AUC 2350.95.
  - Hopper BC -> PPO final 503.05; Walker2d BC -> PPO final 625.78.
  - Hopper BC -> SAC -> PPO final 472.41; Walker2d BC -> SAC -> PPO final 371.66.
  - Hopper Easy SAC -> SAC final 3333.09, AUC 2133.92.
  - Walker2d AWAC -> SAC final 3545.32, AUC 2235.97.

### Writing Style Requirements

- Write as a rigorous CS 224R project paper, not as notes.
- Use cautious, defensible language.
- Prefer claims such as "suggests", "supports", "is consistent with", and "under the tested protocol" when discussing empirical findings.
- Avoid overclaiming from 3-seed or 1-seed rows.
- Make negative results sound useful: they clarify which phase roles and transfer mechanisms do not work under the tested setup.
- Treat "SAC remains strong" as an explanatory anchor, not as a contradiction of the project.
- Keep online-only and offline-assisted comparisons separate.
- If exact formatting is needed, write tables for the main Ryan, Ethan, and Abhinav results and use figures for learning curves/AUC/final return summaries.

### Final Paper North Star

The ideal final paper should leave the reader with this understanding:

> The project began with a simple intuition that SAC and PPO might be combined to exploit complementary strengths. The evidence showed a more nuanced and more interesting story: sequencing helps only when the phase boundary is compatible with the target learner. SAC is hard to beat as a standalone online algorithm, but warm starts and early handoffs into SAC can be effective. PPO, in the current implementation, is not yet a dependable final refinement stage. Therefore, the contribution is a phase-based framework for reasoning about when RL algorithms should switch, supported by empirical evidence on timing, transfer, adaptive switching, and offline/curriculum initialization.

