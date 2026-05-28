# Ethan Experiment Takeaways

## Bottom Line

These experiments are interesting and paper-relevant, but they should be framed as diagnostic evidence about phase effects, not as a claim that PPO -> SAC universally beats every strong baseline.

The most important result is that switch timing matters a lot. Early PPO -> SAC handoff works much better than late PPO -> SAC handoff on both Hopper and Walker2d. This supports the core project hypothesis: sequencing can help when the first phase gives the second phase a useful initialization, state distribution, or representation, while still leaving enough budget for the second phase to improve.

In this experiment, PPO appears most useful as a warm-start phase, not as the dominant training phase. SAC needs enough post-switch budget to exploit the warm start. When PPO consumes too much of the budget, the handoff is weak because SAC does not get enough time to do the main online improvement.

## Important Caveat

Before quoting exact final numbers in the paper, the current summary needs one aggregation fix.

The current summary table appears to have a grouping bug for the Hopper `0.50 self-warmup` row. That row mixes 500k-step and 1M-step handoff runs because the summarizer groups runs by algorithm, environment, policy initialization, value initialization, switch trigger, and switch fraction, but not by total training budget.

The giveaway is:

```text
Hopper 0.50 self-warmup mean_switch_step = 400000
```

For a pure 500k-step run with a 50% switch, the switch step should be:

```text
500000 * 0.50 = 250000
```

So `400000` is inconsistent with the 500k budget and strongly suggests that 1M long-horizon runs were included in the same aggregate row. That does not invalidate the experiment, but it means that the Hopper 50% `self-warmup` row should not be quoted as a clean 500k result until the table is regenerated with total budget included in the grouping key.

The clean next step is to separate rows by `total_timesteps` or by inferred final environment step. Then compare:

- 500k PPO -> SAC schedules against 500k PPO and SAC baselines,
- 1M PPO -> SAC schedules against 1M PPO and SAC baselines,
- fixed switches against adaptive switches within the same budget.

Until that fix is made, the safest interpretation is qualitative rather than exact for the contaminated row.

## Strongest Experimental Signal: Switch Timing

The clearest and most robust signal is the timing sweep.

Early PPO -> SAC at 25% is much better than later PPO -> SAC handoffs on both environments:

| Environment | 25% PPO -> SAC AUC | 50% PPO -> SAC AUC | 75% PPO -> SAC AUC |
|---|---:|---:|---:|
| Hopper-v4 | 1411 | 729 | 393 |
| Walker2d-v4 | 1281 | 598 | 349 |

The Hopper 50% `self-warmup` number is the mixed/buggy row described above, so it should not be treated as final. Even with that caveat, the broad pattern is not subtle: 25% is far stronger than 75% on both environments.

This is exactly the kind of evidence the project was looking for. It says the handoff is not just a cosmetic combination of two algorithms. The ordering and timing change performance substantially.

The result supports this mechanism:

1. PPO can produce a non-random policy and a more useful early state distribution.
2. SAC is more sample-efficient once it receives enough post-handoff budget.
3. If the switch is too late, SAC cannot recover the lost improvement opportunity.

So the result is not "PPO is better than SAC" or "SAC is better than PPO." The more interesting result is that PPO and SAC have phase-specific roles, and the amount of budget assigned to each phase matters.

## Late Handoff Is Bad

The 75% fixed handoff is consistently poor:

- Hopper 75% AUC: `393`
- Walker2d 75% AUC: `349`

This makes intuitive sense. A 75% PPO phase means SAC only receives the final quarter of the environment-step budget. If PPO has already plateaued or learned a suboptimal policy distribution, SAC gets too little time to improve. This is especially damaging because SAC's strength is online sample-efficient improvement, not just inheriting a final PPO policy and making a few small updates.

This is useful for the paper because it shows that sequencing is not automatically beneficial. Bad schedules are bad. The value of sequencing depends on placing the switch where the first algorithm has done enough useful work, but not so much that it starves the second algorithm.

That is a much stronger research story than simply reporting whether a handoff beats a baseline.

## Adaptive Switching Is Genuinely Interesting

The adaptive trigger is one of the most promising parts of the experiment.

The adaptive protocol used here is:

```text
Start with PPO.
Evaluate periodically.
After a minimum PPO phase, switch to SAC if evaluation return has not improved for a fixed patience window.
```

In these Ethan runs:

```text
switch_trigger = no-improve
patience = 3
minimum first phase = 25% of total budget
```

For 500k-step runs, the minimum first phase is 125k steps. The adaptive trigger ended up switching at about 150k steps:

- Hopper mean adaptive switch step: `150000`
- Walker2d mean adaptive switch step: `146667`

That behavior is important. The adaptive rule did not wait until the late 75% fallback point. It identified that PPO was no longer improving enough and moved budget to SAC early.

The adaptive results are strong:

| Environment | Adaptive AUC | Adaptive Final Return |
|---|---:|---:|
| Hopper-v4 | 1131 | 1987 |
| Walker2d-v4 | 1498 | 3778 |

On Walker2d, adaptive PPO -> SAC is the best row in the current table by both AUC and final return. That is very paper-relevant because it supports a scheduler-style claim:

> Fixed switch timing is sensitive, but a simple adaptive trigger can reduce that sensitivity.

This fits the project narrative better than claiming that one fixed switch fraction is universally correct. The adaptive result suggests that the optimal phase boundary can depend on the environment and training dynamics, and that a minimal plateau detector can make the schedule more robust.

The paper should be careful not to overstate this. The claim should not be:

```text
Adaptive switching is always optimal.
```

The more defensible claim is:

```text
Adaptive switching avoids badly mistimed handoffs and can preserve more budget for the algorithm that benefits from it.
```

That is already interesting.

## Value Initialization Result Is Mixed

The value-initialization ablation is informative, but it does not support a simple universal claim.

At the 50% switch, the table compares:

- `random`
- `self-warmup`
- `source-aligned`

The results are environment-dependent.

On Hopper:

- `random` AUC: `641`, final return: `1489`
- `source-aligned` AUC: `689`, final return: `2275`
- `self-warmup` AUC: `729`, final return: `1761`, but this row is contaminated by the 500k/1M grouping bug

The clean Hopper signal is that `source-aligned` improves final return over `random`, and gives a modest AUC improvement. That suggests value transfer may help, but the evidence is not overwhelming.

On Walker2d:

- `random` AUC: `586`, final return: `2227`
- `self-warmup` AUC: `598`, final return: `2090`
- `source-aligned` AUC: `487`, final return: `1546`

Here, `source-aligned` is worse than both `random` and `self-warmup`. That means the paper should not claim that source-aligned value initialization solves PPO -> SAC transfer.

The better interpretation is:

> Value initialization is environment-dependent. The more robust effects in these runs are switch timing and adaptive switching, not a universally superior value-transfer mechanism.

This is still useful. A negative or mixed mechanism result helps sharpen the paper: the handoff is not just about copying a critic or seeding Q-values. The timing and budget allocation may matter more than the particular value initialization method tested here.

## Is This What We Hypothesized?

Broadly, yes.

The main hypothesis was:

> Sequencing helps when the first phase produces a useful initialization, state distribution, or representation for the second phase, while leaving enough budget for the second phase to improve the policy.

The Ethan results support this hypothesis in three ways.

First, early handoff is much better than late handoff. That directly supports the "leaving enough budget for the second phase" part of the hypothesis.

Second, adaptive switching performs well because it switches soon after PPO stops improving. That supports the idea that the handoff should occur when the first phase has extracted most of its useful contribution, not at an arbitrary late budget fraction.

Third, the value-init ablation is mixed, which suggests that not every transfer mechanism is equally useful. That also fits the hypothesis: sequencing only helps when the first phase transfers something useful to the second phase. The result does not say every kind of transfer helps.

So the answer is not "the hypothesis is fully proven." The better answer is:

> The experiments support the phase-budget part of the hypothesis strongly, support adaptive timing as a promising scheduler mechanism, and show that value transfer is more fragile and environment-dependent.

## Is This Informative For A Paper?

Yes. This is informative for a paper, especially as a diagnostic section.

The most paper-worthy claims are:

- PPO -> SAC handoff quality is highly sensitive to switch timing.
- Early handoff can substantially outperform late handoff.
- Adaptive switching can reduce sensitivity to fixed switch fractions.
- Value-transfer mechanisms are not uniformly beneficial across environments.
- Sequencing should be understood as phase allocation under a fixed budget, not as a blanket replacement for SAC or PPO.

The strongest concise claim is:

> PPO -> SAC sequencing exhibits real phase effects: early handoff and adaptive switching can substantially improve over late handoff and PPO-only training, but transfer mechanism and environment determine whether the schedule helps.

This is much more defensible than:

> PPO -> SAC universally beats SAC.

The final paper should compare these rows directly against budget-matched SAC. If SAC remains strongest on final return in some environments, that is not fatal. The story can still be:

> SAC remains a strong standalone baseline, but phase-based schedules reveal meaningful timing effects and can improve early learning, robustness, or average performance when the handoff is well timed.

That is a real research contribution.

## Recommended Next Analysis

Before final reporting, do the following:

1. Fix aggregation by adding total budget to the grouping key.
2. Regenerate the Ethan summary table and plots.
3. Add budget-matched SAC and PPO baselines to the same summary table.
4. Separate 500k and 1M results.
5. Report both AUC and final return.
6. For adaptive switching, report the actual switch-step distribution across seeds.
7. Make the paper claim about timing sensitivity and adaptive scheduling, not universal dominance.

The experiment is already useful. The main thing it needs now is cleaner aggregation and direct baseline comparison.
