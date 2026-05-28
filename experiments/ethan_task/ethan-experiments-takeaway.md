# Ethan Experiments Takeaway

## What These Runs Test

These experiments test whether PPO -> SAC handoffs are useful because PPO provides a warm-start phase and SAC provides the main online-improvement phase. The key variables are:

- when the switch happens,
- whether the SAC critic is initialized with useful value information,
- whether a simple adaptive trigger can choose a better switch point than a fixed fraction.

The adaptive rule used here is a protocol we define: begin with PPO, evaluate periodically, and switch to SAC once evaluation return has stopped improving for a fixed patience window after a minimum PPO budget. In these runs, the trigger is `no-improve`, with patience `3` and minimum first phase `25%` of the total budget.

## Main Takeaway

The strongest signal is timing. Early PPO -> SAC handoff is much better than late PPO -> SAC handoff on both Hopper and Walker2d. A 25% PPO / 75% SAC schedule is far stronger than a 75% PPO / 25% SAC schedule by AUC and final return.

This supports the project hypothesis: sequencing helps when the first phase gives the second phase a useful initialization while still leaving enough budget for the second phase to improve. PPO can be useful as a warm-start phase, but spending too much of the budget in PPO hurts because SAC does not get enough time to exploit the handoff.

## Adaptive Switching

The adaptive `no-improve` trigger is one of the most interesting results. It switches early, around 150k steps in 500k-step runs, and performs strongly:

- On Hopper, adaptive PPO -> SAC has much higher AUC than the late 75% fixed switch.
- On Walker2d, adaptive PPO -> SAC is the best row in the table by both AUC and final return.

This is paper-relevant because it suggests that fixed switch timing is a real sensitivity, and a simple reproducible trigger can reduce that sensitivity. The claim should not be that adaptive switching is universally optimal; the defensible claim is that adaptive switching can avoid badly mistimed handoffs and preserve most of the budget for the algorithm that benefits from it.

## Value Initialization

The value-transfer ablation is mixed. At the 50% switch:

- `source-aligned` improves Hopper final return over `random`.
- `source-aligned` is worse on Walker2d.
- `self-warmup` is not consistently dominant.

This suggests that the robust effect is not simply "better value initialization fixes PPO -> SAC." The cleaner interpretation is that policy transfer and switch timing are currently more reliable than value transfer, while value initialization remains environment-dependent.

## Paper Framing

This experiment is informative for a paper, but it should be framed diagnostically rather than as a leaderboard win.

Good claim:

> PPO -> SAC sequencing exhibits real phase effects: early handoff and adaptive switching can substantially improve over late handoff and PPO-only training, but transfer mechanism and environment determine whether the schedule helps.

Avoid:

> PPO -> SAC universally beats SAC.

The useful contribution is evidence for phase-specific roles: PPO can provide an initial policy/state distribution, SAC can use most of the remaining budget for sample-efficient improvement, and simple adaptive triggers can reduce sensitivity to fixed switch fractions.

## Caveat Before Final Reporting

The current summary should be cleaned before final paper use. The summarizer groups runs by algorithm, environment, transfer settings, trigger, and switch fraction, but not by total budget. As a result, the Hopper 50% `self-warmup` row appears to mix 500k and 1M handoff runs. The giveaway is a mean switch step of `400000`, which is inconsistent with a pure 500k run at a 50% switch.

Before quoting final numbers, regenerate the table with total budget included in the grouping key, and compare the 500k schedules against budget-matched SAC and PPO baselines.
