# Ryan Headline Results Explanation

This file explains the headline plots in `experiments/ryan_task/figures/headline/` and how to interpret the results. The bundle includes all `53/53` completed Ryan runs: SAC and PPO baselines, SAC -> PPO value-initialization ablations on Hopper-v4 and Walker2d-v4, and the Hopper-v4 SAC 1M long-horizon check.

## Overall Takeaway

SAC is the strongest method in this slice by both sample efficiency and final return on Hopper-v4 and Walker2d-v4. The SAC -> PPO handoff arms generally outperform pure PPO, but they do not recover the performance of pure SAC under this 500k-step budget.

The value-initialization ablation is mixed. On Hopper-v4, both `self-warmup` and `source-aligned` value initialization are worse than random value initialization on mean normalized AUC, though their paired bootstrap confidence intervals include zero. On Walker2d-v4, both value-transfer variants improve over random value initialization on mean normalized AUC, especially `source-aligned`, but those confidence intervals also include zero. The safest conclusion is that value initialization affects SAC -> PPO behavior, but these 5-seed results do not support a clean, environment-independent value-transfer win.

## What The Value-Initialization Ablation Tests

In the SAC -> PPO experiments, the first 250k steps train SAC. At the handoff point, the actor policy is transferred into PPO by policy distillation, so the PPO policy starts from behavior meant to imitate the SAC actor. The ablation asks a narrower question: once the policy transfer is fixed, how should PPO's value function be initialized?

PPO uses a value function to estimate how good states are and to compute advantages for policy updates. If that value function is badly initialized at the handoff, PPO can make poor updates even if the policy starts from a useful SAC-like behavior. The value-initialization ablation isolates whether better critic/value initialization helps PPO preserve or improve the SAC warm start.

The three value-initialization variants are:

- `random`: PPO's value function starts from its normal random initialization. The policy is still distilled from SAC, but the value network does not inherit SAC information. This is the baseline for the C1 comparison.
- `self-warmup`: PPO initializes its value function using PPO's own rollout/value-learning procedure before full PPO policy updates begin. This gives PPO a short phase to fit values under its current distilled policy before it starts changing the policy aggressively.
- `source-aligned`: PPO's value function is explicitly regressed toward SAC-derived targets, using SAC's learned Q-functions over replay states and actions from the distilled PPO policy. This is the most direct attempt to transfer SAC's critic knowledge into PPO's value function.

In short: `random` asks "what if only the policy transfers?", `self-warmup` asks "what if PPO calibrates its own value estimates first?", and `source-aligned` asks "what if PPO's value function is initialized from SAC's critic knowledge?"

## `learning_curves_500k`

This plot shows evaluation return over environment steps for the 500k-step runs. Each line is the mean over 5 seeds, and the shaded region is the standard error across seeds. The vertical dashed line at 250k marks the SAC -> PPO handoff point for the transfer arms.

What it means:

- SAC learns much more effectively than the other methods in both environments.
- PPO remains low throughout training, which makes it a weak standalone baseline at this budget.
- The SAC -> PPO arms benefit from the SAC first phase, so they are generally better than PPO before and around the handoff.
- After the handoff, switching from SAC to PPO causes the handoff methods to fall far behind continued SAC training.

Interpretation: the first phase is useful, but the second-phase PPO refinement is not strong enough here to preserve or improve SAC's trajectory. This supports the broader sequencing story that algorithm order and handoff mechanism matter, rather than showing that SAC -> PPO is universally beneficial.

## `auc_summary`

This plot summarizes area under the evaluation-return curve for each method and environment. AUC measures sample efficiency: higher AUC means the method accumulated more reward across the whole training budget, not just at the final checkpoint.

Key values:

- Hopper-v4 normalized AUC: SAC `2293.641`, PPO `324.941`, SAC -> PPO random `1219.303`, self-warmup `1145.765`, source-aligned `1131.139`.
- Walker2d-v4 normalized AUC: SAC `1642.230`, PPO `385.257`, SAC -> PPO random `602.467`, self-warmup `623.689`, source-aligned `718.320`.

What it means:

- SAC has the best AUC in both environments, so it is the best sample-efficiency baseline in Ryan's slice.
- SAC -> PPO improves substantially over pure PPO on AUC, especially on Hopper-v4.
- Among SAC -> PPO value arms, random value initialization is best on Hopper-v4, while source-aligned value initialization is best on Walker2d-v4.

Interpretation: SAC's off-policy online improvement dominates the matched-budget comparison. The handoff arms still reveal a phase effect because they are much stronger than PPO, but they do not beat the strongest single-algorithm baseline.

## `final_return_summary`

This plot compares final evaluation return at the end of the training budget. Final return answers a different question than AUC: it measures end-of-run performance, not how quickly the method learned.

Key values:

- Hopper-v4 final return: SAC `3014.4`, PPO `384.3`, SAC -> PPO random `403.5`, self-warmup `439.2`, source-aligned `384.9`.
- Walker2d-v4 final return: SAC `3310.4`, PPO `426.0`, SAC -> PPO random `442.5`, self-warmup `464.6`, source-aligned `646.6`.

What it means:

- SAC is also the best final-return method in both environments.
- SAC -> PPO final returns are close to PPO on Hopper-v4, suggesting the PPO phase does not preserve SAC's earlier gains there.
- On Walker2d-v4, source-aligned value initialization has the best SAC -> PPO final return, but it is still far below pure SAC.

Interpretation: if the objective is only end-of-budget performance, these results favor SAC. The SAC -> PPO schedules are more useful as mechanism probes than as the top-performing final method.

## `value_init_auc_deltas`

This plot isolates the C1 question: with policy distillation fixed, does the PPO value initialization change SAC -> PPO outcomes? It shows paired-seed normalized AUC differences relative to the SAC -> PPO random-value baseline.

Key values:

- Hopper-v4 self-warmup vs random: `-73.54`, 95% CI `[-165.26, 18.18]`.
- Hopper-v4 source-aligned vs random: `-88.16`, 95% CI `[-188.38, 12.05]`.
- Walker2d-v4 self-warmup vs random: `21.22`, 95% CI `[-7.26, 54.91]`.
- Walker2d-v4 source-aligned vs random: `115.85`, 95% CI `[-25.31, 278.73]`.

What it means:

- Negative values mean the value-transfer variant performed worse than random value initialization.
- Positive values mean the value-transfer variant performed better than random value initialization.
- All four confidence intervals include zero, so none of these differences should be presented as statistically decisive from this run set alone.

Interpretation: value initialization appears environment-sensitive. Hopper-v4 does not benefit from the tested value-transfer mechanisms, while Walker2d-v4 trends positive, especially for source-aligned value warm-up. This is a defensible mixed result: value transfer is not universally beneficial, but it can change handoff dynamics in meaningful ways.

## `hopper_sac_long_horizon`

This plot compares Hopper-v4 SAC at 500k steps and 1M steps for seeds 0..2. It checks whether the 500k SAC baseline had already saturated or whether more SAC training still improves performance.

What it means:

- The 1M SAC curve tests whether SAC continues to make use of additional environment steps after the 500k budget.
- If the 1M curve keeps improving past 500k, then SAC's advantage is not just an early-learning artifact.
- This figure contextualizes the handoff result: switching away from SAC at 250k may be costly if SAC would have continued improving strongly afterward.

Interpretation: the long-horizon check helps explain why SAC -> PPO struggles to beat SAC. If SAC remains productive after the handoff point, replacing it with PPO sacrifices a strong continuing learner.

## How To Use These Results In The Report

The strongest claim is:

> SAC is the strongest single-algorithm baseline in Ryan's experiments, but SAC -> PPO still improves over pure PPO and exposes meaningful environment-dependent effects of value initialization at the handoff.

The value-ablation claim should be cautious:

> With policy distillation fixed, value initialization changes SAC -> PPO outcomes, but the direction is environment-dependent and the paired bootstrap CIs include zero in this 5-seed run.

Avoid claiming:

- SAC -> PPO beats SAC.
- Source-aligned value warm-up is universally better.
- PPO reliably stabilizes SAC after handoff.

The better framing is that Ryan's slice provides a negative or mixed mechanism result: value transfer alone does not explain a robust SAC -> PPO improvement under the matched 500k budget, and the choice of second-phase algorithm remains central.
