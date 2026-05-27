# Critiques — Transfer Mechanism Design Spec

**Target spec:** [`2026-05-26-rl-sequencing-transfer-mechanism-design.md`](./2026-05-26-rl-sequencing-transfer-mechanism-design.md)  
**Date:** 2026-05-26

---

## Issue: C1 ablation confounded by distillation

**Severity:** High — can invalidate the C1 claim if not resolved before implementation.

**Claim under test (C1):** For SAC↔PPO handoffs, transferring **policy alone** (cold value init) underperforms transferring **policy + value**.

**Problem:** The handoff pipeline transfers more than policy and value. At switch time, `train_handoff.py` / `train_reverse_handoff.py` also run **500-step actor distillation** (MSE between source and target actions on replay states). The spec’s variants 3–4 do not state whether distillation is on, off, or part of “policy transfer.”

**What the code actually does (SAC→PPO):**

1. PPO actor starts **random init** (no weight copy; SAC/PPO architectures differ).
2. **500 distillation steps** align PPO actor to SAC deterministic actions on replay observations.
3. **2 critic-only PPO updates** (value warm-up).
4. Full PPO training.

So “policy transfer” in practice = **behavioral distillation**, not checkpoint loading.

**Why this confounds C1:**

| If both variant 3 and 4 keep distillation | You test “distilled policy + cold value” vs “distilled policy + warm value”—not “policy only vs policy+value.” |
| If variant 3 removes value warm-up only | You test whether value alignment helps **given distillation**—a smaller, different claim than C1. |
| Distillation encodes implicit value info | SAC actions on replay states depend on Q and entropy; mimicking them may reduce how much explicit value transfer matters. |
| “Policy only” is undefined | Could mean: (A) weight load, no distill, no value; (B) distill only, no value; (C) distill + cold value vs distill + warm value. Spec does not choose. |
| No reference arm | Missing row for current Experiment 2 protocol (distill + minimal value warm-up), so ablation results cannot be compared to existing handoff. |

**Risk to narrative:** Observing a gap between variants 3 and 4 supports “value warm-up helps after distillation,” not “policy is necessary but not sufficient.” Observing no gap rejects C1 while distillation may have made both arms equally good—confounding mechanism interpretation.

**Resolution options (pick one before implementation):**

1. **Rewrite C1** to match what is testable: *“Given fixed distillation at handoff, does explicit value alignment further improve performance?”*
2. **Expand the ablation matrix** to treat distillation as its own factor, e.g. `{none, distill-only, distill+value, full current protocol}`.
3. **Define `--transfer` explicitly**, e.g. `policy | distill | value | distill+value | full`, and map each matrix row to exact handoff steps (which optimizers reset, which phases run, gradient budget counted how).

**Open decisions for spec author:**

- Is distillation **always on** in variants 3–4?
- Is distillation classified as part of **policy transfer** or a **separate substrate**?
- Which arm is the **reference** for “current best handoff” (Experiment 2)?

---

## Other issues (summary)

### Architecture mismatch blocks BC/IQL transfer

The spec assumes a shared `MLPGaussian` checkpoint loads into SAC and PPO. That class does not exist in the codebase. SAC uses a 256×256 ReLU actor; PPO uses a 64×64 Tanh actor with a different log-std parameterization. BC/IQL checkpoints cannot load directly without a unification refactor first.

### `--transfer value` in harness but not in matrix

The implementation scope lists `policy | value | policy+value | policy+value+replay`, but the matrix never runs **value-only** transfer. Without it, you cannot show value is necessary—only that adding value on top of policy might help.

### Value transfer recipe differs from existing code

The spec proposes supervised Q↔V alignment (up to 5k updates). Current code uses 2 PPO critic-only rollout updates (SAC→PPO) or 1000 Bellman critic warm-up steps on replay (PPO→SAC)—not alignment to the source value function.

### C2 under-controlled

Missing baselines: BC-only eval, random-init online SAC/PPO at matched budget, and internal post-handoff metrics (policy KL from BC, explained variance, advantage stats) to support the “PPO drifts without warm value” story.

### C3 gate logic

IQL is gated on C1 being “clean,” but C1 (online↔online) and C3 (offline→online) test different things. A strong C2/C3 story is possible even if C1 is null.

### Statistical power underspecified

Hopper SAC variance in pilots is very high (e.g. ±795–1135). No pre-specified primary metric, effect-size threshold, or multiple-comparison plan for the Day 2 IQL go/no-go.

### Horizon split

Main matrix at 500k steps; headline comparison at 1M on Hopper only. Mechanism claims from 500k may not support the asymptotic headline question.

### Duplicate compute with in-flight Experiment 2

Experiment 2’s 1M-step grid is already planned/in progress. Re-running pure SAC/PPO baselines at 500k in a new matrix wastes runs; reuse Experiment 2 data where possible.

### D4RL not in dependencies

`pyproject.toml` has no `d4rl`. BC/IQL setup may consume Day 1 morning beyond the spec’s estimate.

### Timeline optimism

As written (100 + 25 runs, BC, new value alignment, IQL conditional, analysis, write-up in ~2 days) is aggressive. Completable if C3 and the 1M headline are dropped and seeds are reduced to 3 for non-headline variants.

### Recommended scope cuts

1. C1 only, both directions, 3 seeds, 500k, Hopper + Walker2d.
2. Explicit ablation: `{policy only, policy+distill, policy+distill+value, full current handoff}`.
3. Reuse Experiment 2 1M data for asymptotics—skip Experiment 5.
4. BC→SAC vs BC→PPO as lightweight C2 if D4RL installs cleanly.
5. Drop IQL unless BC arms show a clear gap by Day 1 evening.
6. Drop replay from the headline claim unless included in the matrix.
