# First Experiments

This document proposes the **first experiments to run** for the RL translational dynamics project (SAC -> PPO handoff).

## Goals for the first week

1. Validate that our training stack is stable and reproducible.
2. Establish strong single-algorithm baselines (SAC, PPO).
3. Run a small fixed-ordering pilot (SAC -> PPO) to verify handoff mechanics.
4. Decide whether adaptive switching is worth prioritizing next.

## Experiment 0: Sanity + Reproducibility

Purpose: de-risk implementation before expensive sweeps.

- Environments:
  - `Hopper-v4`
  - `Walker2d-v4`
  - (optional for early debugging) a small toy continuous-control env
- Runs:
  - 2 seeds per algorithm (`SAC`, `PPO`), short horizon (e.g., 100k steps)
- Checks:
  - learning curves increase above random policy
  - no NaNs / divergence
  - deterministic behavior under fixed seeds (within normal stochastic variation)
  - logging outputs complete and consistent

Exit criteria:

- both SAC and PPO improve reward on Hopper with at least 2 seeds
- no recurring crashes in rollout or training loops

## Experiment 1: Baseline Curves (Core Reference)

Purpose: create the baseline that all switching methods must beat.

- Algorithms:
  - pure `SAC`
  - pure `PPO`
- Environments:
  - `Hopper-v4`, `Walker2d-v4`
- Budget:
  - fixed interaction budget (e.g., 1M env steps)
  - also track total gradient updates for later compute-normalized analysis
- Seeds:
  - at least 5 seeds per method/environment
- Metrics:
  - episodic return vs env steps
  - return vs gradient updates
  - variance across seeds
  - time-to-threshold (sample efficiency)

Deliverable:

- baseline plots with confidence bands and a results table (mean +/- std)

## Experiment 2: Fixed Handoff Pilot (SAC -> PPO)

Purpose: test whether simple algorithm sequencing gives gains before adaptive complexity.

- Pipelines:
  - `SAC(25%) -> PPO(75%)`
  - `SAC(50%) -> PPO(50%)`
  - `SAC(75%) -> PPO(25%)`
- Compare against:
  - pure `SAC`
  - pure `PPO`
- Handoff initialization:
  - initialize PPO actor from SAC policy weights
  - reinitialize PPO optimizer state (do not carry Adam moments by default)
  - evaluate whether value function transfer helps or hurts (ablation if time allows)
- Seeds:
  - 3 seeds for pilot, then expand to 5 for promising settings

Success criteria:

- at least one fixed handoff setting improves either:
  - area under learning curve (AUC), or
  - final return at equal budget

## Experiment 3: Adaptive Trigger Prototype

Purpose: evaluate one internal metric trigger (instead of hand-picked switch time).

Proposed first trigger (from proposal):

- switch when SAC policy KL drift falls below threshold for `N` evaluation windows

Practicalized trigger:

- compute KL between consecutive SAC policy snapshots on a fixed state buffer
- if moving-average KL < `tau` for `k` checks, switch to PPO

Initial sweep:

- `tau` in a small grid (e.g., 0.01, 0.02, 0.05)
- `k` in {2, 3}

Compare to best fixed handoff from Experiment 2.

## Experiment 4: Compute-Constrained Comparison

Purpose: test the core project claim under fixed compute.

- Equalize by one compute proxy (pick one and stick to it):
  - total gradient updates, or
  - estimated FLOPs from (batch size x updates x model cost)
- Methods:
  - pure SAC
  - pure PPO
  - best fixed handoff
  - best adaptive handoff
- Output:
  - return vs compute budget curve
  - final return at matched compute points

## Recommended execution order

1. Experiment 0 (sanity)
2. Experiment 1 (baselines)
3. Experiment 2 (fixed handoff)
4. Experiment 3 (adaptive)
5. Experiment 4 (compute-constrained final comparison)

## Minimal logging schema (do this early)

For each run, log:

- `algorithm`, `env`, `seed`
- `env_steps`, `gradient_updates`, `wall_clock_sec`
- `eval_return_mean`, `eval_return_std`
- handoff metadata: `switched`, `switch_step`, `switch_reason`, `trigger_value`

This keeps later analysis simple and avoids rerunning expensive experiments.

## Risks to watch immediately

- PPO instability after weight transfer from SAC (distribution mismatch).
- Unfair comparisons if compute accounting differs by algorithm.
- Overfitting to one environment (run both Hopper and Walker2d before conclusions).

## Definition of "good first result"

A convincing early result is:

- one SAC -> PPO schedule that beats both pure baselines on at least one environment at equal budget,
- while maintaining similar or lower variance across seeds.
