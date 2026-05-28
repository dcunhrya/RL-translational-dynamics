# Ethan Task Experiment Notes

## Owner

Ethan

## Scope

This directory tracks the PPO -> SAC value-ablation, timing-sweep, adaptive-trigger, and long-horizon runs assigned in `tasks/ethan-task.md`.

## Implemented Arms

- **PPO -> SAC value ablation:** `value=random`, `value=self-warmup`, `value=source-aligned` with `policy=distill`.
- **Timing sweep:** fixed 25% and 75% switches. The 50% point is supplied by the value-ablation arm.
- **Adaptive trigger:** `switch-trigger=no-improve`, with patience and minimum first-phase budget logged.
- **Long horizon:** pure PPO at 1M Hopper plus one PPO -> SAC 1M Hopper arm.

## Smoke Tests

Run before launching the full sweep:

```bash
uv run python src/RL-translational-dynamics/exp0/train_reverse_handoff.py \
  --env-id Hopper-v4 \
  --seed 0 \
  --total-timesteps 50000 \
  --switch-fraction 0.5 \
  --policy-init distill \
  --value-init random \
  --save-dir results/raw/ethan_smoke \
  --no-track
```

Repeat with:

```bash
--value-init self-warmup
--value-init source-aligned
```

Adaptive smoke:

```bash
uv run python src/RL-translational-dynamics/exp0/train_reverse_handoff.py \
  --env-id Hopper-v4 \
  --seed 0 \
  --total-timesteps 50000 \
  --switch-fraction 0.75 \
  --switch-trigger no-improve \
  --patience 1 \
  --min-first-phase 10000 \
  --policy-init distill \
  --value-init self-warmup \
  --eval-interval 5000 \
  --save-dir results/raw/ethan_smoke \
  --no-track
```

## Full Modal Launch

Use detached Modal execution so the run continues after the local machine disconnects:

```bash
modal run --detach src/RL-translational-dynamics/modal/experiment_ethan_modal.py
```

To reduce scope:

```bash
modal run --detach src/RL-translational-dynamics/modal/experiment_ethan_modal.py --no-include-long
```

## Results Reporting

After Modal finishes, pull the raw outputs from the shared Modal volume:

```bash
modal volume get herschethan /raw/ethan_task results/raw/ethan_task --force
modal volume get herschethan /raw/ethan_task_long_horizon_reverse_handoff results/raw/ethan_task_long_horizon_reverse_handoff --force
modal volume get herschethan /raw/ethan_task_long_horizon_ppo results/raw/ethan_task_long_horizon_ppo --force
```

Then generate plots, CSVs, and the final notes:

```bash
uv run python src/RL-translational-dynamics/exp0/summarize_ethan_task.py \
  --results-dir results/raw/ethan_task \
  --extra-results-dir results/raw/ethan_task_long_horizon_reverse_handoff \
  --extra-results-dir results/raw/ethan_task_long_horizon_ppo \
  --output-dir results/processed/ethan_task \
  --notes-dir experiments/ethan_task
```

The generated report is written to `experiments/ethan_task/results.md`.
