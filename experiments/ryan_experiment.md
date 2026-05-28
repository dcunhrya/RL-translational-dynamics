# Ryan Modal Experiment

Ryan owns the Tier 0 SAC -> PPO explanatory spine and matched SAC/PPO baselines.
All training runs for this slice are launched on Modal.

## Arms

- `sac`: pure SAC baseline.
- `ppo`: pure PPO baseline.
- `sac_to_ppo`, `policy_init=distill`, `value_init=random`: fixed 50% handoff, policy distillation only.
- `sac_to_ppo`, `policy_init=distill`, `value_init=self-warmup`: fixed 50% handoff, current PPO rollout value warm-up.
- `sac_to_ppo`, `policy_init=distill`, `value_init=source-aligned`: fixed 50% handoff, PPO value regressed to SAC Q targets over replay states.

## Budgets

- Main baselines: `Hopper-v4`, `Walker2d-v4`, seeds `0..4`, `500000` env steps.
- Main handoff ablation: `Hopper-v4`, `Walker2d-v4`, seeds `0..4`, `500000` env steps, switch at `250000`.
- Long-horizon check: pure SAC, `Hopper-v4`, seeds `0..2`, `1000000` env steps.
- Smoke gate: `Hopper-v4` and `Walker2d-v4`, seed `0`, `50000` env steps for every Ryan arm.

## Modal Commands

Smoke gate:

```shell
modal run src/RL-translational-dynamics/modal/ryan_modal.py --mode smoke --manifest-path experiments/ryan_modal_smoke_manifest.json
```

Full detached launch:

```shell
modal run src/RL-translational-dynamics/modal/ryan_modal.py --mode full --manifest-path experiments/ryan_modal_manifest.json
```

Set `RYAN_MODAL_GPU=L4` or `RYAN_MODAL_GPU=A10G` before launch to choose the GPU class.
The launcher defaults to `L4`.

## Smoke Gate

The full sweep should not launch until smoke jobs complete and show:

- no crash or non-finite metric;
- at least one eval row per run;
- correct `algorithm`, `policy_init`, `value_init`, `arm_name`, `phase`, and switch metadata;
- required diagnostics for handoff arms: distillation loss, policy-retention action MSE/KL proxy, PPO explained variance, advantage stats, value warm-up losses where applicable, and pre/post-handoff eval rows.

## Result Paths

- Raw Modal volume path: `/root/results/raw/ryan_experiment`.
- Local fetch target: `results/raw/ryan_experiment`.
- Processed outputs: `results/processed/ryan_experiment`.
- W&B project: `rl-translational-dynamics`.
- W&B groups are prefixed with `ryan_smoke`, `ryan_full`, or `ryan_long_horizon`.

## Analysis After Reinitialization

After Modal jobs complete, fetch raw metrics and run the Ryan summary with value arms required:

```shell
python src/RL-translational-dynamics/exp2/summarize_experiment_2.py \
  --results-dir results/raw/ryan_experiment \
  --output-dir results/processed/ryan_experiment \
  --switch-fractions 0.5 \
  --value-inits random self-warmup source-aligned
```

Primary metric is AUC over eval returns. Final return, worst seed, collapse frequency,
and seed standard error are secondary diagnostics.
