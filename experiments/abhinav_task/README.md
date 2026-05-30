# Abhinav Task: Offline-Assisted Sequencing

This experiment family covers Abhinav's assigned slice from `tasks/abhinav-task.md`:

- BC pretraining from cached D4RL expert demonstrations.
- BC -> SAC, BC -> PPO, and BC -> SAC -> PPO online transfer arms.
- Interleaved BC anchoring during SAC via `--bc-anchor-interval`.
- Gated AWAC offline pretraining and AWAC -> SAC/PPO transfer.
- Stretch easy-environment SAC pretraining with forgiving termination, then Easy SAC -> real SAC transfer.
- General starter policy compatibility diagnostics for cross-environment transfer.
- Shared Experiment 4 Modal orchestration and analysis.

Offline-assisted runs log offline dataset size and offline updates separately from online environment steps. They should not be described as env-step matched against pure online SAC/PPO without that caveat.

## Local Smoke Commands

```shell
python src/RL-translational-dynamics/exp4/train_bc.py \
  --env-id Hopper-v4 \
  --total-updates 2 \
  --eval-interval 1 \
  --num-eval-episodes 1 \
  --max-demo-samples 1024 \
  --save-dir results/raw/abhinav_task_smoke/bc_pretrain \
  --dataset-cache-dir results/datasets/d4rl_expert_smoke \
  --no-cuda
```

Use the emitted `bc_policy.pt` with:

```shell
python src/RL-translational-dynamics/exp0/train_sac.py \
  --env-id Hopper-v4 \
  --total-timesteps 20 \
  --eval-interval 10 \
  --num-eval-episodes 1 \
  --bc-policy-path <path-to-bc_policy.pt> \
  --offline-policy-source bc \
  --bc-distill-steps 2 \
  --bc-anchor-interval 10 \
  --bc-anchor-steps 1 \
  --no-cuda
```

```shell
python src/RL-translational-dynamics/exp0/train_ppo.py \
  --env-id Hopper-v4 \
  --total-timesteps 20 \
  --num-steps 10 \
  --eval-interval 10 \
  --num-eval-episodes 1 \
  --bc-policy-path <path-to-bc_policy.pt> \
  --offline-policy-source bc \
  --bc-distill-steps 2 \
  --no-cuda
```

```shell
python src/RL-translational-dynamics/exp2/train_handoff.py \
  --env-id Hopper-v4 \
  --total-timesteps 20 \
  --switch-fraction 0.5 \
  --sac-learning-starts 100 \
  --ppo-num-steps 5 \
  --eval-interval 10 \
  --num-eval-episodes 1 \
  --bc-policy-path <path-to-bc_policy.pt> \
  --offline-policy-source bc \
  --bc-init-distill-steps 2 \
  --distill-steps 2 \
  --value-warmup-updates 0 \
  --no-cuda
```

Summarize smoke artifacts:

```shell
python src/RL-translational-dynamics/exp4/summarize_experiment_4.py \
  --results-dir results/raw/abhinav_task_smoke \
  --output-dir results/processed/abhinav_task_smoke \
  --notes-path experiments/abhinav_task/smoke_results.md
```

Check whether a starter checkpoint is cross-environment compatible:

```shell
python src/RL-translational-dynamics/exp4/check_starter_compatibility.py \
  --checkpoint <path-to-policy.pt> \
  --output results/processed/abhinav_task/starter_compatibility.json
```

## Modal Launches

Core Tier 1, after BC pretraining:

```shell
modal run src/RL-translational-dynamics/modal/experiment_4_modal.py \
  --mode core \
  --total-timesteps 500000
```

Interleaved BC K sweep:

```shell
modal run src/RL-translational-dynamics/modal/experiment_4_modal.py \
  --mode interleaved \
  --total-timesteps 500000
```

Long-horizon Hopper check:

```shell
modal run src/RL-translational-dynamics/modal/experiment_4_modal.py \
  --mode long \
  --long-timesteps 1000000
```

Gated AWAC Tier 2:

```shell
modal run src/RL-translational-dynamics/modal/experiment_4_modal.py \
  --mode tier2 \
  --total-timesteps 500000
```

Stretch easy-environment transfer:

```shell
modal run src/RL-translational-dynamics/modal/experiment_4_modal.py \
  --mode easy \
  --total-timesteps 500000
```

All modes:

```shell
modal run src/RL-translational-dynamics/modal/experiment_4_modal.py \
  --mode all \
  --total-timesteps 500000 \
  --long-timesteps 1000000
```

Summarize after jobs finish:

```shell
modal run src/RL-translational-dynamics/modal/experiment_4_modal.py --mode summarize
```

Local summarization after fetching Modal results:

```shell
python src/RL-translational-dynamics/exp4/summarize_experiment_4.py \
  --results-dir results/raw/abhinav_task \
  --output-dir results/processed/abhinav_task \
  --notes-path experiments/abhinav_task/results.md
```
