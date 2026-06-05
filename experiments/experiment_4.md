# Experiment 4: Offline-Assisted Transfer and Compute-Aware Schedules

Experiment 4 implements Abhinav's offline-assisted pipeline and shared run infrastructure.

## Implemented Arms

- `bc`: behavior cloning on D4RL expert demonstrations.
- `bc_to_sac`: BC policy distilled into SAC before online fine-tuning.
- `bc_to_ppo`: BC policy distilled into PPO before online fine-tuning.
- `bc_to_sac_to_ppo`: BC policy distilled into the SAC first phase, followed by fixed SAC -> PPO handoff.
- `bc_anchor_sac`: BC -> SAC with periodic anchoring to the BC policy during SAC updates.
- `awac`: gated value-carrying offline source.
- `awac_to_sac` / `awac_to_ppo`: AWAC actor distilled into the online target.
- `easy_sac` / `easy_sac_to_sac`: stretch curriculum-style starter policy from forgiving-termination MuJoCo.
- starter compatibility diagnostic: checks whether one policy checkpoint can legally transfer to other MuJoCo tasks before running exploratory general-starter pilots.

## Diagnostics

Every arm logs:

- run identity: algorithm, environment, seed, source policy, value init/source;
- online progress: environment steps, gradient updates, wall-clock seconds;
- offline accounting: dataset id, offline dataset size, offline updates where applicable;
- transfer markers: `distill`, `bc_anchor`, `sac`, `ppo`, `ppo_value_warmup`, `source_value_warmup`;
- internal metrics exposed by the underlying SAC/PPO/AWAC loops.

`summarize_experiment_4.py` produces:

- per-arm final return and normalized AUC with bootstrap confidence intervals;
- phase-marked learning curves;
- policy-retention/distillation plots;
- value-quality plots;
- handoff-transient tables for runs with a switch step.

## Execution

See `experiments/abhinav_task/README.md` for exact local smoke and Modal commands.
