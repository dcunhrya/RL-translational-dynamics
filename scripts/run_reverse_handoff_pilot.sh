#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

WANDB_PROJECT="${WANDB_PROJECT:-rl-translational-dynamics}"
WANDB_MODE="${WANDB_MODE:-online}"
SAVE_DIR="${SAVE_DIR:-results/raw/experiment_3_reverse_handoff}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
TRACK="${TRACK:-false}"
DISTILL_STEPS="${DISTILL_STEPS:-500}"
SAC_CRITIC_WARMUP_UPDATES="${SAC_CRITIC_WARMUP_UPDATES:-1000}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-experiment_3_reverse_handoff}"

ENVS=(${ENVS:-Hopper-v4 Walker2d-v4})
SEEDS=(${SEEDS:-0 1 2})
SWITCH_FRACTIONS=(${SWITCH_FRACTIONS:-0.25 0.5 0.75})

track_flag="--no-track"
if [[ "$TRACK" == "true" ]]; then
  track_flag="--track"
fi

export WANDB_PROJECT
export WANDB_MODE

for switch_fraction in "${SWITCH_FRACTIONS[@]}"; do
  switch_pct="$(python - <<PY
switch_fraction = float("${switch_fraction}")
print(int(round(switch_fraction * 100)))
PY
)"
  for env_id in "${ENVS[@]}"; do
    wandb_group="${WANDB_GROUP_PREFIX}__${env_id}__switch_${switch_pct}pct"
    for seed in "${SEEDS[@]}"; do
      echo "Running PPO->SAC on ${env_id}, seed ${seed}, switch_fraction ${switch_fraction}, group ${wandb_group}, ${TOTAL_TIMESTEPS} steps"
      uv run python "src/RL-translational-dynamics/exp0/train_reverse_handoff.py" \
        --env-id "$env_id" \
        --seed "$seed" \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --switch-fraction "$switch_fraction" \
        --eval-interval "$EVAL_INTERVAL" \
        --num-eval-episodes "$NUM_EVAL_EPISODES" \
        --save-dir "$SAVE_DIR" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-group "$wandb_group" \
        --distill-steps "$DISTILL_STEPS" \
        --sac-critic-warmup-updates "$SAC_CRITIC_WARMUP_UPDATES" \
        "$track_flag"
    done
  done
done
