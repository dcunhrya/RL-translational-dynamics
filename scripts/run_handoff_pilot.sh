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
SAVE_DIR="${SAVE_DIR:-results/raw/experiment_2_fixed_handoff}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
TRACK="${TRACK:-false}"
VALUE_WARMUP_UPDATES="${VALUE_WARMUP_UPDATES:-2}"
DISTILL_STEPS="${DISTILL_STEPS:-500}"

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
  for env_id in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running SAC->PPO on ${env_id}, seed ${seed}, switch_fraction ${switch_fraction}, ${TOTAL_TIMESTEPS} steps"
      uv run python "src/RL-translational-dynamics/exp0/train_handoff.py" \
        --env-id "$env_id" \
        --seed "$seed" \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --switch-fraction "$switch_fraction" \
        --eval-interval "$EVAL_INTERVAL" \
        --num-eval-episodes "$NUM_EVAL_EPISODES" \
        --save-dir "$SAVE_DIR" \
        --wandb-project "$WANDB_PROJECT" \
        --value-warmup-updates "$VALUE_WARMUP_UPDATES" \
        --distill-steps "$DISTILL_STEPS" \
        "$track_flag"
    done
  done
done
