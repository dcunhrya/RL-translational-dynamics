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
SAVE_DIR="${SAVE_DIR:-results/raw/experiment_0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
TRACK="${TRACK:-false}"

ENVS=(${ENVS:-Hopper-v4 Walker2d-v4})
SEEDS=(${SEEDS:-0 1})
ALGORITHMS=(${ALGORITHMS:-sac ppo})

track_flag="--no-track"
if [[ "$TRACK" == "true" ]]; then
  track_flag="--track"
fi

export WANDB_PROJECT
export WANDB_MODE

for algorithm in "${ALGORITHMS[@]}"; do
  for env_id in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running ${algorithm} on ${env_id}, seed ${seed}, ${TOTAL_TIMESTEPS} steps"
      uv run python "src/train_${algorithm}.py" \
        --env-id "$env_id" \
        --seed "$seed" \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --eval-interval "$EVAL_INTERVAL" \
        --num-eval-episodes "$NUM_EVAL_EPISODES" \
        --save-dir "$SAVE_DIR" \
        --wandb-project "$WANDB_PROJECT" \
        "$track_flag"
    done
  done
done
