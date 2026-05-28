#!/usr/bin/env bash
# Experiment 2 launcher: local sequential training (default) or detached Modal grid.
#
# Modal (detached grid, 50 jobs: 30 handoff + 20 baselines):
#   LAUNCHER=modal ./scripts/run_experiment_2.sh
#
# After Modal jobs finish, download raw metrics then summarize:
#   ./scripts/fetch_experiment_2_results.sh
#   uv run python src/RL-translational-dynamics/exp2/summarize_experiment_2.py
#
# Local full grid (blocking on this machine):
#   LAUNCHER=local ./scripts/run_experiment_2.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

LAUNCHER="${LAUNCHER:-modal}"
WANDB_PROJECT="${WANDB_PROJECT:-rl-translational-dynamics}"
WANDB_MODE="${WANDB_MODE:-online}"
SAVE_DIR="${SAVE_DIR:-results/raw/experiment_2}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
TRACK="${TRACK:-false}"
VALUE_WARMUP_UPDATES="${VALUE_WARMUP_UPDATES:-2}"
DISTILL_STEPS="${DISTILL_STEPS:-500}"
RUN_HANDOFF="${RUN_HANDOFF:-true}"
RUN_BASELINES="${RUN_BASELINES:-true}"
MODAL_RESULTS_VOLUME="${MODAL_RESULTS_VOLUME:-rl-translational-dynamics-results}"

ENVS=(${ENVS:-Hopper-v4 Walker2d-v4})
SEEDS=(${SEEDS:-0 1 2 3 4})
SWITCH_FRACTIONS=(${SWITCH_FRACTIONS:-0.25 0.5 0.75})

export WANDB_PROJECT
export WANDB_MODE
export MODAL_RESULTS_VOLUME
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::DeprecationWarning}"

if [[ "$LAUNCHER" == "modal" ]]; then
  track_flag="--no-track"
  if [[ "$TRACK" == "true" ]]; then
    track_flag="--track"
  fi

  handoff_flag="--enable-handoff"
  if [[ "$RUN_HANDOFF" != "true" ]]; then
    handoff_flag="--no-enable-handoff"
  fi

  baseline_flag="--include-baselines"
  if [[ "$RUN_BASELINES" != "true" ]]; then
    baseline_flag="--no-include-baselines"
  fi

  envs_csv=$(IFS=,; echo "${ENVS[*]}")
  seeds_csv=$(IFS=,; echo "${SEEDS[*]}")
  fractions_csv=$(IFS=,; echo "${SWITCH_FRACTIONS[*]}")

  echo "Submitting Experiment 2 Modal grid (detached):"
  echo "  handoff=${RUN_HANDOFF} baselines=${RUN_BASELINES} envs=${envs_csv} seeds=${seeds_csv}"
  echo "  fractions=${fractions_csv} steps=${TOTAL_TIMESTEPS} volume=${MODAL_RESULTS_VOLUME}"
  echo ""
  echo "When jobs finish, download results then summarize:"
  echo "  ./scripts/fetch_experiment_2_results.sh"
  echo "  uv run python src/RL-translational-dynamics/exp2/summarize_experiment_2.py --skip-checkpoint-gate"
  echo ""

  uv run modal run --detach \
    src/RL-translational-dynamics/modal/experiment_2_modal.py::main \
    --no-wait \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    --num-eval-episodes "$NUM_EVAL_EPISODES" \
    --wandb-project "$WANDB_PROJECT" \
    --distill-steps "$DISTILL_STEPS" \
    --value-warmup-updates "$VALUE_WARMUP_UPDATES" \
    --save-interval "$SAVE_INTERVAL" \
    --envs "$envs_csv" \
    --seeds "$seeds_csv" \
    --switch-fractions "$fractions_csv" \
    $handoff_flag \
    $baseline_flag \
    "$track_flag"
  exit 0
fi

if [[ "$LAUNCHER" != "local" ]]; then
  echo "Unknown LAUNCHER='${LAUNCHER}'. Use 'modal' or 'local'." >&2
  exit 1
fi

track_flag="--no-track"
if [[ "$TRACK" == "true" ]]; then
  track_flag="--track"
fi

if [[ "$RUN_HANDOFF" == "true" ]]; then
  for switch_fraction in "${SWITCH_FRACTIONS[@]}"; do
    for env_id in "${ENVS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "Running SAC->PPO handoff on ${env_id}, seed ${seed}, switch_fraction ${switch_fraction}, ${TOTAL_TIMESTEPS} steps"
        uv run python "src/RL-translational-dynamics/exp2/train_handoff.py" \
          --env-id "$env_id" \
          --seed "$seed" \
          --total-timesteps "$TOTAL_TIMESTEPS" \
          --switch-fraction "$switch_fraction" \
          --eval-interval "$EVAL_INTERVAL" \
          --num-eval-episodes "$NUM_EVAL_EPISODES" \
          --save-dir "$SAVE_DIR" \
          --wandb-project "$WANDB_PROJECT" \
          --wandb-group experiment_2 \
          --value-warmup-updates "$VALUE_WARMUP_UPDATES" \
          --distill-steps "$DISTILL_STEPS" \
          --save-interval "$SAVE_INTERVAL" \
          "$track_flag"
      done
    done
  done
fi

if [[ "$RUN_BASELINES" == "true" ]]; then
  for algorithm in sac ppo; do
    for env_id in "${ENVS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "Running baseline ${algorithm} on ${env_id}, seed ${seed}, ${TOTAL_TIMESTEPS} steps"
        uv run python "src/RL-translational-dynamics/exp0/train_${algorithm}.py" \
          --env-id "$env_id" \
          --seed "$seed" \
          --total-timesteps "$TOTAL_TIMESTEPS" \
          --eval-interval "$EVAL_INTERVAL" \
          --num-eval-episodes "$NUM_EVAL_EPISODES" \
          --save-interval "$SAVE_INTERVAL" \
          --save-dir "$SAVE_DIR" \
          --wandb-project "$WANDB_PROJECT" \
          --wandb-group experiment_2_baselines \
          "$track_flag"
      done
    done
  done
fi
