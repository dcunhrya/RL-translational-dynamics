#!/usr/bin/env bash
# Download Experiment 2 metrics only (metrics.jsonl per run) from the Modal volume.
# Skips checkpoint .pt files and other large artifacts.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODAL_RESULTS_VOLUME="${MODAL_RESULTS_VOLUME:-rl-translational-dynamics-results}"
REMOTE_PREFIX="${REMOTE_PREFIX:-raw/experiment_2}"
LOCAL_DIR="${LOCAL_DIR:-results/raw/experiment_2}"
# Set to "true" to also download checkpoint_step_*.pt (large).
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-false}"
FORCE="${FORCE:-false}"

mkdir -p "$LOCAL_DIR"

echo "Listing runs in Modal volume '${MODAL_RESULTS_VOLUME}:${REMOTE_PREFIX}'"
RUN_PATHS=()
while IFS= read -r line; do
  line="${line#"${line%%[![:space:]]*}"}"
  [[ -z "$line" ]] && continue
  RUN_PATHS+=("$line")
done < <(uv run modal volume ls "$MODAL_RESULTS_VOLUME" "$REMOTE_PREFIX" 2>/dev/null || true)

if [[ "${#RUN_PATHS[@]}" -eq 0 ]]; then
  echo "No runs found under ${REMOTE_PREFIX}. Check volume name or wait for jobs to finish." >&2
  exit 1
fi

downloaded=0
skipped=0

for remote_run_path in "${RUN_PATHS[@]}"; do
  run_name="$(basename "$remote_run_path")"
  local_run_dir="${LOCAL_DIR}/${run_name}"
  remote_metrics="${remote_run_path}/metrics.jsonl"
  local_metrics="${local_run_dir}/metrics.jsonl"

  mkdir -p "$local_run_dir"

  if [[ -f "$local_metrics" && "$FORCE" != "true" ]]; then
    echo "Skip (exists): ${run_name}/metrics.jsonl"
    skipped=$((skipped + 1))
    continue
  fi

  echo "Downloading ${run_name}/metrics.jsonl"
  if ! uv run modal volume get "$MODAL_RESULTS_VOLUME" "$remote_metrics" "$local_metrics" --force; then
    echo "Warning: failed to download ${remote_metrics}" >&2
    continue
  fi
  downloaded=$((downloaded + 1))

  if [[ "$INCLUDE_CHECKPOINTS" == "true" ]]; then
    while IFS= read -r remote_ckpt; do
      remote_ckpt="${remote_ckpt#"${remote_ckpt%%[![:space:]]*}"}"
      [[ -z "$remote_ckpt" ]] && continue
      [[ "$remote_ckpt" != *checkpoint_step_*.pt ]] && continue
      ckpt_name="$(basename "$remote_ckpt")"
      echo "  Downloading ${run_name}/${ckpt_name}"
      uv run modal volume get "$MODAL_RESULTS_VOLUME" "$remote_ckpt" "${local_run_dir}/${ckpt_name}" --force
    done < <(uv run modal volume ls "$MODAL_RESULTS_VOLUME" "$remote_run_path" 2>/dev/null || true)
  fi
done

echo ""
echo "Done. Downloaded metrics for ${downloaded} run(s), skipped ${skipped} existing."
if [[ "$INCLUDE_CHECKPOINTS" != "true" ]]; then
  echo "(Checkpoints omitted. Set INCLUDE_CHECKPOINTS=true to download .pt files.)"
fi
echo ""
echo "Summarize with:"
echo "  uv run python src/RL-translational-dynamics/exp2/summarize_experiment_2.py --skip-checkpoint-gate"
