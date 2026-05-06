#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "Missing .env. Copy .env.example to .env and set WANDB_API_KEY first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source ".env"
set +a

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is empty in .env." >&2
  exit 1
fi

SECRET_NAME="${WANDB_MODAL_SECRET:-wandb-api-key}"
modal secret create "$SECRET_NAME" WANDB_API_KEY="$WANDB_API_KEY"

echo "Created Modal Secret '${SECRET_NAME}' with WANDB_API_KEY from local .env."
