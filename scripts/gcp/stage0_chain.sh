#!/usr/bin/env bash
# Keep Stage 0 rolling on the VM: wait for batch 1, extract features for every split, then run
# batches 2-7 and the OOD batch back to back into the same output directory. Every step is
# idempotent (finished shards and feature files are skipped), so after a pre-emption the
# supervisor simply re-runs this script.
#
#   SOC_OUTPUT_NAME=rollouts-qwen35-2b-v1 scripts/gcp/supervise.sh stage0 stage0-chain -- \
#     scripts/gcp/stage0_chain.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck source=stage0_shards.sh
source scripts/gcp/stage0_shards.sh
cfg=configs/stage0_rollouts.yaml
log() { echo "[$(date -Is)] chain: $*"; }

# Batch 1 was started on its own; let it finish rather than compete for the GPU.
while [[ -e "$SOC_JOBS_DIR/stage0-batch-1.pid" && ! -e "$SOC_JOBS_DIR/stage0-batch-1.done" ]]; do
  sleep 60
done
log "batch 1 finished"

# Features for every split at once (minutes). Not fatal: rollouts matter more, and features
# can be re-run later.
if ! uv run --extra gpu python -m soc.head.features "$cfg" \
    bigmath_train bigmath_val bigmath_test gpqa_diamond triviaqa simpleqa; then
  log "feature extraction FAILED; continuing with rollouts"
fi

for b in 1 2 3 4 5 6 7 ood; do
  log "batch $b: $(stage0_shards "$b")"
  uv run --extra gpu python -m soc.rollouts.run "$cfg" --shards "$(stage0_shards "$b")"
  log "batch $b done"
done
log "all Stage 0 rollouts done"
