#!/usr/bin/env bash
# Run one 1,000-question batch of the Stage 0 rollouts, on the VM. All batches write into one
# output directory (gs://$SOC_BUCKET/stage0/rollouts-qwen35-2b-v1) and each skips the shards
# earlier batches finished.
#
#   scripts/gcp/stage0_batch.sh 1     # BigMath: +500 train, +250 val, +250 test
#   ...                               # batches 1-4 likewise; 5-7: +1,000 train each
#   scripts/gcp/stage0_batch.sh ood   # GPQA Diamond, TriviaQA, SimpleQA
set -euo pipefail
b="${1:?batch number 1-7 or ood}"
cd "$(dirname "$0")/../.."

if [[ "$b" == "ood" ]]; then
  shards="gpqa_diamond=1,triviaqa=4,simpleqa=4"
elif (( b >= 1 && b <= 4 )); then
  shards="bigmath_train=$((2 * b)),bigmath_val=$b,bigmath_test=$b"
elif (( b >= 5 && b <= 7 )); then
  shards="bigmath_train=$((8 + 4 * (b - 4))),bigmath_val=4,bigmath_test=4"
else
  echo "unknown batch: $b"; exit 1
fi

SOC_OUTPUT_NAME=rollouts-qwen35-2b-v1 exec scripts/gcp/supervise.sh stage0 "stage0-batch-$b" -- \
  uv run --extra gpu python -m soc.rollouts.run configs/stage0_rollouts.yaml --shards "$shards"
