# shellcheck shell=bash
# Cumulative per-split shard limits (250 questions per shard) for Stage 0 batch N.
# Batches 1-4 add 500 train, 250 val, 250 test; 5-7 add 1,000 train; "ood" is the shift sets.
stage0_shards() {
  local b="$1"
  if [[ "$b" == "ood" ]]; then
    echo "gpqa_diamond=1,triviaqa=4,simpleqa=4"
  elif (( b >= 1 && b <= 4 )); then
    echo "bigmath_train=$((2 * b)),bigmath_val=$b,bigmath_test=$b"
  elif (( b >= 5 && b <= 7 )); then
    echo "bigmath_train=$((8 + 4 * (b - 4))),bigmath_val=4,bigmath_test=4"
  else
    echo "unknown batch: $b" >&2
    return 1
  fi
}
