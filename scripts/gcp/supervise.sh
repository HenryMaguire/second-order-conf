#!/usr/bin/env bash
# Resumable job supervisor. Registers a job so it is restarted after pre-emption, then runs it
# until it exits 0 (or fails MAX_TRIES times in a row).
#
#   scripts/gcp/supervise.sh <stage> <run_id> -- <command...>   # register and run
#   scripts/gcp/supervise.sh --resume <run_id>                  # used by vm_startup.sh
#
# The command gets SOC_RUN_ID, SOC_OUTPUT_DIR (local) and SOC_OUTPUT_URI (gs://...) in its
# environment and must itself resume from the latest checkpoint under SOC_OUTPUT_URI.
# Set SOC_OUTPUT_NAME when registering to share one output directory across several jobs
# (e.g. batches of one rollout run); it defaults to the run_id.
set -euo pipefail
# shellcheck source=/dev/null
source /etc/soc.env
MAX_TRIES="${MAX_TRIES:-5}"
jobs="$SOC_JOBS_DIR"

if [[ "${1:-}" == "--resume" ]]; then
  run_id="$2"
else
  stage="${1:?stage}"; run_id="${2:?run_id}"; shift 2
  [[ "${1:-}" == "--" ]] && shift
  [[ -e "$jobs/$run_id.cmd" ]] && { echo "run_id $run_id already registered"; exit 1; }
  printf '%q ' "$@" > "$jobs/$run_id.cmd"
  echo "$stage" > "$jobs/$run_id.stage"
  echo "${SOC_OUTPUT_NAME:-$run_id}" > "$jobs/$run_id.output"
fi

stage="$(cat "$jobs/$run_id.stage")"
output_name="$(cat "$jobs/$run_id.output" 2>/dev/null || echo "$run_id")"
export SOC_RUN_ID="$run_id"
export SOC_OUTPUT_DIR="/home/$SOC_USER/outputs/$output_name"
export SOC_OUTPUT_URI="gs://$SOC_BUCKET/$stage/$output_name"
mkdir -p "$SOC_OUTPUT_DIR"
echo $$ > "$jobs/$run_id.pid"
trap 'rm -f "$jobs/$run_id.pid"' EXIT

cd "$SOC_REPO"
tries=0
while (( tries < MAX_TRIES )); do
  echo "[$(date -Is)] $run_id: attempt $((tries + 1))"
  if bash -lc "$(cat "$jobs/$run_id.cmd")"; then
    date -Is > "$jobs/$run_id.done"
    gcloud storage rsync -r "$SOC_OUTPUT_DIR" "$SOC_OUTPUT_URI" || true
    echo "[$(date -Is)] $run_id: done"
    exit 0
  fi
  tries=$((tries + 1))
  sleep $((60 * tries))
done
echo "[$(date -Is)] $run_id: failed $MAX_TRIES times, giving up"
exit 1
