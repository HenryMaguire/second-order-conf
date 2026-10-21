#!/usr/bin/env bash
# Runs on shutdown, including spot pre-emption (about 30 s of grace). Best-effort push of each
# running job's local checkpoint directory to GCS; jobs should also checkpoint to GCS themselves.
set -uo pipefail
# shellcheck source=/dev/null
source /etc/soc.env 2>/dev/null || exit 0
for spec in "$SOC_JOBS_DIR"/*.cmd; do
  [[ -e "$spec" ]] || continue
  run_id="$(basename "$spec" .cmd)"
  [[ -e "$SOC_JOBS_DIR/$run_id.done" ]] && continue
  local_dir="/home/$SOC_USER/outputs/$run_id"
  [[ -d "$local_dir" ]] || continue
  stage="$(cat "$SOC_JOBS_DIR/$run_id.stage" 2>/dev/null || echo misc)"
  timeout 25 gcloud storage rsync -r "$local_dir" "gs://$SOC_BUCKET/$stage/$run_id" || true
done
