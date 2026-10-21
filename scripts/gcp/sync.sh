#!/usr/bin/env bash
# Sync a run directory with GCS.
#   scripts/gcp/sync.sh push <local_dir> <stage>/<run_id>
#   scripts/gcp/sync.sh pull <stage>/<run_id> <local_dir>
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
[[ -n "${SOC_BUCKET:-}" ]] || source "$here/env.sh"

case "${1:-}" in
  push) gcloud storage rsync -r "${2:?local_dir}" "gs://$SOC_BUCKET/${3:?stage/run_id}" ;;
  pull) mkdir -p "${3:?local_dir}"; gcloud storage rsync -r "gs://$SOC_BUCKET/${2:?stage/run_id}" "$3" ;;
  *) sed -n '2,4p' "$0"; exit 1 ;;
esac
