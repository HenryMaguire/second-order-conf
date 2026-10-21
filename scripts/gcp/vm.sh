#!/usr/bin/env bash
# VM lifecycle. Every command that creates, starts or deletes resources asks for confirmation.
#
#   scripts/gcp/vm.sh create <name> <l4|a100|h100>   # spot VM with idle shutdown + supervisor
#   scripts/gcp/vm.sh start  <name>
#   scripts/gcp/vm.sh stop   <name>
#   scripts/gcp/vm.sh delete <name>
#   scripts/gcp/vm.sh ssh    <name>
#   scripts/gcp/vm.sh list
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$here/env.sh"

confirm() {
  echo "About to run:"; printf '  %q' "$@"; echo
  read -r -p "Proceed? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "Aborted."; exit 1; }
  "$@"
}

gc=(gcloud --project "$SOC_PROJECT")
cmd="${1:-}"; name="${2:-}"

case "$cmd" in
  create)
    gpu="${3:?gpu type: l4, a100 or h100}"
    case "$gpu" in
      l4)   machine=g2-standard-8 ;;   # 1x L4 24GB is built into G2
      a100) machine=a2-highgpu-1g ;;   # 1x A100 40GB
      h100) machine=a3-highgpu-1g ;;   # 1x H100 80GB
      *) echo "unknown gpu: $gpu"; exit 1 ;;
    esac
    confirm "${gc[@]}" compute instances create "$name" \
      --zone "$SOC_ZONE" \
      --machine-type "$machine" \
      --provisioning-model SPOT --instance-termination-action STOP \
      --maintenance-policy TERMINATE \
      --image-project "$SOC_IMAGE_PROJECT" --image-family "$SOC_IMAGE_FAMILY" \
      --boot-disk-size 300GB --boot-disk-type pd-ssd \
      --scopes cloud-platform \
      --metadata "install-nvidia-driver=True,soc-bucket=$SOC_BUCKET,soc-repo-url=$SOC_REPO_URL" \
      --metadata-from-file "startup-script=$here/vm_startup.sh,shutdown-script=$here/vm_shutdown.sh"
    ;;
  start)  confirm "${gc[@]}" compute instances start "$name" --zone "$SOC_ZONE" ;;
  stop)   "${gc[@]}" compute instances stop "$name" --zone "$SOC_ZONE" ;;
  delete) confirm "${gc[@]}" compute instances delete "$name" --zone "$SOC_ZONE" ;;
  ssh)    "${gc[@]}" compute ssh "$name" --zone "$SOC_ZONE" ;;
  list)   "${gc[@]}" compute instances list ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
