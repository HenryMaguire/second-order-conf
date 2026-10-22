#!/usr/bin/env bash
# Get a GPU VM running for the Stage 0 chain after a pre-emption or stockout. Run locally.
#
#   scripts/gcp/acquire_l4.sh [minutes]     # default 15
#
# 1. For up to N minutes, once a minute: try to start the existing spot VM (soc-l4, zone a),
#    and try to create a spot VM (soc-l4-b) in each other zone of the region.
# 2. If none succeed, switch soc-l4 to on-demand (standard) pricing and start it, falling back
#    to the smaller g2-standard-4 host if g2-standard-8 is out of stock.
# Note: the region's SSD quota is 500 GB, so a second VM (soc-l4-b) needs a disk of at most
# 200 GB while soc-l4's 300 GB disk exists.
# On success the chain resumes by itself on soc-l4 (its startup script re-runs unfinished jobs);
# on a newly created VM this script registers and starts the chain over SSH.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$here/env.sh"
G="${GCLOUD:-gcloud}"
minutes="${1:-15}"
home_vm=soc-l4
home_zone="$SOC_ZONE"
alt_vm=soc-l4-b
alt_zones=(us-central1-b us-central1-c)
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

create_alt() {  # $1 = zone
  "$G" --project "$SOC_PROJECT" compute instances create "$alt_vm" --zone "$1" \
    --machine-type g2-standard-8 \
    --provisioning-model SPOT --instance-termination-action STOP \
    --maintenance-policy TERMINATE \
    --image-project "$SOC_IMAGE_PROJECT" --image-family "$SOC_IMAGE_FAMILY" \
    --boot-disk-size 150GB --boot-disk-type pd-ssd --scopes cloud-platform \
    --metadata "install-nvidia-driver=True,soc-bucket=$SOC_BUCKET,soc-repo-url=$SOC_REPO_URL" \
    --metadata-from-file "startup-script=$here/vm_startup.sh,shutdown-script=$here/vm_shutdown.sh" \
    >/dev/null 2>&1
}

start_chain_on() {  # $1 = vm, $2 = zone; new VMs have no registered jobs
  log "waiting for $1 to finish its startup script"
  for _ in $(seq 1 30); do
    if "$G" compute ssh "$1" --zone "$2" --project "$SOC_PROJECT" --quiet \
        --command 'test -f /etc/soc.env' >/dev/null 2>&1; then break; fi
    sleep 20
  done
  "$G" compute ssh "$1" --zone "$2" --project "$SOC_PROJECT" --quiet --command '
    sudo -u soc setsid bash -lc "cd /home/soc/second-order-conf && SOC_OUTPUT_NAME=rollouts-qwen35-2b-v1 scripts/gcp/supervise.sh stage0 stage0-chain -- scripts/gcp/stage0_chain.sh >> /home/soc/soc-jobs/stage0-chain.log 2>&1" < /dev/null > /dev/null 2>&1 &
    sleep 5; sudo ls /home/soc/soc-jobs/' && log "chain started on $1"
}

deadline=$(( $(date +%s) + minutes * 60 ))
attempt=0
while (( $(date +%s) < deadline )); do
  attempt=$((attempt + 1))
  if "$G" compute instances start "$home_vm" --zone "$home_zone" --project "$SOC_PROJECT" \
      -q >/dev/null 2>&1; then
    log "attempt $attempt: spot $home_vm started in $home_zone; chain resumes at boot"
    exit 0
  fi
  for z in "${alt_zones[@]}"; do
    if create_alt "$z"; then
      log "attempt $attempt: spot $alt_vm created in $z"
      start_chain_on "$alt_vm" "$z"
      exit 0
    fi
  done
  log "attempt $attempt: no spot L4 capacity in $home_zone ${alt_zones[*]}"
  sleep 60
done

log "no spot capacity after $minutes min; switching $home_vm to on-demand"
# Both flags are needed: a VM created as spot is also marked preemptible.
"$G" compute instances set-scheduling "$home_vm" --zone "$home_zone" --project "$SOC_PROJECT" \
  --provisioning-model STANDARD --no-preemptible --clear-instance-termination-action ||
  { log "could not switch to on-demand"; exit 1; }
# On-demand L4s can be out of stock too; the smaller g2-standard-4 host (same L4, 4 vCPU,
# 16 GB RAM, enough for a 2B model) draws on a different pool and worked on 2026-10-22.
for machine in g2-standard-8 g2-standard-4; do
  "$G" compute instances set-machine-type "$home_vm" --zone "$home_zone" \
    --project "$SOC_PROJECT" --machine-type "$machine" >/dev/null 2>&1
  if "$G" compute instances start "$home_vm" --zone "$home_zone" --project "$SOC_PROJECT" \
      -q >/dev/null 2>&1; then
    log "on-demand $home_vm ($machine) started; chain resumes at boot"
    exit 0
  fi
  log "no on-demand capacity for $machine in $home_zone"
done
exit 1
