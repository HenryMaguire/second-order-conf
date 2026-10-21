#!/usr/bin/env bash
# Cron job (every 5 min, as root): stop the VM after IDLE_MINUTES with GPU utilisation under
# UTIL_THRESHOLD% and no supervised job running. A stopped VM stops billing for GPU and CPU.
set -uo pipefail
IDLE_MINUTES="${IDLE_MINUTES:-30}"
UTIL_THRESHOLD="${UTIL_THRESHOLD:-5}"
state=/var/run/soc-idle-since
# shellcheck source=/dev/null
source /etc/soc.env 2>/dev/null || true

util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
  | sort -n | tail -n1)"
util="${util:-0}"

job_running=0
for pidfile in "${SOC_JOBS_DIR:-/nonexistent}"/*.pid; do
  [[ -e "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null && job_running=1
done

if (( util >= UTIL_THRESHOLD || job_running )); then
  rm -f "$state"
  exit 0
fi

now="$(date +%s)"
[[ -e "$state" ]] || echo "$now" > "$state"
idle_for=$(( (now - $(cat "$state")) / 60 ))
echo "[$(date -Is)] idle for ${idle_for}m (gpu ${util}%)"
if (( idle_for >= IDLE_MINUTES )); then
  echo "[$(date -Is)] stopping VM"
  rm -f "$state"
  shutdown -h now
fi
