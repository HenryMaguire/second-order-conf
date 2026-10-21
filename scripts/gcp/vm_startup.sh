#!/usr/bin/env bash
# Runs as root on every VM boot, including the restart after a spot pre-emption.
# Clones the repo (first boot), installs the idle-shutdown cron, and resumes unfinished jobs.
set -euo pipefail

md() { curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

user=soc
id -u "$user" &>/dev/null || useradd -m -s /bin/bash "$user"
repo="/home/$user/second-order-conf"
bucket="$(md soc-bucket)"

if [[ ! -d "$repo/.git" ]]; then
  sudo -u "$user" git clone "$(md soc-repo-url)" "$repo"
fi
sudo -u "$user" bash -lc "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"

cat > /etc/soc.env <<EOF
SOC_BUCKET=$bucket
SOC_REPO=$repo
SOC_JOBS_DIR=/home/$user/soc-jobs
SOC_USER=$user
EOF

# Idle shutdown check every 5 minutes.
echo "*/5 * * * * root $repo/scripts/gcp/idle_shutdown.sh >> /var/log/soc-idle.log 2>&1" \
  > /etc/cron.d/soc-idle
chmod 644 /etc/cron.d/soc-idle
rm -f /var/run/soc-idle-since   # a fresh boot is not idle time

# Resume every job that has not finished.
jobs_dir="/home/$user/soc-jobs"
mkdir -p "$jobs_dir" && chown "$user" "$jobs_dir"
for spec in "$jobs_dir"/*.cmd; do
  [[ -e "$spec" ]] || continue
  run_id="$(basename "$spec" .cmd)"
  [[ -e "$jobs_dir/$run_id.done" ]] && continue
  sudo -u "$user" bash -lc \
    "nohup $repo/scripts/gcp/supervise.sh --resume $run_id >> $jobs_dir/$run_id.log 2>&1 &"
done
