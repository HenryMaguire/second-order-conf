# shellcheck shell=bash
# Copy to scripts/gcp/env.sh (git-ignored) and fill in. Sourced by every script here.
export SOC_PROJECT="your-gcp-project"
export SOC_ZONE="us-central1-a"            # a zone with L4/A100 stock and your quota
export SOC_BUCKET="your-soc-bucket"        # regional bucket in the same region, no gs:// prefix
export SOC_REPO_URL="https://github.com/<you>/second-order-conf.git"

# Deep Learning VM image (PyTorch + CUDA). Families change; check the current ones with:
#   gcloud compute images list --project deeplearning-platform-release \
#     --filter="family~pytorch AND family~cu12" --format="value(family)" | sort -u
export SOC_IMAGE_PROJECT="deeplearning-platform-release"
export SOC_IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
