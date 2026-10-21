"""Mirror a local run directory to GCS with the gcloud CLI (present on Deep Learning VM images)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def sync_down(uri: str | None, local: Path) -> None:
    """Pull existing outputs (finished shards) before resuming. No-op without a URI."""
    if not uri:
        return
    local.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(["gcloud", "storage", "ls", uri], capture_output=True, text=True)
    if listing.returncode == 0:
        subprocess.run(["gcloud", "storage", "rsync", "-r", uri, str(local)], check=True)


def sync_up(local: Path, uri: str | None) -> None:
    if uri:
        subprocess.run(["gcloud", "storage", "rsync", "-r", str(local), uri], check=True)
