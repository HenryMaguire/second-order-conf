"""Guards against silently changing the frozen template or splits (PLAN.md: freeze in week 1)."""

import json
from pathlib import Path

import yaml

from soc.data import SPECS, TEMPLATE_HASH

ROOT = Path(__file__).resolve().parents[1]
FROZEN = yaml.safe_load((ROOT / "configs" / "frozen.yaml").read_text())


def test_template_hash_matches_frozen():
    assert TEMPLATE_HASH == FROZEN["prompt_template"]["hash"], (
        "prompt template changed; this invalidates existing rollouts. If intended, bump "
        "TEMPLATE_VERSION and update configs/frozen.yaml."
    )


def test_split_files_match_frozen():
    assert set(FROZEN["splits"]) == {s.name for s in SPECS}
    for name, rec in FROZEN["splits"].items():
        split = json.loads((ROOT / "configs" / "splits" / f"{name}.json").read_text())
        assert split["sha256"] == rec["sha256"] and split["n"] == rec["n"] == len(split["items"])


def test_rollout_settings_match_plan():
    assert FROZEN["rollouts"] == {"n": 32, "temperature": 0.7, "max_new_tokens": 1024}
