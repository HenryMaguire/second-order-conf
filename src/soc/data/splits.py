"""Frozen evaluation and training splits.

Splits are stored in `configs/splits/<split>.json` as ordered question IDs plus content hashes,
never question text (GPQA asks that its questions not be posted in plain text). Loading a split
re-downloads the pinned source and checks every hash, so an upstream edit fails loudly.

    uv run --extra data python -m soc.data.splits freeze   # once, in week 1
    uv run --extra data python -m soc.data.splits check    # verify against the Hub
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from soc.data.sources import Question, load_source

SPLITS_DIR = Path(__file__).resolve().parents[3] / "configs" / "splits"
SEED = 20261021


@dataclass(frozen=True)
class SplitSpec:
    name: str
    source: str
    size: int | None  # None = all remaining questions
    offset: int = 0  # position in the seeded shuffle, so train/val drawn from one pool are disjoint


SPECS = [
    SplitSpec("bigmath_train", "bigmath_train_pool", 5000, offset=0),
    SplitSpec("bigmath_val", "bigmath_train_pool", 1000, offset=5000),
    SplitSpec("bigmath_test", "bigmath_test", None),
    SplitSpec("gpqa_diamond", "gpqa_diamond", None),
    SplitSpec("triviaqa", "triviaqa_val", 1000),
    SplitSpec("simpleqa", "simpleqa", 1000),
]


def _dedupe(questions: list[Question], exclude: set[str] = frozenset()) -> list[Question]:
    seen, out = set(exclude), []
    for q in questions:
        if q.qid not in seen:
            seen.add(q.qid)
            out.append(q)
    return out


def select(spec: SplitSpec, pool: list[Question], seed: int = SEED) -> list[Question]:
    """Deterministic selection: sort by qid, shuffle with a per-source seed, slice."""
    ordered = sorted(pool, key=lambda q: q.qid)
    random.Random(f"{seed}:{spec.source}").shuffle(ordered)
    end = None if spec.size is None else spec.offset + spec.size
    chosen = ordered[spec.offset : end]
    if spec.size is not None and len(chosen) < spec.size:
        raise ValueError(f"{spec.name}: wanted {spec.size}, pool has {len(ordered) - spec.offset}")
    return chosen


def split_hash(questions: list[Question]) -> str:
    text = "\n".join(f"{q.qid}:{q.content_hash}" for q in questions)
    return hashlib.sha256(text.encode()).hexdigest()


def build_all(load=load_source) -> dict[str, list[Question]]:
    pools = {name: load(name) for name in {s.source for s in SPECS}}
    # Questions that also appear in the test set never enter train or val.
    test_ids = {q.qid for q in pools["bigmath_test"]}
    pools = {
        name: _dedupe(qs, exclude=test_ids if name == "bigmath_train_pool" else set())
        for name, qs in pools.items()
    }
    return {spec.name: select(spec, pools[spec.source]) for spec in SPECS}


def freeze(splits: dict[str, list[Question]], out_dir: Path = SPLITS_DIR) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for spec in SPECS:
        qs = splits[spec.name]
        hashes[spec.name] = split_hash(qs)
        record = {
            "split": spec.name,
            "source": spec.source,
            "seed": SEED,
            "n": len(qs),
            "sha256": hashes[spec.name],
            "items": [[q.qid, q.content_hash] for q in qs],
        }
        (out_dir / f"{spec.name}.json").write_text(json.dumps(record, indent=0) + "\n")
    return hashes


def load_split(name: str, splits_dir: Path = SPLITS_DIR, load=load_source) -> list[Question]:
    """Questions of a frozen split, in frozen order, with every content hash verified."""
    record = json.loads((splits_dir / f"{name}.json").read_text())
    by_id = {q.qid: q for q in load(record["source"])}
    out = []
    for qid, chash in record["items"]:
        q = by_id.get(qid)
        if q is None or q.content_hash != chash:
            raise ValueError(f"{name}: question {qid} missing or changed upstream")
        out.append(q)
    assert split_hash(out) == record["sha256"]
    return out


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "freeze":
        if any(SPLITS_DIR.glob("*.json")) and "--force" not in argv:
            sys.exit("splits already frozen; pass --force to overwrite (you almost never want to)")
        for name, h in freeze(build_all()).items():
            print(f"{name}: {h}")
    elif cmd == "check":
        for spec in SPECS:
            qs = load_split(spec.name)
            print(f"{spec.name}: {len(qs)} ok")
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv)
