"""Rollout job: n samples per question -> per-rollout parquet shards -> per-question k, n.

    uv run python -m soc.rollouts.run configs/stage0_rollouts.yaml

Under scripts/gcp/supervise.sh, outputs go to $SOC_OUTPUT_DIR and are mirrored to
$SOC_OUTPUT_URI after every shard. On restart, finished shards are pulled from GCS and skipped,
so a pre-emption loses at most one shard of work.

Layout under the output directory:
    run.json                          config, git commit, template hash, split hashes
    rollouts/<split>/shard-00000.parquet   one row per rollout
    counts/<split>.parquet            one row per question: k, n, mean tokens, truncation rate
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from soc.data import TEMPLATE_HASH, TEMPLATE_VERSION, Question, messages
from soc.rollouts import storage
from soc.rollouts.generate import Generator
from soc.verify import extract_answer, verify

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class RolloutConfig:
    run_id: str
    model: str
    splits: list[str]
    n: int = 32
    temperature: float = 0.7
    max_new_tokens: int = 1024
    seed: int = 0
    shard_size: int = 250
    chat_template_kwargs: dict = field(default_factory=dict)
    vllm: dict = field(default_factory=dict)
    wandb_project: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> RolloutConfig:
        return cls(**yaml.safe_load(Path(path).read_text()))


def check_frozen(cfg: RolloutConfig, frozen_path: Path = ROOT / "configs" / "frozen.yaml"):
    """Target and evaluation rollouts must share template, n and temperature (PLAN.md)."""
    frozen = yaml.safe_load(frozen_path.read_text())
    problems = []
    if TEMPLATE_HASH != frozen["prompt_template"]["hash"]:
        problems.append(f"template hash {TEMPLATE_HASH} != frozen {frozen['prompt_template']}")
    for key in ("n", "temperature", "max_new_tokens"):
        if getattr(cfg, key) != frozen["rollouts"][key]:
            problems.append(f"{key}={getattr(cfg, key)} != frozen {frozen['rollouts'][key]}")
    if problems:
        raise SystemExit(
            "refusing to run, settings differ from configs/frozen.yaml:\n  " + "\n  ".join(problems)
        )
    return frozen


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _shard_seed(base: int, split: str, shard: int) -> int:
    # Stable across restarts and Python processes (unlike hash()).
    return (base * 1_000_003 + sum(map(ord, split)) * 7919 + shard) % (2**31)


def rollout_rows(questions, samples, split, shard):
    rows = []
    for q, outs in zip(questions, samples, strict=True):
        for i, s in enumerate(outs):
            rows.append(
                {
                    "qid": q.qid,
                    "split": split,
                    "shard": shard,
                    "rollout": i,
                    "response": s.text,
                    "answer": extract_answer(s.text),
                    "correct": verify(q, s.text),
                    "n_tokens": s.n_tokens,
                    "truncated": s.finish_reason == "length",
                }
            )
    return rows


def run_split(cfg, split, questions: list[Question], gen: Generator, out: Path, log=print):
    import pandas as pd

    shard_dir = out / "rollouts" / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = [questions[i : i + cfg.shard_size] for i in range(0, len(questions), cfg.shard_size)]
    for idx, chunk in enumerate(shards):
        path = shard_dir / f"shard-{idx:05d}.parquet"
        if path.exists():
            continue
        t0 = time.time()
        samples = gen.generate(
            [messages(q) for q in chunk],
            cfg.n,
            cfg.temperature,
            cfg.max_new_tokens,
            _shard_seed(cfg.seed, split, idx),
        )
        df = pd.DataFrame(rollout_rows(chunk, samples, split, idx))
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.rename(path)  # a shard exists only once fully written
        log(
            {
                "split": split,
                "shard": idx,
                "of": len(shards),
                "secs": round(time.time() - t0, 1),
                "acc": round(df.correct.mean(), 4),
                "trunc": round(df.truncated.mean(), 4),
                "tokens": int(df.n_tokens.sum()),
            }
        )
        yield path
    counts = summarize(shard_dir, [q.qid for q in questions], cfg.n)
    (out / "counts").mkdir(exist_ok=True)
    counts.to_parquet(out / "counts" / f"{split}.parquet", index=False)
    yield out / "counts" / f"{split}.parquet"


def summarize(shard_dir: Path, qids: list[str], n: int):
    import pandas as pd

    df = pd.concat([pd.read_parquet(p) for p in sorted(shard_dir.glob("shard-*.parquet"))])
    g = df.groupby("qid").agg(
        k=("correct", "sum"),
        n=("correct", "size"),
        mean_tokens=("n_tokens", "mean"),
        trunc_rate=("truncated", "mean"),
        parse_rate=("answer", lambda a: a.notna().mean()),
    )
    g = g.reindex(qids)
    if g.n.isna().any() or (g.n != n).any():
        raise ValueError(f"{shard_dir}: missing questions or wrong rollout count")
    g = g.reset_index().astype({"k": int, "n": int})
    return g


def main(config_path: str, generator: Generator | None = None, load=None) -> Path:
    cfg = RolloutConfig.from_yaml(config_path)
    frozen = check_frozen(cfg)
    out = Path(os.environ.get("SOC_OUTPUT_DIR", ROOT / "outputs" / cfg.run_id))
    uri = os.environ.get("SOC_OUTPUT_URI")
    storage.sync_down(uri, out)

    if load is None:
        from soc.data import load_split as load
    if generator is None:
        from soc.rollouts.generate import VLLMGenerator

        generator = VLLMGenerator(cfg.model, cfg.chat_template_kwargs, seed=cfg.seed, **cfg.vllm)

    meta = {
        "config": asdict(cfg),
        "git_commit": git_commit(),
        "template_version": TEMPLATE_VERSION,
        "template_hash": TEMPLATE_HASH,
        "split_hashes": {s: frozen["splits"][s]["sha256"] for s in cfg.splits},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(meta, indent=2))

    log = print
    if cfg.wandb_project:
        import wandb

        wandb.init(
            project=cfg.wandb_project, name=cfg.run_id, id=cfg.run_id, resume="allow", config=meta
        )
        log = lambda rec: (print(rec), wandb.log(rec))  # noqa: E731

    for split in cfg.splits:
        for _ in run_split(cfg, split, load(split), generator, out, log):
            storage.sync_up(out, uri)
    return out


if __name__ == "__main__":
    main(sys.argv[1])
