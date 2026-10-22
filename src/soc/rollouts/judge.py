"""Re-grade short-answer rollouts (TriviaQA, SimpleQA) with the cached LLM judge.

    uv run --extra judge python -m soc.rollouts.judge <run_dir> triviaqa simpleqa

Adds `correct_judged` to each rollout shard and `k_judged` to counts/<split>.parquet. Exact
matches are accepted without a call; everything else goes to the judge once per distinct
(question, answer), since the verdict cache is keyed on the extracted answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from soc.verify.checks import exact_match
from soc.verify.judge import Judge


def judge_split(run_dir: Path, split: str, judge, load=None) -> pd.DataFrame:
    if load is None:
        from soc.data import load_split as load
    questions = {q.qid: q for q in load(split)}
    shards = sorted((run_dir / "rollouts" / split).glob("shard-*.parquet"))
    frames = [pd.read_parquet(p) for p in shards]
    df = pd.concat(frames, ignore_index=True)

    def golds(q):
        return [q.gold, *(a for a in q.aliases if a != q.gold)]

    todo = {
        (r.qid, r.answer)
        for r in df.itertuples()
        if isinstance(r.answer, str) and not exact_match(r.answer, golds(questions[r.qid]))
    }
    items = sorted(todo)
    verdicts = judge.grade_many(
        [(questions[qid].problem, golds(questions[qid]), ans) for qid, ans in items]
    )
    correct_by = {key: v == "CORRECT" for key, v in zip(items, verdicts, strict=True)}

    def grade(qid, ans):
        if not isinstance(ans, str):  # missing answers come back from parquet as None or NaN
            return False
        return exact_match(ans, golds(questions[qid])) or correct_by[(qid, ans)]

    for path, frame in zip(shards, frames, strict=True):
        frame["correct_judged"] = [
            grade(q, a) for q, a in zip(frame.qid, frame.answer, strict=True)
        ]
        frame.to_parquet(path, index=False)
    df["correct_judged"] = [grade(q, a) for q, a in zip(df.qid, df.answer, strict=True)]

    counts_path = run_dir / "counts" / f"{split}.parquet"
    counts = pd.read_parquet(counts_path)
    k = df.groupby("qid").correct_judged.sum()
    counts["k_judged"] = counts.qid.map(k).astype(int)
    counts.to_parquet(counts_path, index=False)
    return counts


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    j = Judge(run_dir / "judge_cache.sqlite")
    for split in sys.argv[2:]:
        c = judge_split(run_dir, split, j)
        print(
            split,
            "mean k/n exact:",
            round((c.k / c.n).mean(), 4),
            "judged:",
            round((c.k_judged / c.n).mean(), 4),
        )
