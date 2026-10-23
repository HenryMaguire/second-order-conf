"""Dataset sources, pinned to Hugging Face revisions, and their conversion to `Question`s."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["math", "mcq", "short"]


@dataclass(frozen=True)
class Question:
    qid: str
    dataset: str
    kind: Kind
    problem: str
    gold: str
    aliases: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def content_hash(self) -> str:
        """Hash of what the model sees and is scored against; detects silent upstream edits."""
        return _sha("\x1f".join([self.problem, self.gold, *self.aliases]))[:16]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class Source:
    name: str
    hf_id: str
    revision: str
    split: str
    to_questions: Callable[[Iterable[dict]], list[Question]]
    config: str | None = None


def _bigmath(dataset: str):
    def convert(rows):
        return [
            Question(
                qid=f"bmd-{_sha(r['problem'])[:12]}",
                dataset=dataset,
                kind="math",
                problem=r["problem"],
                gold=r["answer"].strip(),
                meta={"llama8b_solve_rate": r["llama8b_solve_rate"]},
            )
            for r in rows
        ]

    return convert


def _gpqa(rows):
    # The mirror ships options already shuffled and lettered inside the question text.
    return [
        Question(
            qid=f"gpqa-{_sha(r['question'])[:12]}",
            dataset="gpqa_diamond",
            kind="mcq",
            problem=r["question"],
            gold=r["answer"].strip(),
        )
        for r in rows
    ]


def _triviaqa(rows):
    return [
        Question(
            qid=f"tqa-{r['question_id']}",
            dataset="triviaqa",
            kind="short",
            problem=r["question"],
            gold=r["answer"]["value"],
            aliases=tuple(r["answer"]["aliases"]),
        )
        for r in rows
    ]


def _simpleqa(rows):
    return [
        Question(
            qid=f"sqa-{_sha(r['problem'])[:12]}",
            dataset="simpleqa",
            kind="short",
            problem=r["problem"],
            gold=r["answer"],
        )
        for r in rows
    ]


def _ambigqa(rows):
    """AmbigQA (validation, full config): questions every annotator marked ambiguous
    ("multipleQAs") or every annotator marked single-answer; disagreements are dropped.

    Gold is the original Natural Questions answer, so on an ambiguous question the solve rate
    measures which reading the model prefers. All readings' answers are kept in meta so the
    rollouts can also be scored against "any valid reading". The qid encodes the group
    (aqa-amb-* / aqa-sgl-*) for the per-group spread baseline.
    """
    out = []
    for r in rows:
        ann, nq = r["annotations"], [a for a in r["nq_answer"] if a.strip()]
        types = ann["type"]
        if not nq or not types or len(set(types)) != 1:
            continue
        ambiguous = types[0] == "multipleQAs"
        if ambiguous:
            readings = [list(ans) for pairs in ann["qaPairs"] for ans in pairs["answer"] if ans]
        else:
            readings = [list(a) for a in ann["answer"] if a]
        nq_norm = {_norm(a) for a in nq}
        # Aliases for the NQ answer: the NQ strings plus any reading that contains one of them.
        aliases = list(
            dict.fromkeys(
                [*nq, *(a for rd in readings if nq_norm & {_norm(x) for x in rd} for a in rd)]
            )
        )
        out.append(
            Question(
                qid=f"aqa-{'amb' if ambiguous else 'sgl'}-{r['id']}",
                dataset="ambigqa",
                kind="short",
                problem=r["question"],
                gold=nq[0],
                aliases=tuple(aliases),
                meta={"ambiguous": ambiguous, "readings": readings},
            )
        )
    return out


def _norm(text: str) -> str:
    from soc.verify.checks import normalize

    return normalize(text)


BIGMATH_REV = "c3c94f8eab5b5301bd71d9735c329951d80418a5"

SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        # RLCR's BigMathDigits: Big-Math, numeric answers, Llama-3.1-8B solve rate 0-70%.
        Source(
            "bigmath_train_pool",
            "mehuldamani/big-math-digits",
            BIGMATH_REV,
            "train",
            _bigmath("bigmath"),
        ),
        Source(
            "bigmath_test", "mehuldamani/big-math-digits", BIGMATH_REV, "test", _bigmath("bigmath")
        ),
        # Ungated mirror of GPQA Diamond (198); the official Idavidrein/gpqa is gated.
        Source(
            "gpqa_diamond",
            "fingertap/GPQA-Diamond",
            "68be7564497676e07a77a042fdb587deb88c51c3",
            "test",
            _gpqa,
        ),
        # TriviaQA test answers are hidden, so the subset is drawn from validation.
        Source(
            "triviaqa_val",
            "mandarjoshi/trivia_qa",
            "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
            "validation",
            _triviaqa,
            config="rc.nocontext",
        ),
        # Pilot: ambiguous vs single-answer questions (validation has 2-3 annotators each).
        Source(
            "ambigqa_pool",
            "sewon/ambig_qa",
            "e969d0132f4dd28c2939d55be34f1788c00ccfe7",
            "validation",
            _ambigqa,
            config="full",
        ),
        Source(
            "simpleqa",
            "basicv8vc/SimpleQA",
            "e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94",
            "test",
            _simpleqa,
        ),
    ]
}


def clean_pool(questions: list[Question]) -> list[Question]:
    """Keep one copy of exact duplicates; drop questions whose copies disagree (ambiguous gold).

    BigMathDigits has a few problems listed twice with different answers, and TriviaQA lists
    most questions twice (web and Wikipedia evidence), occasionally with different aliases.
    """
    hashes: dict[str, set[str]] = {}
    for q in questions:
        hashes.setdefault(q.qid, set()).add(q.content_hash)
    seen: set[str] = set()
    out = []
    for q in questions:
        if len(hashes[q.qid]) == 1 and q.qid not in seen:
            seen.add(q.qid)
            out.append(q)
    return out


def load_source(name: str) -> list[Question]:
    """Download a source at its pinned revision and clean it. Needs the `data` extra."""
    from datasets import load_dataset

    src = SOURCES[name]
    ds = load_dataset(src.hf_id, src.config, split=src.split, revision=src.revision)
    return clean_pool(src.to_questions(ds))
