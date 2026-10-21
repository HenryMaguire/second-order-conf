"""Answer verification.

`verify` is the programmatic check used for training targets: numeric/math for BigMathDigits,
letter match for GPQA, normalised exact match against aliases for short answers. For TriviaQA
and SimpleQA evaluation, `verify_with_judge` accepts exact matches without a call and sends
the rest to the cached LLM judge; NOT_ATTEMPTED counts as incorrect.
"""

from __future__ import annotations

from soc.data.sources import Question
from soc.verify.checks import check_math, check_mcq, exact_match, extract_answer


def _golds(q: Question) -> list[str]:
    return [q.gold, *(a for a in q.aliases if a != q.gold)]


def verify(q: Question, response: str) -> bool:
    pred = extract_answer(response)
    if pred is None:
        return False
    if q.kind == "math":
        return check_math(pred, q.gold)
    if q.kind == "mcq":
        return check_mcq(pred, q.gold)
    return exact_match(pred, _golds(q))


def verify_with_judge(q: Question, response: str, judge) -> bool:
    if q.kind != "short":
        return verify(q, response)
    pred = extract_answer(response)
    if pred is None:
        return False
    if exact_match(pred, _golds(q)):
        return True
    return judge.grade(q.problem, _golds(q), pred) == "CORRECT"


__all__ = ["extract_answer", "verify", "verify_with_judge"]
