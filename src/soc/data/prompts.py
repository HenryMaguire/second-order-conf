"""Prompt template for answering rollouts (Stage 0 targets, evaluation, Stage 1 prefix).

The template is frozen in week 1 and its hash recorded; target and evaluation rollouts must use
the same template. Changing any text here changes TEMPLATE_HASH, which the rollout job records.
"""

from __future__ import annotations

import hashlib
import json

from soc.data.sources import Kind, Question

TEMPLATE_VERSION = "answer-v1"

SYSTEM = (
    "Solve the problem. Reason briefly, step by step, then write only the final answer "
    "inside <answer></answer> tags."
)

ANSWER_INSTRUCTIONS: dict[Kind, str] = {
    "math": "The final answer is a number. Write it as an integer or decimal, with no units.",
    "mcq": "The final answer is the letter of the correct option.",
    "short": "The final answer is a short phrase, such as a name, number or date.",
}

TEMPLATE_HASH = hashlib.sha256(
    json.dumps(
        {"version": TEMPLATE_VERSION, "system": SYSTEM, "instructions": ANSWER_INSTRUCTIONS},
        sort_keys=True,
    ).encode()
).hexdigest()[:16]


def messages(q: Question) -> list[dict]:
    """Chat messages for a question; render with the model's own chat template."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{q.problem}\n\n{ANSWER_INSTRUCTIONS[q.kind]}"},
    ]
