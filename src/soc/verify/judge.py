"""Cached LLM judge for short-answer evaluation (TriviaQA, SimpleQA). Never used for training.

Each verdict is cached in SQLite, keyed by judge model, prompt version, question, gold answers
and the extracted answer, so re-scoring a run costs nothing and duplicate answers across
rollouts are judged once. Needs the `judge` extra and Anthropic credentials.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JUDGE_MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "judge-v1"
VERDICTS = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")

PROMPT = """You are grading an answer to a factual question against the reference answer.

Question: {question}
Reference answer: {gold}{aliases}
Answer to grade: {answer}

Grade CORRECT if the answer means the same thing as the reference (ignore capitalisation, \
punctuation, minor spelling differences and extra detail that does not contradict it). \
Grade INCORRECT if it gives a different or contradictory answer, or hedges between several \
candidates. Grade NOT_ATTEMPTED if it declines to answer or says it does not know.

Reply with exactly one word: CORRECT, INCORRECT or NOT_ATTEMPTED."""


class Judge:
    def __init__(self, cache_path: str | Path, client=None, model: str = JUDGE_MODEL):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client, self.model = client, model
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, verdict TEXT)")
        self._db.commit()

    def _key(self, question: str, golds: list[str], answer: str) -> str:
        payload = json.dumps([self.model, PROMPT_VERSION, question, golds, answer])
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cached(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT verdict FROM verdicts WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _store(self, key: str, verdict: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO verdicts VALUES (?, ?)", (key, verdict))
            self._db.commit()

    def grade(self, question: str, golds: list[str], answer: str) -> str:
        """One of VERDICTS."""
        key = self._key(question, golds, answer)
        if (hit := self._cached(key)) is not None:
            return hit
        aliases = f"\nAlso accepted: {'; '.join(golds[1:20])}" if len(golds) > 1 else ""
        prompt = PROMPT.format(question=question, gold=golds[0], aliases=aliases, answer=answer)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"judge refused: {response.stop_details}")
        text = "".join(b.text for b in response.content if b.type == "text")
        found = re.findall(r"NOT_ATTEMPTED|INCORRECT|CORRECT", text.upper())
        if not found:
            raise RuntimeError(f"unparseable judge reply: {text!r}")
        verdict = found[-1]
        self._store(key, verdict)
        return verdict

    def grade_many(self, items: list[tuple[str, list[str], str]], workers: int = 16) -> list[str]:
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(lambda it: self.grade(*it), items))
