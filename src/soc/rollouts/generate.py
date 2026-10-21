"""Generation backends. `VLLMGenerator` runs on the GPU VM; tests use a fake with the same shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Sample:
    text: str
    n_tokens: int
    finish_reason: str  # "stop" or "length" (truncated at max_new_tokens)


class Generator(Protocol):
    name: str

    def generate(
        self,
        conversations: list[list[dict]],
        n: int,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[list[Sample]]: ...


class VLLMGenerator:
    """n samples per conversation with vLLM, rendered through the model's own chat template."""

    def __init__(self, model: str, chat_template_kwargs: dict | None = None, **llm_kwargs):
        from vllm import LLM

        self.name = model
        self.llm = LLM(model=model, **llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.chat_template_kwargs = chat_template_kwargs or {}

    def render(self, conversation: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs
        )

    def generate(self, conversations, n, temperature, max_new_tokens, seed):
        from vllm import SamplingParams

        params = SamplingParams(
            n=n, temperature=temperature, top_p=1.0, max_tokens=max_new_tokens, seed=seed
        )
        outputs = self.llm.generate([self.render(c) for c in conversations], params)
        return [
            [Sample(o.text, len(o.token_ids), o.finish_reason or "stop") for o in out.outputs]
            for out in outputs
        ]
