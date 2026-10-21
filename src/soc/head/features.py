"""Stage 0 features: the frozen model's hidden state at the final prompt token.

That position sees the whole prompt (system, question, answer instruction and the empty
<think></think> the template adds) but no generated token, so it is what the model "knows"
before it starts answering. Saved per split as float16 arrays: last layer and middle layer.

    uv run --extra gpu python -m soc.head.features configs/stage0_rollouts.yaml SPLIT [SPLIT...]

Writes features/<split>.npz (qids, last, middle) into the rollout run's output directory, so
features and counts for a split live side by side.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from soc.data import messages


def load_model(name: str, revision: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    tok.padding_side = "left"  # so position -1 is every prompt's final token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=torch.bfloat16
        )
    except ValueError:  # Qwen3.5 checkpoints are registered as image-text-to-text
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            name, revision=revision, torch_dtype=torch.bfloat16
        )
    return tok, model.cuda().eval()


def final_token_states(hidden_states, layers: list[int]):
    """hidden_states: tuple (embeddings, layer 1, ..., layer L) of [batch, seq, dim] tensors.
    Returns {layer: [batch, dim]} at the last position (left padding puts the prompt end there)."""
    return {layer: hidden_states[layer][:, -1, :].float().cpu().numpy() for layer in layers}


def extract(questions, tok, model, chat_template_kwargs: dict, batch_size: int = 16):
    import torch

    n_layers = model.config.get_text_config().num_hidden_layers
    layers = {"last": n_layers, "middle": n_layers // 2}
    out = {k: [] for k in layers}
    for i in range(0, len(questions), batch_size):
        chunk = questions[i : i + batch_size]
        texts = [
            tok.apply_chat_template(
                messages(q), tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
            for q in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            res = model(**enc, output_hidden_states=True)
        states = final_token_states(res.hidden_states, list(layers.values()))
        for name, layer in layers.items():
            out[name].append(states[layer].astype(np.float16))
    return {k: np.concatenate(v) for k, v in out.items()}, layers


def main(config_path: str, splits: list[str]) -> None:
    from soc.data import load_split
    from soc.rollouts import storage
    from soc.rollouts.run import ROOT, RolloutConfig

    cfg = RolloutConfig.from_yaml(config_path)
    out = Path(os.environ.get("SOC_OUTPUT_DIR", ROOT / "outputs" / cfg.run_id))
    uri = os.environ.get("SOC_OUTPUT_URI")
    (out / "features").mkdir(parents=True, exist_ok=True)
    storage.sync_down(uri, out)
    if all((out / "features" / f"{s}.npz").exists() for s in splits):
        print("all features exist")
        return
    tok, model = load_model(cfg.model, cfg.vllm.get("revision"))
    for split in splits:
        if (out / "features" / f"{split}.npz").exists():
            print(f"{split}: features exist, skipping")
            continue
        qs = load_split(split)
        feats, layers = extract(qs, tok, model, cfg.chat_template_kwargs)
        np.savez(
            out / "features" / f"{split}.npz",
            qids=np.array([q.qid for q in qs]),
            layer_last=layers["last"],
            layer_middle=layers["middle"],
            **feats,
        )
        print(f"{split}: {len(qs)} questions, dim {feats['last'].shape[1]}")
        storage.sync_up(out, uri)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
