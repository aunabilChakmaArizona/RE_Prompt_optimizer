from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _get_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Token '{text}' is not in the tokenizer vocabulary.")
    return token_ids[0]


def run_binary_inference(
    prompts: Sequence[str],
    *,
    model,
    tokenizer,
    batch_size: int = 8,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    yes_token: str = "yes",
    no_token: str = "no",
) -> List[str]:
    if not prompts:
        return []

    yes_token_id = _get_token_id(tokenizer, yes_token)
    no_token_id = _get_token_id(tokenizer, no_token)

    predictions: List[str] = []
    target_device = getattr(model, "device", None)

    for batch in _batched(list(prompts), batch_size):
        if use_chat_template:
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking,
                )
                for prompt in batch
            ]
        else:
            formatted = list(batch)

        model_inputs = tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        )
        if target_device is not None:
            model_inputs = model_inputs.to(target_device)

        with torch.inference_mode():
            outputs = model(**model_inputs, use_cache=False)
            logits = outputs.logits[:, -1, [yes_token_id, no_token_id]]

        yes_logits = logits[:, 0]
        no_logits = logits[:, 1]
        predictions.extend(
            [yes_token if y > n else no_token for y, n in zip(yes_logits, no_logits)]
        )

    return predictions