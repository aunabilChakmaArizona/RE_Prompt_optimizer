from __future__ import annotations

import time
from typing import Iterable, List, Sequence, Tuple

import torch


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _get_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Token '{text}' is not in the tokenizer vocabulary.")
    if len(token_ids) != 1:
        raise ValueError(
            f"Token '{text}' is not a single token for this tokenizer."
        )
    return token_ids[0]


def _resolve_binary_token_id(tokenizer, base_token: str) -> int:
    try:
        return _get_token_id(tokenizer, base_token)
    except ValueError:
        spaced = f" {base_token}"
        return _get_token_id(tokenizer, spaced)


def run_binary_inference(
    prompts: Sequence[str],
    *,
    model,
    tokenizer,
    batch_size: int = 8,
    log_every: int = 20,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    yes_token: str = "yes",
    no_token: str = "no",
    yes_token_id: int | None = None,
    no_token_id: int | None = None,
    evolution_iteration: int | None = None,
    evolution_max_iterations: int | None = None,
) -> List[str]:
    if not prompts:
        return []

    start_time = time.perf_counter()
    total_prompts = len(prompts)
    num_batches = (total_prompts + batch_size - 1) // batch_size
    print(
        f"[agent_binary_inference] run_binary_inference: {total_prompts} prompts, "
        f"batch_size={batch_size}, batches={num_batches}"
    )

    yes_token_id = (
        yes_token_id
        if yes_token_id is not None
        else _resolve_binary_token_id(tokenizer, yes_token)
    )
    no_token_id = (
        no_token_id
        if no_token_id is not None
        else _resolve_binary_token_id(tokenizer, no_token)
    )

    predictions: List[str] = []
    target_device = getattr(model, "device", None)
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    if evolution_iteration is None:
        iteration_prefix = ""
    elif evolution_max_iterations is None:
        iteration_prefix = f"iter {evolution_iteration} "
    else:
        iteration_prefix = f"iter {evolution_iteration}/{evolution_max_iterations} "

    try:
        tokenizer.padding_side = "left"
        for batch_index, batch in enumerate(
            _batched(list(prompts), batch_size),
            start=1,
        ):
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
                [yes_token if y >= n else no_token for y, n in zip(yes_logits, no_logits)]
            )

            if log_every and (batch_index % log_every == 0 or batch_index == num_batches):
                batch_elapsed = time.perf_counter() - start_time
                print(
                    f"[agent_binary_inference] {iteration_prefix} Processed {batch_index}/{num_batches} batches "
                    f"in {batch_elapsed:.2f}s",
                    flush=True,
                )
    finally:
        tokenizer.padding_side = original_padding_side

    elapsed = time.perf_counter() - start_time
    print(f"[agent_binary_inference] run_binary_inference: done in {elapsed:.2f}s")
    print(f"[agent_binary_inference] overall time passed: {elapsed:.2f}s")
    return predictions
