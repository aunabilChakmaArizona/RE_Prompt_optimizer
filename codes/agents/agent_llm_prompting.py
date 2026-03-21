from __future__ import annotations

import math
import time
from typing import Iterable, List, Sequence

import torch

def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def run_prompts(
    prompts: Sequence[str],
    *,
    model,
    tokenizer,
    system_message: str | None = None,
    max_new_tokens: int = 5000,
    batch_size: int = 8,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
    do_log: bool = False,
    **gen_kwargs,
) -> List[str]:
    if not prompts:
        return []

    outputs: List[str] = []
    target_device = getattr(model, "device", None)
    total_batches = math.ceil(len(prompts) / batch_size)
    prompting_start = time.monotonic()

    for batch_index, batch in enumerate(_batched(list(prompts), batch_size), start=1):
        if do_log:
            print(
                f"[agent_llm_prompting] processing prompts "
                f"batch {batch_index}/{total_batches} "
                f"(batch_size={len(batch)}, elapsed={time.monotonic() - prompting_start:.2f}s)"
            )
        if use_chat_template:
            messages_batch = []
            for prompt in batch:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                messages_batch.append(messages)
            formatted = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking,
                )
                for messages in messages_batch
            ]
        else:
            formatted = list(batch)

        model_inputs = tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        )
        if target_device is not None:
            model_inputs = model_inputs.to(target_device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs, max_new_tokens=max_new_tokens, **gen_kwargs
            )

        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        outputs.extend(tokenizer.batch_decode(trimmed, skip_special_tokens=True))

    return outputs


def run_prompt(
    prompt: str,
    *,
    model,
    tokenizer,
    system_message: str | None = None,
    max_new_tokens: int = 5000,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
    **gen_kwargs,
) -> str:
    return run_prompts(
        [prompt],
        model=model,
        tokenizer=tokenizer,
        system_message=system_message,
        max_new_tokens=max_new_tokens,
        batch_size=1,
        use_chat_template=use_chat_template,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        **gen_kwargs,
    )[0]
