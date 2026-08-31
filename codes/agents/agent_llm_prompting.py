from __future__ import annotations

import math
import time
from typing import Iterable, List, Sequence

import torch

from agents.agent_memory import clear_cuda_cache
from agents.agent_token_usage import TokenUsage, create_token_usage

_SAMPLE_LOGGED_LABELS: set[str] = set()


def _log_first_sample(label: str, prompt: str, output: str) -> None:
    """Print one representative prompt and response for a named run."""
    if label in _SAMPLE_LOGGED_LABELS:
        return
    _SAMPLE_LOGGED_LABELS.add(label)
    print(f"\n[{label}] sample first prompt:\n{prompt}", flush=True)
    print(f"\n[{label}] sample first output:\n{output}", flush=True)


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    """Yield fixed-size slices from a sequence of prompts."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _token_id_set(token_ids) -> set[int]:
    """Convert one token ID or a sequence of IDs into an integer set."""
    if token_ids is None:
        return set()
    if isinstance(token_ids, int):
        return {token_ids}
    return {int(token_id) for token_id in token_ids}


def _generated_token_count(output_ids, model, tokenizer) -> int:
    """Count generated model steps through the first stop token, excluding padding."""
    generation_config = getattr(model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token_ids = _token_id_set(eos_token_id)
    pad_token_id = getattr(generation_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

    output_token_count = 0
    token_id_values = output_ids.tolist() if hasattr(output_ids, "tolist") else output_ids
    for token_id in token_id_values:
        token_id = int(token_id)
        if token_id == pad_token_id and token_id not in eos_token_ids:
            break
        output_token_count += 1
        if token_id in eos_token_ids:
            break
    return output_token_count


def run_prompts(
    prompts: Sequence[str],
    *,
    model,
    tokenizer,
    system_message: str | None = None,
    max_new_tokens: int = 10000,
    batch_size: int = 8,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
    do_log: bool = False,
    log_label: str | None = None,
    return_token_usage: bool = False,
    **gen_kwargs,
) -> List[str] | tuple[List[str], list[TokenUsage]]:
    """Generate prompts and optionally return exact per-prompt token counts."""
    if not prompts:
        return ([], []) if return_token_usage else []

    outputs: List[str] = []
    token_usages: list[TokenUsage] = []
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
        batch_outputs = tokenizer.batch_decode(trimmed, skip_special_tokens=True)
        outputs.extend(batch_outputs)
        if return_token_usage:
            input_token_counts = model_inputs.attention_mask.sum(dim=1).tolist()
            token_usages.extend(
                create_token_usage(
                    int(input_token_count),
                    _generated_token_count(output_ids, model, tokenizer),
                )
                for input_token_count, output_ids in zip(input_token_counts, trimmed)
            )
        if log_label and batch and batch_outputs:
            _log_first_sample(log_label, batch[0], batch_outputs[0])
    clear_cuda_cache()

    if return_token_usage:
        return outputs, token_usages
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
    log_label: str | None = None,
    **gen_kwargs,
) -> str:
    """Generate one response with the shared batched prompting implementation."""
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
        log_label=log_label,
        **gen_kwargs,
    )[0]
