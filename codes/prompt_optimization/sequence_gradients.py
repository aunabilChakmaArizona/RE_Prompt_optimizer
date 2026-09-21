"""QA answer gradients, reasoning conditioning, and local token utilities."""

from __future__ import annotations

import math
import random
import re
import time
from typing import Any, Iterable, Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from agents.agent_memory import clear_cuda_cache
from prompt_optimization.meta_prompts import qa_task_description, qa_task_label
from prompt_optimization.qa_task import (
    QAMode,
    choices_as_text,
    context_as_text,
    render_qa_prompt,
)
from prompt_optimization.run_io import format_elapsed


ANSWER_EXTRACTOR_PREFIX = "\n\nTherefore, the final answer is <answer>"
ANSWER_BLOCK_PATTERN = re.compile(
    r"<answer>.*?</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)


def replace_or_append_gold_answer(
    reasoning_trace: str,
    gold_answer: str,
) -> tuple[str, str, str]:
    """Replace the last tagged prediction, or append a gold-answer fallback."""
    matches = list(ANSWER_BLOCK_PATTERN.finditer(reasoning_trace))
    if matches:
        last_match = matches[-1]
        answer_prefix = "<answer>"
        gold_block = f"{answer_prefix}{gold_answer}</answer>"
        assistant_answer = (
            f"{reasoning_trace[:last_match.start()]}"
            f"{gold_block}"
            f"{reasoning_trace[last_match.end():]}"
        )
        return assistant_answer, answer_prefix, "replaced_existing_answer"

    answer_prefix = ANSWER_EXTRACTOR_PREFIX
    assistant_answer = (
        f"{reasoning_trace.rstrip()}{answer_prefix}{gold_answer}</answer>"
    )
    return assistant_answer, answer_prefix, "appended_missing_answer_fallback"


def _log_batch_checkpoint(
    label: str,
    batch_index: int,
    total_batches: int,
    processed: int,
    total_records: int,
    started_at: float,
) -> None:
    """Print the first, last, and roughly ten evenly spaced batch checkpoints."""
    interval = max(1, math.ceil(total_batches / 10))
    if batch_index == 1 or batch_index == total_batches or batch_index % interval == 0:
        print(
            f"[qa:{label}] batch {batch_index}/{total_batches} | "
            f"examples={processed}/{total_records} | "
            f"elapsed={format_elapsed(time.monotonic() - started_at)}",
            flush=True,
        )


def require_torch() -> None:
    """Raise a clear error when gradient code runs outside the ML environment."""
    if torch is None or F is None:
        raise ImportError(
            "QA gradient optimization requires PyTorch in the active environment."
        )


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    """Yield fixed-size slices from a sequence."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def model_device(model) -> torch.device:
    """Return the device holding the target model input embeddings."""
    require_torch()
    return model.get_input_embeddings().weight.device


def freeze_model_parameters(model) -> None:
    """Disable parameter gradients while preserving input-embedding gradients."""
    require_torch()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def render_teacher_forced_qa(
    instruction_prompt: str,
    record: dict[str, Any],
    mode: QAMode,
    tokenizer,
    reasoning_trace: str | None = None,
) -> dict[str, Any]:
    """Render one gold answer, optionally conditioned on generated reasoning."""
    user_prompt = render_qa_prompt(instruction_prompt, record, mode)
    gold_answer = str(record["answer"]).strip()
    if mode.task_name == "openbookqa":
        gold_answer = gold_answer.upper()
    if reasoning_trace is None:
        answer_prefix = "<answer>"
        assistant_answer = f"{answer_prefix}{gold_answer}</answer>"
        answer_target_mode = "direct_gold_answer"
    else:
        assistant_answer, answer_prefix, answer_target_mode = (
            replace_or_append_gold_answer(reasoning_trace, gold_answer)
        )
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_answer},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=mode.enable_thinking,
    )
    instruction_start = rendered.find(instruction_prompt)
    if instruction_start < 0:
        raise ValueError("Instruction text was not found after chat-template rendering.")
    # Chat templates may trim leading whitespace from assistant content.  Align
    # against the invariant answer block rather than the optional fallback
    # preamble, which begins with newlines and is therefore not round-tripped by
    # Gemma's template.
    answer_tag = "<answer>"
    target_text = f"{answer_tag}{gold_answer}</answer>"
    target_start = rendered.rfind(target_text)
    if target_start < 0:
        record_id = record.get("id", "<unknown>")
        raise ValueError(
            "Gold answer target was not found after chat-template rendering "
            f"for record {record_id!r}; target={target_text!r}, "
            f"answer_target_mode={answer_target_mode!r}, "
            f"rendered_suffix={rendered[-500:]!r}."
        )
    answer_start = target_start + len(answer_tag)
    return {
        "text": rendered,
        "instruction_start": instruction_start,
        "instruction_end": instruction_start + len(instruction_prompt),
        "label_start": answer_start,
        "label_end": answer_start + len(gold_answer),
        "gold_label": gold_answer,
        "reasoning_conditioned": reasoning_trace is not None,
        "answer_target_mode": answer_target_mode,
    }


def overlapping_token_positions(
    offsets: Sequence[Sequence[int]],
    start_char: int,
    end_char: int,
) -> list[int]:
    """Find non-empty tokens whose character ranges overlap one text span."""
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start
        and token_start < end_char
        and token_end > start_char
    ]


def _encode_teacher_forced_batch(
    instruction_prompt: str,
    records: Sequence[dict[str, Any]],
    mode: QAMode,
    tokenizer,
    reasoning_traces: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Tokenize a batch and align instruction and gold-answer tokens."""
    if reasoning_traces is not None and len(reasoning_traces) != len(records):
        raise ValueError("Reasoning traces and records must have equal lengths.")
    traces = list(reasoning_traces) if reasoning_traces is not None else [None] * len(records)
    rendered = [
        render_teacher_forced_qa(
            instruction_prompt,
            record,
            mode,
            tokenizer,
            reasoning_trace=trace,
        )
        for record, trace in zip(records, traces)
    ]
    encoded = tokenizer(
        [item["text"] for item in rendered],
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offset_rows = encoded.pop("offset_mapping").tolist()
    instruction_positions: list[list[int]] = []
    answer_positions: list[list[int]] = []
    canonical_instruction_ids: list[int] | None = None
    canonical_offsets: list[tuple[int, int]] | None = None
    for row_index, (item, offsets) in enumerate(zip(rendered, offset_rows)):
        prompt_positions = overlapping_token_positions(
            offsets,
            item["instruction_start"],
            item["instruction_end"],
        )
        label_positions = overlapping_token_positions(
            offsets,
            item["label_start"],
            item["label_end"],
        )
        if not prompt_positions or not label_positions:
            raise ValueError("Unable to align instruction or answer-label tokens.")
        if any(position <= 0 for position in label_positions):
            raise ValueError("Answer-label token has no causal predecessor.")
        prompt_ids = encoded["input_ids"][row_index, prompt_positions].tolist()
        relative_offsets = [
            (
                max(0, int(offsets[position][0]) - item["instruction_start"]),
                min(
                    len(instruction_prompt),
                    int(offsets[position][1]) - item["instruction_start"],
                ),
            )
            for position in prompt_positions
        ]
        if canonical_instruction_ids is None:
            canonical_instruction_ids = prompt_ids
            canonical_offsets = relative_offsets
        elif prompt_ids != canonical_instruction_ids:
            raise ValueError("Instruction tokenization changed across QA records.")
        instruction_positions.append(prompt_positions)
        answer_positions.append(label_positions)
    return {
        "encoded": encoded,
        "instruction_positions": instruction_positions,
        "answer_positions": answer_positions,
        "instruction_token_ids": canonical_instruction_ids or [],
        "instruction_offsets": canonical_offsets or [],
    }


def _encode_teacher_forced_scoring_batch(
    instruction_prompts: Sequence[str],
    records: Sequence[dict[str, Any]],
    mode: QAMode,
    tokenizer,
    reasoning_traces: Sequence[str | None],
) -> dict[str, Any]:
    """Tokenize mixed candidate/example sequences for HF task-loss scoring."""
    if not (
        len(instruction_prompts) == len(records) == len(reasoning_traces)
    ):
        raise ValueError("Prompts, records, and reasoning traces must have equal lengths.")
    rendered = [
        render_teacher_forced_qa(
            instruction_prompt,
            record,
            mode,
            tokenizer,
            reasoning_trace=reasoning_trace,
        )
        for instruction_prompt, record, reasoning_trace in zip(
            instruction_prompts,
            records,
            reasoning_traces,
        )
    ]
    encoded = tokenizer(
        [item["text"] for item in rendered],
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offset_rows = encoded.pop("offset_mapping").tolist()
    answer_positions = []
    for item, offsets in zip(rendered, offset_rows):
        positions = overlapping_token_positions(
            offsets,
            item["label_start"],
            item["label_end"],
        )
        if not positions or any(position <= 0 for position in positions):
            raise ValueError("Unable to align causal gold-answer tokens for HF scoring.")
        answer_positions.append(positions)
    return {"encoded": encoded, "answer_positions": answer_positions}


def _label_loss_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_positions: Sequence[Sequence[int]],
    reduction: str,
) -> torch.Tensor:
    """Average gold-answer token loss per problem, then reduce problems."""
    require_torch()
    example_losses = []
    for row_index, positions in enumerate(answer_positions): #aunabil2nd: this should be weired if the answer is wrong: we reasoned one thing and put the ground truth another thing.
        if not positions:
            raise ValueError("Every problem must provide at least one answer token.")
        selected_logits = torch.stack(
            [logits[row_index, token_position - 1, :] for token_position in positions]
        )
        selected_targets = torch.stack(
            [input_ids[row_index, token_position] for token_position in positions]
        )
        example_losses.append(
            F.cross_entropy(selected_logits, selected_targets, reduction="mean")
        )
    if not example_losses:
        raise ValueError("No gold-answer tokens were selected for task loss.")
    losses = torch.stack(example_losses)
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    if reduction == "none":
        return losses
    raise ValueError(f"Unsupported answer-loss reduction: {reduction!r}")


def collect_instruction_gradients(
    instruction_prompt: str,
    records: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    model,
    tokenizer,
    batch_size: int,
    reasoning_traces: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Average gold-answer gradients over instruction token embeddings."""
    require_torch()
    if not records:
        raise ValueError("Gradient records must not be empty.")
    if batch_size <= 0:
        raise ValueError("Gradient batch size must be positive.")
    if reasoning_traces is not None and len(reasoning_traces) != len(records):
        raise ValueError("Reasoning traces and gradient records must have equal lengths.")
    freeze_model_parameters(model)
    device = model_device(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gradient_sums: torch.Tensor | None = None
    embedding_sums: torch.Tensor | None = None
    canonical_ids: list[int] | None = None
    canonical_offsets: list[tuple[int, int]] | None = None
    total_loss = 0.0
    processed = 0
    total_batches = math.ceil(len(records) / batch_size)
    started_at = time.monotonic()
    record_list = list(records)
    trace_list = list(reasoning_traces) if reasoning_traces is not None else None
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for batch_index, start in enumerate(
            range(0, len(records), batch_size),
            start=1,
        ):
            chunk = record_list[start : start + batch_size]
            trace_chunk = ( #aunabil2nd: what is chunk vs trace_chunk
                trace_list[start : start + batch_size]
                if trace_list is not None
                else None
            )
            payload = _encode_teacher_forced_batch(
                instruction_prompt,
                chunk,
                mode,
                tokenizer,
                reasoning_traces=trace_chunk,
            )
            encoded = payload["encoded"]
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            base_embeddings = model.get_input_embeddings()(input_ids).detach()
            input_embeddings = base_embeddings.clone().requires_grad_(True)
            outputs = model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                use_cache=False,
            )
            loss_sum = _label_loss_from_logits(
                outputs.logits,
                input_ids,
                payload["answer_positions"],
                reduction="sum",
            )
            scaled_loss = loss_sum / len(records)
            scaled_loss.backward()
            if input_embeddings.grad is None:
                raise RuntimeError("Input embeddings did not receive QA task gradients.")

            chunk_ids = list(payload["instruction_token_ids"])
            chunk_offsets = list(payload["instruction_offsets"])
            if canonical_ids is None:
                canonical_ids = chunk_ids
                canonical_offsets = chunk_offsets
                hidden_size = input_embeddings.size(-1)
                gradient_sums = torch.zeros(
                    (len(chunk_ids), hidden_size),
                    dtype=torch.float32,
                    device=device,
                )
                embedding_sums = torch.zeros_like(gradient_sums)
            elif chunk_ids != canonical_ids:
                raise ValueError("Instruction tokenization changed between gradient batches.")

            assert gradient_sums is not None and embedding_sums is not None
            for row_index, positions in enumerate(payload["instruction_positions"]):
                gradient_sums += input_embeddings.grad[row_index, positions, :].float()
                embedding_sums += base_embeddings[row_index, positions, :].float()
            total_loss += float(loss_sum.detach().cpu())
            processed += len(chunk)
            _log_batch_checkpoint(
                "gradient",
                batch_index,
                total_batches,
                processed,
                len(records),
                started_at,
            )
            del encoded, input_ids, attention_mask, base_embeddings, input_embeddings
            del outputs, loss_sum, scaled_loss
            clear_cuda_cache()
    finally:
        tokenizer.padding_side = original_padding_side

    if gradient_sums is None or embedding_sums is None or canonical_ids is None:
        raise RuntimeError("No prompt gradients were accumulated.")
    gradient_means = gradient_sums
    embedding_means = embedding_sums / processed
    norms = gradient_means.norm(dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(canonical_ids)
    peak_allocated_bytes = None #aunabil2nd: why these peack_ memory variables for? what is the use?
    peak_reserved_bytes = None
    if device.type == "cuda":
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    return {
        "instruction_prompt": instruction_prompt,
        "num_records": processed,
        "mean_task_loss": total_loss / processed,
        "reasoning_conditioned": reasoning_traces is not None,
        "gradient_peak_allocated_bytes": peak_allocated_bytes,
        "gradient_peak_reserved_bytes": peak_reserved_bytes,
        "token_ids": canonical_ids,
        "tokens": tokens,
        "offsets": canonical_offsets or [],
        "gradient_norms": norms.detach().cpu().tolist(),
        "gradient_vectors": gradient_means.detach(),
        "embedding_vectors": embedding_means.detach(),
        "token_gradients": [
            {
                "token_index": index,
                "token_id": token_id,
                "token": token,
                "char_start": (canonical_offsets or [])[index][0],
                "char_end": (canonical_offsets or [])[index][1],
                "gradient_norm": float(norms[index].detach().cpu()),
            }
            for index, (token_id, token) in enumerate(zip(canonical_ids, tokens))
        ],
    }


def score_instruction_task_loss(
    instruction_prompt: str,
    records: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    model,
    tokenizer,
    batch_size: int,
    reasoning_traces: Sequence[str] | None = None,
) -> float:
    """Measure mean per-problem gold-answer cross entropy for one prompt."""
    return score_instruction_task_losses(
        [instruction_prompt],
        records,
        mode=mode,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        reasoning_traces=reasoning_traces,
    )[0]


def score_instruction_task_losses(
    instruction_prompts: Sequence[str],
    records: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    model,
    tokenizer,
    batch_size: int,
    reasoning_traces: Sequence[str] | None = None,
) -> list[float]:
    """Batch candidate/example pairs and calculate every task loss with HF."""
    require_torch()
    prompts = list(instruction_prompts)
    if not prompts:
        return []
    if not records:
        raise ValueError("Task-loss records must not be empty.")
    if batch_size <= 0:
        raise ValueError("Task-loss batch size must be positive.")
    if reasoning_traces is not None and len(reasoning_traces) != len(records):
        raise ValueError("Reasoning traces and scored records must have equal lengths.")

    record_list = list(records)
    trace_list: list[str | None] = (
        list(reasoning_traces) if reasoning_traces is not None else [None] * len(records)
    )
    work_items = [
        (candidate_index, prompt, record, reasoning_trace)
        for candidate_index, prompt in enumerate(prompts)
        for record, reasoning_trace in zip(record_list, trace_list)
    ]
    loss_sums = [0.0] * len(prompts)
    loss_counts = [0] * len(prompts)
    device = model_device(model)
    total_batches = math.ceil(len(work_items) / batch_size)
    scoring_started_at = time.monotonic()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        with torch.inference_mode():
            for batch_index, chunk in enumerate(
                batched(work_items, batch_size),
                start=1,
            ):
                payload = _encode_teacher_forced_scoring_batch(
                    [item[1] for item in chunk],
                    [item[2] for item in chunk],
                    mode,
                    tokenizer,
                    [item[3] for item in chunk],
                )
                encoded = payload["encoded"]
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                losses = _label_loss_from_logits(
                    outputs.logits,
                    input_ids,
                    payload["answer_positions"],
                    reduction="none",
                )
                for item, loss in zip(chunk, losses.detach().cpu().tolist()):
                    candidate_index = int(item[0])
                    loss_sums[candidate_index] += float(loss)
                    loss_counts[candidate_index] += 1
                del encoded, input_ids, attention_mask, outputs, losses
                clear_cuda_cache()
                _log_batch_checkpoint(
                    "candidate-task-loss",
                    batch_index,
                    total_batches,
                    min(batch_index * batch_size, len(work_items)),
                    len(work_items),
                    scoring_started_at,
                )
    finally:
        tokenizer.padding_side = original_padding_side
    if any(count != len(records) for count in loss_counts):
        raise RuntimeError("HF task-loss scoring did not process every candidate example.")
    return [
        loss_sum / count
        for loss_sum, count in zip(loss_sums, loss_counts)
    ]


def score_instruction_nll(instruction_prompt: str, model, tokenizer) -> float:
    """Measure mean next-token negative log likelihood of the instruction."""
    require_torch()
    device = model_device(model)
    encoded = tokenizer(
        instruction_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    if input_ids.size(1) < 2:
        return 0.0
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)
        loss = F.cross_entropy(
            outputs.logits[:, :-1, :].reshape(-1, outputs.logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            reduction="mean",
        )
    value = float(loss.detach().cpu())
    del encoded, input_ids, outputs, loss
    clear_cuda_cache()
    return value


def score_instruction_nlls(
    instruction_prompts: Sequence[str],
    model,
    tokenizer,
    *,
    batch_size: int,
) -> list[float]:
    """Measure prompt fluency for several candidate instructions in HF batches."""
    require_torch()
    if not instruction_prompts:
        return []
    if batch_size <= 0:
        raise ValueError("Fluency batch size must be positive.")
    device = model_device(model)
    values = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        with torch.inference_mode():
            for chunk in batched(list(instruction_prompts), batch_size):
                encoded = tokenizer(
                    list(chunk),
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                for row_index in range(input_ids.size(0)):
                    valid_positions = torch.nonzero(
                        attention_mask[row_index],
                        as_tuple=False,
                    ).flatten()
                    if valid_positions.numel() < 2:
                        values.append(0.0)
                        continue
                    source_positions = valid_positions[:-1]
                    target_positions = valid_positions[1:]
                    loss = F.cross_entropy(
                        outputs.logits[row_index, source_positions, :],
                        input_ids[row_index, target_positions],
                        reduction="mean",
                    )
                    values.append(float(loss.detach().cpu()))
                del encoded, input_ids, attention_mask, outputs
                clear_cuda_cache()
    finally:
        tokenizer.padding_side = original_padding_side
    return values


def score_combined_objective(
    instruction_prompt: str,
    records: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    model,
    tokenizer,
    batch_size: int,
    fluency_lambda: float,
    reasoning_traces: Sequence[str] | None = None,
) -> dict[str, float]:
    """Combine gold-answer loss with the weighted prompt-fluency penalty."""
    task_loss = score_instruction_task_loss(
        instruction_prompt,
        records,
        mode=mode,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        reasoning_traces=reasoning_traces,
    )
    instruction_nll = score_instruction_nll(instruction_prompt, model, tokenizer)
    return {
        "task_loss": task_loss,
        "instruction_nll": instruction_nll,
        "combined_score": task_loss + fluency_lambda * instruction_nll,
    }


def score_combined_objectives(
    instruction_prompts: Sequence[str],
    records: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    model,
    tokenizer,
    batch_size: int,
    fluency_lambda: float,
    reasoning_traces: Sequence[str] | None = None,
) -> list[dict[str, float]]:
    """Score all candidate task losses and fluency penalties in HF batches."""
    prompts = list(instruction_prompts)
    if not prompts:
        return []
    task_losses = score_instruction_task_losses(
        prompts,
        records,
        mode=mode,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        reasoning_traces=reasoning_traces,
    )
    instruction_nlls = score_instruction_nlls(
        prompts,
        model,
        tokenizer,
        batch_size=batch_size,
    )
    return [
        {
            "task_loss": task_loss,
            "instruction_nll": instruction_nll,
            "combined_score": task_loss + fluency_lambda * instruction_nll,
        }
        for task_loss, instruction_nll in zip(task_losses, instruction_nlls)
    ]


def _differentiable_instruction_nll(
    instruction_token_ids: Sequence[int],
    token_index: int,
    selected_embedding: torch.Tensor,
    *,
    model,
) -> torch.Tensor:
    """Calculate instruction NLL through one relaxed candidate embedding."""
    require_torch()
    if len(instruction_token_ids) < 2:
        return selected_embedding.sum() * 0.0
    device = model_device(model)
    input_ids = torch.tensor(
        [list(instruction_token_ids)],
        dtype=torch.long,
        device=device,
    )
    base_embeddings = model.get_input_embeddings()(input_ids).detach()
    replacement_embeddings = torch.zeros_like(base_embeddings)
    replacement_mask = torch.zeros(
        base_embeddings.shape[:2],
        dtype=base_embeddings.dtype,
        device=device,
    )
    replacement_embeddings[0, token_index, :] = selected_embedding[0]
    replacement_mask[0, token_index] = 1.0
    input_embeddings = (
        base_embeddings * (1.0 - replacement_mask.unsqueeze(-1))
        + replacement_embeddings * replacement_mask.unsqueeze(-1)
    )
    outputs = model(inputs_embeds=input_embeddings, use_cache=False)
    shifted_logits = outputs.logits[:, :-1, :].contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        reduction="mean",
    )


def _editable_token(
    record: dict[str, Any],
    instruction_prompt: str,
    check_isalnum: bool,
) -> bool:
    """Optionally require an aligned token to contain a letter or number."""
    if not check_isalnum:
        return True
    start = int(record["char_start"])
    end = int(record["char_end"])
    return any(character.isalnum() for character in instruction_prompt[start:end])


def build_gradient_region_pool(
    gradient_analysis: dict[str, Any],
    *,
    max_region_tokens: int,
    expansion_threshold_ratio: float,
    check_isalnum: bool = True,
) -> list[dict[str, Any]]:
    """Expand gradient peaks into non-overlapping editable local regions."""
    if max_region_tokens <= 0:
        raise ValueError("Maximum region token length must be positive.")
    if not 0.0 <= expansion_threshold_ratio <= 1.0:
        raise ValueError("Region expansion threshold must be between zero and one.")
    prompt = str(gradient_analysis["instruction_prompt"])
    records = list(gradient_analysis["token_gradients"])
    editable = {
        int(record["token_index"])
        for record in records
        if _editable_token(record, prompt, check_isalnum)
    }
    scores = {int(record["token_index"]): float(record["gradient_norm"]) for record in records}
    ranked_peaks = sorted(editable, key=lambda index: (scores[index], -index), reverse=True)
    claimed: set[int] = set()
    regions: list[dict[str, Any]] = []
    for peak in ranked_peaks:
        if peak in claimed:
            continue
        indices = [peak]
        threshold = scores[peak] * expansion_threshold_ratio
        while len(indices) < max_region_tokens:
            candidates = []
            left = min(indices) - 1
            right = max(indices) + 1
            if left in editable and left not in claimed and left not in indices:
                candidates.append(left)
            if right in editable and right not in claimed and right not in indices:
                candidates.append(right)
            if not candidates:
                break
            best_neighbor = max(candidates, key=lambda index: scores[index])
            if scores[best_neighbor] < threshold:
                break
            indices.append(best_neighbor)
            indices.sort()
        claimed.update(indices)
        start_index = min(indices)
        end_index = max(indices)
        start_char = int(records[start_index]["char_start"])
        end_char = int(records[end_index]["char_end"])
        region_text = prompt[start_char:end_char]
        if not region_text.strip():
            continue
        regions.append(
            {
                "peak_token_index": peak,
                "start_token": start_index,
                "end_token": end_index,
                "token_indices": indices,
                "token_count": len(indices),
                "start_char": start_char,
                "end_char": end_char,
                "region_text": region_text,
                "gradient_score": max(scores[index] for index in indices),
                "gradient_norms": [scores[index] for index in indices],
            }
        )
    return sorted(regions, key=lambda item: item["gradient_score"], reverse=True)


def select_gradient_regions(
    region_pool: Sequence[dict[str, Any]],
    *,
    count: int,
    selection_mode: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Choose highest-gradient or random regions from the common region pool."""
    if count <= 0:
        raise ValueError("Selected region count must be positive.")
    available = list(region_pool)
    if not available:
        raise ValueError("No editable gradient regions were found.")
    selection_count = min(count, len(available))
    if selection_mode == "top_gradient":
        selected = available[:selection_count]
    elif selection_mode == "random":
        selected = rng.sample(available, selection_count)
        selected.sort(key=lambda item: int(item["start_char"]))
    else:
        raise ValueError(f"Unsupported region selection mode: {selection_mode!r}")
    output = []
    for rank, region in enumerate(selected, start=1):
        output.append({**region, "region_rank": rank, "selection_mode": selection_mode})
    return output


def mark_selected_regions(
    instruction_prompt: str,
    regions: Sequence[dict[str, Any]],
) -> str:
    """Insert stable numbered span tags around selected prompt regions."""
    marked = instruction_prompt
    for region in sorted(regions, key=lambda item: int(item["start_char"]), reverse=True):
        rank = int(region["region_rank"])
        start = int(region["start_char"])
        end = int(region["end_char"])
        marked = (
            marked[:start]
            + f"<span_{rank}>"
            + marked[start:end]
            + f"</span_{rank}>"
            + marked[end:]
        )
    return marked


def align_replacement_whitespace(source_text: str, replacement: str) -> str:
    """Preserve boundary whitespace from a selected source region."""
    leading = source_text[: len(source_text) - len(source_text.lstrip())]
    trailing = source_text[len(source_text.rstrip()) :]
    return leading + replacement.strip() + trailing


def replace_selected_regions(
    instruction_prompt: str,
    regions: Sequence[dict[str, Any]],
    replacements: dict[int, str],
) -> str:
    """Apply local replacements from right to left using original character spans."""
    revised = instruction_prompt
    for region in sorted(regions, key=lambda item: int(item["start_char"]), reverse=True):
        rank = int(region["region_rank"])
        if rank not in replacements:
            continue
        start = int(region["start_char"])
        end = int(region["end_char"])
        replacement = align_replacement_whitespace(
            str(region["region_text"]),
            replacements[rank],
        )
        revised = revised[:start] + replacement + revised[end:]
    return revised.strip()


def _allowed_candidate_token(
    tokenizer,
    token_id: int,
    check_isalnum: bool,
) -> bool:
    """Filter invalid candidates and optionally require word-like token text."""
    if token_id in set(tokenizer.all_special_ids):
        return False
    text = tokenizer.decode([token_id], skip_special_tokens=True)
    if not text.strip():
        return False
    if check_isalnum and not any(character.isalnum() for character in text):
        return False
    return text.isascii() and "\n" not in text and "\r" not in text


def same_length_single_token_replacement_prompt(
    tokenizer,
    source_token_ids: Sequence[int],
    token_index: int,
    candidate_token_id: int,
) -> str | None:
    """Return a prompt when one-token replacement preserves total token count."""
    candidate_token_ids = [int(value) for value in source_token_ids]
    candidate_token_ids[token_index] = int(candidate_token_id)
    candidate_prompt = tokenizer.decode(
        candidate_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    retokenized_ids = tokenizer.encode(
        candidate_prompt,
        add_special_tokens=False,
    )
    if len(retokenized_ids) != len(candidate_token_ids):
        return None
    return candidate_prompt


def qa_proposal_header(record: dict[str, Any], mode: QAMode) -> str:
    """Build GreaTer's task-specific proposal context around one example."""
    example_lines = []
    if mode.task_name == "hotpotqa":
        example_lines.extend(["Context:", context_as_text(record)])
    example_lines.append(f"Question: {record['question']}")
    if mode.task_name == "openbookqa":
        example_lines.append(f"Choices: {choices_as_text(record)}")
    example_input = "\n".join(example_lines)
    return (
        f"You are optimizing an instruction prompt for a {qa_task_label(mode)} "
        "model.\n\n"
        f"{qa_task_description(mode)}\n\n"
        f"Example model input:\n{example_input}\n\n"
        "Write an instruction that should appear before this type of input and help "
        "the model solve the task.\n\nInstruction:\n"
    )


def proposal_token_candidates(
    gradient_analysis: dict[str, Any],
    token_index: int,
    proposal_records: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    model,
    tokenizer,
    top_k: int,
    min_candidates: int,
    check_isalnum: bool,
) -> tuple[list[int], dict[str, Any]]:
    """Intersect or frequency-rank GreaTer token proposals across QA examples."""
    require_torch()
    if not proposal_records:
        raise ValueError("GreaTer proposal records must not be empty.")
    prompt = str(gradient_analysis["instruction_prompt"])
    token_record = gradient_analysis["token_gradients"][token_index]
    prefix = prompt[: int(token_record["char_start"])]
    source_token_ids = [int(value) for value in gradient_analysis["token_ids"]]
    current_id = source_token_ids[token_index]
    device = model_device(model)
    candidate_sets: list[list[int]] = []
    unstable_candidate_ids: set[int] = set()
    with torch.inference_mode():
        for record in proposal_records:
            context = qa_proposal_header(record, mode) + prefix
            encoded = tokenizer(
                context,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(device)
            if input_ids.size(1) == 0:
                raise ValueError("GreaTer proposal context tokenized to an empty sequence.")
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1, :].float()
            for special_id in tokenizer.all_special_ids:
                if 0 <= special_id < logits.numel():
                    logits[special_id] = float("-inf")
            pool_size = min(max(top_k * 20, top_k + 10), logits.numel())
            ranked_ids = torch.topk(logits, k=pool_size).indices.tolist()
            candidates = [current_id]
            seen_text = {
                tokenizer.decode([current_id], skip_special_tokens=True).strip().casefold()
            }
            for candidate_id in ranked_ids:
                candidate_id = int(candidate_id)
                if not _allowed_candidate_token(
                    tokenizer,
                    candidate_id,
                    check_isalnum,
                ):
                    continue
                if same_length_single_token_replacement_prompt(
                    tokenizer,
                    source_token_ids,
                    token_index,
                    candidate_id,
                ) is None:
                    unstable_candidate_ids.add(candidate_id)
                    continue
                text = tokenizer.decode([candidate_id], skip_special_tokens=True)
                normalized = text.strip().casefold()
                if normalized in seen_text:
                    continue
                seen_text.add(normalized)
                candidates.append(candidate_id)
                if len(candidates) >= top_k + 1:
                    break
            candidate_sets.append(candidates)

    strict_intersection = set(candidate_sets[0])
    for candidates in candidate_sets[1:]:
        strict_intersection.intersection_update(candidates)
    counts: dict[int, int] = {}
    first_ranks: dict[int, int] = {}
    for candidates in candidate_sets:
        for rank, candidate_id in enumerate(candidates):
            counts[candidate_id] = counts.get(candidate_id, 0) + 1
            first_ranks.setdefault(candidate_id, rank)
    ranked_by_frequency = sorted(
        counts,
        key=lambda candidate_id: (
            candidate_id in strict_intersection,
            counts[candidate_id],
            -first_ranks[candidate_id],
        ),
        reverse=True,
    )
    selected = []
    target_size = max(min_candidates, len(strict_intersection), 1)
    for candidate_id in [current_id, *ranked_by_frequency]:
        if candidate_id in selected:
            continue
        if candidate_id in strict_intersection or len(strict_intersection) < min_candidates:
            selected.append(candidate_id)
        if len(selected) >= target_size:
            break
    return selected, {
        "check_isalnum": check_isalnum,
        "proposal_example_count": len(proposal_records),
        "per_example_candidate_counts": [len(values) for values in candidate_sets],
        "strict_intersection_size": len(strict_intersection),
        "strict_intersection_token_ids": sorted(strict_intersection),
        "used_frequency_fallback": len(strict_intersection) < min_candidates,
        "candidate_frequency": {str(key): value for key, value in counts.items()},
        "unstable_retokenization_count": len(unstable_candidate_ids),
        "unstable_retokenization_token_ids": sorted(unstable_candidate_ids),
        "selected_token_ids": selected,
    }


def rank_fixed_token_candidates(
    gradient_analysis: dict[str, Any],
    token_index: int,
    candidate_token_ids: Sequence[int],
    records: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    model,
    tokenizer,
    batch_size: int,
    fluency_lambda: float,
    reasoning_traces: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank fixed GreaTer candidates with a combined-loss one-hot gradient."""
    require_torch()
    if not records:
        raise ValueError("GreaTer gradient records must not be empty.")
    if batch_size <= 0:
        raise ValueError("GreaTer gradient batch size must be positive.")
    if fluency_lambda < 0:
        raise ValueError("GreaTer fluency weight must be non-negative.")
    if reasoning_traces is not None and len(reasoning_traces) != len(records):
        raise ValueError("Reasoning traces and ranking records must have equal lengths.")
    instruction_prompt = str(gradient_analysis["instruction_prompt"])
    instruction_token_ids = [int(value) for value in gradient_analysis["token_ids"]]
    if not 0 <= token_index < len(instruction_token_ids):
        raise IndexError(f"Prompt token index is out of range: {token_index}")
    current_id = int(gradient_analysis["token_ids"][token_index])
    candidate_ids = list(dict.fromkeys(int(value) for value in candidate_token_ids))
    if current_id not in candidate_ids:
        raise ValueError("Current prompt token must be included in GreaTer candidates.")

    freeze_model_parameters(model)
    device = model_device(model)
    embedding_layer = model.get_input_embeddings()
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    candidate_embeddings = embedding_layer.weight.detach()[candidate_tensor]
    one_hot = torch.zeros(
        (1, len(candidate_ids)),
        dtype=candidate_embeddings.dtype,
        device=device,
    )
    current_candidate_index = candidate_ids.index(current_id)
    one_hot[0, current_candidate_index] = 1.0
    one_hot.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    total_batches = math.ceil(len(records) / batch_size)
    processed = 0
    started_at = time.monotonic()
    record_list = list(records)
    trace_list = list(reasoning_traces) if reasoning_traces is not None else None
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for chunk_index, start in enumerate(range(0, len(records), batch_size)):
            chunk = record_list[start : start + batch_size]
            trace_chunk = (
                trace_list[start : start + batch_size]
                if trace_list is not None
                else None
            )
            payload = _encode_teacher_forced_batch(
                instruction_prompt,
                chunk,
                mode,
                tokenizer,
                reasoning_traces=trace_chunk,
            )
            if list(payload["instruction_token_ids"]) != instruction_token_ids:
                raise ValueError(
                    "Instruction tokenization changed during GreaTer one-hot ranking."
                )
            encoded = payload["encoded"]
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            selected_embedding = one_hot @ candidate_embeddings
            base_embeddings = embedding_layer(input_ids).detach()
            replacement_embeddings = torch.zeros_like(base_embeddings)
            replacement_mask = torch.zeros(
                base_embeddings.shape[:2],
                dtype=base_embeddings.dtype,
                device=device,
            )
            for row_index, positions in enumerate(payload["instruction_positions"]):
                prompt_position = int(positions[token_index])
                replacement_embeddings[row_index, prompt_position, :] = (
                    selected_embedding[0]
                )
                replacement_mask[row_index, prompt_position] = 1.0
            input_embeddings = (
                base_embeddings * (1.0 - replacement_mask.unsqueeze(-1))
                + replacement_embeddings * replacement_mask.unsqueeze(-1)
            )
            outputs = model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                use_cache=False,
            )
            task_loss = _label_loss_from_logits(
                outputs.logits,
                input_ids,
                payload["answer_positions"],
                reduction="sum",
            ) / len(records)
            combined_loss = task_loss
            instruction_nll = None
            if fluency_lambda > 0 and chunk_index == 0:
                instruction_nll = _differentiable_instruction_nll(
                    instruction_token_ids,
                    token_index,
                    selected_embedding,
                    model=model,
                )
                combined_loss = combined_loss + fluency_lambda * instruction_nll
            combined_loss.backward()
            processed += len(chunk)
            _log_batch_checkpoint(
                "greater-gradient-ranking",
                chunk_index + 1,
                total_batches,
                processed,
                len(records),
                started_at,
            )
            del encoded, input_ids, attention_mask, selected_embedding
            del base_embeddings, replacement_embeddings, replacement_mask
            del input_embeddings, outputs, task_loss, combined_loss
            if instruction_nll is not None:
                del instruction_nll
            clear_cuda_cache()
    finally:
        tokenizer.padding_side = original_padding_side

    if one_hot.grad is None:
        raise RuntimeError("No GreaTer candidate one-hot gradient was produced.")
    gradient = one_hot.grad.detach()[0].float()
    ranked = []
    for candidate_index, candidate_id in enumerate(candidate_ids):
        one_hot_gradient = gradient[candidate_index]
        ranked.append(
            {
                "token_id": int(candidate_id),
                "token_text": tokenizer.decode(
                    [int(candidate_id)],
                    skip_special_tokens=True,
                ),
                "one_hot_gradient": float(one_hot_gradient.detach().cpu()),
                "is_original": int(candidate_id) == current_id,
            }
        )
    del one_hot, candidate_embeddings, candidate_tensor, gradient
    clear_cuda_cache()
    return sorted(
        ranked,
        key=lambda item: float(item["one_hot_gradient"]),
    )


def tensor_free_gradient_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """Remove large tensors before writing gradient metadata to JSON."""
    return {
        key: value
        for key, value in analysis.items()
        if key not in {"gradient_vectors", "embedding_vectors"}
    }
