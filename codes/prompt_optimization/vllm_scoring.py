"""Forward-only candidate objectives using vLLM input-token log probabilities."""

from __future__ import annotations

import importlib.metadata
import math
import time
from typing import Any, Sequence

from prompt_optimization.qa_task import QAMode
from prompt_optimization.sequence_gradients import (
    _log_batch_checkpoint,
    batched,
    overlapping_token_positions,
    render_teacher_forced_qa,
)


def validate_vllm_scoring_version() -> str:
    """Require the tested v2 release without changing the original HF scorer."""
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("vLLM scoring requires the re_prompt_optimization_vllm_v2 environment.") from error
    if version != "0.11.0":
        raise RuntimeError(
            f"vLLM objective scoring requires vllm==0.11.0, found {version}. "
            "Activate re_prompt_optimization_vllm_v2; HF scoring supports the old environment."
        )
    return version


def encode_scoring_sequences(
    prompts: Sequence[str],
    records: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    tokenizer,
    reasoning_traces: Sequence[str | None],
) -> tuple[list[list[int]], list[list[int]]]:
    """Reuse HF rendering and answer masks, returning unpadded token ID sequences."""
    if not (len(prompts) == len(records) == len(reasoning_traces)):
        raise ValueError("Prompts, records, and reasoning traces must have equal lengths.")
    rendered = [
        render_teacher_forced_qa(prompt, record, mode, tokenizer, reasoning_trace=trace)
        for prompt, record, trace in zip(prompts, records, reasoning_traces)
    ]
    encoded = tokenizer(
        [item["text"] for item in rendered],
        padding=False,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_rows = [list(row) for row in encoded["input_ids"]]
    positions = [
        overlapping_token_positions(offsets, item["label_start"], item["label_end"])
        for item, offsets in zip(rendered, encoded["offset_mapping"])
    ]
    if len(token_rows) != len(prompts) or len(positions) != len(prompts):
        raise RuntimeError("Tokenizer did not return every scoring sequence.")
    if any(not row or any(index <= 0 for index in row) for row in positions):
        raise ValueError("Unable to align causal gold-answer tokens for vLLM scoring.")
    return token_rows, positions


def score_token_sequences(
    model,
    token_rows: Sequence[Sequence[int]],
    target_positions: Sequence[Sequence[int]],
) -> list[float]:
    """Return mean NLL at supplied positions, ignoring the one generated token."""
    if len(token_rows) != len(target_positions):
        raise ValueError("Token rows and target-position rows must have equal lengths.")
    if not token_rows:
        return []
    for tokens, positions in zip(token_rows, target_positions):
        if not positions or any(position <= 0 or position >= len(tokens) for position in positions):
            raise ValueError("Every scored token needs a preceding causal token.")
    from vllm import SamplingParams

    # Zero requests no alternative tokens; vLLM still returns the supplied token.
    # v0.11.0 requires max_tokens >= 1. Its generated output is not a loss target.
    params = SamplingParams(
        prompt_logprobs=0,
        max_tokens=1,
        temperature=0.0,
        detokenize=False,
    )
    outputs = model.generate(
        [{"prompt_token_ids": list(tokens)} for tokens in token_rows],
        sampling_params=params,
        use_tqdm=False,
    )
    if len(outputs) != len(token_rows):
        raise RuntimeError("vLLM did not return every scoring request.")
    losses = []
    for output, tokens, positions in zip(outputs, token_rows, target_positions):
        if list(output.prompt_token_ids) != list(tokens):
            raise RuntimeError("vLLM changed the supplied scoring token IDs.")
        logprobs = output.prompt_logprobs
        if logprobs is None or len(logprobs) != len(tokens):
            raise RuntimeError("vLLM returned missing or incomplete input-token logprobs.")
        selected = []
        for position in positions:
            entries = logprobs[position]
            token_id = tokens[position]
            if entries is None or token_id not in entries:
                raise RuntimeError(f"Missing supplied-token logprob at position {position}.")
            value = float(entries[token_id].logprob)
            if not math.isfinite(value):
                raise RuntimeError("vLLM returned a non-finite scoring logprob.")
            selected.append(-value)
        losses.append(math.fsum(selected) / len(selected))
    return losses


def score_combined_objectives_vllm(
    instruction_prompts: Sequence[str],
    records: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    model,
    tokenizer,
    batch_size: int,
    fluency_lambda: float,
    reasoning_traces: Sequence[str] | None = None,
    instruction_nlls: Sequence[float] | None = None,
) -> list[dict[str, float]]:
    """Batch task loss with vLLM and combine supplied or vLLM fluency NLLs."""
    prompts = list(instruction_prompts)
    if not prompts:
        return []
    if not records or batch_size <= 0:
        raise ValueError("Scoring records must not be empty and batch size must be positive.")
    if reasoning_traces is not None and len(reasoning_traces) != len(records):
        raise ValueError("Reasoning traces and scored records must have equal lengths.")
    validate_vllm_scoring_version()
    traces = list(reasoning_traces) if reasoning_traces is not None else [None] * len(records)
    # Interleave candidates so one submission can cover several beam nodes.
    work = [(candidate, example) for example in range(len(records)) for candidate in range(len(prompts))]
    sums = [0.0] * len(prompts)
    counts = [0] * len(prompts)
    started = time.monotonic()
    for index, chunk in enumerate(batched(work, batch_size), start=1):
        tokens, positions = encode_scoring_sequences(
            [prompts[candidate] for candidate, _ in chunk],
            [records[example] for _, example in chunk],
            mode=mode,
            tokenizer=tokenizer,
            reasoning_traces=[traces[example] for _, example in chunk],
        )
        losses = score_token_sequences(model, tokens, positions)
        for (candidate, _), loss in zip(chunk, losses):
            sums[candidate] += loss
            counts[candidate] += 1
        _log_batch_checkpoint(
            "vllm-candidate-task-loss", index, math.ceil(len(work) / batch_size),
            min(index * batch_size, len(work)), len(work), started,
        )
    if any(count != len(records) for count in counts):
        raise RuntimeError("vLLM scoring did not process every candidate/example pair.")
    if instruction_nlls is not None:
        fluencies = [float(value) for value in instruction_nlls]
        if len(fluencies) != len(prompts):
            raise ValueError("Supplied instruction NLLs must match the candidate prompts.")
        if any(not math.isfinite(value) for value in fluencies):
            raise ValueError("Supplied instruction NLLs must all be finite.")
    else:
        fluencies = [0.0] * len(prompts)
        for indices in batched(list(range(len(prompts))), batch_size):
            # Match score_instruction_nlls: raw instruction, no chat/BOS, no truncation.
            encoded = tokenizer(
                [prompts[index] for index in indices],
                padding=False,
                add_special_tokens=False,
            )
            scored = [(index, list(tokens)) for index, tokens in zip(indices, encoded["input_ids"]) if len(tokens) >= 2]
            losses = score_token_sequences(
                model,
                [tokens for _, tokens in scored],
                [list(range(1, len(tokens))) for _, tokens in scored],
            )
            for (index, _), loss in zip(scored, losses):
                fluencies[index] = loss
    return [
        {
            "task_loss": total / count,
            "instruction_nll": fluency,
            "combined_score": total / count + fluency_lambda * fluency,
        }
        for total, count, fluency in zip(sums, counts, fluencies)
    ]
