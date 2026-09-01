"""QA answer-label gradients, region selection, and local token utilities."""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from agents.agent_memory import clear_cuda_cache
from prompt_optimization.qa_task import QAMode, choices_as_text, render_qa_prompt


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
) -> dict[str, Any]:
    """Render a user question and gold tagged assistant answer for label loss."""
    user_prompt = render_qa_prompt(instruction_prompt, record, mode)
    gold_label = str(record["answer"]).strip().upper()
    assistant_answer = f"<answer>{gold_label}</answer>"
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
    answer_start = rendered.rfind(assistant_answer)
    if answer_start < 0:
        raise ValueError("Tagged gold answer was not found after chat-template rendering.")
    label_start = answer_start + len("<answer>")
    return {
        "text": rendered,
        "instruction_start": instruction_start,
        "instruction_end": instruction_start + len(instruction_prompt),
        "label_start": label_start,
        "label_end": label_start + len(gold_label),
        "gold_label": gold_label,
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
) -> dict[str, Any]:
    """Tokenize a teacher-forced batch and align instruction and answer tokens."""
    rendered = [
        render_teacher_forced_qa(instruction_prompt, record, mode, tokenizer)
        for record in records
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


def _label_loss_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_positions: Sequence[Sequence[int]],
    reduction: str,
) -> torch.Tensor:
    """Calculate causal cross entropy only for gold option-label tokens."""
    require_torch()
    selected_logits = []
    selected_targets = []
    for row_index, positions in enumerate(answer_positions):
        for token_position in positions:
            selected_logits.append(logits[row_index, token_position - 1, :])
            selected_targets.append(input_ids[row_index, token_position])
    if not selected_logits:
        raise ValueError("No answer-label tokens were selected for task loss.")
    return F.cross_entropy(
        torch.stack(selected_logits),
        torch.stack(selected_targets),
        reduction=reduction,
    )


def collect_instruction_gradients(
    instruction_prompt: str,
    records: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    model,
    tokenizer,
    batch_size: int,
) -> dict[str, Any]:
    """Average answer-label loss gradients over instruction token embeddings."""
    require_torch()
    if not records:
        raise ValueError("Gradient records must not be empty.")
    if batch_size <= 0:
        raise ValueError("Gradient batch size must be positive.")
    freeze_model_parameters(model)
    device = model_device(model)
    gradient_sums: torch.Tensor | None = None
    embedding_sums: torch.Tensor | None = None
    canonical_ids: list[int] | None = None
    canonical_offsets: list[tuple[int, int]] | None = None
    total_loss = 0.0
    processed = 0
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for chunk in batched(list(records), batch_size):
            payload = _encode_teacher_forced_batch(
                instruction_prompt,
                chunk,
                mode,
                tokenizer,
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
    return {
        "instruction_prompt": instruction_prompt,
        "num_records": processed,
        "mean_task_loss": total_loss / processed,
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
) -> float:
    """Measure mean teacher-forced answer-label cross entropy for one prompt."""
    require_torch()
    if not records:
        raise ValueError("Task-loss records must not be empty.")
    device = model_device(model)
    total_loss = 0.0
    label_token_count = 0
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        with torch.inference_mode():
            for chunk in batched(list(records), batch_size):
                payload = _encode_teacher_forced_batch(
                    instruction_prompt,
                    chunk,
                    mode,
                    tokenizer,
                )
                encoded = payload["encoded"]
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                loss_sum = _label_loss_from_logits(
                    outputs.logits,
                    input_ids,
                    payload["answer_positions"],
                    reduction="sum",
                )
                total_loss += float(loss_sum.detach().cpu())
                label_token_count += sum(len(row) for row in payload["answer_positions"])
                del encoded, input_ids, attention_mask, outputs, loss_sum
                clear_cuda_cache()
    finally:
        tokenizer.padding_side = original_padding_side
    return total_loss / max(1, label_token_count)


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


def score_combined_objective(
    instruction_prompt: str,
    records: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    model,
    tokenizer,
    batch_size: int,
    fluency_lambda: float,
) -> dict[str, float]:
    """Combine QA label loss with the weighted log-perplexity penalty."""
    task_loss = score_instruction_task_loss(
        instruction_prompt,
        records,
        mode=mode,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    instruction_nll = score_instruction_nll(instruction_prompt, model, tokenizer)
    return {
        "task_loss": task_loss,
        "instruction_nll": instruction_nll,
        "combined_score": task_loss + fluency_lambda * instruction_nll,
    }


def _editable_token(record: dict[str, Any], instruction_prompt: str) -> bool:
    """Check that an aligned token covers at least one letter or number."""
    start = int(record["char_start"])
    end = int(record["char_end"])
    return any(character.isalnum() for character in instruction_prompt[start:end])


def build_gradient_region_pool(
    gradient_analysis: dict[str, Any],
    *,
    max_region_tokens: int,
    expansion_threshold_ratio: float,
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
        if _editable_token(record, prompt)
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


def _allowed_candidate_token(tokenizer, token_id: int) -> bool:
    """Filter special, empty, control, and non-ASCII replacement tokens."""
    if token_id in set(tokenizer.all_special_ids):
        return False
    text = tokenizer.decode([token_id], skip_special_tokens=True)
    if not text.strip() or not any(character.isalnum() for character in text):
        return False
    return text.isascii() and "\n" not in text and "\r" not in text


def qa_proposal_header(record: dict[str, Any], mode: QAMode) -> str:
    """Build GreaTer's instruction-generation context around one QA example."""
    return (
        "You are optimizing an instruction prompt for a multiple-choice question "
        "answering model.\n\n"
        "The instruction will appear before a fixed answer-format instruction, the "
        "question, and its labeled choices. The model must select the single best "
        "option.\n\n"
        f"Fixed answer instruction:\n{mode.answer_instruction}\n\n"
        "Example model input:\n"
        f"Question: {record['question']}\n"
        f"Choices: {choices_as_text(record)}\n\n"
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
) -> tuple[list[int], dict[str, Any]]:
    """Intersect or frequency-rank GreaTer token proposals across QA examples."""
    require_torch()
    if not proposal_records:
        raise ValueError("GreaTer proposal records must not be empty.")
    prompt = str(gradient_analysis["instruction_prompt"])
    token_record = gradient_analysis["token_gradients"][token_index]
    prefix = prompt[: int(token_record["char_start"])]
    current_id = int(gradient_analysis["token_ids"][token_index])
    device = model_device(model)
    candidate_sets: list[list[int]] = []
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
                if not _allowed_candidate_token(tokenizer, candidate_id):
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
        "proposal_example_count": len(proposal_records),
        "per_example_candidate_counts": [len(values) for values in candidate_sets],
        "strict_intersection_size": len(strict_intersection),
        "strict_intersection_token_ids": sorted(strict_intersection),
        "used_frequency_fallback": len(strict_intersection) < min_candidates,
        "candidate_frequency": {str(key): value for key, value in counts.items()},
        "selected_token_ids": selected,
    }


def rank_fixed_token_candidates(
    gradient_analysis: dict[str, Any],
    token_index: int,
    candidate_token_ids: Sequence[int],
    *,
    model,
    tokenizer,
) -> list[dict[str, Any]]:
    """Rank a fixed GreaTer proposal set by first-order QA task-loss change."""
    require_torch()
    gradient = gradient_analysis["gradient_vectors"][token_index].to(model_device(model))
    embeddings = model.get_input_embeddings().weight.detach()
    current_id = int(gradient_analysis["token_ids"][token_index])
    current_projection = torch.dot(embeddings[current_id].float(), gradient.float())
    ranked = []
    for candidate_id in candidate_token_ids:
        score = torch.dot(embeddings[int(candidate_id)].float(), gradient.float())
        ranked.append(
            {
                "token_id": int(candidate_id),
                "token_text": tokenizer.decode(
                    [int(candidate_id)],
                    skip_special_tokens=True,
                ),
                "first_order_score": float((score - current_projection).detach().cpu()),
                "is_original": int(candidate_id) == current_id,
            }
        )
    return sorted(
        ranked,
        key=lambda item: (float(item["first_order_score"]), bool(not item["is_original"])),
    )


def tensor_free_gradient_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """Remove large tensors before writing gradient metadata to JSON."""
    return {
        key: value
        for key, value in analysis.items()
        if key not in {"gradient_vectors", "embedding_vectors"}
    }
