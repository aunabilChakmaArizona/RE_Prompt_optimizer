#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
CODES_DIR = CURRENT_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.append(str(CODES_DIR))

from agents.agent_data_loader import DEFAULT_DATA_DIR, load_split_episodes
from agents.agent_models import load_model_and_tokenizer
from agents.agent_prompts import (
    INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1,
    INFERENCE_INPUT_PROMPT_V1,
    INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    compose_inference_prompt,
)
from agents.agent_relation_utils import get_relation_description
from agents.agent_utils import build_support_block, get_sentence_with_tags

TOKEN_CANDIDATE_MODE_NEAREST_UPDATED = "nearest_updated"
TOKEN_CANDIDATE_MODE_FIRST_ORDER_LOSS_APPROX = "first_order_loss_approx"
TOKEN_CANDIDATE_MODE_CHOICES = [
    TOKEN_CANDIDATE_MODE_NEAREST_UPDATED,
    TOKEN_CANDIDATE_MODE_FIRST_ORDER_LOSS_APPROX,
]


def _resolve_binary_token_id(tokenizer, base_token: str) -> int:
    token_ids = tokenizer.encode(base_token, add_special_tokens=False)
    if len(token_ids) == 1:
        return token_ids[0]

    spaced_token_ids = tokenizer.encode(f" {base_token}", add_special_tokens=False)
    if len(spaced_token_ids) == 1:
        return spaced_token_ids[0]

    raise ValueError(
        f"Token '{base_token}' is not a single token for tokenizer {tokenizer.__class__.__name__}."
    )


@dataclass
class CandidateToken:
    token_id: int
    token: str
    similarity_score: float


@dataclass
class TokenGradientRecord:
    token_index: int
    token_id: int
    token: str
    gradient_norm: float
    candidates: List[CandidateToken]


@dataclass
class TokenRegion:
    start: int
    end: int
    score: float
    tokens: List[str]
    token_indices: List[int]


@dataclass
class BatchPromptGradientAnalysis:
    prompt_template: str
    num_instances: int
    formatted_prompts: List[str]
    predicted_labels: List[str]
    target_labels: List[str]
    target_probabilities: List[float]
    prompt_tokens: List[str]
    token_gradients: List[TokenGradientRecord]
    top_regions: List[TokenRegion]


def _ensure_supported_inference_mode(inference_mode: str) -> None:
    if inference_mode != INFERENCE_MODE_SEPARATE_NO_EXAMPLES:
        raise ValueError(
            "agent_gradient_token_analysis only supports "
            "INFERENCE_MODE_SEPARATE_NO_EXAMPLES."
        )


def build_relation_prompt(
    *,
    instruction_prompt: str,
    relation: str,
    relation_description: str,
    support_sentence: str,
    query_sentence: str,
    inference_mode: str = INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    answer_instruction_prompt: str = INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1,
    input_prompt: str = INFERENCE_INPUT_PROMPT_V1,
) -> str:
    support_block = build_support_block([support_sentence])
    
    return compose_inference_prompt(
        inference_mode=inference_mode,
        inference_prompt="",
        inference_instruction_prompt=instruction_prompt,
        inference_answer_instruction_prompt=answer_instruction_prompt,
        inference_input_prompt=input_prompt,
        relation=relation,
        relation_description=relation_description,
        support_block=support_block,
        query_sentence=query_sentence,
        example_query_sentence=support_sentence,
    )


def _build_instruction_token_map(
    *,
    instruction_prompt: str,
    rendered_prompt: str,
    tokenizer,
) -> Dict[str, Any]:
    instruction_ids = tokenizer.encode(instruction_prompt, add_special_tokens=False)
    encoded_rendered = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    rendered_ids = encoded_rendered["input_ids"][0].tolist()
    offset_mapping = encoded_rendered["offset_mapping"][0].tolist()

    def _log_alignment_failure() -> None:
        print("[agent_gradient_token_analysis] instruction alignment failed")
    start_char = rendered_prompt.find(instruction_prompt)
    if start_char < 0:
        _log_alignment_failure()
        raise ValueError("Instruction prompt text not found within rendered prompt.")
    end_char = start_char + len(instruction_prompt)

    try:
        instruction_start = _find_subsequence(
            rendered_ids,
            instruction_ids,
            start_index=0,
        )
        instruction_end = instruction_start + len(instruction_ids)
        instruction_positions = list(range(instruction_start, instruction_end))
    except ValueError:
        if len(instruction_ids) < 2:
            _log_alignment_failure()
            raise

        prefix_ids = instruction_ids[:-1]
        try:
            instruction_start = _find_subsequence(
                rendered_ids,
                prefix_ids,
                start_index=0,
            )
        except ValueError:
            _log_alignment_failure()
            raise

        instruction_positions = list(
            range(instruction_start, instruction_start + len(prefix_ids))
        )
        candidate_index = instruction_start + len(prefix_ids)
        if candidate_index < len(offset_mapping):
            token_start, token_end = offset_mapping[candidate_index]
            if token_start < end_char and token_end > token_start:
                instruction_positions.append(candidate_index)

        if not instruction_positions:
            _log_alignment_failure()
            raise ValueError("Unable to recover instruction token positions.")

    return {
        "prompt_token_ids": [rendered_ids[idx] for idx in instruction_positions],
        "prompt_token_positions": instruction_positions,
        "prompt_tokens": tokenizer.convert_ids_to_tokens(
            [rendered_ids[idx] for idx in instruction_positions]
        ),
        "rendered_prompt_ids": rendered_ids,
    }


def _find_subsequence(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    start_index: int = 0,
) -> int:
    if not target_ids:
        return start_index

    max_start = len(source_ids) - len(target_ids)
    for candidate_start in range(start_index, max_start + 1):
        if list(source_ids[candidate_start : candidate_start + len(target_ids)]) == list(target_ids):
            return candidate_start
    raise ValueError("Unable to align prompt tokens within the rendered input.")


def build_binary_pair_payload(
    *,
    instruction_prompt: str,
    support: Dict[str, Any],
    query: Dict[str, Any],
    pair_index: int,
    tokenizer,
    dataset_type: str = "fs_tacred",
    inference_mode: str = INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    answer_instruction_prompt: str = INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1,
    input_prompt: str = INFERENCE_INPUT_PROMPT_V1,
) -> Dict[str, Any]:
    relation = support["relation"]
    label = "yes" if query["relation"] == relation else "no"
    support_sentence = get_sentence_with_tags(support).strip()
    query_sentence = get_sentence_with_tags(query).strip()
    relation_description = get_relation_description(relation, dt=dataset_type)

    prompt = build_relation_prompt(
        instruction_prompt=instruction_prompt,
        relation=relation,
        relation_description=relation_description,
        support_sentence=support_sentence,
        query_sentence=query_sentence,
        inference_mode=inference_mode,
        answer_instruction_prompt=answer_instruction_prompt,
        input_prompt=input_prompt,
    )
    prompt_token_map = _build_instruction_token_map(
        instruction_prompt=instruction_prompt,
        rendered_prompt=prompt,
        tokenizer=tokenizer,
    )
    return {
        "pair_index": pair_index,
        "support_id": support.get("id"),
        "query_id": query.get("id"),
        "relation": relation,
        "label": label,
        "prompt": prompt,
        "prompt_token_ids": prompt_token_map["prompt_token_ids"],
        "prompt_token_positions": prompt_token_map["prompt_token_positions"],
        "prompt_tokens": prompt_token_map["prompt_tokens"],
        "rendered_prompt_ids": prompt_token_map["rendered_prompt_ids"],
    }


def build_binary_pair_payload_from_episode(
    *,
    instruction_prompt: str,
    episode: Dict[str, Any],
    shots: Dict[str, Dict[str, Any]],
    queries: Dict[str, Dict[str, Any]],
    pair_index: int,
    tokenizer,
    dataset_type: str = "fs_tacred",
    way_index: int = 0,
    query_index: int = 0,
    inference_mode: str = INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    answer_instruction_prompt: str = INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1,
    input_prompt: str = INFERENCE_INPUT_PROMPT_V1,
) -> Dict[str, Any]:
    support_id = episode["meta_train"][way_index][0]
    query_id = episode["meta_test"][query_index]
    support = shots[support_id]
    query = queries[query_id]
    payload = build_binary_pair_payload(
        instruction_prompt=instruction_prompt,
        support=support,
        query=query,
        pair_index=pair_index,
        tokenizer=tokenizer,
        dataset_type=dataset_type,
        inference_mode=inference_mode,
        answer_instruction_prompt=answer_instruction_prompt,
        input_prompt=input_prompt,
    )
    payload["support_id"] = support_id
    payload["query_id"] = query_id
    return payload


def _prepare_model_inputs(
    prompt: str,
    *,
    tokenizer,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> Dict[str, torch.Tensor | str]:
    if use_chat_template:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    else:
        formatted_prompt = prompt

    encoded = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    return {
        "formatted_prompt": formatted_prompt,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def _candidate_tokens_for_position(
    *,
    token_id: int,
    current_embedding: torch.Tensor,
    gradient: torch.Tensor,
    updated_embedding: torch.Tensor,
    embedding_weight: torch.Tensor,
    tokenizer,
    num_candidates: int,
    candidate_mode: str,
) -> List[CandidateToken]:
    embedding_weight_work = embedding_weight.float()
    current_embedding_work = current_embedding.float().to(embedding_weight.device)
    gradient_work = gradient.float().to(embedding_weight.device)
    updated_embedding_work = updated_embedding.float().to(embedding_weight.device)

    if candidate_mode == TOKEN_CANDIDATE_MODE_NEAREST_UPDATED:
        normalized_weight = F.normalize(embedding_weight_work, dim=-1)
        normalized_query = F.normalize(updated_embedding_work.unsqueeze(0), dim=-1)
        scores = torch.matmul(normalized_query, normalized_weight.T).squeeze(0)
    elif candidate_mode == TOKEN_CANDIDATE_MODE_FIRST_ORDER_LOSS_APPROX:
        displacement = embedding_weight.float() - current_embedding.float().unsqueeze(0)
        scores = -torch.matmul(displacement, gradient.float())
    else:
        raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")

    scores[token_id] = float("-inf")
    for special_id in tokenizer.all_special_ids:
        if 0 <= special_id < scores.numel():
            scores[special_id] = float("-inf")

    k = min(num_candidates, scores.numel())
    top_scores, top_ids = torch.topk(scores, k=k)
    top_scores = top_scores.detach().cpu()
    top_ids = top_ids.detach().cpu()

    return [
        CandidateToken(
            token_id=int(candidate_id.item()),
            token=tokenizer.convert_ids_to_tokens(int(candidate_id.item())),
            similarity_score=float(candidate_score.item()),
        )
        for candidate_id, candidate_score in zip(top_ids, top_scores)
    ]


def _select_top_regions(
    gradient_scores: Sequence[float],
    tokens: Sequence[str],
    *,
    max_regions: int,
    max_total_tokens: int = 5,
    expansion_threshold_ratio: float = 0.6,
) -> List[TokenRegion]:
    if max_regions <= 0 or max_total_tokens <= 0:
        return []

    remaining = set(range(len(gradient_scores)))
    ranked = sorted(
        range(len(gradient_scores)),
        key=lambda idx: gradient_scores[idx],
        reverse=True,
    )
    regions: List[TokenRegion] = []
    selected_token_count = 0

    for peak in ranked:
        if peak not in remaining:
            continue
        peak_score = gradient_scores[peak]
        if peak_score <= 0:
            continue

        threshold = peak_score * expansion_threshold_ratio
        start = peak
        end = peak

        while start - 1 in remaining and gradient_scores[start - 1] >= threshold:
            start -= 1
        while end + 1 in remaining and gradient_scores[end + 1] >= threshold:
            end += 1

        for idx in range(start, end + 1):
            remaining.discard(idx)

        remaining_token_budget = max_total_tokens - selected_token_count
        if remaining_token_budget <= 0:
            break

        token_indices = [peak]
        left = peak - 1
        right = peak + 1
        while len(token_indices) < remaining_token_budget and (left >= start or right <= end):
            left_score = gradient_scores[left] if left >= start else float("-inf")
            right_score = gradient_scores[right] if right <= end else float("-inf")

            if right_score > left_score:
                token_indices.append(right)
                right += 1
            else:
                if left >= start:
                    token_indices.insert(0, left)
                    left -= 1
                elif right <= end:
                    token_indices.append(right)
                    right += 1

        if not token_indices:
            continue

        selected_token_count += len(token_indices)
        regions.append(
            TokenRegion(
                start=token_indices[0],
                end=token_indices[-1],
                score=float(sum(gradient_scores[idx] for idx in token_indices)),
                tokens=[tokens[idx] for idx in token_indices],
                token_indices=token_indices,
            )
        )
        if len(regions) == max_regions or selected_token_count >= max_total_tokens:
            break

    return regions


def analyze_relation_extraction_binary_pairs(
    *,
    instruction_prompt: str,
    binary_pairs: Sequence[Dict[str, Any]],
    model,
    tokenizer,
    dataset_type: str = "fs_tacred",
    num_candidates: int = 5,
    max_regions: int = 1,
    max_total_region_tokens: int = 10,
    embedding_step_size: float = 1.0,
    candidate_mode: str = TOKEN_CANDIDATE_MODE_NEAREST_UPDATED,
    inference_mode: str = INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    answer_instruction_prompt: str = INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1,
    input_prompt: str = INFERENCE_INPUT_PROMPT_V1,
) -> Dict[str, Any]:
    _ensure_supported_inference_mode(inference_mode)

    batch_payloads: List[Dict[str, Any]] = []
    for pair_index, pair in enumerate(binary_pairs):
        batch_payloads.append(
            build_binary_pair_payload(
                instruction_prompt=instruction_prompt,
                support=pair["support"],
                query=pair["query"],
                pair_index=pair_index,
                dataset_type=dataset_type,
                inference_mode=inference_mode,
                answer_instruction_prompt=answer_instruction_prompt,
                input_prompt=input_prompt,
                tokenizer=tokenizer,
            )
        )

    if not batch_payloads:
        return asdict(
            BatchPromptGradientAnalysis(
                prompt_template=instruction_prompt,
                num_instances=0,
                formatted_prompts=[],
                predicted_labels=[],
                target_labels=[],
                target_probabilities=[],
                prompt_tokens=[],
                token_gradients=[],
                top_regions=[],
            )
        )

    prompt_token_ids = batch_payloads[0]["prompt_token_ids"]
    prompt_tokens = batch_payloads[0]["prompt_tokens"]
    for payload in batch_payloads[1:]:
        if payload["prompt_token_ids"] != prompt_token_ids:
            raise ValueError(
                "Prompt-only tokenization is not aligned across episodes. "
                "Use a stable prompt template so prompt tokens are identical for the batch."
            )

    encoded_inputs = [
        _prepare_model_inputs(
            payload["prompt"],
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        for payload in batch_payloads
    ]

    formatted_prompts = [item["formatted_prompt"] for item in encoded_inputs]
    tokenized_batch = tokenizer.pad(
        [
            {
                "input_ids": item["input_ids"][0],
                "attention_mask": item["attention_mask"][0],
            }
            for item in encoded_inputs
        ],
        return_tensors="pt",
    )

    embedding_layer = model.get_input_embeddings()
    embed_device = embedding_layer.weight.device
    input_ids = tokenized_batch["input_ids"].to(embed_device)
    attention_mask = tokenized_batch["attention_mask"].to(embed_device)

    yes_token_id = _resolve_binary_token_id(tokenizer, "yes")
    no_token_id = _resolve_binary_token_id(tokenizer, "no")
    target_class_indices = torch.tensor(
        [0 if payload["label"] == "yes" else 1 for payload in batch_payloads],
        device=embed_device,
    )

    model.zero_grad(set_to_none=True)
    inputs_embeds = embedding_layer(input_ids).detach().clone()
    inputs_embeds.requires_grad_(True)

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    decision_logits = outputs.logits[:, -1, [yes_token_id, no_token_id]]
    probabilities = torch.softmax(decision_logits, dim=-1)
    predicted_indices = torch.argmax(probabilities, dim=-1)
    predicted_labels = ["yes" if index.item() == 0 else "no" for index in predicted_indices]
    target_probabilities = [
        float(probabilities[row_index, class_index].item())
        for row_index, class_index in enumerate(target_class_indices.tolist())
    ]

    loss = F.cross_entropy(decision_logits, target_class_indices, reduction="mean")
    loss.backward()

    gradients = inputs_embeds.grad
    prompt_position_lists = [payload["prompt_token_positions"] for payload in batch_payloads]
    rendered_prompt_ids_list = [payload["rendered_prompt_ids"] for payload in batch_payloads]

    prompt_gradient_sums = torch.zeros(
        (len(prompt_token_ids), gradients.size(-1)),
        device=gradients.device,
        dtype=gradients.dtype,
    )
    prompt_embedding_sums = torch.zeros_like(prompt_gradient_sums)

    seq_len = input_ids.size(1)
    for batch_index, positions in enumerate(prompt_position_lists):
        formatted_ids = encoded_inputs[batch_index]["input_ids"][0].tolist()
        rendered_prompt_start = _find_subsequence(
            formatted_ids,
            rendered_prompt_ids_list[batch_index],
            start_index=0,
        )
        rendered_length = len(formatted_ids)
        left_pad = seq_len - rendered_length
        shifted_positions = [
            left_pad + rendered_prompt_start + position for position in positions
        ]
        prompt_gradient_sums += gradients[batch_index, shifted_positions, :]
        prompt_embedding_sums += inputs_embeds.detach()[batch_index, shifted_positions, :]

    prompt_gradient_means = prompt_gradient_sums / len(batch_payloads)
    prompt_embedding_means = prompt_embedding_sums / len(batch_payloads)
    gradient_norms = prompt_gradient_means.float().norm(dim=-1).detach().cpu().tolist()
    updated_embeddings = prompt_embedding_means - embedding_step_size * prompt_gradient_means

    embedding_weight = embedding_layer.weight.detach()
    token_gradients: List[TokenGradientRecord] = []
    for index, (token_id, token, gradient_norm, current_embedding, gradient_vector, updated_embedding) in enumerate(
        zip(
            prompt_token_ids,
            prompt_tokens,
            gradient_norms,
            prompt_embedding_means,
            prompt_gradient_means,
            updated_embeddings,
        ),
    ):
        candidates = _candidate_tokens_for_position(
            token_id=token_id,
            current_embedding=current_embedding,
            gradient=gradient_vector,
            updated_embedding=updated_embedding,
            embedding_weight=embedding_weight,
            tokenizer=tokenizer,
            num_candidates=num_candidates,
            candidate_mode=candidate_mode,
        )
        token_gradients.append(
            TokenGradientRecord(
                token_index=index,
                token_id=token_id,
                token=token,
                gradient_norm=float(gradient_norm),
                candidates=candidates,
            )
        )

    top_regions = _select_top_regions(
        gradient_norms,
        prompt_tokens,
        max_regions=max_regions,
        max_total_tokens=max_total_region_tokens,
    )

    return asdict(
        BatchPromptGradientAnalysis(
            prompt_template=instruction_prompt,
            num_instances=len(batch_payloads),
            formatted_prompts=formatted_prompts,
            predicted_labels=predicted_labels,
            target_labels=[payload["label"] for payload in batch_payloads],
            target_probabilities=target_probabilities,
            prompt_tokens=prompt_tokens,
            token_gradients=token_gradients,
            top_regions=top_regions,
        )
    )


def analyze_relation_extraction_dataset(
    *,
    instruction_prompt: str,
    episodes: Sequence[Dict[str, Any]],
    shots: Dict[str, Dict[str, Any]],
    queries: Dict[str, Dict[str, Any]],
    model,
    tokenizer,
    dataset_type: str = "fs_tacred",
    query_index: int = 0,
    num_candidates: int = 5,
    max_regions: int = 1,
    max_total_region_tokens: int = 10,
    embedding_step_size: float = 1.0,
    candidate_mode: str = TOKEN_CANDIDATE_MODE_NEAREST_UPDATED,
    inference_mode: str = INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    answer_instruction_prompt: str = INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1,
    input_prompt: str = INFERENCE_INPUT_PROMPT_V1,
) -> Dict[str, Any]:
    binary_pairs = [
        {
            "support": shots[episode["meta_train"][0][0]],
            "query": queries[episode["meta_test"][query_index]],
        }
        for episode in episodes
    ]
    return analyze_relation_extraction_binary_pairs(
        instruction_prompt=instruction_prompt,
        binary_pairs=binary_pairs,
        model=model,
        tokenizer=tokenizer,
        dataset_type=dataset_type,
        num_candidates=num_candidates,
        max_regions=max_regions,
        max_total_region_tokens=max_total_region_tokens,
        embedding_step_size=embedding_step_size,
        candidate_mode=candidate_mode,
        inference_mode=inference_mode,
        use_chat_template=use_chat_template,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        answer_instruction_prompt=answer_instruction_prompt,
        input_prompt=input_prompt,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prompt token gradients for relation extraction episodes."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--dataset-type", default="fs_tacred")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--ep-start", type=int, default=0)
    parser.add_argument("--ep-end", type=int, default=5)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--max-regions", type=int, default=1)
    parser.add_argument("--max-total-region-tokens", type=int, default=10)
    parser.add_argument("--embedding-step-size", type=float, default=1.0)
    parser.add_argument(
        "--candidate-mode",
        choices=TOKEN_CANDIDATE_MODE_CHOICES,
        default=TOKEN_CANDIDATE_MODE_NEAREST_UPDATED,
    )
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--disable-chat-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.prompt_file, "r", encoding="utf-8") as handle:
        instruction_prompt = handle.read().strip()

    dataset = load_split_episodes(
        split=args.split,
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        ep_start=args.ep_start,
        ep_end=args.ep_end,
    )
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        device_map=args.device_map,
    )
    results = analyze_relation_extraction_dataset(
        instruction_prompt=instruction_prompt,
        episodes=dataset["episodes"],
        shots=dataset["shots"],
        queries=dataset["queries"],
        model=model,
        tokenizer=tokenizer,
        dataset_type=args.dataset_type,
        query_index=args.query_index,
        num_candidates=args.num_candidates,
        max_regions=args.max_regions,
        max_total_region_tokens=args.max_total_region_tokens,
        embedding_step_size=args.embedding_step_size,
        candidate_mode=args.candidate_mode,
        use_chat_template=not args.disable_chat_template,
    )

    payload = {
        "model_id": args.model_id,
        "split": args.split,
        "dataset_type": args.dataset_type,
        "candidate_mode": args.candidate_mode,
        "num_instances": results["num_instances"],
        "results": results,
    }
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
