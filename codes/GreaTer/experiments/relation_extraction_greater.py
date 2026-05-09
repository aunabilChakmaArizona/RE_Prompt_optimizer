#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
CODES_DIR = REPO_ROOT / "codes"
GREATER_DIR = CODES_DIR / "GreaTer"
for path in (CODES_DIR, GREATER_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from agents.agent_binary_inference import run_binary_inference
from agents.agent_data_loader import DEFAULT_DATA_DIR, load_split_episodes, load_train_samples
from agents.agent_gradient_eval_debug import (
    _attach_predictions_to_pairs,
    _build_binary_pairs_from_feedback_samples,
    _build_binary_prompts_for_instruction,
    _build_evaluation_report,
    _build_prf_scores,
    _build_score_payload_from_pairs,
    _build_train_sample_index,
    _sample_balanced_pairs_from_bucketed_pool,
    build_full_binary_pairs,
)
from agents.agent_gradient_token_analysis import _build_instruction_token_map
from agents.agent_memory import clear_cuda_cache
from agents.agent_models import load_model_and_tokenizer
from agents.agent_prompts import INFERENCE_INSTRUCTION_PROMPT_V1
from agents.agent_sample_feedback import sample_feedback_fn
from agents.agent_utils import load_json_file

PROMPT_SOURCE_SCRATCH = "scratch"
PROMPT_SOURCE_POPULATION = "population"
FLUENCY_SCOPE_INSTRUCTION = "instruction"
FLUENCY_SCOPE_PREFIX = "prefix"
FLUENCY_SCOPE_PROPOSAL_PREFIX = "proposal_prefix"
FLUENCY_METRIC_NLL = "nll"
FLUENCY_METRIC_PPL = "ppl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal GReaTer-style token optimizer for binary relation "
            "extraction instruction prompts."
        )
    )
    parser.add_argument("--model", required=True, help="HF model id to load.")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--prompt-source",
        choices=[PROMPT_SOURCE_SCRATCH, PROMPT_SOURCE_POPULATION],
        default=PROMPT_SOURCE_SCRATCH,
        help=(
            "scratch uses INFERENCE_INSTRUCTION_PROMPT_V1 from agent_prompts.py; "
            "population loads a node prompt from population.json."
        ),
    )
    parser.add_argument(
        "--population-path",
        type=Path,
        default=None,
        help="Path to population.json when --prompt-source=population.",
    )
    parser.add_argument(
        "--prompt-node-id",
        type=int,
        default=None,
        help="Node id to load from population.json.",
    )
    parser.add_argument("--dataset-type", default="fs_tacred")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-samples", default="fs_tacred_train_samples.pkl")
    parser.add_argument("--full-eval-split", default="dev")
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-gradient-sample-size",
        type=int,
        default=128,
        help="Number of train feedback samples to draw before TP/TN/FP/FN balancing.",
    )
    parser.add_argument("--gradient-batch-size", type=int, default=4)
    parser.add_argument("--selection-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--proposal-top-k", type=int, default=32)
    parser.add_argument(
        "--proposal-example-size",
        type=int,
        default=16,
        help=(
            "Number of balanced train examples D_q used for per-example candidate "
            "proposal/intersection at each step. Set <=0 to use all balanced pairs."
        ),
    )
    parser.add_argument(
        "--proposal-min-candidates",
        type=int,
        default=3,
        help=(
            "Minimum candidate count, including the current token. If the strict "
            "intersection is smaller, fall back to frequent candidates from the "
            "per-example top-k sets."
        ),
    )
    parser.add_argument(
        "--selection-top-mu",
        type=int,
        default=3,
        help="Number of gradient-ranked candidate tokens to forward-verify.",
    )
    parser.add_argument(
        "--fluency-lambda",
        type=float,
        default=0.2,
        help="Weight for instruction fluency in candidate selection.",
    )
    parser.add_argument(
        "--fluency-scope",
        choices=[
            FLUENCY_SCOPE_INSTRUCTION,
            FLUENCY_SCOPE_PREFIX,
            FLUENCY_SCOPE_PROPOSAL_PREFIX,
        ],
        default=FLUENCY_SCOPE_INSTRUCTION,
        help=(
            "instruction matches GreaTer's full-control fluency regularizer. "
            "prefix uses p_<=i. proposal_prefix uses task header + p_<=i."
        ),
    )
    parser.add_argument(
        "--fluency-metric",
        choices=[FLUENCY_METRIC_NLL, FLUENCY_METRIC_PPL],
        default=FLUENCY_METRIC_NLL,
        help="Metric multiplied by --fluency-lambda during candidate selection.",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument("--allow-non-ascii", action="store_true")
    parser.add_argument(
        "--output-root-dir",
        type=Path,
        default=Path("codes/greater_experiments"),
    )
    parser.add_argument("--output-substring", default="")
    return parser.parse_args()


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _create_run_dir(output_root_dir: Path, output_substring: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = output_substring.strip().replace("/", "-")
    run_name = f"{stamp}_{suffix}" if suffix else stamp
    run_dir = output_root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _namespace_to_json_dict(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def _format_prf_for_print(prf: Dict[str, float]) -> str:
    return (
        f"P={float(prf.get('precision', 0.0)):.2f} +/- "
        f"{float(prf.get('precision_std', 0.0)):.2f}, "
        f"R={float(prf.get('recall', 0.0)):.2f} +/- "
        f"{float(prf.get('recall_std', 0.0)):.2f}, "
        f"F1={float(prf.get('f1', 0.0)):.2f} +/- "
        f"{float(prf.get('f1_std', 0.0)):.2f}"
    )


def _resolve_initial_instruction_prompt(args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    if args.prompt_source == PROMPT_SOURCE_SCRATCH:
        return INFERENCE_INSTRUCTION_PROMPT_V1, {
            "prompt_source": PROMPT_SOURCE_SCRATCH,
            "source_symbol": "INFERENCE_INSTRUCTION_PROMPT_V1",
            "source_file": "codes/agents/agent_prompts.py",
        }

    if args.population_path is None:
        raise ValueError("--population-path is required with --prompt-source=population.")
    if args.prompt_node_id is None:
        raise ValueError("--prompt-node-id is required with --prompt-source=population.")

    payload = load_json_file(args.population_path)
    for node in payload.get("population", []):
        if node.get("node_id") == args.prompt_node_id:
            prompt = (
                node.get("inference_instruction_prompt")
                or node.get("inference_prompt")
                or ""
            )
            if not prompt:
                raise ValueError(
                    f"Node {args.prompt_node_id} does not contain an instruction prompt."
                )
            return prompt, {
                "prompt_source": PROMPT_SOURCE_POPULATION,
                "population_path": str(args.population_path),
                "prompt_node_id": args.prompt_node_id,
                "best_node_id": payload.get("best_node_id"),
                "node_val_score": node.get("val_score"),
            }

    raise KeyError(f"Could not find node_id={args.prompt_node_id} in population.")


def _resolve_binary_token_id(tokenizer, base_token: str) -> int:
    token_ids = tokenizer.encode(base_token, add_special_tokens=False)
    if len(token_ids) == 1:
        return token_ids[0]
    spaced_token_ids = tokenizer.encode(f" {base_token}", add_special_tokens=False)
    if len(spaced_token_ids) == 1:
        return spaced_token_ids[0]
    raise ValueError(f"Token {base_token!r} is not a single token for this tokenizer.")


def _decode_instruction(tokenizer, instruction_token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(instruction_token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _tokenize_instruction(tokenizer, instruction_prompt: str) -> List[int]:
    token_ids = tokenizer.encode(instruction_prompt, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Instruction prompt tokenized to an empty sequence.")
    return list(token_ids)


def _stable_retokenizes(tokenizer, instruction_token_ids: Sequence[int]) -> bool:
    decoded = _decode_instruction(tokenizer, instruction_token_ids)
    return _tokenize_instruction(tokenizer, decoded) == list(instruction_token_ids)


def _is_allowed_token(tokenizer, token_id: int, allow_non_ascii: bool) -> bool:
    if token_id in set(tokenizer.all_special_ids):
        return False
    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
    if not token_text:
        return False
    if not allow_non_ascii and (not token_text.isascii() or not token_text.isprintable()):
        return False
    return True


def _build_proposal_header(example_pair: Dict[str, Any]) -> str:
    support = example_pair["support"]

    return (
        "You are optimizing an instruction prompt for a binary relation extraction  classifier.\n\n"

        "The instruction prompt will appear before the classifier input.\n"
        "Later, the classifier will receive an input similar to the following example "
        "and must decide whether the target relation is expressed between the "
        "Subject and Object entities in the query sentence.\n\n"

        "The classifier must answer with exactly one token: \"yes\" or \"no\".\n\n"

        "Example classifier input:\n\n"

        f"Relation name: {support['relation']} ({example_pair['relation_description']})\n"
        f"Support Sentence: {example_pair['support_sentence']}\n"
        f"Query Sentence: {example_pair['query_sentence']}\n\n"

        "Write a instruction prompt that should appear before this type of input and help the classifier solve the relation extraction problem.\n\n"

        "Instruction:\n"
    )


@torch.inference_mode()
def _proposal_candidate_set_for_example(
    *,
    instruction_token_ids: Sequence[int],
    position: int,
    example_pair: Dict[str, Any],
    model,
    tokenizer,
    top_k: int,
    allow_non_ascii: bool,
) -> Tuple[List[int], str]:
    proposal_header = _build_proposal_header(example_pair)
    prefix = _decode_instruction(tokenizer, instruction_token_ids[:position])
    context = proposal_header + prefix
    model_inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False)
    target_device = getattr(model, "device", None)
    if target_device is not None:
        model_inputs = model_inputs.to(target_device)
    outputs = model(**model_inputs, use_cache=False)
    logits = outputs.logits[0, -1, :].float()

    current_token = int(instruction_token_ids[position])
    blocked = set(tokenizer.all_special_ids)
    if position > 0:
        blocked.add(int(instruction_token_ids[position - 1]))
    for token_id in blocked:
        if 0 <= token_id < logits.numel():
            logits[token_id] = float("-inf")
    proposal_pool_size = min(max(top_k * 20, top_k + 10), logits.numel())
    candidate_ids = torch.topk(logits, k=proposal_pool_size).indices.tolist()
    deduped: List[int] = []
    for token_id in [current_token] + [int(token_id) for token_id in candidate_ids]:
        if token_id in deduped:
            continue
        if not _is_allowed_token(tokenizer, token_id, allow_non_ascii=allow_non_ascii):
            continue
        test_ids = list(instruction_token_ids)
        test_ids[position] = token_id
        if not _stable_retokenizes(tokenizer, test_ids):
            continue
        deduped.append(token_id)
        if len(deduped) >= top_k + 1:
            break
    return deduped, proposal_header


def _sample_proposal_examples(
    *,
    sampled_pairs: Sequence[Dict[str, Any]],
    proposal_example_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if proposal_example_size <= 0 or proposal_example_size >= len(sampled_pairs):
        return list(sampled_pairs)

    buckets: Dict[str, List[Dict[str, Any]]] = {"tp": [], "tn": [], "fp": [], "fn": []}
    for pair in sampled_pairs:
        bucket_name = pair.get("confusion_bucket")
        if bucket_name in buckets:
            buckets[bucket_name].append(pair)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    nonempty_buckets = [name for name, bucket in buckets.items() if bucket]
    per_bucket = max(1, proposal_example_size // max(1, len(nonempty_buckets)))
    for bucket_name in nonempty_buckets:
        bucket = buckets[bucket_name]
        selected.extend(rng.sample(bucket, min(per_bucket, len(bucket))))

    remaining = proposal_example_size - len(selected)
    if remaining > 0:
        selected_ids = {id(pair) for pair in selected}
        rest = [pair for pair in sampled_pairs if id(pair) not in selected_ids]
        selected.extend(rng.sample(rest, min(remaining, len(rest))))

    rng.shuffle(selected)
    return selected[:proposal_example_size]


def _proposal_candidates_for_examples(
    *,
    instruction_token_ids: Sequence[int],
    position: int,
    proposal_examples: Sequence[Dict[str, Any]],
    model,
    tokenizer,
    top_k: int,
    min_candidates: int,
    allow_non_ascii: bool,
) -> Tuple[List[int], Dict[str, Any]]:
    current_token = int(instruction_token_ids[position])
    if not proposal_examples:
        raise ValueError("proposal_examples must not be empty.")

    candidate_sets: List[List[int]] = []
    proposal_headers: List[str] = []
    for example_pair in proposal_examples:
        candidate_ids, proposal_header = _proposal_candidate_set_for_example(
            instruction_token_ids=instruction_token_ids,
            position=position,
            example_pair=example_pair,
            model=model,
            tokenizer=tokenizer,
            top_k=top_k,
            allow_non_ascii=allow_non_ascii,
        )
        candidate_sets.append(candidate_ids)
        proposal_headers.append(proposal_header)

    strict_intersection = set(candidate_sets[0])
    for candidate_ids in candidate_sets[1:]:
        strict_intersection &= set(candidate_ids)

    candidate_counts: Dict[int, int] = {}
    for candidate_ids in candidate_sets:
        for token_id in candidate_ids:
            candidate_counts[int(token_id)] = candidate_counts.get(int(token_id), 0) + 1

    ranked_by_frequency = sorted(
        candidate_counts,
        key=lambda token_id: (
            token_id in strict_intersection,
            candidate_counts[token_id],
            -candidate_sets[0].index(token_id) if token_id in candidate_sets[0] else -10**9,
        ),
        reverse=True,
    )

    selected: List[int] = []
    for token_id in [current_token] + ranked_by_frequency:
        if token_id in selected:
            continue
        if token_id in strict_intersection or len(strict_intersection) < min_candidates:
            selected.append(int(token_id))
        if len(selected) >= max(min_candidates, len(strict_intersection), 1):
            break

    if current_token not in selected:
        selected.insert(0, current_token)

    metadata = {
        "proposal_example_count": len(proposal_examples),
        "per_example_candidate_counts": [len(candidate_ids) for candidate_ids in candidate_sets],
        "strict_intersection_size": len(strict_intersection),
        "strict_intersection": sorted(int(token_id) for token_id in strict_intersection),
        "used_frequency_fallback": len(strict_intersection) < min_candidates,
        "candidate_counts": {
            str(token_id): count
            for token_id, count in sorted(candidate_counts.items(), key=lambda item: item[0])
        },
        "proposal_headers": proposal_headers,
    }
    return selected, metadata


def _format_real_prompt(
    prompt: str,
    *,
    tokenizer,
    use_chat_template: bool,
) -> str:
    if not use_chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _candidate_one_hot_gradient(
    *,
    instruction_token_ids: Sequence[int],
    position: int,
    candidate_token_ids: Sequence[int],
    sampled_pairs: Sequence[Dict[str, Any]],
    model,
    tokenizer,
    yes_token_id: int,
    no_token_id: int,
    batch_size: int,
    use_chat_template: bool,
    fluency_lambda: float,
    fluency_scope: str,
) -> torch.Tensor:
    current_token = int(instruction_token_ids[position])
    if current_token not in candidate_token_ids:
        raise ValueError("Current token must be included in candidate_token_ids.")

    embedding_layer = model.get_input_embeddings()
    embed_device = embedding_layer.weight.device
    candidate_ids = torch.tensor(candidate_token_ids, device=embed_device, dtype=torch.long)
    candidate_embeds = embedding_layer.weight.detach()[candidate_ids]
    one_hot = torch.zeros(
        (1, len(candidate_token_ids)),
        device=embed_device,
        dtype=candidate_embeds.dtype,
    )
    current_index = candidate_token_ids.index(current_token)
    one_hot[0, current_index] = 1.0
    one_hot.requires_grad_(True)

    instruction_prompt = _decode_instruction(tokenizer, instruction_token_ids)
    total_pairs = len(sampled_pairs)
    if total_pairs == 0:
        raise ValueError("sampled_pairs must not be empty.")

    model.zero_grad(set_to_none=True)
    original_padding_side = getattr(tokenizer, "padding_side", "left")
    tokenizer.padding_side = "left"
    try:
        for chunk_index, chunk in enumerate(_batched(list(sampled_pairs), batch_size)):
            prompts, target_labels = _build_binary_prompts_for_instruction(
                instruction_prompt=instruction_prompt,
                binary_pairs=chunk,
            )
            formatted_prompts = [
                _format_real_prompt(
                    prompt,
                    tokenizer=tokenizer,
                    use_chat_template=use_chat_template,
                )
                for prompt in prompts
            ]
            encoded = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = encoded["input_ids"].to(embed_device)
            attention_mask = encoded["attention_mask"].to(embed_device)

            prompt_positions: List[int] = []
            for row, formatted_prompt in enumerate(formatted_prompts):
                token_map = _build_instruction_token_map(
                    instruction_prompt=instruction_prompt,
                    rendered_prompt=formatted_prompt,
                    tokenizer=tokenizer,
                )
                prompt_token_ids = token_map["prompt_token_ids"]
                if len(prompt_token_ids) != len(instruction_token_ids):
                    raise ValueError(
                        "Instruction token count changed inside formatted prompt. "
                        f"standalone_len={len(instruction_token_ids)} "
                        f"rendered_len={len(prompt_token_ids)}"
                    )
                if prompt_token_ids[position] != instruction_token_ids[position]:
                    raise ValueError(
                        "The instruction token being optimized changed inside the "
                        "formatted prompt. This usually means the current token is "
                        "at a prompt/input boundary and cannot be safely optimized "
                        "with the current token-level replacement path. "
                        f"position={position} "
                        f"standalone_token={instruction_token_ids[position]} "
                        f"rendered_token={prompt_token_ids[position]}"
                    )
                rendered_len = tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                )["input_ids"].size(1)
                left_pad = input_ids.size(1) - rendered_len
                prompt_positions.append(left_pad + token_map["prompt_token_positions"][position])

            selected_embed = one_hot @ candidate_embeds
            base_embeds = embedding_layer(input_ids).detach()
            replacement_embeds = torch.zeros_like(base_embeds)
            replacement_mask = torch.zeros(
                base_embeds.shape[:2],
                device=base_embeds.device,
                dtype=base_embeds.dtype,
            )
            for row, prompt_position in enumerate(prompt_positions):
                replacement_embeds[row, prompt_position, :] = selected_embed[0]
                replacement_mask[row, prompt_position] = 1.0
            inputs_embeds = (
                base_embeds * (1.0 - replacement_mask.unsqueeze(-1))
                + replacement_embeds * replacement_mask.unsqueeze(-1)
            )

            target_class_indices = torch.tensor(
                [0 if label == "yes" else 1 for label in target_labels],
                device=embed_device,
                dtype=torch.long,
            )
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            decision_logits = outputs.logits[:, -1, [yes_token_id, no_token_id]]
            loss = F.cross_entropy(
                decision_logits,
                target_class_indices,
                reduction="sum",
            ) / total_pairs

            if fluency_lambda > 0 and chunk_index == 0:
                fluency_loss = _differentiable_instruction_fluency_loss(
                    instruction_token_ids=instruction_token_ids,
                    position=position,
                    selected_embed=selected_embed,
                    model=model,
                    tokenizer=tokenizer,
                    scope=fluency_scope,
                )
                loss = loss + fluency_lambda * fluency_loss
            loss.backward()

            del encoded, input_ids, attention_mask, base_embeds, replacement_embeds
            del replacement_mask, inputs_embeds, outputs, decision_logits
            clear_cuda_cache()
    finally:
        tokenizer.padding_side = original_padding_side

    if one_hot.grad is None:
        raise RuntimeError("No gradient was produced for candidate one-hot selector.")
    grad = one_hot.grad.detach().clone()
    del one_hot, candidate_embeds
    clear_cuda_cache()
    return grad


def _differentiable_instruction_fluency_loss(
    *,
    instruction_token_ids: Sequence[int],
    position: int,
    selected_embed: torch.Tensor,
    model,
    tokenizer,
    scope: str,
) -> torch.Tensor:
    if scope == FLUENCY_SCOPE_PREFIX:
        token_ids = list(instruction_token_ids[: position + 1])
        replacement_position = len(token_ids) - 1
    else:
        if scope == FLUENCY_SCOPE_PROPOSAL_PREFIX:
            # GReaTer regularizes the optimized control string itself. The
            # proposal header is only artificial context for candidate proposal,
            # so use the instruction control tokens for the differentiable term.
            pass
        token_ids = list(instruction_token_ids)
        replacement_position = position

    if len(token_ids) < 2:
        return selected_embed.sum() * 0.0

    embed_device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([token_ids], device=embed_device, dtype=torch.long)
    base_embeds = model.get_input_embeddings()(input_ids).detach()
    replacement_embeds = torch.zeros_like(base_embeds)
    replacement_mask = torch.zeros(
        base_embeds.shape[:2],
        device=base_embeds.device,
        dtype=base_embeds.dtype,
    )
    replacement_embeds[0, replacement_position, :] = selected_embed[0]
    replacement_mask[0, replacement_position] = 1.0
    inputs_embeds = (
        base_embeds * (1.0 - replacement_mask.unsqueeze(-1))
        + replacement_embeds * replacement_mask.unsqueeze(-1)
    )

    outputs = model(inputs_embeds=inputs_embeds, use_cache=False)
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return loss


def _fluency_payload(
    *,
    instruction_token_ids: Sequence[int],
    position: int,
    proposal_header: str,
    model,
    tokenizer,
    scope: str,
) -> Dict[str, float]:
    if scope == FLUENCY_SCOPE_INSTRUCTION:
        token_ids = list(instruction_token_ids)
    elif scope == FLUENCY_SCOPE_PREFIX:
        token_ids = list(instruction_token_ids[: position + 1])
    elif scope == FLUENCY_SCOPE_PROPOSAL_PREFIX:
        text = proposal_header + _decode_instruction(
            tokenizer,
            instruction_token_ids[: position + 1],
        )
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        raise ValueError(f"Unsupported fluency scope: {scope}")

    if len(token_ids) < 2:
        return {"nll": 0.0, "ppl": 1.0}

    target_device = getattr(model, "device", None)
    input_ids = torch.tensor([token_ids], device=target_device or "cpu")
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
    nll_value = float(nll.item())
    ppl_value = float(torch.exp(nll).item())
    del input_ids, outputs, shift_logits, shift_labels, nll
    clear_cuda_cache()
    return {"nll": nll_value, "ppl": ppl_value}


def _score_binary_prompts_with_task_ce(
    *,
    prompts: Sequence[str],
    target_labels: Sequence[str],
    model,
    tokenizer,
    batch_size: int,
    use_chat_template: bool,
) -> Dict[str, Any]:
    if len(prompts) != len(target_labels):
        raise ValueError("prompts/target_labels length mismatch")
    if not prompts:
        return {
            "predicted_labels": [],
            "target_labels": [],
            "cross_entropy_losses": [],
            "mean_cross_entropy": 0.0,
            "task_perplexity": 1.0,
        }

    yes_token_id = _resolve_binary_token_id(tokenizer, "yes")
    no_token_id = _resolve_binary_token_id(tokenizer, "no")
    target_device = getattr(model, "device", None)
    original_padding_side = getattr(tokenizer, "padding_side", "left")
    predicted_labels: List[str] = []
    cross_entropy_losses: List[float] = []

    tokenizer.padding_side = "left"
    try:
        for prompt_batch, label_batch in zip(
            _batched(list(prompts), batch_size),
            _batched(list(target_labels), batch_size),
        ):
            if use_chat_template:
                formatted_batch = [
                    _format_real_prompt(
                        prompt,
                        tokenizer=tokenizer,
                        use_chat_template=True,
                    )
                    for prompt in prompt_batch
                ]
            else:
                formatted_batch = list(prompt_batch)

            model_inputs = tokenizer(
                formatted_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            if target_device is not None:
                model_inputs = model_inputs.to(target_device)

            target_class_indices = torch.tensor(
                [0 if label == "yes" else 1 for label in label_batch],
                device=model_inputs["input_ids"].device,
                dtype=torch.long,
            )

            with torch.inference_mode():
                outputs = model(**model_inputs, use_cache=False)
                decision_logits = outputs.logits[:, -1, [yes_token_id, no_token_id]]
                batch_losses = F.cross_entropy(
                    decision_logits,
                    target_class_indices,
                    reduction="none",
                )
                batch_predictions = torch.argmax(decision_logits, dim=-1)

            predicted_labels.extend(
                "yes" if prediction.item() == 0 else "no"
                for prediction in batch_predictions
            )
            cross_entropy_losses.extend(float(loss.item()) for loss in batch_losses)

            del model_inputs, target_class_indices, outputs, decision_logits
            del batch_losses, batch_predictions
            clear_cuda_cache()
    finally:
        tokenizer.padding_side = original_padding_side

    mean_cross_entropy = sum(cross_entropy_losses) / len(cross_entropy_losses)
    return {
        "predicted_labels": predicted_labels,
        "target_labels": list(target_labels),
        "cross_entropy_losses": cross_entropy_losses,
        "mean_cross_entropy": mean_cross_entropy,
        "task_perplexity": float(torch.exp(torch.tensor(mean_cross_entropy)).item()),
    }


def _score_candidate_instruction(
    *,
    instruction_token_ids: Sequence[int],
    position: int,
    sampled_pairs: Sequence[Dict[str, Any]],
    proposal_header: str,
    model,
    tokenizer,
    batch_size: int,
    use_chat_template: bool,
    fluency_lambda: float,
    fluency_scope: str,
    fluency_metric: str,
) -> Dict[str, Any]:
    instruction_prompt = _decode_instruction(tokenizer, instruction_token_ids)
    prompts, target_labels = _build_binary_prompts_for_instruction(
        instruction_prompt=instruction_prompt,
        binary_pairs=sampled_pairs,
    )
    task_metrics = _score_binary_prompts_with_task_ce(
        prompts=prompts,
        target_labels=target_labels,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        use_chat_template=use_chat_template,
    )
    fluency = _fluency_payload(
        instruction_token_ids=instruction_token_ids,
        position=position,
        proposal_header=proposal_header,
        model=model,
        tokenizer=tokenizer,
        scope=fluency_scope,
    )
    fluency_value = fluency[fluency_metric]
    total_loss = (
        float(task_metrics["mean_cross_entropy"])
        + fluency_lambda * float(fluency_value)
    )

    print(f"[scores] loss={float(task_metrics['mean_cross_entropy']):.3f}, fluency={float(fluency_value):.3f}, total_score={float(total_loss):.3f}")
    prf = _build_prf_scores(
        predicted_labels=task_metrics["predicted_labels"],
        target_labels=task_metrics["target_labels"],
    )
    return {
        "instruction_prompt": instruction_prompt,
        "mean_cross_entropy": float(task_metrics["mean_cross_entropy"]),
        "task_perplexity": float(task_metrics["task_perplexity"]),
        "prf": prf,
        "fluency": fluency,
        "fluency_metric": fluency_metric,
        "fluency_lambda": fluency_lambda,
        "fluency_loss_component": fluency_lambda * float(fluency_value),
        "total_loss": total_loss,
        "combined_score": total_loss,
        "predicted_labels": task_metrics["predicted_labels"],
        "target_labels": task_metrics["target_labels"],
    }


def _evaluate_full_split(
    *,
    instruction_prompt: str,
    full_eval_pairs: Sequence[Dict[str, Any]],
    model,
    tokenizer,
    batch_size: int,
    use_chat_template: bool,
    log_every: int,
) -> Dict[str, Any]:
    prompts, target_labels = _build_binary_prompts_for_instruction(
        instruction_prompt=instruction_prompt,
        binary_pairs=full_eval_pairs,
    )
    predicted_labels = run_binary_inference(
        prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        log_every=log_every,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _build_evaluation_report(
        score_payload={
            "predicted_labels": predicted_labels,
            "target_labels": target_labels,
        },
        binary_pairs=full_eval_pairs,
        prf_mode="episode",
    )


def _sample_balanced_train_pairs_for_step(
    *,
    instruction_prompt: str,
    train_samples: Dict[str, Any],
    train_sample_index: Dict[str, dict],
    dataset_type: str,
    sample_size: int,
    seed: int,
    model,
    tokenizer,
    batch_size: int,
    use_chat_template: bool,
    log_every: int,
    log_label: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, float]]:
    rng = random.Random(seed)
    train_feedback_samples = sample_feedback_fn(
        train_samples,
        k=sample_size,
        rng=rng,
    )
    train_pair_pool = _build_binary_pairs_from_feedback_samples(
        feedback_samples=train_feedback_samples,
        shots_by_id=train_sample_index,
        dataset_type=dataset_type,
    )
    train_prompts, train_target_labels = _build_binary_prompts_for_instruction(
        instruction_prompt=instruction_prompt,
        binary_pairs=train_pair_pool,
    )
    train_predicted_labels = run_binary_inference(
        train_prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        log_every=log_every,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        log_label=log_label,
    )
    bucketed_pairs = _attach_predictions_to_pairs(
        pairs=train_pair_pool,
        predicted_labels=train_predicted_labels,
    )
    sampled_pairs, bucket_stats = _sample_balanced_pairs_from_bucketed_pool(
        bucketed_pairs=bucketed_pairs,
        seed=seed,
    )
    baseline_score_payload = _build_score_payload_from_pairs(sampled_pairs)
    baseline_prf = _build_prf_scores(
        predicted_labels=baseline_score_payload["predicted_labels"],
        target_labels=baseline_score_payload["target_labels"],
    )
    bucket_stats = {
        **bucket_stats,
        "train_pair_pool_size": len(train_pair_pool),
        "balanced_pair_count": len(sampled_pairs),
    }
    return sampled_pairs, bucket_stats, baseline_prf


def _freeze_model_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def main() -> None:
    args = _parse_args()
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive.")
    if args.train_gradient_sample_size <= 0:
        raise ValueError("--train-gradient-sample-size must be positive.")
    if args.fluency_lambda < 0:
        raise ValueError("--fluency-lambda must be non-negative.")

    start_time = time.monotonic()
    run_dir = _create_run_dir(args.output_root_dir, args.output_substring)
    steps_path = run_dir / "steps.jsonl"
    _save_json(run_dir / "args.json", _namespace_to_json_dict(args))

    instruction_prompt, prompt_source_info = _resolve_initial_instruction_prompt(args)
    print("[relation_extraction_greater] run_dir:", run_dir)
    print("[relation_extraction_greater] prompt_source:", prompt_source_info)
    print("[relation_extraction_greater] loading model")
    model, tokenizer = load_model_and_tokenizer(args.model, device_map=args.device_map)
    _freeze_model_parameters(model)
    use_chat_template = not args.disable_chat_template

    raw_instruction_token_ids = _tokenize_instruction(tokenizer, instruction_prompt)
    initial_instruction_prompt = _decode_instruction(tokenizer, raw_instruction_token_ids)
    instruction_token_ids = _tokenize_instruction(tokenizer, initial_instruction_prompt)
    if instruction_token_ids != raw_instruction_token_ids:
        print(
            "[relation_extraction_greater] warning: initial instruction does not "
            "exactly round-trip through decode/tokenize; continuing with normalized "
            "tokenized form."
        )
    (run_dir / "initial_prompt.txt").write_text(initial_instruction_prompt, encoding="utf-8")
    _save_json(run_dir / "prompt_source.json", prompt_source_info)
    print(
        "[relation_extraction_greater] initial tokens:",
        len(instruction_token_ids),
    )

    yes_token_id = _resolve_binary_token_id(tokenizer, "yes")
    no_token_id = _resolve_binary_token_id(tokenizer, "no")

    print("[relation_extraction_greater] loading train samples")
    train_samples = load_train_samples(
        data_dir=args.data_dir,
        filename=args.train_samples,
    )
    train_sample_index = _build_train_sample_index(train_samples)
    print("[relation_extraction_greater] building initial train buckets")
    sampled_pairs, bucket_stats, baseline_prf = _sample_balanced_train_pairs_for_step(
        instruction_prompt=initial_instruction_prompt,
        train_samples=train_samples,
        train_sample_index=train_sample_index,
        dataset_type=args.dataset_type,
        sample_size=args.train_gradient_sample_size,
        seed=args.seed,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.selection_batch_size,
        use_chat_template=use_chat_template,
        log_every=args.log_every,
        log_label="greater_train_bucket_initial",
    )
    _save_json(
        run_dir / "balanced_train_sample.json",
        {
            "bucket_stats": bucket_stats,
            "baseline_prf": baseline_prf,
            "sampled_pairs": sampled_pairs,
        },
    )
    print("[relation_extraction_greater] balanced bucket stats:", bucket_stats)

    print("[relation_extraction_greater] loading full eval split:", args.full_eval_split)
    full_eval_dataset = load_split_episodes(
        split=args.full_eval_split,
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
    )
    full_eval_pairs = build_full_binary_pairs(
        dataset=full_eval_dataset,
        dataset_type=args.dataset_type,
        query_index=args.query_index,
    )

    position = args.start_position % len(instruction_token_ids)
    best_instruction_ids = list(instruction_token_ids)
    best_selection_score = float("inf")
    best_full_eval: Dict[str, Any] | None = None

    for step_index in range(1, args.n_steps + 1):
        step_start = time.monotonic()
        current_instruction = _decode_instruction(tokenizer, instruction_token_ids)
        current_token = int(instruction_token_ids[position])
        print(
            "[relation_extraction_greater] step",
            f"{step_index}/{args.n_steps}",
            f"position={position}",
            f"token={tokenizer.decode([current_token])!r}",
        )
        print(f"[Prompt to optimize] prompt = {current_instruction}")
        sampled_pairs, bucket_stats, baseline_prf = _sample_balanced_train_pairs_for_step(
            instruction_prompt=current_instruction,
            train_samples=train_samples,
            train_sample_index=train_sample_index,
            dataset_type=args.dataset_type,
            sample_size=args.train_gradient_sample_size,
            seed=args.seed + step_index,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.selection_batch_size,
            use_chat_template=use_chat_template,
            log_every=args.log_every,
            log_label=f"greater_train_bucket_step_{step_index}",
        )
        proposal_examples = _sample_proposal_examples(
            sampled_pairs=sampled_pairs,
            proposal_example_size=args.proposal_example_size,
            seed=args.seed + 10_000 + step_index,
        )
        proposal_header_for_scoring = _build_proposal_header(proposal_examples[0])
        baseline_selection = _score_candidate_instruction(
            instruction_token_ids=instruction_token_ids,
            position=position,
            sampled_pairs=sampled_pairs,
            proposal_header=proposal_header_for_scoring,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.selection_batch_size,
            use_chat_template=use_chat_template,
            fluency_lambda=args.fluency_lambda,
            fluency_scope=args.fluency_scope,
            fluency_metric=args.fluency_metric,
        )
        if baseline_selection["combined_score"] < best_selection_score:
            best_selection_score = float(baseline_selection["combined_score"])
            best_instruction_ids = list(instruction_token_ids)
        pre_step_selection = dict(baseline_selection)
        print(
            "[relation_extraction_greater] balanced train before step:",
            f"step={step_index}",
            _format_prf_for_print(pre_step_selection["prf"]),
            f"loss={pre_step_selection['combined_score']:.6f}",
        )

        candidate_token_ids, proposal_metadata = _proposal_candidates_for_examples(
            instruction_token_ids=instruction_token_ids,
            position=position,
            proposal_examples=proposal_examples,
            model=model,
            tokenizer=tokenizer,
            top_k=args.proposal_top_k,
            min_candidates=args.proposal_min_candidates,
            allow_non_ascii=args.allow_non_ascii,
        )

        current_tok = _decode_instruction(tokenizer, [instruction_token_ids[position]])
        cands_toks = []
        for tokid in candidate_token_ids:
            new_tok = _decode_instruction(tokenizer, [tokid])
            cands_toks.append(new_tok)
        print(f"[Candidates tokens] current tok: '{current_tok}' , replacement: {cands_toks}")

        gradient = _candidate_one_hot_gradient(
            instruction_token_ids=instruction_token_ids,
            position=position,
            candidate_token_ids=candidate_token_ids,
            sampled_pairs=sampled_pairs,
            model=model,
            tokenizer=tokenizer,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            batch_size=args.gradient_batch_size,
            use_chat_template=use_chat_template,
            fluency_lambda=args.fluency_lambda,
            fluency_scope=args.fluency_scope,
        )
        mu = min(args.selection_top_mu, len(candidate_token_ids))
        selected_indices = torch.topk(-gradient[0], k=mu).indices.detach().cpu().tolist()
        selected_candidate_ids = [candidate_token_ids[index] for index in selected_indices]
        if current_token not in selected_candidate_ids:
            selected_candidate_ids.append(current_token)

        candidate_records: List[Dict[str, Any]] = []
        best_step_record: Dict[str, Any] | None = None
        best_step_ids = list(instruction_token_ids)
        for candidate_id in selected_candidate_ids:
            trial_ids = list(instruction_token_ids)
            current_tok = _decode_instruction(tokenizer, [trial_ids[position]])

            trial_ids[position] = int(candidate_id)
            test_tok = _decode_instruction(tokenizer, [candidate_id])

            trial_prompt = _decode_instruction(tokenizer, trial_ids)
            print(f"[Token replacing] current tok: '{current_tok}' , replacement: {test_tok}")
            print(f"[trailing prompt]: {trial_prompt}")

            score_payload = _score_candidate_instruction(
                instruction_token_ids=trial_ids,
                position=position,
                sampled_pairs=sampled_pairs,
                proposal_header=proposal_header_for_scoring,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.selection_batch_size,
                use_chat_template=use_chat_template,
                fluency_lambda=args.fluency_lambda,
                fluency_scope=args.fluency_scope,
                fluency_metric=args.fluency_metric,
            )
            record = {
                "token_id": int(candidate_id),
                "token_text": tokenizer.decode([int(candidate_id)], skip_special_tokens=False),
                "gradient": float(gradient[0, candidate_token_ids.index(int(candidate_id))].item()),
                "mean_cross_entropy": score_payload["mean_cross_entropy"],
                "task_perplexity": score_payload["task_perplexity"],
                "prf": score_payload["prf"],
                "fluency": score_payload["fluency"],
                "fluency_loss_component": score_payload["fluency_loss_component"],
                "total_loss": score_payload["total_loss"],
                "combined_score": score_payload["combined_score"],
                "changed": int(candidate_id) != current_token,
            }
            candidate_records.append(record)
            if best_step_record is None or record["combined_score"] < best_step_record["combined_score"]:
                best_step_record = record
                best_step_ids = trial_ids

        if best_step_record is None:
            raise RuntimeError("No candidate was scored.")

        accepted = best_step_record["combined_score"] <= pre_step_selection["combined_score"]
        if accepted:
            instruction_token_ids = best_step_ids
            baseline_selection = _score_candidate_instruction(
                instruction_token_ids=instruction_token_ids,
                position=position,
                sampled_pairs=sampled_pairs,
                proposal_header=proposal_header_for_scoring,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.selection_batch_size,
                use_chat_template=use_chat_template,
                fluency_lambda=args.fluency_lambda,
                fluency_scope=args.fluency_scope,
                fluency_metric=args.fluency_metric,
            )
            if best_step_record["combined_score"] < best_selection_score:
                best_selection_score = best_step_record["combined_score"]
                best_instruction_ids = list(instruction_token_ids)
        post_step_prf = baseline_selection["prf"]
        print(
            "[relation_extraction_greater] balanced train after step:",
            f"step={step_index}",
            _format_prf_for_print(post_step_prf),
            f"loss={baseline_selection['combined_score']:.6f}",
            f"accepted={accepted}",
        )

        full_eval = None
        if accepted and args.eval_every > 0 and (
            step_index % args.eval_every == 0 or step_index == args.n_steps
        ):
            print(f"[new prompt updated!!!!!!]")
            print(f"[Prompt to full eval] prompt = {_decode_instruction(tokenizer, instruction_token_ids)}")
            full_eval = _evaluate_full_split(
                instruction_prompt=_decode_instruction(tokenizer, instruction_token_ids),
                full_eval_pairs=full_eval_pairs,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.eval_batch_size,
                use_chat_template=use_chat_template,
                log_every=args.log_every,
            )
            best_full_eval = full_eval
            print(
                "[relation_extraction_greater] dev evaluation:",
                f"step={step_index}",
                f"split={args.full_eval_split}",
                _format_prf_for_print(full_eval["prf"]),
            )
        elif args.eval_every > 1:
            print(
                "[relation_extraction_greater] dev evaluation skipped:",
                f"step={step_index}",
                f"next_eval_every={args.eval_every}",
                "set --eval-every 1 to print dev scores every iteration",
            )

        step_payload = {
            "step": step_index,
            "position": position,
            "input_prompt": current_instruction,
            "current_token_id": current_token,
            "current_token_text": tokenizer.decode([current_token], skip_special_tokens=False),
            "step_bucket_stats": bucket_stats,
            "step_baseline_prf": baseline_prf,
            "step_baseline_selection": {
                "mean_cross_entropy": pre_step_selection["mean_cross_entropy"],
                "task_perplexity": pre_step_selection["task_perplexity"],
                "prf": pre_step_selection["prf"],
                "fluency": pre_step_selection["fluency"],
                "fluency_loss_component": pre_step_selection["fluency_loss_component"],
                "total_loss": pre_step_selection["total_loss"],
                "combined_score": pre_step_selection["combined_score"],
            },
            "proposal_metadata": proposal_metadata,
            "proposal_candidate_count": len(candidate_token_ids),
            "proposal_candidates": [
                {
                    "token_id": int(token_id),
                    "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                }
                for token_id in candidate_token_ids
            ],
            "selected_candidate_ids": [int(token_id) for token_id in selected_candidate_ids],
            "candidate_scores": candidate_records,
            "accepted": accepted,
            "selected_token": best_step_record,
            "output_prompt": _decode_instruction(tokenizer, instruction_token_ids),
            "selection_score_after_step": baseline_selection["combined_score"],
            "balanced_train_after_step_prf": post_step_prf,
            "full_evaluation": full_eval,
            "elapsed_seconds": time.monotonic() - step_start,
        }
        _append_jsonl(steps_path, step_payload)
        (run_dir / "latest_prompt.txt").write_text(
            step_payload["output_prompt"],
            encoding="utf-8",
        )
        print(
            "[relation_extraction_greater] step result:",
            f"accepted={accepted}",
            f"selected={best_step_record['token_text']!r}",
            f"score={best_step_record['combined_score']:.6f}",
            f"elapsed={step_payload['elapsed_seconds']:.2f}s",
        )

        position = (position + 1) % len(instruction_token_ids)

    final_prompt = _decode_instruction(tokenizer, instruction_token_ids)
    best_prompt = _decode_instruction(tokenizer, best_instruction_ids)
    (run_dir / "final_prompt.txt").write_text(final_prompt, encoding="utf-8")
    (run_dir / "best_selection_prompt.txt").write_text(best_prompt, encoding="utf-8")
    summary = {
        "run_dir": str(run_dir),
        "prompt_source": prompt_source_info,
        "initial_prompt": initial_instruction_prompt,
        "final_prompt": final_prompt,
        "best_selection_prompt": best_prompt,
        "best_selection_score": best_selection_score,
        "last_full_evaluation": best_full_eval,
        "bucket_stats": bucket_stats,
        "baseline_prf": baseline_prf,
        "elapsed_seconds": time.monotonic() - start_time,
    }
    _save_json(run_dir / "summary.json", summary)
    print("[relation_extraction_greater] done:", run_dir)


if __name__ == "__main__":
    main()
