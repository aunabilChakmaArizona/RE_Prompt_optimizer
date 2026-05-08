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
from agents.agent_gradient_token_analysis import score_binary_prompts_with_ce_and_perplexity
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


def _build_proposal_header(example_pair: Dict[str, Any] | None = None) -> str:
    header = (
        "You are optimizing an instruction prompt for a binary relation extraction "
        "classifier.\n\n"
        "The final classifier input appears after the instruction and has this schema:\n\n"
        "Relation name: <relation name> (<relation definition>)\n"
        "Support Sentence: <sentence with subject/object tags>\n"
        "Query Sentence: <sentence with subject/object tags>\n\n"
        "The classifier should answer whether the target relation is expressed "
        "between the Subject and Object entities in the query sentence.\n"
    )
    if example_pair is not None:
        support = example_pair["support"]
        header += (
            "\nExample relation input:\n"
            f"Relation name: {support['relation']} "
            f"({example_pair['relation_description']})\n"
            f"Support Sentence: {example_pair['support_sentence']}\n"
            f"Query Sentence: {example_pair['query_sentence']}\n"
        )
    return header + "\nWrite the instruction that should appear before the input.\n\nInstruction:\n"


@torch.inference_mode()
def _proposal_candidates_for_position(
    *,
    instruction_token_ids: Sequence[int],
    position: int,
    proposal_header: str,
    model,
    tokenizer,
    top_k: int,
    allow_non_ascii: bool,
) -> List[int]:
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
    if current_token not in deduped:
        deduped.insert(0, current_token)
    return deduped


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
        for chunk in _batched(list(sampled_pairs), batch_size):
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
                if token_map["prompt_token_ids"] != list(instruction_token_ids):
                    raise ValueError(
                        "Instruction tokenization changed inside formatted prompt. "
                        "Use a stable instruction prompt before optimizing."
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
    task_metrics = score_binary_prompts_with_ce_and_perplexity(
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
    combined_score = (
        float(task_metrics["mean_cross_entropy"])
        + fluency_lambda * float(fluency_value)
    )
    return {
        "instruction_prompt": instruction_prompt,
        "mean_cross_entropy": float(task_metrics["mean_cross_entropy"]),
        "task_perplexity": float(task_metrics["mean_perplexity"]),
        "fluency": fluency,
        "fluency_metric": fluency_metric,
        "combined_score": combined_score,
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
    rng = random.Random(args.seed)
    train_feedback_samples = sample_feedback_fn(
        train_samples,
        k=args.train_gradient_sample_size,
        rng=rng,
    )
    train_pair_pool = _build_binary_pairs_from_feedback_samples(
        feedback_samples=train_feedback_samples,
        shots_by_id=train_sample_index,
        dataset_type=args.dataset_type,
    )

    print("[relation_extraction_greater] building initial train buckets")
    train_prompts, train_target_labels = _build_binary_prompts_for_instruction(
        instruction_prompt=initial_instruction_prompt,
        binary_pairs=train_pair_pool,
    )
    train_predicted_labels = run_binary_inference(
        train_prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.selection_batch_size,
        log_every=args.log_every,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        log_label="greater_train_bucket",
    )
    bucketed_pairs = _attach_predictions_to_pairs(
        pairs=train_pair_pool,
        predicted_labels=train_predicted_labels,
    )
    sampled_pairs, bucket_stats = _sample_balanced_pairs_from_bucketed_pool(
        bucketed_pairs=bucketed_pairs,
        seed=args.seed,
    )
    baseline_score_payload = _build_score_payload_from_pairs(sampled_pairs)
    baseline_prf = _build_prf_scores(
        predicted_labels=baseline_score_payload["predicted_labels"],
        target_labels=baseline_score_payload["target_labels"],
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
    proposal_header = _build_proposal_header(sampled_pairs[0] if sampled_pairs else None)
    (run_dir / "proposal_header.txt").write_text(proposal_header, encoding="utf-8")

    position = args.start_position % len(instruction_token_ids)
    baseline_selection = _score_candidate_instruction(
        instruction_token_ids=instruction_token_ids,
        position=position,
        sampled_pairs=sampled_pairs,
        proposal_header=proposal_header,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.selection_batch_size,
        use_chat_template=use_chat_template,
        fluency_lambda=args.fluency_lambda,
        fluency_scope=args.fluency_scope,
        fluency_metric=args.fluency_metric,
    )
    best_instruction_ids = list(instruction_token_ids)
    best_selection_score = float(baseline_selection["combined_score"])
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

        candidate_token_ids = _proposal_candidates_for_position(
            instruction_token_ids=instruction_token_ids,
            position=position,
            proposal_header=proposal_header,
            model=model,
            tokenizer=tokenizer,
            top_k=args.proposal_top_k,
            allow_non_ascii=args.allow_non_ascii,
        )
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
            trial_ids[position] = int(candidate_id)
            score_payload = _score_candidate_instruction(
                instruction_token_ids=trial_ids,
                position=position,
                sampled_pairs=sampled_pairs,
                proposal_header=proposal_header,
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
                "fluency": score_payload["fluency"],
                "combined_score": score_payload["combined_score"],
                "changed": int(candidate_id) != current_token,
            }
            candidate_records.append(record)
            if best_step_record is None or record["combined_score"] < best_step_record["combined_score"]:
                best_step_record = record
                best_step_ids = trial_ids

        if best_step_record is None:
            raise RuntimeError("No candidate was scored.")

        accepted = best_step_record["combined_score"] <= baseline_selection["combined_score"]
        if accepted:
            instruction_token_ids = best_step_ids
            baseline_selection = _score_candidate_instruction(
                instruction_token_ids=instruction_token_ids,
                position=position,
                sampled_pairs=sampled_pairs,
                proposal_header=proposal_header,
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

        full_eval = None
        if args.eval_every > 0 and (
            step_index % args.eval_every == 0 or step_index == args.n_steps
        ):
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

        step_payload = {
            "step": step_index,
            "position": position,
            "input_prompt": current_instruction,
            "current_token_id": current_token,
            "current_token_text": tokenizer.decode([current_token], skip_special_tokens=False),
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
