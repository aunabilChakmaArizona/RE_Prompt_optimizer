#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

CURRENT_DIR = Path(__file__).resolve().parent
CODES_DIR = CURRENT_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.append(str(CODES_DIR))

from agents.agent_data_loader import DEFAULT_DATA_DIR, load_split_episodes
from agents.agent_gradient_token_analysis import (
    TOKEN_CANDIDATE_MODE_CHOICES,
    TOKEN_CANDIDATE_MODE_NEAREST_UPDATED,
    analyze_relation_extraction_binary_pairs,
    build_relation_prompt,
)
from agents.agent_llm_prompting import run_prompts
from agents.agent_models import load_model_and_tokenizer
from agents.agent_prompts import (
    GRADIENT_REGION_MUTATION_PROMPT_V1,
    INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
)
from agents.agent_relation_utils import get_relation_description
from agents.agent_utils import get_sentence_with_tags


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _namespace_to_json_dict(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _resolve_dataset_split(saved_split: str | None) -> str:
    if not saved_split:
        return "dev"
    if saved_split == "validation":
        return "dev"
    return saved_split


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample correct and incorrect TACRED binary inference pairs from a saved "
            "eval output file and run prompt-token gradient analysis on them."
        )
    )
    parser.add_argument("--model", required=True, help="HF model id to load")
    parser.add_argument(
        "--meta-prompt-model",
        default=None,
        help=(
            "Optional HF model id used only for generating prompt variants from the "
            "meta prompt. If omitted, --model is reused."
        ),
    )
    parser.add_argument(
        "--device-map",
        default="cuda:0",
        help="Device map override (e.g. cuda:0, cuda:1, cpu, auto)",
    )
    parser.add_argument(
        "--eval-output-path",
        type=Path,
        required=True,
        help="Path to EVALID_*_labels_predictions.json",
    )
    parser.add_argument(
        "--prompt-source-path",
        type=Path,
        required=True,
        help="Path to generated_prompt_candidates.json or population.json",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=None,
        help="0-based prompt index for generated_prompt_candidates.json",
    )
    parser.add_argument(
        "--prompt-node-id",
        type=int,
        default=None,
        help="node_id for population.json",
    )
    parser.add_argument("--dataset-type", default="fs_tacred")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to load. Defaults to split saved inside the eval output file.",
    )
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--num-correct", type=int, default=4)
    parser.add_argument("--num-mistakes", type=int, default=4)
    parser.add_argument(
        "--mistake-coverage",
        type=float,
        default=0.0,
        help="If > 0, sample this coverage fraction of all mistakes and match correct count to it.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient-batch-size",
        type=int,
        default=8,
        help="Optional batch size used while accumulating gradients over the sampled pairs.",
    )
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument(
        "--candidate-mode",
        choices=TOKEN_CANDIDATE_MODE_CHOICES,
        default=TOKEN_CANDIDATE_MODE_NEAREST_UPDATED,
    )
    parser.add_argument("--max-regions", type=int, default=5)
    parser.add_argument("--max-total-region-tokens", type=int, default=15)
    parser.add_argument(
        "--region-expansion-threshold-ratio",
        type=float,
        default=0.6,
        help="Threshold ratio used when expanding a peak token into a larger gradient region.",
    )
    parser.add_argument("--embedding-step-size", type=float, default=1.0)
    parser.add_argument(
        "--num-edit-regions",
        type=int,
        default=1,
        help="Use the top X ranked gradient regions for localized prompt editing.",
    )
    parser.add_argument(
        "--num-generated-prompts",
        type=int,
        default=0,
        help="How many prompt variants to generate from the selected gradient region.",
    )
    parser.add_argument(
        "--meta-prompt-max-new-tokens",
        type=int,
        default=10000,
        help="Generation budget for each region-edit meta prompt run.",
    )
    parser.add_argument(
        "--meta-prompt-batch-size",
        type=int,
        default=1,
        help="Batch size for running the region-edit meta prompt repeatedly.",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=8,
        help="Batch size for validating generated prompts on the sampled pairs.",
    )
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def _resolve_prompt_from_generated_candidates(
    payload: Dict[str, Any],
    prompt_index: int | None,
) -> Tuple[str, Dict[str, Any]]:
    if prompt_index is None:
        raise ValueError("--prompt-index is required for generated_prompt_candidates.json")

    prompts = payload.get("candidate_prompts", [])
    metadata_list = payload.get("candidate_metadata", [])
    if not 0 <= prompt_index < len(prompts):
        raise IndexError(
            f"prompt_index={prompt_index} is out of range for {len(prompts)} prompts."
        )

    metadata: Dict[str, Any] = {}
    candidate_index = prompt_index + 1
    for item in metadata_list:
        if item.get("candidate_index") == candidate_index:
            metadata = item
            break

    return prompts[prompt_index], {
        "prompt_source_type": "generated_prompt_candidates",
        "prompt_index": prompt_index,
        "candidate_index": candidate_index,
        "candidate_metadata": metadata,
    }


def _resolve_prompt_from_population(
    payload: Dict[str, Any],
    prompt_node_id: int | None,
) -> Tuple[str, Dict[str, Any]]:
    if prompt_node_id is None:
        raise ValueError("--prompt-node-id is required for population.json")

    population = payload.get("population", [])
    for node in population:
        if node.get("node_id") == prompt_node_id:
            return node.get("inference_instruction_prompt", ""), {
                "prompt_source_type": "population",
                "prompt_node_id": prompt_node_id,
                "best_node_id": payload.get("best_node_id"),
                "node_summary": {
                    "node_id": node.get("node_id"),
                    "parent_id": node.get("parent_id"),
                    "children_ids": node.get("children_ids", []),
                    "inference_mode": node.get("inference_mode"),
                    "val_score": node.get("val_score"),
                },
            }
    raise KeyError(f"Could not find node_id={prompt_node_id} in population.json")


def resolve_instruction_prompt(
    prompt_source_path: Path,
    *,
    prompt_index: int | None,
    prompt_node_id: int | None,
) -> Tuple[str, Dict[str, Any]]:
    payload = _load_json(prompt_source_path)
    if prompt_source_path.name == "generated_prompt_candidates.json":
        return _resolve_prompt_from_generated_candidates(payload, prompt_index)
    if prompt_source_path.name == "population.json":
        return _resolve_prompt_from_population(payload, prompt_node_id)
    raise ValueError(
        "Unsupported prompt source file. Expected generated_prompt_candidates.json "
        "or population.json."
    )


def _sample_indices(
    indices: Sequence[int],
    *,
    k: int,
    rng: random.Random,
    label: str,
) -> List[int]:
    unique_indices = sorted(set(indices))
    if len(indices) < k:
        raise ValueError(f"Need {k} {label} samples, but found only {len(indices)}.")
    if len(unique_indices) < k:
        raise ValueError(
            f"Need {k} unique {label} samples, but found only {len(unique_indices)}."
        )
    return sorted(rng.sample(unique_indices, k))


def _bucket_pair_indices(
    labels: Sequence[str],
    predictions: Sequence[str],
) -> Dict[str, List[int]]:
    buckets = {
        "tp": [],
        "tn": [],
        "fp": [],
        "fn": [],
    }
    for idx, (label, prediction) in enumerate(zip(labels, predictions)):
        if label == "yes" and prediction == "yes":
            buckets["tp"].append(idx)
        elif label == "no" and prediction == "no":
            buckets["tn"].append(idx)
        elif label == "no" and prediction == "yes":
            buckets["fp"].append(idx)
        elif label == "yes" and prediction == "no":
            buckets["fn"].append(idx)
    return buckets


def sample_eval_pair_indices(
    eval_payload: Dict[str, Any],
    *,
    num_correct: int,
    num_mistakes: int,
    mistake_coverage: float,
    seed: int,
) -> Dict[str, List[int]]:
    labels = eval_payload.get("pair_labels", [])
    predictions = eval_payload.get("pair_predictions", [])
    if len(labels) != len(predictions):
        raise ValueError("eval output pair_labels/pair_predictions length mismatch")

    buckets = _bucket_pair_indices(labels, predictions)
    correct_indices = buckets["tp"] + buckets["tn"]
    mistake_indices = buckets["fp"] + buckets["fn"]

    if not 0.0 <= mistake_coverage <= 1.0:
        raise ValueError("--mistake-coverage must be in the [0, 1] range.")

    if mistake_coverage > 0.0:
        resolved_num_mistakes = math.ceil(len(mistake_indices) * mistake_coverage)
        resolved_num_mistakes = max(1, resolved_num_mistakes)
    else:
        if num_mistakes > len(mistake_indices):
            raise ValueError(
                f"num_mistakes={num_mistakes} exceeds available mistakes={len(mistake_indices)}."
            )
        resolved_num_mistakes = num_mistakes

    rng = random.Random(seed)
    selected_per_bucket = min(
        len(buckets["tp"]),
        len(buckets["tn"]),
        len(buckets["fp"]),
        len(buckets["fn"]),
        resolved_num_mistakes // 2,
    )
    if resolved_num_mistakes > 0 and selected_per_bucket == 0:
        raise ValueError(
            "Unable to sample balanced tp/tn/fp/fn buckets. "
            f"available_tp={len(buckets['tp'])}, available_tn={len(buckets['tn'])}, "
            f"available_fp={len(buckets['fp'])}, available_fn={len(buckets['fn'])}, "
            f"requested_mistakes={resolved_num_mistakes}."
        )

    selected_fp = _sample_indices(
        buckets["fp"],
        k=selected_per_bucket,
        rng=rng,
        label="false_positive",
    ) if selected_per_bucket > 0 else []
    selected_fn = _sample_indices(
        buckets["fn"],
        k=selected_per_bucket,
        rng=rng,
        label="false_negative",
    ) if selected_per_bucket > 0 else []
    selected_mistake_indices = sorted(selected_fp + selected_fn)

    selected_tp = _sample_indices(
        buckets["tp"],
        k=selected_per_bucket,
        rng=rng,
        label="true_positive",
    ) if selected_per_bucket > 0 else []
    selected_tn = _sample_indices(
        buckets["tn"],
        k=selected_per_bucket,
        rng=rng,
        label="true_negative",
    ) if selected_per_bucket > 0 else []

    result = {
        "correct_indices": sorted(selected_tp + selected_tn),
        "mistake_indices": sorted(selected_mistake_indices),
        "bucket_stats": {
            "available": {
                "tp": len(buckets["tp"]),
                "tn": len(buckets["tn"]),
                "fp": len(buckets["fp"]),
                "fn": len(buckets["fn"]),
            },
            "selected": {
                "tp": len(selected_tp),
                "tn": len(selected_tn),
                "fp": len(selected_fp),
                "fn": len(selected_fn),
            },
        },
    }
    print(
        "[agent_gradient_eval_debug] pair pool:",
        f"requested_mistakes={resolved_num_mistakes}",
        f"available_tp={result['bucket_stats']['available']['tp']}",
        f"available_tn={result['bucket_stats']['available']['tn']}",
        f"available_fp={result['bucket_stats']['available']['fp']}",
        f"available_fn={result['bucket_stats']['available']['fn']}",
        f"selected_tp={result['bucket_stats']['selected']['tp']}",
        f"selected_tn={result['bucket_stats']['selected']['tn']}",
        f"selected_fp={result['bucket_stats']['selected']['fp']}",
        f"selected_fn={result['bucket_stats']['selected']['fn']}",
        f"selected_mistakes={len(result['mistake_indices'])}",
        f"selected_corrects={len(result['correct_indices'])}",
    )
    return result


def _resolve_num_ways(episodes: Sequence[Dict[str, Any]]) -> int:
    if not episodes:
        raise ValueError("Dataset episodes are empty.")
    num_ways = len(episodes[0]["meta_train"])
    if num_ways <= 0:
        raise ValueError("Episodes contain no candidate ways.")
    return num_ways


def _pair_index_to_episode_way(pair_index: int, num_ways: int) -> Tuple[int, int]:
    return divmod(pair_index, num_ways)


def _pair_confusion_bucket(label: str, prediction: str) -> str:
    if label == "yes" and prediction == "yes":
        return "tp"
    if label == "no" and prediction == "no":
        return "tn"
    if label == "no" and prediction == "yes":
        return "fp"
    if label == "yes" and prediction == "no":
        return "fn"
    raise ValueError(f"Unsupported label/prediction pair: label={label}, prediction={prediction}")


def _extract_between(text: str, open_tag: str, close_tag: str) -> str:
    pattern = rf"{re.escape(open_tag)}(.*?){re.escape(close_tag)}"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return text.strip()
    return match.group(1).strip()


def _resolve_region_details(
    *,
    instruction_prompt: str,
    tokenizer,
    gradient_results: Dict[str, Any],
    num_edit_regions: int,
) -> Dict[str, Any]:
    top_regions = gradient_results.get("top_regions", [])
    if not top_regions:
        raise ValueError("No top_regions were returned by the gradient analysis.")
    if num_edit_regions <= 0:
        raise ValueError("--num-edit-regions must be positive.")

    encoded = tokenizer(
        instruction_prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoded["offset_mapping"]
    if not offsets:
        raise ValueError("Instruction prompt produced no tokenizer offsets.")

    selected_regions: List[Dict[str, Any]] = []
    for region_rank, region in enumerate(top_regions[:num_edit_regions], start=1):
        start_token = min(region["start"], len(offsets) - 1)
        end_token = min(region["end"], len(offsets) - 1)
        start_char = int(offsets[start_token][0])
        end_char = int(offsets[end_token][1])
        selected_regions.append(
            {
                "region_rank": region_rank,
                "start_token": start_token,
                "end_token": end_token,
                "start_char": start_char,
                "end_char": end_char,
                "region_text": instruction_prompt[start_char:end_char],
                "region_score": region["score"],
                "region_tokens": region["tokens"],
                "region_token_indices": region["token_indices"],
            }
        )

    marked_prompt = instruction_prompt
    for region in sorted(selected_regions, key=lambda item: item["start_char"], reverse=True):
        marked_prompt = (
            marked_prompt[: region["start_char"]]
            + f"<edit_start_{region['region_rank']}>"
            + marked_prompt[region["start_char"] : region["end_char"]]
            + f"<edit_end_{region['region_rank']}>"
            + marked_prompt[region["end_char"] :]
        )

    return {
        "num_edit_regions": len(selected_regions),
        "selected_regions": selected_regions,
        "marked_prompt": marked_prompt,
    }


def _build_gradient_region_meta_prompt(
    *,
    instruction_prompt: str,
    region_details: Dict[str, Any],
) -> str:
    prompt = GRADIENT_REGION_MUTATION_PROMPT_V1
    prompt = prompt.replace("#INFERENCE_PROMPT#", instruction_prompt)
    prompt = prompt.replace("#MARKED_PROMPT#", region_details["marked_prompt"])
    return prompt


def _resolve_binary_token_id(tokenizer, base_token: str) -> int:
    token_ids = tokenizer.encode(base_token, add_special_tokens=False)
    if len(token_ids) == 1:
        return token_ids[0]
    spaced_token_ids = tokenizer.encode(f" {base_token}", add_special_tokens=False)
    if len(spaced_token_ids) == 1:
        return spaced_token_ids[0]
    raise ValueError(f"Token '{base_token}' is not a single token for this tokenizer.")


def _build_binary_prompts_for_instruction(
    *,
    instruction_prompt: str,
    binary_pairs: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    prompts: List[str] = []
    target_labels: List[str] = []
    for pair in binary_pairs:
        support = pair["support"]
        query = pair["query"]
        relation = support["relation"]
        target_label = "yes" if query["relation"] == relation else "no"
        prompts.append(
            build_relation_prompt(
                instruction_prompt=instruction_prompt,
                relation=relation,
                relation_description=pair["relation_description"],
                support_sentence=pair["support_sentence"],
                query_sentence=pair["query_sentence"],
                inference_mode=INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
            )
        )
        target_labels.append(target_label)
    return prompts, target_labels


def _score_binary_prompts(
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

    yes_token_id = _resolve_binary_token_id(tokenizer, "yes")
    no_token_id = _resolve_binary_token_id(tokenizer, "no")
    target_device = getattr(model, "device", None)
    original_padding_side = getattr(tokenizer, "padding_side", "right")

    formatted_prompts: List[str] = []
    predicted_labels: List[str] = []
    yes_probabilities: List[float] = []
    no_probabilities: List[float] = []
    target_probabilities: List[float] = []

    try:
        tokenizer.padding_side = "left"
        for start in range(0, len(prompts), batch_size):
            chunk_prompts = list(prompts[start : start + batch_size])
            chunk_targets = list(target_labels[start : start + batch_size])
            if use_chat_template:
                formatted = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for prompt in chunk_prompts
                ]
            else:
                formatted = chunk_prompts
            formatted_prompts.extend(formatted)

            model_inputs = tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            if target_device is not None:
                model_inputs = model_inputs.to(target_device)

            with torch.inference_mode():
                outputs = model(**model_inputs, use_cache=False)
                logits = outputs.logits[:, -1, [yes_token_id, no_token_id]]
                probabilities = torch.softmax(logits, dim=-1)

            predicted_indices = torch.argmax(probabilities, dim=-1).detach().cpu().tolist()
            prob_rows = probabilities.detach().cpu().tolist()
            for predicted_index, prob_row, target_label in zip(
                predicted_indices,
                prob_rows,
                chunk_targets,
            ):
                predicted_label = "yes" if predicted_index == 0 else "no"
                predicted_labels.append(predicted_label)
                yes_probabilities.append(float(prob_row[0]))
                no_probabilities.append(float(prob_row[1]))
                target_probabilities.append(
                    float(prob_row[0] if target_label == "yes" else prob_row[1])
                )
    finally:
        tokenizer.padding_side = original_padding_side

    return {
        "formatted_prompts": formatted_prompts,
        "predicted_labels": predicted_labels,
        "target_labels": list(target_labels),
        "yes_probabilities": yes_probabilities,
        "no_probabilities": no_probabilities,
        "target_probabilities": target_probabilities,
    }


def _summarize_validation_by_bucket(
    *,
    sampled_pairs: Sequence[Dict[str, Any]],
    score_payload: Dict[str, Any],
) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, Any]] = {
        "overall": {"count": 0, "correct": 0, "target_probability_sum": 0.0},
        "tp": {"count": 0, "correct": 0, "target_probability_sum": 0.0},
        "tn": {"count": 0, "correct": 0, "target_probability_sum": 0.0},
        "fp": {"count": 0, "correct": 0, "target_probability_sum": 0.0},
        "fn": {"count": 0, "correct": 0, "target_probability_sum": 0.0},
    }
    fixes_from_mistakes = 0
    regressions_from_correct = 0

    for pair, predicted_label, target_label, target_probability in zip(
        sampled_pairs,
        score_payload["predicted_labels"],
        score_payload["target_labels"],
        score_payload["target_probabilities"],
    ):
        confusion_bucket = pair["confusion_bucket"]
        is_correct = predicted_label == target_label
        for bucket_name in ("overall", confusion_bucket):
            buckets[bucket_name]["count"] += 1
            buckets[bucket_name]["correct"] += int(is_correct)
            buckets[bucket_name]["target_probability_sum"] += float(target_probability)

        if confusion_bucket in {"fp", "fn"} and is_correct:
            fixes_from_mistakes += 1
        if confusion_bucket in {"tp", "tn"} and not is_correct:
            regressions_from_correct += 1

    summary: Dict[str, Any] = {}
    for bucket_name, stats in buckets.items():
        count = stats["count"]
        summary[bucket_name] = {
            "count": count,
            "accuracy": (stats["correct"] / count) if count else 0.0,
            "avg_target_probability": (
                stats["target_probability_sum"] / count if count else 0.0
            ),
        }
    summary["fixes_from_mistakes"] = fixes_from_mistakes
    summary["regressions_from_correct"] = regressions_from_correct
    return summary


def _compute_summary_delta(
    *,
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    for bucket_name in ("overall", "tp", "tn", "fp", "fn"):
        delta[bucket_name] = {
            "accuracy": (
                candidate_summary[bucket_name]["accuracy"]
                - baseline_summary[bucket_name]["accuracy"]
            ),
            "avg_target_probability": (
                candidate_summary[bucket_name]["avg_target_probability"]
                - baseline_summary[bucket_name]["avg_target_probability"]
            ),
        }
    delta["fixes_from_mistakes"] = (
        candidate_summary["fixes_from_mistakes"] - baseline_summary["fixes_from_mistakes"]
    )
    delta["regressions_from_correct"] = (
        candidate_summary["regressions_from_correct"]
        - baseline_summary["regressions_from_correct"]
    )
    return delta


def _build_bucket_comparison(
    *,
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}
    for bucket_name in ("tp", "tn", "fp", "fn"):
        baseline_count = int(baseline_summary[bucket_name]["count"])
        baseline_accuracy = float(baseline_summary[bucket_name]["accuracy"])
        candidate_count = int(candidate_summary[bucket_name]["count"])
        candidate_accuracy = float(candidate_summary[bucket_name]["accuracy"])
        comparison[bucket_name] = {
            "original": {
                "count": baseline_count,
                "accuracy": baseline_accuracy,
                "num_correct": int(round(baseline_count * baseline_accuracy)),
            },
            "updated": {
                "count": candidate_count,
                "accuracy": candidate_accuracy,
                "num_correct": int(round(candidate_count * candidate_accuracy)),
            },
        }
    return comparison


def _build_count_only_bucket_summary(summary: Dict[str, Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for bucket_name in ("tp", "tn", "fp", "fn"):
        count = int(summary[bucket_name]["count"])
        accuracy = float(summary[bucket_name]["accuracy"])
        result[bucket_name] = int(round(count * accuracy))
    return result


def _build_summary_payload(
    *,
    instruction_prompt: str,
    prompt_editing_payload: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_validation = prompt_editing_payload.get("baseline_validation", {})
    generated_variants = prompt_editing_payload.get("generated_prompt_variants", [])
    summary_variants: List[Dict[str, Any]] = []
    for variant in generated_variants:
        validation = variant.get("validation")
        summary_variants.append(
            {
                "generation_index": variant.get("generation_index"),
                "revised_prompt": variant.get("revised_prompt"),
                "scores": (
                    _build_count_only_bucket_summary(validation["summary"])
                    if validation
                    else None
                ),
            }
        )

    return {
        "original_prompt": instruction_prompt,
        "original_scores": (
            _build_count_only_bucket_summary(baseline_validation)
            if baseline_validation
            else {}
        ),
        "meta_prompt": prompt_editing_payload.get("meta_prompt"),
        "generated_prompts": summary_variants,
    }


def _generate_region_prompt_variants(
    *,
    meta_prompt: str,
    instruction_prompt: str,
    num_generated_prompts: int,
    model,
    tokenizer,
    max_new_tokens: int,
    batch_size: int,
    use_chat_template: bool,
) -> List[Dict[str, Any]]:
    if num_generated_prompts <= 0:
        return []

    raw_outputs = run_prompts(
        [meta_prompt] * num_generated_prompts,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        do_sample=True,
        do_log=True,
        log_label="agent_gradient_eval_debug_meta_prompt",
    )
    variants: List[Dict[str, Any]] = []
    for generation_index, raw_output in enumerate(raw_outputs):
        revised_prompt = _extract_between(raw_output, "<p>", "</p>")
        variants.append(
            {
                "generation_index": generation_index,
                "raw_output": raw_output,
                "revised_prompt": revised_prompt,
                "changed": bool(revised_prompt) and revised_prompt != instruction_prompt,
            }
        )
    return variants


def _binary_pair_debug_record(
    *,
    pair_index: int,
    episode_index: int,
    way_index: int,
    episode: Dict[str, Any],
    shots: Dict[str, Dict[str, Any]],
    queries: Dict[str, Dict[str, Any]],
    label: str,
    prediction: str,
    sampled_bucket: str,
    query_index: int,
) -> Dict[str, Any]:
    support_id = episode["meta_train"][way_index][0]
    query_id = episode["meta_test"][query_index]
    support = shots[support_id]
    query = queries[query_id]
    return {
        "pair_index": pair_index,
        "episode_index": episode_index,
        "way_index": way_index,
        "sampled_bucket": sampled_bucket,
        "confusion_bucket": _pair_confusion_bucket(label, prediction),
        "is_correct": label == prediction,
        "label": label,
        "prediction": prediction,
        "support_id": support_id,
        "query_id": query_id,
        "support_relation": support.get("relation"),
        "query_relation": query.get("relation"),
        "support_sentence": get_sentence_with_tags(support).strip(),
        "query_sentence": get_sentence_with_tags(query).strip(),
    }


def build_sampled_batch(
    *,
    eval_payload: Dict[str, Any],
    dataset: Dict[str, Any],
    dataset_type: str,
    query_index: int,
    sampled_indices: Dict[str, List[int]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    episodes = dataset["episodes"]
    shots = dataset["shots"]
    queries = dataset["queries"]
    labels = eval_payload["pair_labels"]
    predictions = eval_payload["pair_predictions"]
    num_ways = _resolve_num_ways(episodes)

    expected_pairs = len(episodes) * num_ways
    if len(labels) != expected_pairs or len(predictions) != expected_pairs:
        raise ValueError(
            "Flattened binary outputs do not match the dataset shape: "
            f"expected {expected_pairs} pair labels/predictions from "
            f"{len(episodes)} episodes x {num_ways} ways, got "
            f"{len(labels)} labels and {len(predictions)} predictions."
        )

    ordered_indices = sampled_indices["mistake_indices"] + sampled_indices["correct_indices"]
    rng = random.Random(seed)
    rng.shuffle(ordered_indices)
    sampled_pairs = []
    for pair_index in ordered_indices:
        episode_index, way_index = _pair_index_to_episode_way(pair_index, num_ways)
        episode = episodes[episode_index]
        support_id = episode["meta_train"][way_index][0]
        query_id = episode["meta_test"][query_index]
        label = labels[pair_index]
        prediction = predictions[pair_index]
        sampled_pairs.append(
            {
                "pair_index": pair_index,
                "episode_index": episode_index,
                "way_index": way_index,
                "support": shots[support_id],
                "query": queries[query_id],
                "label": label,
                "prediction": prediction,
                "confusion_bucket": _pair_confusion_bucket(label, prediction),
                "support_sentence": get_sentence_with_tags(shots[support_id]).strip(),
                "query_sentence": get_sentence_with_tags(queries[query_id]).strip(),
                "relation_description": get_relation_description(
                    shots[support_id]["relation"],
                    dt=dataset_type,
                ),
            }
        )

    debug_records: List[Dict[str, Any]] = []
    for bucket_name, indices in (
        ("mistake", sampled_indices["mistake_indices"]),
        ("correct", sampled_indices["correct_indices"]),
    ):
        for pair_index in indices:
            episode_index, way_index = _pair_index_to_episode_way(pair_index, num_ways)
            debug_records.append(
                _binary_pair_debug_record(
                    pair_index=pair_index,
                    episode_index=episode_index,
                    episode=episodes[episode_index],
                    way_index=way_index,
                    shots=shots,
                    queries=queries,
                    label=labels[pair_index],
                    prediction=predictions[pair_index],
                    sampled_bucket=bucket_name,
                    query_index=query_index,
                )
            )
    return sampled_pairs, debug_records


def main() -> None:
    args = _parse_args()
    print("[agent_gradient_eval_debug] starting")

    instruction_prompt, prompt_info = resolve_instruction_prompt(
        args.prompt_source_path,
        prompt_index=args.prompt_index,
        prompt_node_id=args.prompt_node_id,
    )
    print(
        "[agent_gradient_eval_debug] prompt resolved:",
        prompt_info.get("prompt_source_type"),
    )
    print(
        "[agent_gradient_eval_debug] initial instruction prompt:\n"
        f"{instruction_prompt}"
    )

    eval_payload = _load_json(args.eval_output_path)
    saved_split = args.split or eval_payload.get("split") or "validation"
    split = _resolve_dataset_split(saved_split)
    print(
        "[agent_gradient_eval_debug] eval output loaded:",
        args.eval_output_path,
        f"(saved_split={saved_split}, dataset_split={split})",
    )
    sampled_indices = sample_eval_pair_indices(
        eval_payload,
        num_correct=args.num_correct,
        num_mistakes=args.num_mistakes,
        mistake_coverage=args.mistake_coverage,
        seed=args.seed,
    )
    print(
        "[agent_gradient_eval_debug] sampled binary pairs:",
        f"mistakes={len(sampled_indices['mistake_indices'])}",
        f"correct={len(sampled_indices['correct_indices'])}",
    )

    dataset = load_split_episodes(
        split=split,
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
    )
    print(
        "[agent_gradient_eval_debug] dataset loaded:",
        f"episodes={len(dataset['episodes'])}",
        f"dataset_type={args.dataset_type}",
    )
    pair_labels = eval_payload.get("pair_labels", [])
    num_ways = _resolve_num_ways(dataset["episodes"])
    if len(pair_labels) != len(dataset["episodes"]) * num_ways:
        raise ValueError(
            "Loaded split shape does not match the eval output pair count: "
            f"{len(dataset['episodes'])} episodes x {num_ways} ways vs {len(pair_labels)} pairs."
        )

    sampled_pairs, sampled_examples = build_sampled_batch(
        eval_payload=eval_payload,
        dataset=dataset,
        dataset_type=args.dataset_type,
        query_index=args.query_index,
        sampled_indices=sampled_indices,
        seed=args.seed,
    )
    print(
        "[agent_gradient_eval_debug] batch built:",
        f"pairs={len(sampled_pairs)}",
    )

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model,
        device_map=args.device_map,
    )
    print("[agent_gradient_eval_debug] model and tokenizer loaded")

    meta_prompt_model = model
    meta_prompt_tokenizer = tokenizer
    meta_prompt_model_id = args.meta_prompt_model or args.model
    if args.num_generated_prompts > 0 and args.meta_prompt_model:
        meta_prompt_model, meta_prompt_tokenizer = load_model_and_tokenizer(
            model_id=args.meta_prompt_model,
            device_map=args.device_map,
        )
        print(
            "[agent_gradient_eval_debug] meta-prompt model and tokenizer loaded:",
            args.meta_prompt_model,
        )

    print("[agent_gradient_eval_debug] running gradient analysis")
    gradient_results = analyze_relation_extraction_binary_pairs(
        instruction_prompt=instruction_prompt,
        binary_pairs=sampled_pairs,
        model=model,
        tokenizer=tokenizer,
        dataset_type=args.dataset_type,
        gradient_batch_size=args.gradient_batch_size,
        num_candidates=args.num_candidates,
        candidate_mode=args.candidate_mode,
        max_regions=args.max_regions,
        max_total_region_tokens=args.max_total_region_tokens,
        region_expansion_threshold_ratio=args.region_expansion_threshold_ratio,
        embedding_step_size=args.embedding_step_size,
        use_chat_template=not args.disable_chat_template,
    )
    print(
        "[agent_gradient_eval_debug] gradient analysis done:",
        f"instances={gradient_results['num_instances']}",
        f"candidate_mode={args.candidate_mode}",
        f"token_gradients={len(gradient_results['token_gradients'])}",
        f"top_regions={len(gradient_results['top_regions'])}",
    )

    baseline_score_payload = {
        "formatted_prompts": gradient_results["formatted_prompts"],
        "predicted_labels": gradient_results["predicted_labels"],
        "target_labels": gradient_results["target_labels"],
        "target_probabilities": gradient_results["target_probabilities"],
    }
    baseline_validation = _summarize_validation_by_bucket(
        sampled_pairs=sampled_pairs,
        score_payload=baseline_score_payload,
    )
    print(
        "[agent_gradient_eval_debug] baseline validation:",
        f"overall_accuracy={baseline_validation['overall']['accuracy']:.4f}",
        f"overall_avg_target_probability={baseline_validation['overall']['avg_target_probability']:.4f}",
        f"fixes_from_mistakes={baseline_validation['fixes_from_mistakes']}",
        f"regressions_from_correct={baseline_validation['regressions_from_correct']}",
    )

    prompt_editing_payload: Dict[str, Any] = {
        "baseline_validation": baseline_validation,
        "selected_regions": [],
        "meta_prompt": None,
        "generated_prompt_variants": [],
    }
    if args.num_generated_prompts > 0:
        region_details = _resolve_region_details(
            instruction_prompt=instruction_prompt,
            tokenizer=tokenizer,
            gradient_results=gradient_results,
            num_edit_regions=args.num_edit_regions,
        )
        meta_prompt = _build_gradient_region_meta_prompt(
            instruction_prompt=instruction_prompt,
            region_details=region_details,
        )
        print(
            "[agent_gradient_eval_debug] meta prompt:\n"
            f"{meta_prompt}"
        )
        print(
            "[agent_gradient_eval_debug] generating prompt variants:",
            f"num_edit_regions={region_details['num_edit_regions']}",
            f"region_ranks={[region['region_rank'] for region in region_details['selected_regions']]}",
            f"num_generated_prompts={args.num_generated_prompts}",
        )
        generated_variants = _generate_region_prompt_variants(
            meta_prompt=meta_prompt,
            instruction_prompt=instruction_prompt,
            num_generated_prompts=args.num_generated_prompts,
            model=meta_prompt_model,
            tokenizer=meta_prompt_tokenizer,
            max_new_tokens=args.meta_prompt_max_new_tokens,
            batch_size=args.meta_prompt_batch_size,
            use_chat_template=not args.disable_chat_template,
        )
        validated_variants: List[Dict[str, Any]] = []
        for variant in generated_variants:
            revised_prompt = variant["revised_prompt"]
            if not revised_prompt:
                print(
                    "[agent_gradient_eval_debug] prompt variant generated:",
                    f"generation_index={variant['generation_index']}",
                    "revised_prompt=<empty>",
                    "validation_skipped=True",
                )
                validated_variants.append(
                    {
                        **variant,
                        "validation": None,
                        "delta_vs_baseline": None,
                    }
                )
                continue

            prompts, target_labels = _build_binary_prompts_for_instruction(
                instruction_prompt=revised_prompt,
                binary_pairs=sampled_pairs,
            )
            score_payload = _score_binary_prompts(
                prompts=prompts,
                target_labels=target_labels,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.validation_batch_size,
                use_chat_template=not args.disable_chat_template,
            )
            validation_summary = _summarize_validation_by_bucket(
                sampled_pairs=sampled_pairs,
                score_payload=score_payload,
            )
            delta_vs_baseline = _compute_summary_delta(
                baseline_summary=baseline_validation,
                candidate_summary=validation_summary,
            )
            print(
                "[agent_gradient_eval_debug] prompt variant text:\n"
                f"{revised_prompt}"
            )
            print(
                "[agent_gradient_eval_debug] prompt variant validated:",
                f"generation_index={variant['generation_index']}",
                f"overall_accuracy={validation_summary['overall']['accuracy']:.4f}",
                f"overall_avg_target_probability={validation_summary['overall']['avg_target_probability']:.4f}",
                f"delta_accuracy={delta_vs_baseline['overall']['accuracy']:.4f}",
                f"delta_avg_target_probability={delta_vs_baseline['overall']['avg_target_probability']:.4f}",
            )
            validated_variants.append(
                {
                    **variant,
                    "validation": {
                        "predicted_labels": score_payload["predicted_labels"],
                        "target_labels": score_payload["target_labels"],
                        "target_probabilities": score_payload["target_probabilities"],
                        "summary": validation_summary,
                        "bucket_comparison": _build_bucket_comparison(
                            baseline_summary=baseline_validation,
                            candidate_summary=validation_summary,
                        ),
                    },
                    "delta_vs_baseline": delta_vs_baseline,
                }
            )

        prompt_editing_payload["selected_regions"] = region_details["selected_regions"]
        prompt_editing_payload["meta_prompt"] = meta_prompt
        prompt_editing_payload["generated_prompt_variants"] = validated_variants
    else:
        print("[agent_gradient_eval_debug] prompt variant generation skipped")

    payload = {
        "args": _namespace_to_json_dict(args),
        "model": args.model,
        "meta_prompt_model": meta_prompt_model_id,
        "device_map": args.device_map,
        "dataset_type": args.dataset_type,
        "candidate_mode": args.candidate_mode,
        "split": saved_split,
        "dataset_split": split,
        "eval_output_path": str(args.eval_output_path),
        "prompt_source_path": str(args.prompt_source_path),
        "prompt_info": prompt_info,
        "sampling": {
            "seed": args.seed,
            "num_correct": args.num_correct,
            "num_mistakes": args.num_mistakes,
            "mistake_coverage": args.mistake_coverage,
            "correct_pair_indices": sampled_indices["correct_indices"],
            "mistake_pair_indices": sampled_indices["mistake_indices"],
            "bucket_stats": sampled_indices.get("bucket_stats", {}),
        },
        "sampled_examples": sampled_examples,
        "gradient_analysis": gradient_results,
        "prompt_region_editing": prompt_editing_payload,
    }
    summary_payload = _build_summary_payload(
        instruction_prompt=instruction_prompt,
        prompt_editing_payload=prompt_editing_payload,
    )

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"saved output to {args.output_file}")
        summary_output_file = args.output_file.with_name(
            f"{args.output_file.stem}_summary{args.output_file.suffix}"
        )
        with summary_output_file.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, ensure_ascii=False)
        print(f"saved summary output to {summary_output_file}")


if __name__ == "__main__":
    main()



# python3 codes/agents/agent_gradient_eval_debug.py \
#   --model "Qwen/Qwen3-4B" \
#   --device-map "cuda:0" \
#   --eval-output-path "trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
#   --prompt-source-path "trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
#   --prompt-node-id 12 \
#   --output-file "gradients_experiments/gradient_debug_node12_f1-37.json"

# python3 codes/agents/agent_gradient_eval_debug.py \
#   --model "google/gemma-3-4b-it" \
#   --device-map "cuda:0" \
#   --eval-output-path "trainings/20260128_151114_Qwen-Qwen3-4B/eval_outputs/EVALID_0_labels_predictions.json" \
#   --prompt-source-path "unified_optimization_results/unified_opt_fs_tacred_20260325_215139/generated_prompt_candidates.json" \
#   --prompt-index 0 \
# --output-file "gradients_experiments/gradient_debug_node9.json"
