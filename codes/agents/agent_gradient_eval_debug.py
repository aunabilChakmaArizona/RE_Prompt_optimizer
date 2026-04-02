#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
CODES_DIR = CURRENT_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.append(str(CODES_DIR))

from agents.agent_data_loader import DEFAULT_DATA_DIR, load_split_episodes, load_train_samples
from agents.agent_binary_inference import run_binary_inference
from agents.agent_gradient_token_analysis import (
    TOKEN_CANDIDATE_MODE_CHOICES,
    TOKEN_CANDIDATE_MODE_NEAREST_UPDATED,
    analyze_relation_extraction_binary_pairs,
    build_relation_prompt,
)
from agents.agent_llm_prompting import run_prompts
from agents.agent_metrics import compute_prf_stats
from agents.agent_models import load_model_and_tokenizer
from agents.agent_prompts import (
    GRADIENT_REGION_MUTATION_PROMPT_V1,
    INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
)
from agents.agent_relation_utils import get_relation_description
from agents.agent_sample_feedback import sample_feedback_fn
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample train feedback-style TACRED binary pairs, run prompt-token "
            "gradient analysis on a balanced subset, generate prompt edits, and "
            "evaluate the edited prompts on the balanced train subset and full dev set."
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
        "--train-samples",
        default="fs_tacred_train_samples.pkl",
        help="Train samples pickle used by the trainer-style feedback sampler.",
    )
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-gradient-sample-size",
        type=int,
        default=None,
        help=(
            "New mode only. Sample D binary pairs from the train split, bucket them "
            "by fresh predictions from the original prompt, and use a balanced subset "
            "for gradient collection."
        ),
    )
    parser.add_argument(
        "--full-eval-split",
        default="dev",
        help=(
            "Split used for full evaluation of generated prompts in the new mode. "
            "Defaults to dev."
        ),
    )
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
        buckets[_pair_confusion_bucket(label, prediction)].append(idx)
    return buckets


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


def _build_prf_scores(
    *,
    predicted_labels: Sequence[str],
    target_labels: Sequence[str],
) -> Dict[str, float]:
    stats = compute_prf_stats(target_labels, predicted_labels, n_chunks=1)
    return {
        "precision": float(stats["precision_mean"] * 100.0),
        "recall": float(stats["recall_mean"] * 100.0),
        "f1": float(stats["f1_mean"] * 100.0),
    }

def _build_binary_pairs_from_feedback_samples(
    *,
    feedback_samples,
    shots_by_id: Dict[str, dict],
    dataset_type: str,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for pair_index, sample in enumerate(feedback_samples.all_samples):
        support = shots_by_id[sample.id_1shot]
        query = shots_by_id[sample.id_query]
        label = sample.label
        pairs.append(
            {
                "pair_index": pair_index,
                "episode_index": None,
                "way_index": None,
                "support": support,
                "query": query,
                "label": label,
                "prediction": None,
                "confusion_bucket": None,
                "support_sentence": get_sentence_with_tags(support).strip(),
                "query_sentence": get_sentence_with_tags(query).strip(),
                "relation_description": get_relation_description(
                    support["relation"],
                    dt=dataset_type,
                ),
            }
        )
    return pairs


def _attach_predictions_to_pairs(
    *,
    pairs: Sequence[Dict[str, Any]],
    predicted_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    if len(pairs) != len(predicted_labels):
        raise ValueError("pairs/predicted_labels length mismatch")

    updated_pairs: List[Dict[str, Any]] = []
    for pair, predicted_label in zip(pairs, predicted_labels):
        updated_pair = dict(pair)
        updated_pair["prediction"] = predicted_label
        updated_pair["confusion_bucket"] = _pair_confusion_bucket(
            updated_pair["label"],
            predicted_label,
        )
        updated_pairs.append(updated_pair)
    return updated_pairs


def _sample_balanced_pairs_from_bucketed_pool(
    *,
    bucketed_pairs: Sequence[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "tp": [],
        "tn": [],
        "fp": [],
        "fn": [],
    }
    for pair in bucketed_pairs:
        buckets[pair["confusion_bucket"]].append(pair)

    selected_per_bucket = min(len(buckets["tp"]), len(buckets["tn"]), len(buckets["fp"]), len(buckets["fn"]))
    if selected_per_bucket <= 0:
        raise ValueError(
            "Unable to sample balanced tp/tn/fp/fn buckets from the train D set. "
            f"available_tp={len(buckets['tp'])}, available_tn={len(buckets['tn'])}, "
            f"available_fp={len(buckets['fp'])}, available_fn={len(buckets['fn'])}."
        )

    rng = random.Random(seed)
    selected_pairs: List[Dict[str, Any]] = []
    selected_counts: Dict[str, int] = {}
    for bucket_name in ("tp", "tn", "fp", "fn"):
        chosen = rng.sample(buckets[bucket_name], selected_per_bucket)
        selected_counts[bucket_name] = len(chosen)
        selected_pairs.extend(chosen)

    rng.shuffle(selected_pairs)
    bucket_stats = {
        "available": {bucket_name: len(bucket) for bucket_name, bucket in buckets.items()},
        "selected": selected_counts,
    }
    return selected_pairs, bucket_stats


def _build_evaluation_report(
    *,
    score_payload: Dict[str, Any],
) -> Dict[str, Any]:
    confusion_matrix = _build_confusion_matrix_counts(
        predicted_labels=score_payload["predicted_labels"],
        target_labels=score_payload["target_labels"],
    )
    return {
        "predicted_labels": score_payload["predicted_labels"],
        "target_labels": score_payload["target_labels"],
        "confusion_matrix": confusion_matrix,
        "prf": _build_prf_scores(
            predicted_labels=score_payload["predicted_labels"],
            target_labels=score_payload["target_labels"],
        ),
    }


def _build_score_payload_from_pairs(
    pairs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "predicted_labels": [pair["prediction"] for pair in pairs],
        "target_labels": [pair["label"] for pair in pairs],
    }


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


def _build_binary_prompts_for_instruction(
    *,
    instruction_prompt: str,
    binary_pairs: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    prompts: List[str] = []
    target_labels: List[str] = []
    for pair in binary_pairs:
        support = pair["support"]
        relation = support["relation"]
        target_label = pair["label"]
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


def _predict_binary_prompts(
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
    predicted_labels = run_binary_inference(
        prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        log_label="gradient_eval_debug",
    )

    return {
        "predicted_labels": predicted_labels,
        "target_labels": list(target_labels),
    }


def _summarize_validation_by_bucket(
    *,
    sampled_pairs: Sequence[Dict[str, Any]],
    score_payload: Dict[str, Any],
) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, Any]] = {
        "overall": {"count": 0, "correct": 0},
        "tp": {"count": 0, "correct": 0},
        "tn": {"count": 0, "correct": 0},
        "fp": {"count": 0, "correct": 0},
        "fn": {"count": 0, "correct": 0},
    }
    fixes_from_mistakes = 0
    regressions_from_correct = 0

    for pair, predicted_label, target_label in zip(
        sampled_pairs,
        score_payload["predicted_labels"],
        score_payload["target_labels"],
    ):
        confusion_bucket = pair["confusion_bucket"]
        is_correct = predicted_label == target_label
        for bucket_name in ("overall", confusion_bucket):
            buckets[bucket_name]["count"] += 1
            buckets[bucket_name]["correct"] += int(is_correct)

        if confusion_bucket in {"fp", "fn"} and is_correct:
            fixes_from_mistakes += 1
        if confusion_bucket in {"tp", "tn"} and not is_correct:
            regressions_from_correct += 1

    summary: Dict[str, Any] = {}
    for bucket_name, stats in buckets.items():
        count = stats["count"]
        summary[bucket_name] = {
            "total": count,
            "count": stats["correct"],
            "accuracy": (stats["correct"] / count) if count else 0.0,
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
        }
    delta["fixes_from_mistakes"] = (
        candidate_summary["fixes_from_mistakes"] - baseline_summary["fixes_from_mistakes"]
    )
    delta["regressions_from_correct"] = (
        candidate_summary["regressions_from_correct"]
        - baseline_summary["regressions_from_correct"]
    )
    return delta


def _build_confusion_matrix_counts(
    *,
    predicted_labels: Sequence[str],
    target_labels: Sequence[str],
) -> Dict[str, int]:
    if len(predicted_labels) != len(target_labels):
        raise ValueError("predicted_labels/target_labels length mismatch")

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for prediction, target in zip(predicted_labels, target_labels):
        if target == "yes" and prediction == "yes":
            counts["tp"] += 1
        elif target == "no" and prediction == "no":
            counts["tn"] += 1
        elif target == "no" and prediction == "yes":
            counts["fp"] += 1
        elif target == "yes" and prediction == "no":
            counts["fn"] += 1
        else:
            raise ValueError(
                f"Unsupported target/prediction pair: target={target}, prediction={prediction}"
            )
    return counts


def _build_summary_payload(
    *,
    instruction_prompt: str,
    prompt_editing_payload: Dict[str, Any],
    baseline_confusion_matrix: Dict[str, int],
) -> Dict[str, Any]:
    generated_variants = prompt_editing_payload.get("generated_prompt_variants", [])
    summary_variants: List[Dict[str, Any]] = []
    for variant in generated_variants:
        validation = variant.get("validation")
        summary_variants.append(
            {
                "generation_index": variant.get("generation_index"),
                "revised_prompt": variant.get("revised_prompt"),
                "updated_scores": (
                    _build_confusion_matrix_counts(
                        predicted_labels=validation["predicted_labels"],
                        target_labels=validation["target_labels"],
                    )
                    if validation
                    else None
                ),
            }
        )

    return {
        "original_prompt": instruction_prompt,
        "original_scores": baseline_confusion_matrix,
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


def build_full_binary_pairs(
    *,
    dataset: Dict[str, Any],
    dataset_type: str,
    query_index: int,
) -> List[Dict[str, Any]]:
    episodes = dataset["episodes"]
    shots = dataset["shots"]
    queries = dataset["queries"]
    num_ways = _resolve_num_ways(episodes)

    full_pairs: List[Dict[str, Any]] = []
    for pair_index in range(len(episodes) * num_ways):
        episode_index, way_index = _pair_index_to_episode_way(pair_index, num_ways)
        episode = episodes[episode_index]
        support_id = episode["meta_train"][way_index][0]
        query_id = episode["meta_test"][query_index]
        support = shots[support_id]
        query = queries[query_id]
        label = "yes" if support["relation"] == query["relation"] else "no"
        prediction = None
        full_pairs.append(
            {
                "pair_index": pair_index,
                "episode_index": episode_index,
                "way_index": way_index,
                "support": support,
                "query": query,
                "label": label,
                "prediction": prediction,
                "confusion_bucket": None,
                "support_sentence": get_sentence_with_tags(support).strip(),
                "query_sentence": get_sentence_with_tags(query).strip(),
                "relation_description": get_relation_description(
                    support["relation"],
                    dt=dataset_type,
                ),
            }
        )
    return full_pairs


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

    sampled_indices: Dict[str, Any] = {}
    sampled_pairs: List[Dict[str, Any]] = []
    sampled_examples: List[Dict[str, Any]] = []
    full_eval_pairs: List[Dict[str, Any]] = []
    train_d_pairs: List[Dict[str, Any]] = []
    if args.train_gradient_sample_size is None or args.train_gradient_sample_size <= 0:
        raise ValueError("--train-gradient-sample-size must be set to a positive integer.")

    train_samples = load_train_samples(
        data_dir=args.data_dir,
        filename=args.train_samples,
    )
    print(
        "[agent_gradient_eval_debug] train samples loaded:",
        f"relations={len(train_samples)}",
        f"dataset_type={args.dataset_type}",
    )
    rng = random.Random(args.seed)
    train_feedback_samples = sample_feedback_fn(
        train_samples,
        k=args.train_gradient_sample_size,
        rng=rng,
    )
    train_d_pairs = _build_binary_pairs_from_feedback_samples(
        feedback_samples=train_feedback_samples,
        shots_by_id=train_samples,
        dataset_type=args.dataset_type,
    )
    print(
        "[agent_gradient_eval_debug] train D sample built:",
        f"D={len(train_d_pairs)}",
    )

    train_d_prompts, train_d_target_labels = _build_binary_prompts_for_instruction(
        instruction_prompt=instruction_prompt,
        binary_pairs=train_d_pairs,
    )
    train_d_predicted_labels = run_binary_inference(
        train_d_prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.validation_batch_size,
        use_chat_template=not args.disable_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        log_label="train_gradient_collection",
    )

    train_d_pairs = _attach_predictions_to_pairs(
        pairs=train_d_pairs,
        predicted_labels=train_d_predicted_labels,
    )
    train_d_report = {
        "confusion_matrix": _build_confusion_matrix_counts(
            predicted_labels=train_d_predicted_labels,
            target_labels=train_d_target_labels,
        ),
        "prf": _build_prf_scores(
            predicted_labels=train_d_predicted_labels,
            target_labels=train_d_target_labels,
        ),
    }
    sampled_pairs, bucket_stats = _sample_balanced_pairs_from_bucketed_pool(
        bucketed_pairs=train_d_pairs,
        seed=args.seed,
    )
    sampled_examples = [dict(pair) for pair in sampled_pairs]
    sampled_indices = {
        "correct_indices": [
            pair["pair_index"] for pair in sampled_pairs if pair["confusion_bucket"] in {"tp", "tn"}
        ],
        "mistake_indices": [
            pair["pair_index"] for pair in sampled_pairs if pair["confusion_bucket"] in {"fp", "fn"}
        ],
        "bucket_stats": bucket_stats,
    }
    train_gradient_collection = {
        "gradient_split": "train",
        "evaluation_split": args.full_eval_split,
        "train_gradient_sample_size": args.train_gradient_sample_size,
        "train_d_pair_indices": [pair["pair_index"] for pair in train_d_pairs],
        "train_d_bucket_stats": {
            "available": bucket_stats["available"],
            "selected_for_gradients": bucket_stats["selected"],
        },
        "train_d_evaluation_original_prompt": train_d_report,
    }
    print(
        "[agent_gradient_eval_debug] train D pair pool:",
        f"available_tp={bucket_stats['available']['tp']}",
        f"available_tn={bucket_stats['available']['tn']}",
        f"available_fp={bucket_stats['available']['fp']}",
        f"available_fn={bucket_stats['available']['fn']}",
        f"selected_tp={bucket_stats['selected']['tp']}",
        f"selected_tn={bucket_stats['selected']['tn']}",
        f"selected_fp={bucket_stats['selected']['fp']}",
        f"selected_fn={bucket_stats['selected']['fn']}",
    )
    print(
        "[agent_gradient_eval_debug] train gradient batch built:",
        f"pairs={len(sampled_pairs)}",
    )

    full_eval_dataset = load_split_episodes(
        split=args.full_eval_split,
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
    )
    print(
        "[agent_gradient_eval_debug] full evaluation dataset loaded:",
        f"split={args.full_eval_split}",
        f"episodes={len(full_eval_dataset['episodes'])}",
    )
    full_eval_pairs = build_full_binary_pairs(
        dataset=full_eval_dataset,
        dataset_type=args.dataset_type,
        query_index=args.query_index,
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

    baseline_score_payload = _build_score_payload_from_pairs(sampled_pairs)
    baseline_confusion_matrix = _build_confusion_matrix_counts(
        predicted_labels=baseline_score_payload["predicted_labels"],
        target_labels=baseline_score_payload["target_labels"],
    )
    baseline_validation = _summarize_validation_by_bucket(
        sampled_pairs=sampled_pairs,
        score_payload=baseline_score_payload,
    )
    print(
        "[agent_gradient_eval_debug] baseline validation:",
        f"overall_accuracy={baseline_validation['overall']['accuracy']:.4f}",
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
                        "full_evaluation": None,
                        "delta_vs_baseline": None,
                    }
                )
                continue

            prompts, target_labels = _build_binary_prompts_for_instruction(
                instruction_prompt=revised_prompt,
                binary_pairs=sampled_pairs,
            )
            score_payload = _predict_binary_prompts(
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
            updated_confusion_matrix = _build_confusion_matrix_counts(
                predicted_labels=score_payload["predicted_labels"],
                target_labels=score_payload["target_labels"],
            )
            delta_vs_baseline = _compute_summary_delta(
                baseline_summary=baseline_validation,
                candidate_summary=validation_summary,
            )
            full_variant_prompts, full_variant_targets = _build_binary_prompts_for_instruction(
                instruction_prompt=revised_prompt,
                binary_pairs=full_eval_pairs,
            )
            full_variant_score_payload = _predict_binary_prompts(
                prompts=full_variant_prompts,
                target_labels=full_variant_targets,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.validation_batch_size,
                use_chat_template=not args.disable_chat_template,
            )
            full_evaluation = _build_evaluation_report(
                score_payload=full_variant_score_payload
            )
            print(
                "[agent_gradient_eval_debug] prompt variant text:\n"
                f"{revised_prompt}"
            )
            print(
                "[agent_gradient_eval_debug] prompt variant validated:",
                f"generation_index={variant['generation_index']}",
                f"overall_accuracy={validation_summary['overall']['accuracy']:.4f}",
                f"delta_accuracy={delta_vs_baseline['overall']['accuracy']:.4f}",
                f"balanced_train_f1={_build_prf_scores(predicted_labels=score_payload['predicted_labels'], target_labels=score_payload['target_labels'])['f1']:.4f}",
                f"full_eval_f1={full_evaluation['prf']['f1']:.4f}",
            )
            validated_variants.append(
                {
                    **variant,
                    "validation": {
                        "predicted_labels": score_payload["predicted_labels"],
                        "target_labels": score_payload["target_labels"],
                        "summary": validation_summary,
                        "prf": _build_prf_scores(
                            predicted_labels=score_payload["predicted_labels"],
                            target_labels=score_payload["target_labels"],
                        ),
                        "confusion_matrix": {
                            "original": baseline_confusion_matrix,
                            "updated": updated_confusion_matrix,
                        },
                    },
                    "full_evaluation": full_evaluation,
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
        "mode": "train_gradient_sampling",
        "dataset_split": "train",
        "prompt_source_path": str(args.prompt_source_path),
        "prompt_info": prompt_info,
        "sampling": {
            "seed": args.seed,
            "correct_pair_indices": sampled_indices["correct_indices"],
            "mistake_pair_indices": sampled_indices["mistake_indices"],
            "bucket_stats": sampled_indices.get("bucket_stats", {}),
        },
        "sampled_examples": sampled_examples,
        "gradient_analysis": gradient_results,
        "train_gradient_collection": train_gradient_collection,
        "prompt_region_editing": prompt_editing_payload,
    }
    summary_payload = _build_summary_payload(
        instruction_prompt=instruction_prompt,
        prompt_editing_payload=prompt_editing_payload,
        baseline_confusion_matrix=baseline_confusion_matrix,
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
