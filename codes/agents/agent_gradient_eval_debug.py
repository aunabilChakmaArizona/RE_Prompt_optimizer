#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
CODES_DIR = CURRENT_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.append(str(CODES_DIR))

from agents.agent_data_loader import DEFAULT_DATA_DIR, load_split_episodes
from agents.agent_gradient_token_analysis import (
    TOKEN_CANDIDATE_MODE_CHOICES,
    TOKEN_CANDIDATE_MODE_NEAREST_UPDATED,
    analyze_relation_extraction_binary_pairs,
)
from agents.agent_models import load_model_and_tokenizer
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument(
        "--candidate-mode",
        choices=TOKEN_CANDIDATE_MODE_CHOICES,
        default=TOKEN_CANDIDATE_MODE_NEAREST_UPDATED,
    )
    parser.add_argument("--max-regions", type=int, default=3)
    parser.add_argument("--max-total-region-tokens", type=int, default=10)
    parser.add_argument("--embedding-step-size", type=float, default=1.0)
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
    if len(indices) < k:
        raise ValueError(f"Need {k} {label} samples, but found only {len(indices)}.")
    return sorted(rng.sample(list(indices), k))


def sample_eval_pair_indices(
    eval_payload: Dict[str, Any],
    *,
    num_correct: int,
    num_mistakes: int,
    seed: int,
) -> Dict[str, List[int]]:
    labels = eval_payload.get("pair_labels", [])
    predictions = eval_payload.get("pair_predictions", [])
    if len(labels) != len(predictions):
        raise ValueError("eval output pair_labels/pair_predictions length mismatch")

    correct_indices = [idx for idx, (label, pred) in enumerate(zip(labels, predictions)) if label == pred]
    mistake_indices = [idx for idx, (label, pred) in enumerate(zip(labels, predictions)) if label != pred]

    rng = random.Random(seed)
    return {
        "correct_indices": _sample_indices(correct_indices, k=num_correct, rng=rng, label="correct"),
        "mistake_indices": _sample_indices(mistake_indices, k=num_mistakes, rng=rng, label="mistake"),
    }


def _resolve_num_ways(episodes: Sequence[Dict[str, Any]]) -> int:
    if not episodes:
        raise ValueError("Dataset episodes are empty.")
    num_ways = len(episodes[0]["meta_train"])
    if num_ways <= 0:
        raise ValueError("Episodes contain no candidate ways.")
    return num_ways


def _pair_index_to_episode_way(pair_index: int, num_ways: int) -> Tuple[int, int]:
    return divmod(pair_index, num_ways)


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
    query_index: int,
    sampled_indices: Dict[str, List[int]],
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
    sampled_pairs = []
    for pair_index in ordered_indices:
        episode_index, way_index = _pair_index_to_episode_way(pair_index, num_ways)
        episode = episodes[episode_index]
        support_id = episode["meta_train"][way_index][0]
        query_id = episode["meta_test"][query_index]
        sampled_pairs.append(
            {
                "pair_index": pair_index,
                "episode_index": episode_index,
                "way_index": way_index,
                "support": shots[support_id],
                "query": queries[query_id],
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
        query_index=args.query_index,
        sampled_indices=sampled_indices,
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
    print("[agent_gradient_eval_debug] running gradient analysis")
    gradient_results = analyze_relation_extraction_binary_pairs(
        instruction_prompt=instruction_prompt,
        binary_pairs=sampled_pairs,
        model=model,
        tokenizer=tokenizer,
        dataset_type=args.dataset_type,
        num_candidates=args.num_candidates,
        candidate_mode=args.candidate_mode,
        max_regions=args.max_regions,
        max_total_region_tokens=args.max_total_region_tokens,
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

    payload = {
        "args": _namespace_to_json_dict(args),
        "model": args.model,
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
            "correct_pair_indices": sampled_indices["correct_indices"],
            "mistake_pair_indices": sampled_indices["mistake_indices"],
        },
        "sampled_examples": sampled_examples,
        "gradient_analysis": gradient_results,
    }

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"saved output to {args.output_file}")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


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
