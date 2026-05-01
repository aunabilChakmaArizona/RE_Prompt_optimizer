#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

DEFAULT_DATA_DIR = str((CURRENT_DIR.parent / "data").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate top prompts from an evolutionary-search run or a gradient "
            "experiment run on the final-step FS-TACRED dev episodes."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help=(
            "Output directory from codes/core_trainer_evolutionary_search.py or "
            "codes/agents/agent_gradient_eval_debug.py."
        ),
    )
    parser.add_argument(
        "top_k",
        type=int,
        help="Number of top validation/dev-F1 prompts to evaluate.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing FS-TACRED pickle files.",
    )
    parser.add_argument(
        "--final-dev-split",
        default="final_step_dev_6000",
        help=(
            "Split name resolved as "
            "'{dataset_type}_{split}_episodes_1shots.pkl' and "
            "'{dataset_type}_{split}_episodes_shots_details.pkl'."
        ),
    )
    parser.add_argument("--dataset-type", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--query-index", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-n-chunks", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--output-root-dir",
        type=Path,
        default=Path("trainings_final_dev"),
        help="Root directory for validation result subfolders.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_population_path(run_dir: Path) -> Path:
    population_path = run_dir / "population.json"
    if not population_path.is_file():
        raise FileNotFoundError(f"Could not find population.json at {population_path}")
    return population_path


def load_run_args(run_dir: Path) -> Dict[str, Any]:
    args_path = run_dir / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"Could not find args.json at {args_path}")
    return read_json(args_path)


def detect_source_type(run_dir: Path) -> str:
    if (run_dir / "population.json").is_file():
        return "evolutionary_population"
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if isinstance(summary.get("iterations"), list):
            return "gradient_experiment"
    raise FileNotFoundError(
        "Could not detect supported run type. Expected population.json for "
        "evolutionary search, or summary.json with iterations for gradient runs."
    )


def f1_score(node_payload: Dict[str, Any]) -> float | None:
    val_score = node_payload.get("val_score")
    if not isinstance(val_score, dict):
        return None
    value = val_score.get("f1_mean")
    if value is None:
        return None
    return float(value)


def _gradient_prf_to_val_score(prf: Dict[str, Any] | None) -> Dict[str, float] | None:
    if not isinstance(prf, dict) or prf.get("f1") is None:
        return None
    return {
        "precision_mean": float(prf.get("precision", 0.0)) / 100.0,
        "precision_std": 0.0,
        "recall_mean": float(prf.get("recall", 0.0)) / 100.0,
        "recall_std": 0.0,
        "f1_mean": float(prf.get("f1", 0.0)) / 100.0,
        "f1_std": 0.0,
    }


def gradient_prompt_to_node_payload(prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
    val_score = _gradient_prf_to_val_score(prompt_payload.get("dev_prf"))
    if val_score is None:
        raise ValueError("Gradient prompt is missing dev_prf.f1")
    iteration_index = prompt_payload.get("iteration_index")
    return {
        "node_id": f"gradient_iter_{iteration_index}",
        "parent_id": None,
        "children_ids": [],
        "is_in_final_population": False,
        "inference_prompt": prompt_payload.get("prompt", ""),
        "inference_mode": "separate_no_examples",
        "inference_instruction_prompt": prompt_payload.get("prompt", ""),
        "inference_answer_instruction_prompt": "",
        "inference_example_prompt": "",
        "inference_input_prompt": "",
        "val_score": val_score,
        "test_score": None,
        "gradient_iteration_index": iteration_index,
        "gradient_generation_index": prompt_payload.get("generation_index"),
        "gradient_beam_index": prompt_payload.get("beam_index"),
        "gradient_selection_strategy": prompt_payload.get("selection_strategy"),
        "gradient_source_prompt": prompt_payload,
    }


def select_gradient_nodes(summary_payload: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    candidates: List[Dict[str, Any]] = []
    for iteration in summary_payload.get("iterations", []):
        if not isinstance(iteration, dict):
            continue
        selected_prompt = iteration.get("selected_prompt")
        if not isinstance(selected_prompt, dict):
            continue
        node_payload = gradient_prompt_to_node_payload(selected_prompt)
        candidates.append({
            "node": node_payload,
            "selection_reason": "gradient_iteration_selected_prompt",
        })

    if not candidates:
        raise ValueError("No iteration selected prompts found in gradient summary.json")

    candidates.sort(key=lambda item: f1_score(item["node"]), reverse=True)
    return candidates[: min(top_k, len(candidates))]


def select_validation_nodes(
    population_payload: Dict[str, Any],
    top_k: int,
    *,
    initial_node_id: int = 0,
) -> List[Dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    population = population_payload.get("population")
    if not isinstance(population, list):
        raise ValueError("population.json does not contain a list at key 'population'")

    initial_node = None
    for node in population:
        if isinstance(node, dict) and node.get("node_id") == initial_node_id:
            initial_node = node
            break
    if initial_node is None:
        raise ValueError(f"Could not find initial node_id={initial_node_id}")

    scored_nodes = [
        node for node in population
        if isinstance(node, dict) and f1_score(node) is not None
    ]
    scored_nodes.sort(key=lambda node: f1_score(node), reverse=True)

    top_nodes = scored_nodes[:top_k]
    initial_is_in_top_k = any(
        node.get("node_id") == initial_node_id for node in top_nodes
    )

    if initial_is_in_top_k:
        selected = [
            {"node": node, "selection_reason": "top_validation_f1"}
            for node in scored_nodes[: top_k + 1]
        ]
    else:
        selected = [
            {"node": node, "selection_reason": "top_validation_f1"}
            for node in top_nodes
        ]
        selected.append({
            "node": initial_node,
            "selection_reason": "initial_prompt_baseline",
        })

    if len(selected) != top_k + 1:
        raise ValueError(
            f"Could only select {len(selected)} unique nodes; expected {top_k + 1}"
        )

    initial_items = [
        item for item in selected
        if item["node"].get("node_id") == initial_node_id
    ]
    other_items = [
        item for item in selected
        if item["node"].get("node_id") != initial_node_id
    ]
    other_items.sort(key=lambda item: f1_score(item["node"]), reverse=True)
    return initial_items + other_items


def node_from_payload(node_payload: Dict[str, Any]):
    from agents.agent_graph_node import GraphNode

    node = GraphNode(
        inference_prompt=node_payload.get("inference_prompt", ""),
        inference_mode=node_payload.get("inference_mode", ""),
        inference_instruction_prompt=node_payload.get("inference_instruction_prompt", ""),
        inference_answer_instruction_prompt=node_payload.get(
            "inference_answer_instruction_prompt", ""
        ),
        inference_example_prompt=node_payload.get("inference_example_prompt", ""),
        inference_input_prompt=node_payload.get("inference_input_prompt", ""),
        feedback=node_payload.get("feedback", ""),
        raw_feedback_texts=list(node_payload.get("raw_feedback_texts", [])),
        feedback_prompts_used=list(node_payload.get("feedback_prompts_used", [])),
        feedback_prompt=node_payload.get("feedback_prompt", ""),
        mutation_prompt=node_payload.get("mutation_prompt", ""),
        example_generation_prompt=node_payload.get("example_generation_prompt", ""),
        mutation_prompt_used=node_payload.get("mutation_prompt_used", ""),
        raw_mutation_response=node_payload.get("raw_mutation_response"),
        differentiation_prompt_used=node_payload.get("differentiation_prompt_used", ""),
        raw_differentiation_response=node_payload.get("raw_differentiation_response"),
        differentiation=node_payload.get("differentiation", ""),
        node_id=node_payload.get("node_id"),
        val_score=node_payload.get("val_score"),
        test_score=node_payload.get("test_score"),
    )
    node.is_dead = bool(node_payload.get("is_dead", False))
    node.mutation_failures = int(node_payload.get("mutation_failures", 0))
    return node


def resolve_config(args: argparse.Namespace, run_args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": args.model or run_args.get("model"),
        "device_map": args.device_map or run_args.get("device_map"),
        "dataset_type": args.dataset_type or run_args.get("dataset_type", "fs_tacred"),
        "query_index": (
            args.query_index
            if args.query_index is not None
            else int(run_args.get("query_index", 0))
        ),
        "eval_batch_size": (
            args.eval_batch_size
            if args.eval_batch_size is not None
            else int(run_args.get("eval_batch_size", 8))
        ),
        "eval_n_chunks": args.eval_n_chunks,
        "log_every": (
            args.log_every
            if args.log_every is not None
            else int(run_args.get("log_every", 100))
        ),
    }


def create_output_dir(output_root_dir: Path, run_dir: Path, top_k: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_dir.resolve().name
    output_dir = output_root_dir / f"{stamp}_{run_name}_top{top_k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def require_config_value(config: Dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value: {key}")
    return value


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_split_episodes_local(
    *,
    split: str,
    data_dir: str,
    dataset_type: str,
) -> Dict[str, Any]:
    data_path = Path(data_dir)
    episodes_path = data_path / f"{dataset_type}_{split}_episodes_1shots.pkl"
    details_path = data_path / f"{dataset_type}_{split}_episodes_shots_details.pkl"
    episodes = read_pickle(episodes_path)
    details = read_pickle(details_path)
    return {
        "episodes": episodes,
        "shots": details.get("shots", {}),
        "queries": details.get("queries", {}),
        "umbc_shots": details.get("umbc_shots", {}),
    }


def main() -> None:
    start_time = time.monotonic()
    args = parse_args()

    run_dir = args.run_dir.resolve()
    source_type = detect_source_type(run_dir)
    run_args = load_run_args(run_dir)
    population_path = None
    summary_path = None
    if source_type == "evolutionary_population":
        population_path = resolve_population_path(run_dir)
        population_payload = read_json(population_path)
        selected_nodes = select_validation_nodes(population_payload, args.top_k)
    elif source_type == "gradient_experiment":
        summary_path = run_dir / "summary.json"
        summary_payload = read_json(summary_path)
        selected_nodes = select_gradient_nodes(summary_payload, args.top_k)
    else:
        raise ValueError(f"Unsupported source_type={source_type!r}")
    config = resolve_config(args, run_args)
    output_dir = create_output_dir(args.output_root_dir, run_dir, args.top_k)
    eval_output_dir = output_dir / "eval_outputs"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    model_id = require_config_value(config, "model")
    dataset_type = require_config_value(config, "dataset_type")

    print("[final_dev_validator] run_dir:", run_dir)
    print("[final_dev_validator] source_type:", source_type)
    if population_path is not None:
        print("[final_dev_validator] population:", population_path)
    if summary_path is not None:
        print("[final_dev_validator] gradient_summary:", summary_path)
    print("[final_dev_validator] output_dir:", output_dir)
    print("[final_dev_validator] model:", model_id)
    print("[final_dev_validator] dataset_type:", dataset_type)
    print("[final_dev_validator] final_dev_split:", args.final_dev_split)
    print(
        "[final_dev_validator] selected_nodes:",
        [
            {
                "node_id": item["node"].get("node_id"),
                "selection_reason": item["selection_reason"],
            }
            for item in selected_nodes
        ],
    )

    final_dev_data = load_split_episodes_local(
        split=args.final_dev_split,
        data_dir=args.data_dir,
        dataset_type=dataset_type,
    )
    print(
        "[final_dev_validator] final_dev_episodes:",
        len(final_dev_data.get("episodes", [])),
    )

    from agents.agent_evaluate import evaluate_fn
    from agents.agent_models import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(model_id, config.get("device_map"))
    yes_token_id = _resolve_binary_token_id(tokenizer, "yes")
    no_token_id = _resolve_binary_token_id(tokenizer, "no")

    results: List[Dict[str, Any]] = []
    for rank, selected_item in enumerate(selected_nodes, start=1):
        node_payload = selected_item["node"]
        node = node_from_payload(node_payload)
        eval_id = f"rank{rank}_node{node.node_id}"
        print(
            f"[final_dev_validator] evaluating rank={rank} "
            f"node_id={node.node_id} source_f1={f1_score(node_payload):.6f}"
        )
        final_dev_score = evaluate_fn(
            node,
            args.final_dev_split,
            dataset_type=dataset_type,
            model=model,
            tokenizer=tokenizer,
            episodes=final_dev_data["episodes"],
            shots=final_dev_data,
            query_index=int(config["query_index"]),
            batch_size=int(config["eval_batch_size"]),
            n_chunks=int(config["eval_n_chunks"]),
            eval_id=eval_id,
            output_dir=str(eval_output_dir),
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            log_every=int(config["log_every"]),
        )
        results.append(
            {
                "rank": rank,
                "selection_reason": selected_item["selection_reason"],
                "node_id": node_payload.get("node_id"),
                "parent_id": node_payload.get("parent_id"),
                "children_ids": node_payload.get("children_ids", []),
                "is_in_final_population": node_payload.get("is_in_final_population", False),
                "source_validation_score": node_payload.get("val_score"),
                "gradient_iteration_index": node_payload.get("gradient_iteration_index"),
                "gradient_generation_index": node_payload.get("gradient_generation_index"),
                "gradient_beam_index": node_payload.get("gradient_beam_index"),
                "gradient_selection_strategy": node_payload.get("gradient_selection_strategy"),
                "gradient_source_prompt": node_payload.get("gradient_source_prompt"),
                "final_dev_score": final_dev_score,
                "inference_mode": node_payload.get("inference_mode"),
                "inference_prompt": node_payload.get("inference_prompt", ""),
                "inference_instruction_prompt": node_payload.get(
                    "inference_instruction_prompt", ""
                ),
                "inference_answer_instruction_prompt": node_payload.get(
                    "inference_answer_instruction_prompt", ""
                ),
                "inference_example_prompt": node_payload.get("inference_example_prompt", ""),
                "inference_input_prompt": node_payload.get("inference_input_prompt", ""),
                "eval_predictions_file": str(
                    eval_output_dir / f"EVALID_{eval_id}_labels_predictions.json"
                ),
            }
        )

    payload = {
        "source_run_dir": str(run_dir),
        "source_type": source_type,
        "source_population_path": str(population_path) if population_path is not None else None,
        "source_gradient_summary_path": str(summary_path) if summary_path is not None else None,
        "top_k": args.top_k,
        "validated_prompt_count": len(selected_nodes),
        "initial_node_id": 0 if source_type == "evolutionary_population" else None,
        "initial_node_always_included": source_type == "evolutionary_population",
        "selection_metric": (
            "val_score.f1_mean"
            if source_type == "evolutionary_population"
            else "selected_prompt.dev_prf.f1"
        ),
        "source_validation_split": (
            run_args.get("dev_split", "dev")
            if source_type == "evolutionary_population"
            else run_args.get("full_eval_split", "dev")
        ),
        "final_dev_split": args.final_dev_split,
        "data_dir": str(Path(args.data_dir).resolve()),
        "config": config,
        "final_dev_episode_count": len(final_dev_data.get("episodes", [])),
        "elapsed_seconds": time.monotonic() - start_time,
        "results": results,
    }
    output_path = output_dir / "final_dev_validation_results.json"
    write_json(output_path, payload)
    print("[final_dev_validator] wrote:", output_path)


def _resolve_binary_token_id(tokenizer, base_token: str) -> int:
    token_ids = tokenizer.encode(base_token, add_special_tokens=False)
    if len(token_ids) == 1:
        return token_ids[0]
    spaced_token_ids = tokenizer.encode(f" {base_token}", add_special_tokens=False)
    if len(spaced_token_ids) == 1:
        return spaced_token_ids[0]
    raise ValueError(
        f"Token '{base_token}' is not a single token for this tokenizer."
    )


if __name__ == "__main__":
    main()
