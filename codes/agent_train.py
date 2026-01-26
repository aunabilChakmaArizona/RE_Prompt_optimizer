from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List

from agent_data_loader import load_split_episodes, load_train_samples
from agent_data_utils import get_sentence_with_tags
from agent_evaluate import evaluate_fn as _evaluate_fn
from agent_evolutionary_search import EvolutionarySearch
from agent_generate_feedback import generate_feedback_fn as _generate_feedback_fn
from agent_graph_node import GraphNode
from agent_models import load_model_and_tokenizer
from agent_mutate_prompt import mutate_prompt_fn as _mutate_prompt_fn
from agent_prompts import (
    EXAMPLE_GENERATION_PROMPT_V1,
    FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1,
    INFERENCE_PROMPT_V1,
    MUTATION_PROMPT_V1,
)
from agent_run_inference import run_inference_fn as _run_inference_fn
from agent_sample_feedback import sample_feedback_fn as _sample_feedback_fn


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _build_feedback_pool(shots: Dict) -> Dict[str, List[dict]]:
    if not shots:
        return {}
    first_value = next(iter(shots.values()))
    if isinstance(first_value, list):
        return shots

    pool: Dict[str, List[dict]] = {}
    for shot in shots.values():
        relation = shot.get("relation")
        if relation is None:
            continue
        pool.setdefault(relation, []).append(shot)
    return pool


def _build_shot_index(shots: Dict) -> Dict[str, dict]:
    if not shots:
        return {}
    first_value = next(iter(shots.values()))
    if isinstance(first_value, list):
        shot_index: Dict[str, dict] = {}
        for shot_list in shots.values():
            for shot in shot_list:
                shot_id = shot.get("id")
                if shot_id is not None:
                    shot_index[shot_id] = shot
        return shot_index

    return shots


def _enrich_feedback_samples(feedback_samples, shots_by_id: Dict[str, dict]) -> None:
    for sample in feedback_samples.all_samples:
        support = shots_by_id.get(sample.id_1shot)
        query = shots_by_id.get(sample.id_query)
        if support:
            sample.relation = support.get("relation", "")
            sample.support_sentence = get_sentence_with_tags(support).strip()
        if query:
            sample.query_sentence = get_sentence_with_tags(query).strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evolutionary prompt optimization.")
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--dataset-type", default="fs_tacred", help="Dataset type")
    parser.add_argument("--data-dir", default=None, help="Path to data dir")
    parser.add_argument(
        "--train-samples",
        default="fs_tacred_train_samples.pkl",
        help="Train samples pkl filename",
    )
    parser.add_argument("--dataset-prefix", default="fs_tacred", help="Dataset prefix")
    parser.add_argument("--dev-split", default="dev", help="Dev split name")
    parser.add_argument("--test-split", default="test", help="Test split name")
    parser.add_argument("--dev-ep-start", type=int, default=0)
    parser.add_argument("--dev-ep-end", type=int, default=None)
    parser.add_argument("--test-ep-start", type=int, default=0)
    parser.add_argument("--test-ep-end", type=int, default=None)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--feedback-sample-size", type=int, default=100)
    parser.add_argument("--selection-mode", default="mixed")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--feedback-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-n-chunks", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--trainings-dir",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "trainings")
        ),
        help="Output root for training runs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "data")
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "-")
    run_dir = os.path.join(args.trainings_dir, f"{run_stamp}_{model_tag}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")

    log_file = open(log_path, "w", encoding="utf-8")
    stdout_tee = _Tee(sys.stdout, log_file)
    stderr_tee = _Tee(sys.stderr, log_file)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    try:
        print("[train] run_dir:", run_dir)
        print("[train] model:", args.model)
        print("[train] dataset_type:", args.dataset_type)

        with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2)

        rng = random.Random(args.seed)

        print("[train] loading model/tokenizer")
        model, tokenizer = load_model_and_tokenizer(args.model)

        print("[train] loading train samples")
        train_samples = load_train_samples(
            data_dir=data_dir, filename=args.train_samples
        )
        train_shots = train_samples.get("shots", {})
        feedback_pool = _build_feedback_pool(train_shots)
        train_shot_index = _build_shot_index(train_shots)

        print("[train] loading dev episodes")
        dev_data = load_split_episodes(
            split=args.dev_split,
            data_dir=data_dir,
            dataset_prefix=args.dataset_prefix,
            ep_start=args.dev_ep_start,
            ep_end=args.dev_ep_end,
        )
        print("[train] loading test episodes")
        test_data = load_split_episodes(
            split=args.test_split,
            data_dir=data_dir,
            dataset_prefix=args.dataset_prefix,
            ep_start=args.test_ep_start,
            ep_end=args.test_ep_end,
        )

        eval_output_dir = os.path.join(run_dir, "eval_outputs")
        os.makedirs(eval_output_dir, exist_ok=True)

        def sample_feedback(k: int):
            samples = _sample_feedback_fn(feedback_pool, k=k, rng=rng)
            _enrich_feedback_samples(samples, train_shot_index)
            samples.selected_samples = list(samples.all_samples)
            return samples

        def run_inference(node: GraphNode, feedback_samples):
            feedback_samples.selected_samples = list(feedback_samples.all_samples)
            return _run_inference_fn(
                node,
                feedback_samples,
                dataset_type=args.dataset_type,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.inference_batch_size,
            )

        def generate_feedback(node: GraphNode, feedback_samples):
            return _generate_feedback_fn(
                node,
                feedback_samples,
                dataset_type=args.dataset_type,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.feedback_batch_size,
            )

        def mutate_prompt(node: GraphNode, feedback_samples):
            return _mutate_prompt_fn(
                node,
                feedback_samples,
                dataset_type=args.dataset_type,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
            )

        def evaluate(node: GraphNode, split: str):
            eval_id = node.node_id if node.node_id is not None else 0
            if split == "validation":
                episodes = dev_data["episodes"]
                shots = dev_data
            else:
                episodes = test_data["episodes"]
                shots = test_data
            return _evaluate_fn(
                node,
                split,
                dataset_type=args.dataset_type,
                model=model,
                tokenizer=tokenizer,
                episodes=episodes,
                shots=shots,
                relation_descriptions={},
                query_index=args.query_index,
                batch_size=args.eval_batch_size,
                n_chunks=args.eval_n_chunks,
                eval_id=eval_id,
                output_dir=eval_output_dir,
            )

        root = GraphNode(
            inference_prompt=INFERENCE_PROMPT_V1,
            feedback_prompt=FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1,
            mutation_prompt=MUTATION_PROMPT_V1,
            example_generation_prompt=EXAMPLE_GENERATION_PROMPT_V1,
            node_id=0,
        )

        search = EvolutionarySearch(
            root=root,
            max_iterations=args.max_iterations,
            feedback_sample_size=args.feedback_sample_size,
            temperature=args.temperature,
            feedback_prompt=FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1,
            mutation_prompt=MUTATION_PROMPT_V1,
            example_generation_prompt=EXAMPLE_GENERATION_PROMPT_V1,
            dataset_type=args.dataset_type,
            rng=rng,
        )

        best_node, population = search.run(
            sample_feedback_fn=sample_feedback,
            run_inference_fn=run_inference,
            generate_feedback_fn=generate_feedback,
            mutate_prompt_fn=mutate_prompt,
            evaluate_fn=evaluate,
            selection_mode=args.selection_mode,
        )

        print("[train] evaluating best on test")
        best_test_metrics = evaluate(best_node, "test")

        def serialize_node(node: GraphNode) -> Dict[str, object]:
            return {
                "node_id": node.node_id,
                "parent_id": node.parent.node_id if node.parent else None,
                "children_ids": [child.node_id for child in node.children],
                "inference_prompt": node.inference_prompt,
                "feedback": node.feedback,
                "val_score": node.val_score,
                "test_score": node.test_score,
            }

        population_data = [serialize_node(node) for node in population]
        best_node_data = serialize_node(best_node)
        with open(os.path.join(run_dir, "population.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "best_node_id": best_node.node_id,
                    "population": population_data,
                    "best_node": best_node_data,
                },
                handle,
                indent=2,
            )

        summary = {
            "run_dir": run_dir,
            "best_val_score": best_node.val_score,
            "best_test_metrics": best_test_metrics,
            "population_size": len(population),
        }
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print("[train] done")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
