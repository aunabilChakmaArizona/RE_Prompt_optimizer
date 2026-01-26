from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

from agent_data_loader import load_split_episodes, load_train_samples
from agent_data_utils import get_sentence_with_tags
from agent_evaluate import evaluate_fn as _evaluate_fn
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


def build_feedback_pool(shots: Dict) -> Dict[str, List[dict]]:
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


def build_shot_index(shots: Dict) -> Dict[str, dict]:
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


def enrich_feedback_samples(feedback_samples, shots_by_id: Dict[str, dict]) -> None:
    for sample in feedback_samples.all_samples:
        support = shots_by_id.get(sample.id_1shot)
        query = shots_by_id.get(sample.id_query)
        if support:
            sample.relation = support.get("relation", "")
            sample.support_sentence = get_sentence_with_tags(support).strip()
        if query:
            sample.query_sentence = get_sentence_with_tags(query).strip()


def load_data_assets(args, data_dir: str) -> Tuple[Dict, Dict, Dict, Dict]:
    train_samples = load_train_samples(data_dir=data_dir, filename=args.train_samples)
    train_shots = train_samples.get("shots", {})
    feedback_pool = build_feedback_pool(train_shots)
    train_shot_index = build_shot_index(train_shots)

    dev_data = load_split_episodes(
        split=args.dev_split,
        data_dir=data_dir,
        dataset_prefix=args.dataset_prefix,
        ep_start=args.dev_ep_start,
        ep_end=args.dev_ep_end,
    )
    test_data = load_split_episodes(
        split=args.test_split,
        data_dir=data_dir,
        dataset_prefix=args.dataset_prefix,
        ep_start=args.test_ep_start,
        ep_end=args.test_ep_end,
    )

    return feedback_pool, train_shot_index, dev_data, test_data


def build_training_functions(
    *,
    args,
    model,
    tokenizer,
    feedback_pool: Dict[str, List[dict]],
    train_shot_index: Dict[str, dict],
    dev_data: Dict,
    test_data: Dict,
    eval_output_dir: str,
    rng: random.Random,
):
    def sample_feedback(k: int):
        samples = _sample_feedback_fn(feedback_pool, k=k, rng=rng)
        enrich_feedback_samples(samples, train_shot_index)
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
            query_index=args.query_index,
            batch_size=args.eval_batch_size,
            n_chunks=args.eval_n_chunks,
            eval_id=eval_id,
            output_dir=eval_output_dir,
        )

    return sample_feedback, run_inference, generate_feedback, mutate_prompt, evaluate


def build_root_node() -> GraphNode:
    return GraphNode(
        inference_prompt=INFERENCE_PROMPT_V1,
        feedback_prompt=FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1,
        mutation_prompt=MUTATION_PROMPT_V1,
        example_generation_prompt=EXAMPLE_GENERATION_PROMPT_V1,
        node_id=0,
    )


def load_model_and_data(args, data_dir: str, eval_output_dir: str, rng_seed: int):
    rng = random.Random(rng_seed)
    model, tokenizer = load_model_and_tokenizer(args.model)
    feedback_pool, train_shot_index, dev_data, test_data = load_data_assets(
        args, data_dir
    )
    functions = build_training_functions(
        args=args,
        model=model,
        tokenizer=tokenizer,
        feedback_pool=feedback_pool,
        train_shot_index=train_shot_index,
        dev_data=dev_data,
        test_data=test_data,
        eval_output_dir=eval_output_dir,
        rng=rng,
    )
    return model, tokenizer, rng, feedback_pool, train_shot_index, dev_data, test_data, functions
