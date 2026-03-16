from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Sequence

from agents.agent_binary_inference import run_binary_inference
from agents.agent_graph_node import GraphNode
from agents.agent_prompts import compose_inference_prompt
from agents.agent_metrics import compute_prf_stats
from agents.agent_scorer import NO_RELATION
from agents.agent_relation_utils import get_relation_description
from agents.agent_utils import build_support_block, get_sentence_with_tags, resolve_way_shots


def _build_inference_prompt(
    node: GraphNode,
    relation: str,
    relation_description: str,
    support_sentences: Sequence[str],
    query_sentence: str,
) -> str:
    support_block = build_support_block(support_sentences)
    example_query_sentence = support_sentences[0] if support_sentences else query_sentence
    return compose_inference_prompt(
        inference_mode=node.inference_mode,
        inference_prompt=node.inference_prompt,
        inference_instruction_prompt=node.inference_instruction_prompt,
        inference_answer_instruction_prompt=node.inference_answer_instruction_prompt,
        inference_example_prompt=node.inference_example_prompt,
        inference_input_prompt=node.inference_input_prompt,
        relation=relation,
        relation_description=relation_description,
        support_block=support_block,
        query_sentence=query_sentence,
        example_query_sentence=example_query_sentence,
    )


def evaluate_fn(
    node: GraphNode,
    split: str,
    *,
    dataset_type: str,
    model,
    tokenizer,
    episodes: List[Dict],
    shots: Dict,
    query_index: int = 0,
    batch_size: int = 8,
    n_chunks: int = 1,
    eval_id: str | int | None = None,
    output_dir: str | None = None,
    yes_token_id: int | None = None,
    no_token_id: int | None = None,
    log_every: int = 100,
    evolution_iteration: int | None = None,
    evolution_max_iterations: int | None = None,
) -> float:
    if output_dir and eval_id is None:
        raise ValueError("eval_id is required when output_dir is provided.")

    start_time = time.perf_counter()
    print(f"[agent_evaluate] evaluate_fn: insturction_prompt=\n{node.inference_prompt}")

    prompts: List[str] = []
    pair_labels: List[str] = []
    episode_relations: List[List[str]] = []
    episode_labels: List[str] = []

    for episode in episodes:
        ways = episode["meta_train"]
        query_id = episode["meta_test"][query_index]
        query = shots["queries"][query_id]
        query_relation = query["relation"]
        query_sentence = get_sentence_with_tags(query).strip()

        relations: List[str] = []
        for way in ways:
            way_shots = resolve_way_shots(way, shots)
            relation = way_shots[0]["relation"]
            relations.append(relation)
            support_sentences = [get_sentence_with_tags(s).strip() for s in way_shots]
            relation_description = get_relation_description(relation, dt=dataset_type)

            prompts.append(
                _build_inference_prompt(
                    node=node,
                    relation=relation,
                    relation_description=relation_description,
                    support_sentences=support_sentences,
                    query_sentence=query_sentence,
                )
            )
            pair_labels.append("yes" if relation == query_relation else "no")

        episode_relations.append(relations)
        if query_relation in relations:
            episode_labels.append(query_relation)
        else:
            episode_labels.append(NO_RELATION)

    if prompts:
        print(f"[agent_evaluate] evaluate_fn: sample full prompt=\n{prompts[0]}")

    print(
        f"[agent_evaluate] evaluate_fn: split={split}, episodes={len(episodes)}, "
        f"prompts={len(prompts)}, batch_size={batch_size}, "
        f"n_chunks={n_chunks}, eval_id={eval_id}"
    )

    pair_predictions = run_binary_inference(
        prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        log_every=log_every,
        evolution_iteration=evolution_iteration,
        evolution_max_iterations=evolution_max_iterations,
    )

    episode_predictions: List[str] = []
    offset = 0
    for relations in episode_relations:
        chunk = pair_predictions[offset : offset + len(relations)]
        offset += len(relations)
        predicted_relation = NO_RELATION
        for relation, pred in zip(relations, chunk):
            if pred == "yes":
                predicted_relation = relation
                break
        episode_predictions.append(predicted_relation)

    assert len(episode_labels) == len(episode_predictions) == len(episodes)

    metrics = compute_prf_stats(episode_labels, episode_predictions, n_chunks=n_chunks)
    elapsed = time.perf_counter() - start_time
    print(
        f"[agent_evaluate] evaluate_fn: done in {elapsed:.2f}s, "
        f"precision={metrics['precision_mean'] * 100:.2f}±{metrics['precision_std'] * 100:.2f}, "
        f"recall={metrics['recall_mean'] * 100:.2f}±{metrics['recall_std'] * 100:.2f}, "
        f"f1={metrics['f1_mean'] * 100:.2f}±{metrics['f1_std'] * 100:.2f}"
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"EVALID_{eval_id}_labels_predictions.json"
        )
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "eval_id": eval_id,
                    "split": split,
                    "labels": episode_labels,
                    "predictions": episode_predictions,
                    "pair_labels": pair_labels,
                    "pair_predictions": pair_predictions,
                    "metrics": metrics,
                },
                handle,
            )
        print(f"[agent_evaluate] evaluate_fn: saved labels/predictions to {output_path}")
    return metrics
