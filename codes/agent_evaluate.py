from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Sequence

from agent_binary_inference import run_binary_inference
from agent_graph_node import GraphNode
from agent_prompts import INFERENCE_PROMPT_V1
from agent_data_utils import build_support_block, get_sentence_with_tags, resolve_way_shots
from agent_metrics import compute_prf_stats


def _format_inference_prompt(
    base_prompt: str,
    relation: str,
    relation_description: str,
    support_sentences: Sequence[str],
    query_sentence: str,
) -> str:
    support_block = build_support_block(support_sentences)
    prompt = base_prompt
    prompt = prompt.replace("#RELATION#", relation)
    prompt = prompt.replace("#RELATION_DESCRIPTION#", relation_description)
    prompt = prompt.replace("#SUPPORT_SENTENCE_BLOCK#", support_block)
    prompt = prompt.replace("#QUERY_SENTENCE#", query_sentence)
    return prompt


def evaluate_fn(
    node: GraphNode,
    split: str,
    *,
    model,
    tokenizer,
    episodes: List[Dict],
    shots: Dict,
    relation_descriptions: Dict[str, str],
    query_index: int = 0,
    batch_size: int = 8,
    n_chunks: int = 1,
    eval_id: str | int | None = None,
    output_dir: str | None = None,
) -> float:
    if not episodes:
        return 0.0
    if output_dir and eval_id is None:
        raise ValueError("eval_id is required when output_dir is provided.")

    start_time = time.perf_counter()
    base_prompt = node.inference_prompt or INFERENCE_PROMPT_V1

    prompts: List[str] = []
    labels: List[str] = []

    for episode in episodes:
        ways = episode["meta_train"]
        query_id = episode["meta_test"][query_index]
        query = shots["queries"][query_id]
        query_relation = query["relation"]
        query_sentence = get_sentence_with_tags(query).strip()

        for way in ways:
            way_shots = resolve_way_shots(way, shots)
            relation = way_shots[0]["relation"]
            support_sentences = [get_sentence_with_tags(s).strip() for s in way_shots]
            relation_description = relation_descriptions.get(relation, "")

            prompts.append(
                _format_inference_prompt(
                    base_prompt=base_prompt,
                    relation=relation,
                    relation_description=relation_description,
                    support_sentences=support_sentences,
                    query_sentence=query_sentence,
                )
            )
            labels.append("yes" if relation == query_relation else "no")

    print(
        f"evaluate_fn: split={split}, episodes={len(episodes)}, "
        f"prompts={len(prompts)}, batch_size={batch_size}, "
        f"n_chunks={n_chunks}, eval_id={eval_id}"
    )

    predictions = run_binary_inference(
        prompts, model=model, tokenizer=tokenizer, batch_size=batch_size
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
                    "labels": labels,
                    "predictions": predictions,
                },
                handle,
            )
        print(f"evaluate_fn: saved labels/predictions to {output_path}")

    metrics = compute_prf_stats(labels, predictions, n_chunks=n_chunks)
    elapsed = time.perf_counter() - start_time
    print(
        f"evaluate_fn: done in {elapsed:.2f}s, "
        f"f1_mean={metrics['f1_mean']:.4f}"
    )
    return metrics
