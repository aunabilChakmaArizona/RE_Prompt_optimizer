from __future__ import annotations

from typing import Dict, List, Sequence

from binary_inference import run_binary_inference
from graph_node import GraphNode
from prompts import INFERENCE_PROMPT_V1
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
) -> float:
    if not episodes:
        return 0.0

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

    predictions = run_binary_inference(
        prompts, model=model, tokenizer=tokenizer, batch_size=batch_size
    )
    stats = compute_prf_stats(labels, predictions, n_chunks=n_chunks)
    return stats["f1_mean"]
