from __future__ import annotations

from typing import List

from binary_inference import run_binary_inference
from feedback_samples import FeedbackSamples
from graph_node import GraphNode
from prompts import INFERENCE_PROMPT_V1
from agent_data_utils import build_support_block


def _format_inference_prompt(
    base_prompt: str,
    relation: str,
    relation_description: str,
    support_sentence: str,
    query_sentence: str,
) -> str:
    support_block = build_support_block([support_sentence])
    prompt = base_prompt
    prompt = prompt.replace("#RELATION#", relation)
    prompt = prompt.replace("#RELATION_DESCRIPTION#", relation_description)
    prompt = prompt.replace("#SUPPORT_SENTENCE_BLOCK#", support_block)
    prompt = prompt.replace("#QUERY_SENTENCE#", query_sentence)
    prompt = prompt.replace("#QUERY#", query_sentence)
    return prompt


def run_inference_fn(
    node: GraphNode,
    feedback_samples: FeedbackSamples,
    *,
    model,
    tokenizer,
    batch_size: int = 8,
) -> FeedbackSamples:
    if not feedback_samples.selected_samples:
        return feedback_samples

    base_prompt = node.inference_prompt or INFERENCE_PROMPT_V1

    prompts: List[str] = []
    for sample in feedback_samples.selected_samples:
        prompt = _format_inference_prompt(
            base_prompt=base_prompt,
            relation=getattr(sample, "relation", ""),
            relation_description=getattr(sample, "relation_description", ""),
            support_sentence=getattr(sample, "support_sentence", ""),
            query_sentence=getattr(sample, "query_sentence", ""),
        )
        prompts.append(prompt)

    predictions = run_binary_inference(
        prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    for sample, pred in zip(feedback_samples.selected_samples, predictions):
        sample.inference = pred

    return feedback_samples
