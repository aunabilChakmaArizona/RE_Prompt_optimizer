from __future__ import annotations

from typing import List

from agents.agent_binary_inference import run_binary_inference
from agents.agent_feedback_samples import FeedbackSamples
from agents.agent_graph_node import GraphNode
from agents.agent_prompts import compose_inference_prompt
from agents.agent_relation_utils import get_relation_description
from agents.agent_utils import build_support_block


def _build_inference_prompt(
    node: GraphNode,
    relation: str,
    relation_description: str,
    support_sentence: str,
    query_sentence: str,
) -> str:
    support_block = build_support_block([support_sentence])
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
        example_query_sentence=support_sentence,
    )


def run_inference_fn(
    node: GraphNode,
    feedback_samples: FeedbackSamples,
    *,
    dataset_type: str,
    model,
    tokenizer,
    batch_size: int = 8,
    yes_token_id: int | None = None,
    no_token_id: int | None = None,
    log_every: int = 100,
    evolution_iteration: int | None = None,
    evolution_max_iterations: int | None = None,
) -> FeedbackSamples:
    prompts: List[str] = []
    for sample in feedback_samples.all_samples:
        relation = sample.relation
        prompt = _build_inference_prompt(
            node=node,
            relation=relation,
            relation_description=get_relation_description(relation, dt=dataset_type),
            support_sentence=sample.support_sentence,
            query_sentence=sample.query_sentence,
        )
        prompts.append(prompt)

    predictions = run_binary_inference(
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

    for sample, pred in zip(feedback_samples.all_samples, predictions):
        sample.inference = pred

    return feedback_samples
