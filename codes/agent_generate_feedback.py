from __future__ import annotations

from typing import List

from agent_feedback_samples import FeedbackSamples
from agent_graph_node import GraphNode
from agent_llm_prompting import run_prompts
from agent_prompts import FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1
from agent_relation_utils import get_relation_description


def _format_feedback_prompt(
    base_prompt: str,
    relation: str,
    relation_description: str,
    support_sentence: str,
    query_sentence: str,
    label: str,
    inference: str,
) -> str:
    prompt = base_prompt
    prompt = prompt.replace("#RELATION#", relation)
    prompt = prompt.replace("#RELATION_DESCRIPTION#", relation_description)
    prompt = prompt.replace("#SUPPORT_INSTANCE#", support_sentence)
    prompt = prompt.replace("#QUERY#", query_sentence)
    prompt = prompt.replace("#LABEL#", label)
    prompt = prompt.replace("#INFERENCE#", inference)
    return prompt


def _extract_feedback(text: str) -> str:
    start = text.find("<f>")
    if start == -1:
        return text.strip()
    end = text.find("</f>", start + 3)
    if end == -1:
        return text.strip()
    return text[start + 3 : end].strip()


def generate_feedback_fn(
    node: GraphNode,
    feedback_samples: FeedbackSamples,
    *,
    dataset_type: str,
    model,
    tokenizer,
    batch_size: int = 4,
) -> FeedbackSamples:
    base_prompt = node.feedback_prompt or FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1

    prompts: List[str] = []
    for sample in feedback_samples.selected_samples:
        relation = sample.relation
        prompts.append(
            _format_feedback_prompt(
                base_prompt=base_prompt,
                relation=relation,
                relation_description=get_relation_description(relation, dt=dataset_type),
                support_sentence=sample.support_sentence,
                query_sentence=sample.query_sentence,
                label=sample.label,
                inference=sample.inference,
            )
        )

    raw_feedback_texts = run_prompts(
        prompts, model=model, tokenizer=tokenizer, batch_size=batch_size
    )
    feedback_texts = [_extract_feedback(text) for text in raw_feedback_texts]
    feedback_samples.raw_feedback_texts = raw_feedback_texts
    feedback_samples.feedback_texts = feedback_texts
    for sample, raw_feedback_text, feedback_text in zip(
        feedback_samples.selected_samples, raw_feedback_texts, feedback_texts
    ):
        sample.raw_feedback_text = raw_feedback_text
        sample.feedback_text = feedback_text

    return feedback_samples
