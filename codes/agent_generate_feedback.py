from __future__ import annotations

from typing import List

from agent_feedback_samples import FeedbackSamples
from agent_graph_node import GraphNode
from agent_llm_prompting import run_prompts
from agent_prompts import FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1


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


def generate_feedback_fn(
    node: GraphNode,
    feedback_samples: FeedbackSamples,
    *,
    model,
    tokenizer,
    batch_size: int = 4,
) -> FeedbackSamples:
    if not feedback_samples.selected_samples:
        feedback_samples.feedback_texts = []
        return ""

    base_prompt = node.feedback_prompt or FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1

    prompts: List[str] = []
    for sample in feedback_samples.selected_samples:
        prompts.append(
            _format_feedback_prompt(
                base_prompt=base_prompt,
                relation=getattr(sample, "relation", ""),
                relation_description=getattr(sample, "relation_description", ""),
                support_sentence=getattr(sample, "support_sentence", ""),
                query_sentence=getattr(sample, "query_sentence", ""),
                label=sample.label,
                inference=sample.inference,
            )
        )

    feedback_texts = run_prompts(
        prompts, model=model, tokenizer=tokenizer, batch_size=batch_size
    )
    feedback_samples.feedback_texts = feedback_texts
    for sample, feedback_text in zip(feedback_samples.selected_samples, feedback_texts):
        sample.feedback_text = feedback_text

    return feedback_samples
