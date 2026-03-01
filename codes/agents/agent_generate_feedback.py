from __future__ import annotations

from typing import List

from agents.agent_feedback_samples import FeedbackSamples
from agents.agent_graph_node import GraphNode
from agents.agent_llm_prompting import run_prompts
from agents.agent_relation_utils import get_relation_description


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


def _extract_feedback(text: str, open_tag: str, close_tag: str) -> str:
    start = text.find(open_tag)
    if start == -1:
        return text.strip()
    end = text.find(close_tag, start + len(open_tag))
    if end == -1:
        return text.strip()
    return text[start + len(open_tag) : end].strip()


def _has_feedback_tag(text: str, open_tag: str, close_tag: str) -> bool:
    start = text.find(open_tag)
    if start == -1:
        return False
    end = text.find(close_tag, start + len(open_tag))
    return end != -1


def generate_feedback_fn(
    node: GraphNode,
    feedback_samples: FeedbackSamples,
    *,
    dataset_type: str,
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = 3000,
    feedback_open_tag: str = "<f>",
    feedback_close_tag: str = "</f>",
) -> FeedbackSamples:
    base_prompt = node.feedback_prompt

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

    max_attempts = 3
    raw_feedback_texts: List[str] = []
    feedback_texts: List[str] = []
    for attempt in range(1, max_attempts + 1):
        raw_feedback_texts = run_prompts(
            prompts,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
        feedback_texts = [
            _extract_feedback(text, feedback_open_tag, feedback_close_tag)
            for text in raw_feedback_texts
        ]
        if all(
            _has_feedback_tag(text, feedback_open_tag, feedback_close_tag)
            for text in raw_feedback_texts
        ):
            break
        if attempt < max_attempts:
            print(
                f"[agent_generate_feedback] missing {feedback_open_tag} tag; retry {attempt}/{max_attempts}"
            )
    feedback_samples.feedback_prompts = prompts
    feedback_samples.raw_feedback_texts = raw_feedback_texts
    feedback_samples.feedback_texts = feedback_texts
    for sample, prompt, raw_feedback_text, feedback_text in zip(
        feedback_samples.selected_samples, prompts, raw_feedback_texts, feedback_texts
    ):
        sample.feedback_prompt = prompt
        sample.raw_feedback_text = raw_feedback_text
        sample.feedback_text = feedback_text

    return feedback_samples
