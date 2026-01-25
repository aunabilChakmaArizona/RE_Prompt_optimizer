from __future__ import annotations

import re
from typing import List

from feedback_samples import FeedbackSamples
from graph_node import GraphNode
from llm_prompting import run_prompt
from prompts import MUTATION_PROMPT_V1


def _extract_between(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return text.strip()
    return match.group(1).strip()


def _pad_list(items: List[str], size: int) -> List[str]:
    if len(items) >= size:
        return items[:size]
    return items + [""] * (size - len(items))


def mutate_prompt_fn(
    node: GraphNode,
    feedback_samples: FeedbackSamples,
    feedback_text: str,
    *,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> str:
    base_prompt = node.mutation_prompt or MUTATION_PROMPT_V1

    samples = feedback_samples.selected_samples
    feedback_texts = getattr(feedback_samples, "feedback_texts", None)
    if feedback_texts is None:
        feedback_texts = [t for t in feedback_text.split("\n") if t.strip()]
    feedback_texts = _pad_list(list(feedback_texts), 3)

    def get_attr(i: int, name: str) -> str:
        if i >= len(samples):
            return ""
        return getattr(samples[i], name, "")

    prompt = base_prompt
    prompt = prompt.replace("#INFERENCE_PROMPT#", node.inference_prompt)

    for idx in range(3):
        prompt = prompt.replace(f"#RELATION_{idx+1}#", get_attr(idx, "relation"))
        prompt = prompt.replace(
            f"#RELATION_DESCRIPTION_{idx+1}#", get_attr(idx, "relation_description")
        )
        prompt = prompt.replace(
            f"#SUPPORT_INSTANCE_{idx+1}#", get_attr(idx, "support_sentence")
        )
        prompt = prompt.replace(f"#QUERY_{idx+1}#", get_attr(idx, "query_sentence"))
        prompt = prompt.replace(f"#LABEL_{idx+1}#", get_attr(idx, "label"))
        prompt = prompt.replace(f"#INFERENCE_{idx+1}#", get_attr(idx, "inference"))
        prompt = prompt.replace(f"#FEEDBACK_{idx+1}#", feedback_texts[idx])

    raw_response = run_prompt(
        prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    return _extract_between(raw_response, "p")
