from __future__ import annotations

import re
from typing import List

from agent_feedback_samples import FeedbackSamples
from agent_graph_node import GraphNode
from agent_llm_prompting import run_prompt
from agent_prompts import MUTATION_PROMPT_V1
from agent_relation_utils import get_relation_description


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
    *,
    dataset_type: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> str:
    base_prompt = node.mutation_prompt or MUTATION_PROMPT_V1

    samples = feedback_samples.selected_samples
    feedback_texts = [
        getattr(sample, "feedback_text", "") for sample in feedback_samples.selected_samples
    ]
    feedback_texts = _pad_list([t for t in feedback_texts if t.strip()], 3)

    def get_attr(i: int, name: str) -> str:
        if i >= len(samples):
            return ""
        return getattr(samples[i], name, "")

    prompt = base_prompt
    prompt = prompt.replace("#INFERENCE_PROMPT#", node.inference_prompt)

    for idx in range(3):
        relation = get_attr(idx, "relation")
        prompt = prompt.replace(f"#RELATION_{idx+1}#", relation)
        prompt = prompt.replace(
            f"#RELATION_DESCRIPTION_{idx+1}#",
            get_relation_description(relation, dt=dataset_type) if relation else "",
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
