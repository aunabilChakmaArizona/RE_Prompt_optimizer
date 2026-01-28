from __future__ import annotations

import re
from typing import List, Optional, Tuple

from agents.agent_feedback_samples import FeedbackSamples
from agents.agent_graph_node import GraphNode
from agents.agent_llm_prompting import run_prompt
from agents.agent_prompts import MUTATION_PROMPT_V1, INFERENCE_PROMPT_PLACEHODERS_V1
from agents.agent_relation_utils import get_relation_description


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


def _contains_placeholders(prompt: str, placeholders: List[str]) -> bool:
    return all(ph in prompt for ph in placeholders)


def mutate_prompt_fn(
    node: GraphNode,
    feedback_samples: FeedbackSamples,
    *,
    dataset_type: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> Optional[Tuple[str, str, str]]:
    base_prompt = node.mutation_prompt or MUTATION_PROMPT_V1

    samples = feedback_samples.selected_samples
    feedback_texts = [
        getattr(sample, "feedback_text", "") for sample in feedback_samples.selected_samples
    ]
    # feedback_texts = _pad_list([t for t in feedback_texts if t.strip()], 3)

    def get_attr(i: int, name: str) -> str:
        # if i >= len(samples):
        #     return ""
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

    prompt = prompt.replace("#LIST_OF_PLACEHOLDERS#", str(INFERENCE_PROMPT_PLACEHODERS_V1))

    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        raw_response = run_prompt(
            prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        if not re.search(r"<p>.*?</p>", raw_response, flags=re.DOTALL | re.IGNORECASE):
            print(f"[agent_mutate_prompt] missing <p> tag; retry {attempt}/{max_attempts}")
            continue
        candidate = _extract_between(raw_response, "p")
        if _contains_placeholders(candidate, INFERENCE_PROMPT_PLACEHODERS_V1):
            return candidate, raw_response, prompt

        print(f"[agent_mutate_prompt] missing placeholders; retry {attempt}/{max_attempts}")

    node.is_dead = True
    print("[agent_mutate_prompt] node marked dead (too many placeholder failures)")
    return None
