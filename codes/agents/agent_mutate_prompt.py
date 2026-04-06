from __future__ import annotations

from typing import List, Optional, Tuple

from agents.agent_feedback_samples import FeedbackSamples
from agents.agent_graph_node import GraphNode
from agents.agent_llm_prompting import run_prompt
from agents.agent_prompts import (
    INFERENCE_MODE_NON_SEPARATE,
    INFERENCE_PROMPT_PLACEHODERS_V1,
    MUTATION_TRACES_DIFFERENCES_PROMPT_SEGMENT_V1,
    MUTATION_TRACES_PROMPT_SEGMENT_V1,
)
from agents.agent_relation_utils import get_relation_description
from agents.agent_utils import differentiate_prompts, extract_tagged_text


def _pad_list(items: List[str], size: int) -> List[str]:
    if len(items) >= size:
        return items[:size]
    return items + [""] * (size - len(items))


def _contains_placeholders(prompt: str, placeholders: List[str]) -> bool:
    return all(ph in prompt for ph in placeholders)


def _format_node_score(node: GraphNode) -> str:
    score = getattr(node, "val_score", None)
    if score is None:
        return "N/A"
    if isinstance(score, dict):
        f1_mean = score.get("f1_mean")
        if f1_mean is not None:
            return f"{float(f1_mean) * 100:.2f}"
    try:
        return f"{float(score) * 100:.2f}"
    except (TypeError, ValueError):
        return str(score)


def _build_inference_prompt_traces(node: GraphNode, max_depth: int = 3) -> str:
    lineage: List[GraphNode] = []
    cursor: Optional[GraphNode] = node
    while cursor is not None and len(lineage) < max_depth:
        lineage.append(cursor)
        cursor = cursor.parent

    # Present oldest -> latest to show evolution order.
    lineage.reverse()

    segments: List[str] = []
    for idx, history_node in enumerate(lineage, start=1):
        segment = MUTATION_TRACES_PROMPT_SEGMENT_V1
        segment = segment.replace("#PROMPT_NUMBER#", str(idx))
        segment = segment.replace("#PROMPT#", history_node.inference_prompt)
        segment = segment.replace("#SCORE#", _format_node_score(history_node))
        segments.append(segment.strip())
    return "\n\n".join(segments)


def _build_inference_prompt_traces_with_differences(
    node: GraphNode, max_depth: int = 3
) -> str:
    lineage: List[GraphNode] = []
    cursor: Optional[GraphNode] = node
    while cursor is not None and len(lineage) < max_depth:
        lineage.append(cursor)
        cursor = cursor.parent

    # Present oldest -> latest to show evolution order.
    lineage.reverse()

    segments: List[str] = []
    for idx, history_node in enumerate(lineage, start=1):
        segment = MUTATION_TRACES_DIFFERENCES_PROMPT_SEGMENT_V1
        segment = segment.replace("#PROMPT_NUMBER#", str(idx))

        if idx == 1:
            prompt_content = history_node.inference_prompt
        else:
            prompt_content = (getattr(history_node, "differentiation", "") or "").strip()

        segment = segment.replace("#PROMPT_CONTENT#", prompt_content)
        segment = segment.replace("#SCORE#", _format_node_score(history_node))
        segments.append(segment.strip())
    return "\n\n".join(segments)


def _should_use_difference_traces(mutation_prompt_key: str) -> bool:
    return "diff" in mutation_prompt_key


def mutate_prompt_fn(
    node: GraphNode,
    feedback_samples: FeedbackSamples,
    *,
    dataset_type: str,
    model,
    tokenizer,
    max_new_tokens: int = 5000,
    do_sample: bool = True,
    prompt_open_tag: str = "<p>",
    prompt_close_tag: str = "</p>",
    mutation_prompt_override: Optional[str] = None,
    mutation_prompt_key_override: Optional[str] = None,
) -> Optional[Tuple[str, str, str, str, str, str]]:
    base_prompt = mutation_prompt_override or node.mutation_prompt

    samples = feedback_samples.selected_samples
    feedback_texts = _pad_list(
        [getattr(sample, "feedback_text", "") for sample in feedback_samples.selected_samples],
        3,
    )

    def get_attr(i: int, name: str) -> str:
        if i >= len(samples):
            return ""
        return getattr(samples[i], name, "")

    prompt = base_prompt
    prompt = prompt.replace("#INFERENCE_PROMPT#", node.inference_prompt)
    if _should_use_difference_traces(mutation_prompt_key_override or ""):
        traces = _build_inference_prompt_traces_with_differences(node)
    else:
        traces = _build_inference_prompt_traces(node)
    prompt = prompt.replace("#INFERENCE_PROMPT_TRACES#", traces)

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

    print(f"[agent_mutate_prompt] mutation prompt:\n{prompt}")

    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        raw_response = run_prompt(
            prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        tag_pattern = rf"{re.escape(prompt_open_tag)}(.*?){re.escape(prompt_close_tag)}"
        if not re.search(tag_pattern, raw_response, flags=re.DOTALL | re.IGNORECASE):
            print(
                f"[agent_mutate_prompt] missing {prompt_open_tag} tag; retry {attempt}/{max_attempts}"
            )
            continue
        candidate = extract_tagged_text(raw_response, prompt_open_tag, prompt_close_tag)
        (
            raw_differentiation_response,
            core_difference,
            differentiate_prompt,
        ) = differentiate_prompts(
            node.inference_prompt,
            candidate,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        if node.inference_mode != INFERENCE_MODE_NON_SEPARATE:
            print(f"[agent_mutate_prompt] new inference prompt:\n{candidate}")
            return (
                candidate,
                raw_response,
                prompt,
                raw_differentiation_response,
                differentiate_prompt,
                core_difference,
            )

        if _contains_placeholders(candidate, INFERENCE_PROMPT_PLACEHODERS_V1):
            print(f"[agent_mutate_prompt] new inference prompt:\n{candidate}")
            return (
                candidate,
                raw_response,
                prompt,
                raw_differentiation_response,
                differentiate_prompt,
                core_difference,
            )

        print(f"[agent_mutate_prompt] missing placeholders; retry {attempt}/{max_attempts}")

    node.is_dead = True
    print("[agent_mutate_prompt] node marked dead (too many placeholder failures)")
    return None
