from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from agents.agent_graph_node import GraphNode
from agents.agent_prompts import INFERENCE_MODE_NON_SEPARATE
from agents.agent_sample_feedback import sample_feedback_fn
from agents.agent_select_feedback import select_feedback_samples


_DONE_RE = re.compile(
    r"precision=([0-9.]+).([0-9.]+), "
    r"recall=([0-9.]+).([0-9.]+), "
    r"f1=([0-9.]+).([0-9.]+)"
)
_ITER_END_RE = re.compile(r"\[agent_evolutionary_search\] iteration (\d+) end")
_PRUNED_RE = re.compile(r"removed node_ids=\[([^\]]*)\]")


@dataclass
class NohupResumeState:
    initial_population: List[GraphNode]
    initial_population_history: List[GraphNode]
    completed_iterations: int
    next_node_id: int
    source_path: str


def _parse_metric_line(line: str) -> Optional[dict]:
    match = _DONE_RE.search(line)
    if not match:
        return None
    precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std = (
        float(value) / 100.0 for value in match.groups()
    )
    return {
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
    }


def _extract_code_fence_after(text: str, marker: str) -> str:
    marker_index = text.rfind(marker)
    if marker_index < 0:
        return ""
    after_marker = text[marker_index + len(marker) :]
    fence_start = after_marker.find("```")
    if fence_start < 0:
        return ""
    content_start = fence_start + 3
    fence_end = after_marker.find("```", content_start)
    if fence_end < 0:
        return ""
    return after_marker[content_start:fence_end].strip()


def _extract_new_prompt(lines: List[str], start_index: int) -> str:
    prompt_lines: List[str] = []
    for line in lines[start_index + 1 :]:
        if line.startswith("[agent_evolutionary_search] evaluate child"):
            break
        if line.startswith("[agent_evaluate] evaluate_fn: insturction_prompt="):
            break
        prompt_lines.append(line)
    return "\n".join(prompt_lines).strip()


def _parse_removed_node_ids(line: str) -> List[int]:
    match = _PRUNED_RE.search(line)
    if not match:
        return []
    raw_ids = match.group(1).strip()
    if not raw_ids:
        return []
    return [int(part.strip()) for part in raw_ids.split(",") if part.strip()]


def load_nohup_resume_state(
    source_path: str,
    *,
    root: GraphNode,
    population_size: Optional[int],
    feedback_prompt: str,
    mutation_prompt: str,
    example_generation_prompt: str,
) -> NohupResumeState:
    with open(source_path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    root.val_score = None
    nodes_by_id: Dict[int, GraphNode] = {0: root}
    nodes_by_prompt: Dict[str, GraphNode] = {root.inference_prompt.strip(): root}
    active_node_ids = [0]
    completed_iterations = 0
    pending_parent_prompt = ""
    pending_child_prompt = ""

    for index, line in enumerate(lines):
        if line.startswith("[agent_mutate_prompt] mutation prompt:"):
            mutation_prompt_text: List[str] = []
            for mutation_line in lines[index + 1 :]:
                if mutation_line.startswith("[agent_mutate_prompt] new inference prompt:"):
                    break
                mutation_prompt_text.append(mutation_line)
            pending_parent_prompt = _extract_code_fence_after(
                "\n".join(mutation_prompt_text),
                "You are given below a prompt that is used by another LLM to make an inference for the task:",
            )

        elif line.startswith("[agent_mutate_prompt] new inference prompt:"):
            pending_child_prompt = _extract_new_prompt(lines, index)

        metric = _parse_metric_line(line)
        if metric is not None:
            if pending_child_prompt:
                node_id = len(nodes_by_id)
                parent = nodes_by_prompt.get(pending_parent_prompt.strip(), root)
                child_instruction_prompt = parent.inference_instruction_prompt
                if parent.inference_mode != INFERENCE_MODE_NON_SEPARATE:
                    child_instruction_prompt = pending_child_prompt
                child = GraphNode(
                    inference_prompt=pending_child_prompt,
                    inference_mode=parent.inference_mode,
                    inference_instruction_prompt=child_instruction_prompt,
                    inference_answer_instruction_prompt=parent.inference_answer_instruction_prompt,
                    inference_example_prompt=parent.inference_example_prompt,
                    inference_input_prompt=parent.inference_input_prompt,
                    parent=parent,
                    feedback_prompt=feedback_prompt,
                    mutation_prompt=mutation_prompt,
                    example_generation_prompt=example_generation_prompt,
                    node_id=node_id,
                    val_score=metric,
                )
                parent.children.append(child)
                nodes_by_id[node_id] = child
                nodes_by_prompt[child.inference_prompt.strip()] = child
                active_node_ids.append(node_id)
                pending_parent_prompt = ""
                pending_child_prompt = ""
            elif root.val_score is None:
                root.val_score = metric

        iter_end_match = _ITER_END_RE.search(line)
        if iter_end_match:
            completed_iterations = max(completed_iterations, int(iter_end_match.group(1)))

        if "[agent_evolutionary_search] pruned active population" in line:
            removed_ids = set(_parse_removed_node_ids(line))
            active_node_ids = [node_id for node_id in active_node_ids if node_id not in removed_ids]

    if root.val_score is None:
        raise ValueError(f"Could not parse root validation metrics from '{source_path}'")

    completed_child_ids = set(range(1, completed_iterations + 1))
    missing_child_ids = sorted(node_id for node_id in completed_child_ids if node_id not in nodes_by_id)
    if missing_child_ids:
        raise ValueError(
            f"Could not parse completed child node(s) {missing_child_ids} from '{source_path}'"
        )

    if population_size is not None and len(active_node_ids) > population_size:
        active_node_ids = active_node_ids[-population_size:]

    initial_population = [nodes_by_id[node_id] for node_id in active_node_ids]
    initial_population_history = [
        nodes_by_id[node_id] for node_id in sorted(nodes_by_id)
        if node_id <= completed_iterations
    ]
    return NohupResumeState(
        initial_population=initial_population,
        initial_population_history=initial_population_history,
        completed_iterations=completed_iterations,
        next_node_id=completed_iterations + 1,
        source_path=source_path,
    )


def catch_up_rng_from_log(
    rng,
    *,
    feedback_pool: dict,
    completed_iterations: int,
    feedback_sample_size: int,
    selection_mode: str,
    extra_draws_per_iteration: int,
) -> None:
    for _ in range(completed_iterations):
        feedback_samples = sample_feedback_fn(feedback_pool, k=feedback_sample_size, rng=rng)
        select_feedback_samples(
            feedback_samples,
            selection_mode=selection_mode,
            rng=rng,
        )
        for _ in range(max(0, extra_draws_per_iteration)):
            rng.random()
