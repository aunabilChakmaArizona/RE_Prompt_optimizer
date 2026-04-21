from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

from agents.agent_graph_node import GraphNode
from agents.agent_feedback_samples import FeedbackSample, FeedbackSamples
from agents.agent_train_config import get_arg_defaults


class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        for stream in self._streams:
            isatty = getattr(stream, "isatty", None)
            if callable(isatty) and isatty():
                return True
        return False


def create_run_dir(trainings_dir: str, model_id: str) -> str:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = model_id.replace("/", "-")
    run_dir = os.path.join(trainings_dir, f"{run_stamp}_{model_tag}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_logging(run_dir: str) -> Tuple[object, object, object]:
    log_path = os.path.join(run_dir, "train.log")

    print(f"[agent_train_io] Logging file: {log_path}")

    log_file = open(log_path, "w", encoding="utf-8")
    stdout_tee = Tee(sys.stdout, log_file)
    stderr_tee = Tee(sys.stderr, log_file)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    return log_file, original_stdout, original_stderr


def restore_logging(
    log_file: object, original_stdout: object, original_stderr: object
) -> None:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def write_args(run_dir: str, args: object) -> None:
    full_args = get_arg_defaults()
    full_args.update(vars(args))
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as handle:
        json.dump(full_args, handle, indent=2)


def _resolve_population_path(source_path: str) -> str:
    population_path = os.path.abspath(source_path)
    if os.path.isdir(population_path):
        population_path = os.path.join(population_path, "population.json")
    if os.path.basename(population_path) != "population.json":
        raise ValueError(
            "--initial-prompt-source-path must point to a run directory or population.json"
        )
    if not os.path.isfile(population_path):
        raise FileNotFoundError(
            f"Could not find population.json at '{population_path}'"
        )
    return population_path


def load_initial_prompt_node(
    source_path: str,
    prompt_node_id: int | None,
    expected_inference_mode: str,
) -> Tuple[Dict[str, object], str]:
    population_path = _resolve_population_path(source_path)
    with open(population_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    population = payload.get("population", [])
    selected_node_id = (
        prompt_node_id if prompt_node_id is not None else payload.get("best_node_id")
    )
    selected_node = None
    for node in population:
        if node.get("node_id") == selected_node_id:
            selected_node = node
            break

    if selected_node is None and prompt_node_id is None:
        selected_node = payload.get("best_node")

    if selected_node is None:
        raise KeyError(
            f"Could not find node_id={selected_node_id} in '{population_path}'"
        )

    node_inference_mode = selected_node.get("inference_mode")
    if node_inference_mode != expected_inference_mode:
        raise ValueError(
            "Initial prompt inference_mode mismatch: "
            f"source={node_inference_mode!r}, requested={expected_inference_mode!r}"
        )

    return selected_node, population_path


def serialize_feedback_sample(sample: FeedbackSample) -> Dict[str, object]:
    return {
        "id_1shot": sample.id_1shot,
        "id_query": sample.id_query,
        "inference": sample.inference,
        "label": sample.label,
        "relation": sample.relation,
        "support_sentence": sample.support_sentence,
        "query_sentence": sample.query_sentence,
        "feedback_prompt": sample.feedback_prompt,
        "raw_feedback_text": sample.raw_feedback_text,
        "feedback_text": sample.feedback_text,
    }


def serialize_feedback_samples(samples: FeedbackSamples) -> Dict[str, object]:
    return {
        "all_samples": [serialize_feedback_sample(s) for s in samples.all_samples],
        "selected_samples": [serialize_feedback_sample(s) for s in samples.selected_samples],
        "feedback_prompts": list(samples.feedback_prompts),
        "raw_feedback_texts": list(samples.raw_feedback_texts),
        "feedback_texts": list(samples.feedback_texts),
    }


def serialize_node(
    node: GraphNode,
    *,
    final_population_node_ids: Optional[Set[int]] = None,
) -> Dict[str, object]:
    data_entries: List[Dict[str, Optional[object]]] = []
    for feedback_samples, child in node.data:
        data_entries.append(
            {
                "feedback_samples": serialize_feedback_samples(feedback_samples),
                "child_id": child.node_id if child is not None else None,
            }
        )
    return {
        "node_id": node.node_id,
        "parent_id": node.parent.node_id if node.parent else None,
        "children_ids": [child.node_id for child in node.children],
        "inference_prompt": node.inference_prompt,
        "inference_mode": node.inference_mode,
        "inference_instruction_prompt": node.inference_instruction_prompt,
        "inference_answer_instruction_prompt": node.inference_answer_instruction_prompt,
        "inference_example_prompt": node.inference_example_prompt,
        "inference_input_prompt": node.inference_input_prompt,
        "is_dead": node.is_dead,
        "mutation_failures": node.mutation_failures,
        "feedback": node.feedback,
        "raw_feedback_texts": list(node.raw_feedback_texts),
        "feedback_prompts_used": list(node.feedback_prompts_used),
        "feedback_prompt": node.feedback_prompt,
        "mutation_prompt": node.mutation_prompt,
        "example_generation_prompt": node.example_generation_prompt,
        "mutation_prompt_used": node.mutation_prompt_used,
        "raw_mutation_response": node.raw_mutation_response,
        "differentiation_prompt_used": node.differentiation_prompt_used,
        "raw_differentiation_response": node.raw_differentiation_response,
        "differentiation": node.differentiation,
        "val_score": node.val_score,
        "test_score": node.test_score,
        "is_in_final_population": (
            node.node_id in final_population_node_ids
            if final_population_node_ids is not None and node.node_id is not None
            else False
        ),
        "data": data_entries,
    }


def save_population(
    run_dir: str,
    population: Iterable[GraphNode],
    best_node: GraphNode,
    final_population: Optional[Iterable[GraphNode]] = None,
) -> None:
    final_population_node_ids = {
        node.node_id for node in (final_population or []) if node.node_id is not None
    }
    population_data = [
        serialize_node(node, final_population_node_ids=final_population_node_ids)
        for node in population
    ]
    best_node_data = serialize_node(
        best_node,
        final_population_node_ids=final_population_node_ids,
    )
    with open(os.path.join(run_dir, "population.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_node_id": best_node.node_id,
                "final_population_node_ids": sorted(final_population_node_ids),
                "population": population_data,
                "best_node": best_node_data,
            },
            handle,
            indent=2,
        )


def save_summary(run_dir: str, summary: Dict[str, object]) -> None:
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def save_json(run_dir: str, filename: str, payload: Dict[str, object]) -> None:
    with open(os.path.join(run_dir, filename), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
