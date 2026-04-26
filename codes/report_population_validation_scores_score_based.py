#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report population summaries every 10 iterations using only validation "
            "scores to infer surviving prompts."
        )
    )
    parser.add_argument(
        "population_dirs",
        nargs="+",
        help="Training run directories containing population.json and summary.json.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _node_f1_mean(node: dict | None) -> float | None:
    if node is None:
        return None
    val_score = node.get("val_score") or {}
    value = val_score.get("f1_mean")
    if value is None:
        return None
    return float(value)


def _format_node_f1(node: dict | None) -> str:
    if node is None:
        return "NA"
    val_score = node.get("val_score") or {}
    mean = val_score.get("f1_mean")
    std = val_score.get("f1_std")
    if mean is None or std is None:
        return "NA"
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def _format_mean_std(values: list[float]) -> str:
    if not values:
        return "NA"
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_mean_difference(current_values: list[float], previous_values: list[float]) -> str:
    current_mean = _mean(current_values)
    previous_mean = _mean(previous_values)
    if current_mean is None or previous_mean is None:
        return "NA"
    return f"{(current_mean - previous_mean) * 100:.2f}"


def _format_survivor_scores(active_ids: list[int], population_by_id: dict[int, dict]) -> str:
    parts: list[str] = []
    for node_id in active_ids:
        node = population_by_id[node_id]
        f1_mean = _node_f1_mean(node)
        if f1_mean is None:
            parts.append(f"{node_id}: NA")
        else:
            parts.append(f"{node_id}: {f1_mean * 100:.2f}")
    return ", ".join(parts)


def _node_f1_values_for_ids(node_ids: range, population_by_id: dict[int, dict]) -> list[float]:
    values: list[float] = []
    for node_id in node_ids:
        value = _node_f1_mean(population_by_id.get(node_id))
        if value is not None:
            values.append(value)
    return values


def _score_based_active_ids(
    iteration_number: int,
    population_size: int,
    population_by_id: dict[int, dict],
) -> list[int]:
    generated_nodes = [
        node
        for node_id, node in population_by_id.items()
        if node_id <= iteration_number and _node_f1_mean(node) is not None
    ]
    ranked_nodes = sorted(
        generated_nodes,
        key=lambda node: (-float(_node_f1_mean(node)), int(node["node_id"])),
    )
    return sorted(int(node["node_id"]) for node in ranked_nodes[:population_size])


def _render_iteration_block(
    iteration_number: int,
    active_ids: list[int],
    population_by_id: dict[int, dict],
) -> list[str]:
    created_ids = sorted(node_id for node_id in population_by_id if node_id <= iteration_number)
    node_zero = population_by_id.get(0)
    node_zero_f1 = _node_f1_mean(node_zero)
    improved_nodes = [
        population_by_id[node_id]
        for node_id in created_ids
        if node_id != 0
        and node_zero_f1 is not None
        and _node_f1_mean(population_by_id[node_id]) is not None
        and _node_f1_mean(population_by_id[node_id]) > node_zero_f1
    ]
    improved_values = [
        value for value in (_node_f1_mean(node) for node in improved_nodes) if value is not None
    ]
    active_values = [
        value for value in (_node_f1_mean(population_by_id[node_id]) for node_id in active_ids)
        if value is not None
    ]

    lines: list[str] = []
    lines.append(f"Iteration {iteration_number}")
    lines.append("Summary")
    lines.append(f"Score-Based Surviving Node IDs: {active_ids}")
    lines.append(
        "Score-Based Surviving Node F1s: "
        f"{_format_survivor_scores(active_ids, population_by_id)}"
    )
    lines.append(f"Improved Prompts After Node 0: {len(improved_nodes)}")
    lines.append(f"Improved Prompt Avg F1: {_format_mean_std(improved_values)}")
    lines.append(f"Score-Based Surviving Population F1 Mean ± Std: {_format_mean_std(active_values)}")
    if iteration_number > 10:
        previous_node_ids = range(iteration_number - 19, iteration_number - 9)
        current_node_ids = range(iteration_number - 9, iteration_number + 1)
        previous_values = _node_f1_values_for_ids(previous_node_ids, population_by_id)
        current_values = _node_f1_values_for_ids(current_node_ids, population_by_id)
        lines.append(
            "Previous 10 Generated Prompts F1 Mean ± Std: "
            f"{_format_mean_std(previous_values)}"
        )
        lines.append(
            "Current 10 Generated Prompts F1 Mean ± Std: "
            f"{_format_mean_std(current_values)}"
        )
        lines.append(
            "Current - Previous 10 Generated Prompts Avg F1 Difference: "
            f"{_format_mean_difference(current_values, previous_values)}"
        )
    lines.append(f"Node 0: {_format_node_f1(node_zero)}")
    return lines


def _render_population(run_dir: str) -> list[str]:
    run_path = Path(run_dir).resolve()
    payload = _read_json(run_path / "population.json")
    summary = _read_json(run_path / "summary.json")
    population = payload.get("population", [])
    population_by_id = {int(node["node_id"]): node for node in population}
    max_iteration = max(population_by_id)
    milestone_iterations = list(range(10, max_iteration + 1, 10))
    population_size = int(summary["population_size"])

    lines: list[str] = [str(run_path)]
    lines.append("Inference Method: top validation F1 among prompts generated so far")
    lines.append(f"Population Size: {population_size}")
    for index, milestone in enumerate(milestone_iterations):
        if index:
            lines.append("")
        active_ids = _score_based_active_ids(milestone, population_size, population_by_id)
        lines.extend(_render_iteration_block(milestone, active_ids, population_by_id))
    return lines


def main() -> None:
    args = _parse_args()
    output_lines: list[str] = []
    for index, population_dir in enumerate(args.population_dirs):
        if index:
            output_lines.append("")
            output_lines.append("=" * 80)
            output_lines.append("")
        output_lines.extend(_render_population(population_dir))
    print("\n".join(output_lines).rstrip())


if __name__ == "__main__":
    main()
