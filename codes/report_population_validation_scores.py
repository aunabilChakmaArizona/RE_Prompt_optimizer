#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


PRUNE_RE = re.compile(r"pruned active population to \d+; removed node_ids=\[(\d+)\]")
ITER_END_RE = re.compile(r"iteration (\d+) end")
INITIAL_SOURCE_RE = re.compile(r"initial population source: (.+/population\.json)")
INITIAL_NODE_IDS_RE = re.compile(r"initial population node_ids: \[([^\]]*)\]")
RESUMED_NEXT_NODE_ID_RE = re.compile(r"resumed next_node_id: (\d+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report score-only population summaries every 10 iterations."
    )
    parser.add_argument(
        "population_dirs",
        nargs="+",
        help="Training run directories containing population.json and train.log.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _node_f1_mean(node: dict) -> float | None:
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


def _parse_log(log_path: Path) -> dict:
    initial_source_dir: Path | None = None
    initial_active_ids: list[int] | None = None
    resumed_next_node_id: int | None = None
    removed_ids: list[int] = []
    max_local_iteration = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if initial_source_dir is None:
                match = INITIAL_SOURCE_RE.search(line)
                if match:
                    initial_source_dir = Path(match.group(1)).resolve().parent
            if initial_active_ids is None:
                match = INITIAL_NODE_IDS_RE.search(line)
                if match:
                    raw_ids = match.group(1).strip()
                    initial_active_ids = [
                        int(value.strip()) for value in raw_ids.split(",") if value.strip()
                    ]
            if resumed_next_node_id is None:
                match = RESUMED_NEXT_NODE_ID_RE.search(line)
                if match:
                    resumed_next_node_id = int(match.group(1))
            match = PRUNE_RE.search(line)
            if match:
                removed_ids.append(int(match.group(1)))
            match = ITER_END_RE.search(line)
            if match:
                max_local_iteration = max(max_local_iteration, int(match.group(1)))

    return {
        "initial_source_dir": initial_source_dir,
        "initial_active_ids": initial_active_ids,
        "resumed_next_node_id": resumed_next_node_id,
        "removed_ids": removed_ids,
        "max_local_iteration": max_local_iteration,
    }


def _build_run_chain(final_run_dir: Path) -> list[dict]:
    chain: list[dict] = []
    current = final_run_dir.resolve()
    while True:
        log_info = _parse_log(current / "train.log")
        summary = _read_json(current / "summary.json")
        chain.append(
            {
                "run_dir": current,
                "population_size": int(summary["population_size"]),
                "initial_source_dir": log_info["initial_source_dir"],
                "initial_active_ids": log_info["initial_active_ids"],
                "resumed_next_node_id": log_info["resumed_next_node_id"],
                "removed_ids": log_info["removed_ids"],
                "max_local_iteration": log_info["max_local_iteration"],
            }
        )
        if log_info["initial_source_dir"] is None:
            break
        current = log_info["initial_source_dir"]
    chain.reverse()
    return chain


def _simulate_snapshots(chain: list[dict]) -> dict[int, list[int]]:
    active_ids = [0]
    next_node_id = 1
    overall_iteration = 0
    snapshots: dict[int, list[int]] = {}

    for index, run_info in enumerate(chain):
        if index > 0:
            expected_active = sorted(run_info["initial_active_ids"] or [])
            if expected_active:
                active_ids = expected_active[:]
            if run_info["resumed_next_node_id"] is not None:
                next_node_id = run_info["resumed_next_node_id"]

        remove_index = 0
        for local_iteration in range(1, run_info["max_local_iteration"] + 1):
            active_ids.append(next_node_id)
            next_node_id += 1
            if len(active_ids) > run_info["population_size"]:
                removed_id = run_info["removed_ids"][remove_index]
                remove_index += 1
                active_ids.remove(removed_id)
            overall_iteration += 1
            snapshots[overall_iteration] = sorted(active_ids)

    return snapshots


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
    lines.append(f"Improved Prompts After Node 0: {len(improved_nodes)}")
    lines.append(f"Improved Prompt Avg F1: {_format_mean_std(improved_values)}")
    lines.append(f"Surviving Population F1 Mean ± Std: {_format_mean_std(active_values)}")
    lines.append(f"Node 0: {_format_node_f1(node_zero)}")
    return lines


def _render_population(run_dir: str) -> list[str]:
    run_path = Path(run_dir).resolve()
    payload = _read_json(run_path / "population.json")
    summary = _read_json(run_path / "summary.json")
    population = payload.get("population", [])
    population_by_id = {int(node["node_id"]): node for node in population}

    chain = _build_run_chain(run_path)
    snapshots = _simulate_snapshots(chain)
    max_iteration = max(snapshots)
    milestone_iterations = list(range(10, max_iteration + 1, 10))

    lines: list[str] = [str(run_path)]
    lines.append(f"Population Size: {summary['population_size']}")
    for index, milestone in enumerate(milestone_iterations):
        if index:
            lines.append("")
        lines.extend(_render_iteration_block(milestone, snapshots[milestone], population_by_id))
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
