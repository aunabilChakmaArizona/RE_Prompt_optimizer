#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_POPULATION_DIRS = [
    "trainings/20260424_092931_Qwen-Qwen3-4B",
    "trainings/20260424_093047_Qwen-Qwen3-4B",
    "trainings/20260424_093215_Qwen-Qwen3-4B",
    "trainings/20260424_093253_Qwen-Qwen3-4B",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report dev F1 scores for generated prompt nodes and each node's "
            "recursive parent chain back to root node 0."
        )
    )
    parser.add_argument(
        "population_dirs",
        nargs="*",
        default=DEFAULT_POPULATION_DIRS,
        help=(
            "Training run directories containing population.json. Defaults to the "
            "four 20260424 Qwen3-4B runs used by the 80-prompt score-based report."
        ),
    )
    parser.add_argument(
        "--max-node",
        type=int,
        default=80,
        help="Highest generated prompt node_id to include. Node 0 is shown as baseline only.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_node_f1(node: dict | None) -> str:
    if node is None:
        return "NA"
    val_score = node.get("val_score") or {}
    mean = val_score.get("f1_mean")
    std = val_score.get("f1_std")
    if mean is None:
        return "NA"
    if std is None:
        return f"{float(mean) * 100:.2f}"
    return f"{float(mean) * 100:.2f} +/- {float(std) * 100:.2f}"


def _node_f1_mean(node: dict | None) -> float | None:
    if node is None:
        return None
    val_score = node.get("val_score") or {}
    mean = val_score.get("f1_mean")
    if mean is None:
        return None
    return float(mean)


def _parent_chain_to_root(node_id: int, population_by_id: dict[int, dict]) -> list[str]:
    node = population_by_id.get(node_id)
    if node is None:
        return ["missing-node"]

    chain: list[str] = []
    visited = {node_id}
    parent_id = node.get("parent_id")
    while parent_id is not None:
        try:
            parent_id_int = int(parent_id)
        except (TypeError, ValueError):
            chain.append(f"invalid-parent:{parent_id}")
            return chain

        if parent_id_int in visited:
            chain.append(f"cycle:{parent_id_int}")
            return chain

        chain.append(str(parent_id_int))
        if parent_id_int == 0:
            return chain

        visited.add(parent_id_int)
        parent_node = population_by_id.get(parent_id_int)
        if parent_node is None:
            chain.append(f"missing-parent:{parent_id_int}")
            return chain
        parent_id = parent_node.get("parent_id")

    return chain or ["root"]


def _render_population(run_dir: str, max_node: int) -> list[str]:
    run_path = Path(run_dir).resolve()
    payload = _read_json(run_path / "population.json")
    summary = _read_json(run_path / "summary.json")
    population = payload.get("population", [])
    population_by_id = {int(node["node_id"]): node for node in population}

    lines: list[str] = [str(run_path)]
    lines.append(f"Population Size: {summary.get('population_size', 'NA')}")
    lines.append(f"Prompt Node IDs: 1..{max_node}")
    lines.append(f"Node 0 Baseline Dev F1: {_format_node_f1(population_by_id.get(0))}")
    lines.append("Sort: descending dev_f1_mean, then ascending node_id")
    lines.append("Columns: rank | node_id | dev_f1_mean +/- std | parent_ids_to_root")
    node_ids = list(range(1, max_node + 1))
    sorted_node_ids = sorted(
        node_ids,
        key=lambda node_id: (
            _node_f1_mean(population_by_id.get(node_id)) is None,
            -(_node_f1_mean(population_by_id.get(node_id)) or 0.0),
            node_id,
        ),
    )
    for rank, node_id in enumerate(sorted_node_ids, start=1):
        node = population_by_id.get(node_id)
        parent_chain = " -> ".join(_parent_chain_to_root(node_id, population_by_id))
        lines.append(f"{rank} | {node_id} | {_format_node_f1(node)} | {parent_chain}")
    return lines


def main() -> None:
    args = _parse_args()
    output_lines: list[str] = []
    for index, population_dir in enumerate(args.population_dirs):
        if index:
            output_lines.append("")
            output_lines.append("=" * 80)
            output_lines.append("")
        output_lines.extend(_render_population(population_dir, args.max_node))
    print("\n".join(output_lines).rstrip())


if __name__ == "__main__":
    main()
