#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_POPULATION_DIRS = [
    "trainings/20260422_202830_Qwen-Qwen3-4B",
    "trainings/20260422_202858_Qwen-Qwen3-4B",
    "trainings/20260422_202907_Qwen-Qwen3-4B",
    "trainings/20260422_202923_Qwen-Qwen3-4B",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report cumulative score-based average validation F1 per iteration "
            "and detect stalls."
        )
    )
    parser.add_argument(
        "population_dirs",
        nargs="*",
        default=DEFAULT_POPULATION_DIRS,
        help=(
            "Training run directories containing population.json and summary.json. "
            "Defaults to the four 20260422 Qwen3-4B runs."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Number of most recent score improvements to average.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.002,
        help="Raw F1 average-gain threshold below which a window counts as stalled.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of consecutive stalled windows required before stopping.",
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


def _top_prompt_at_iteration(
    iteration_number: int,
    population_by_id: dict[int, dict],
) -> tuple[int, float] | None:
    generated_nodes = [
        node
        for node_id, node in population_by_id.items()
        if node_id <= iteration_number and _node_f1_mean(node) is not None
    ]
    if not generated_nodes:
        return None
    top_node = min(
        generated_nodes,
        key=lambda node: (-float(_node_f1_mean(node)), int(node["node_id"])),
    )
    return int(top_node["node_id"]), float(_node_f1_mean(top_node))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _score_based_avg_f1(
    iteration_number: int,
    population_size: int,
    population_by_id: dict[int, dict],
) -> float | None:
    active_ids = _score_based_active_ids(iteration_number, population_size, population_by_id)
    values = [
        value
        for value in (_node_f1_mean(population_by_id[node_id]) for node_id in active_ids)
        if value is not None
    ]
    return _mean(values)


def _cumulative_scores(
    population_size: int,
    population_by_id: dict[int, dict],
) -> list[tuple[int, float]]:
    max_iteration = max(population_by_id)
    scores: list[tuple[int, float]] = []
    for iteration_number in range(0, max_iteration + 1):
        score = _score_based_avg_f1(iteration_number, population_size, population_by_id)
        if score is not None:
            scores.append((iteration_number, score))
    return scores


def _find_stop_iteration(
    scores: list[tuple[int, float]],
    window: int,
    epsilon: float,
    patience: int,
) -> tuple[int, float] | None:
    raw_scores = [score for _, score in scores]
    stall_count = 0

    for t in range(window, len(raw_scores)):
        improvements = [
            raw_scores[i] - raw_scores[i - 1]
            for i in range(t - window + 1, t + 1)
        ]
        avg_gain = sum(improvements) / window

        if avg_gain < epsilon:
            stall_count += 1
        else:
            stall_count = 0

        if stall_count >= patience:
            iteration_number = scores[t][0]
            return iteration_number, avg_gain

    return None


def _render_population(
    run_dir: str,
    window: int,
    epsilon: float,
    patience: int,
) -> list[str]:
    run_path = Path(run_dir).resolve()
    payload = _read_json(run_path / "population.json")
    summary = _read_json(run_path / "summary.json")
    population = payload.get("population", [])
    population_by_id = {int(node["node_id"]): node for node in population}
    population_size = int(summary["population_size"])
    scores = _cumulative_scores(population_size, population_by_id)
    stop = _find_stop_iteration(scores, window, epsilon, patience)

    lines: list[str] = [str(run_path)]
    lines.append("Inference Method: avg F1 of top validation-F1 prompts generated so far")
    lines.append(f"Population Size: {population_size}")
    lines.append(f"Stall Detection: window={window}, epsilon={epsilon}, patience={patience}")
    lines.append("Cumulative Avg F1 Scores:")
    for iteration_number, score in scores:
        lines.append(f"Iteration {iteration_number}: {score * 100:.2f}")

    if stop is None:
        lines.append("Stop: not triggered")
    else:
        stop_iteration, avg_gain = stop
        top_prompt = _top_prompt_at_iteration(stop_iteration, population_by_id)
        lines.append(
            f"Stop at iteration {stop_iteration} "
            f"(last window avg gain={avg_gain:.6f})"
        )
        if top_prompt is not None:
            top_node_id, top_f1 = top_prompt
            lines.append(f"Top Prompt at Stop: node_id={top_node_id}, f1={top_f1 * 100:.2f}")
    return lines


def main() -> None:
    args = _parse_args()
    output_lines: list[str] = []
    for index, population_dir in enumerate(args.population_dirs):
        if index:
            output_lines.append("")
            output_lines.append("=" * 80)
            output_lines.append("")
        output_lines.extend(
            _render_population(
                population_dir,
                window=args.window,
                epsilon=args.epsilon,
                patience=args.patience,
            )
        )
    print("\n".join(output_lines).rstrip())


if __name__ == "__main__":
    main()
