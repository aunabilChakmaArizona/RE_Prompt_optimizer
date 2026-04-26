#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
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
    parser.add_argument(
        "--bootstrap-min-gain",
        type=float,
        default=0.002,
        help="Minimum useful raw F1 gain for bootstrap stopping.",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.05,
        help="Bootstrap confidence tail probability.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples per stopping check.",
    )
    parser.add_argument(
        "--bootstrap-rule",
        choices=["upper", "lower", "majority"],
        default="upper",
        help=(
            "Bootstrap stop rule. upper: stop when the upper confidence bound is "
            "below min gain. lower: stop when the lower confidence bound is below "
            "min gain. majority: stop when fewer than half of bootstrap means are "
            "at least min gain."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=12345,
        help="Random seed used for deterministic bootstrap resampling.",
    )
    parser.add_argument(
        "--compare-bootstrap-rules",
        action="store_true",
        help="Report both upper and lower bootstrap stop rules in the same output.",
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


def _bootstrap_mean_gain_stats(
    gains: list[float],
    min_gain: float,
    alpha: float,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    window = len(gains)
    bootstrap_means = [
        sum(rng.choice(gains) for _ in range(window)) / window
        for _ in range(samples)
    ]
    bootstrap_means.sort()
    lower_index = int(alpha * (samples - 1))
    upper_index = int((1.0 - alpha) * (samples - 1))
    lower_bound = bootstrap_means[lower_index]
    upper_bound = bootstrap_means[upper_index]
    prob_good = sum(mean >= min_gain for mean in bootstrap_means) / samples
    return lower_bound, upper_bound, prob_good


def _find_bootstrap_stop_iteration(
    scores: list[tuple[int, float]],
    window: int,
    min_gain: float,
    alpha: float,
    patience: int,
    samples: int,
    rule: str,
    seed: int,
    start_iteration: int,
) -> tuple[int, float, float, float, float] | None:
    raw_scores = [score for _, score in scores]
    stall_count = 0

    for t in range(window, len(raw_scores)):
        iteration_number = scores[t][0]
        if iteration_number < start_iteration:
            continue

        gains = [
            raw_scores[i] - raw_scores[i - 1]
            for i in range(t - window + 1, t + 1)
        ]
        avg_gain = sum(gains) / window
        lower_bound, upper_bound, prob_good = _bootstrap_mean_gain_stats(
            gains,
            min_gain=min_gain,
            alpha=alpha,
            samples=samples,
            seed=seed + iteration_number,
        )

        if rule == "upper":
            stalled = upper_bound < min_gain
        elif rule == "lower":
            stalled = lower_bound < min_gain
        elif rule == "majority":
            stalled = prob_good < 0.5
        else:
            raise ValueError(f"Unsupported bootstrap rule: {rule}")

        if stalled:
            stall_count += 1
        else:
            stall_count = 0

        if stall_count >= patience:
            return iteration_number, avg_gain, lower_bound, upper_bound, prob_good

    return None


def _format_top_prompt_line(
    label: str,
    iteration_number: int,
    population_by_id: dict[int, dict],
) -> str | None:
    top_prompt = _top_prompt_at_iteration(iteration_number, population_by_id)
    if top_prompt is None:
        return None
    top_node_id, top_f1 = top_prompt
    return f"{label}: node_id={top_node_id}, f1={top_f1 * 100:.2f}"


def _append_threshold_stop_lines(
    lines: list[str],
    stop: tuple[int, float] | None,
    population_by_id: dict[int, dict],
) -> None:
    if stop is None:
        lines.append("Threshold Stop: not triggered")
        return

    stop_iteration, avg_gain = stop
    lines.append(
        f"Threshold Stop at iteration {stop_iteration} "
        f"(last window avg gain={avg_gain:.6f})"
    )
    top_prompt_line = _format_top_prompt_line(
        "Top Prompt at Threshold Stop",
        stop_iteration,
        population_by_id,
    )
    if top_prompt_line is not None:
        lines.append(top_prompt_line)


def _append_bootstrap_stop_lines(
    lines: list[str],
    label: str,
    bootstrap_stop: tuple[int, float, float, float, float] | None,
    population_by_id: dict[int, dict],
) -> None:
    if bootstrap_stop is None:
        lines.append(f"{label}: not triggered")
        return

    (
        bootstrap_iteration,
        bootstrap_avg_gain,
        lower_bound,
        upper_bound,
        prob_good,
    ) = bootstrap_stop
    lines.append(
        f"{label} at iteration {bootstrap_iteration} "
        f"(avg_gain={bootstrap_avg_gain:.6f}, "
        f"lower={lower_bound:.6f}, upper={upper_bound:.6f}, "
        f"prob_gain_at_least_min={prob_good:.3f})"
    )
    top_prompt_line = _format_top_prompt_line(
        f"Top Prompt at {label}",
        bootstrap_iteration,
        population_by_id,
    )
    if top_prompt_line is not None:
        lines.append(top_prompt_line)


def _render_population(
    run_dir: str,
    window: int,
    epsilon: float,
    patience: int,
    bootstrap_min_gain: float,
    bootstrap_alpha: float,
    bootstrap_samples: int,
    bootstrap_rule: str,
    bootstrap_seed: int,
    compare_bootstrap_rules: bool,
) -> list[str]:
    run_path = Path(run_dir).resolve()
    payload = _read_json(run_path / "population.json")
    summary = _read_json(run_path / "summary.json")
    population = payload.get("population", [])
    population_by_id = {int(node["node_id"]): node for node in population}
    population_size = int(summary["population_size"])
    scores = _cumulative_scores(population_size, population_by_id)
    stop = _find_stop_iteration(scores, window, epsilon, patience)
    bootstrap_start_iteration = population_size + window
    if compare_bootstrap_rules:
        bootstrap_stops = {
            rule: _find_bootstrap_stop_iteration(
                scores,
                window=window,
                min_gain=bootstrap_min_gain,
                alpha=bootstrap_alpha,
                patience=patience,
                samples=bootstrap_samples,
                rule=rule,
                seed=bootstrap_seed,
                start_iteration=bootstrap_start_iteration,
            )
            for rule in ("upper", "lower")
        }
    else:
        bootstrap_stops = {
            bootstrap_rule: _find_bootstrap_stop_iteration(
                scores,
                window=window,
                min_gain=bootstrap_min_gain,
                alpha=bootstrap_alpha,
                patience=patience,
                samples=bootstrap_samples,
                rule=bootstrap_rule,
                seed=bootstrap_seed,
                start_iteration=bootstrap_start_iteration,
            )
        }

    lines: list[str] = [str(run_path)]
    lines.append("Inference Method: avg F1 of top validation-F1 prompts generated so far")
    lines.append(f"Population Size: {population_size}")
    lines.append(f"Threshold Stop Rule: window={window}, epsilon={epsilon}, patience={patience}")
    if compare_bootstrap_rules:
        lines.append(
            "Bootstrap Stop Rules: "
            f"rules=upper,lower, window={window}, min_gain={bootstrap_min_gain}, "
            f"alpha={bootstrap_alpha}, patience={patience}, samples={bootstrap_samples}, "
            f"start_iteration={bootstrap_start_iteration}"
        )
    else:
        lines.append(
            "Bootstrap Stop Rule: "
            f"rule={bootstrap_rule}, window={window}, min_gain={bootstrap_min_gain}, "
            f"alpha={bootstrap_alpha}, patience={patience}, samples={bootstrap_samples}, "
            f"start_iteration={bootstrap_start_iteration}"
        )
    lines.append("Cumulative Avg F1 Scores:")
    for iteration_number, score in scores:
        lines.append(f"Iteration {iteration_number}: {score * 100:.2f}")

    _append_threshold_stop_lines(lines, stop, population_by_id)
    if compare_bootstrap_rules:
        _append_bootstrap_stop_lines(
            lines,
            "Bootstrap Upper Stop",
            bootstrap_stops["upper"],
            population_by_id,
        )
        _append_bootstrap_stop_lines(
            lines,
            "Bootstrap Lower Stop",
            bootstrap_stops["lower"],
            population_by_id,
        )
    else:
        _append_bootstrap_stop_lines(
            lines,
            "Bootstrap Stop",
            bootstrap_stops[bootstrap_rule],
            population_by_id,
        )
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
                bootstrap_min_gain=args.bootstrap_min_gain,
                bootstrap_alpha=args.bootstrap_alpha,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_rule=args.bootstrap_rule,
                bootstrap_seed=args.bootstrap_seed,
                compare_bootstrap_rules=args.compare_bootstrap_rules,
            )
        )
    print("\n".join(output_lines).rstrip())


if __name__ == "__main__":
    main()
