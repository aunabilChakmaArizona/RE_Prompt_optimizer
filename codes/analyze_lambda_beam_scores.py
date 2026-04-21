#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read iterative beam-search summary.json files and write a plain-text "
            "report of loss, perplexity, and combined scores by lambda, iteration, "
            "and beam step."
        )
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Path to a run directory containing summary.json. Can be passed multiple times.",
    )
    parser.add_argument(
        "--expected-lambda",
        action="append",
        default=[],
        help=(
            "Optional lambda value expected in the analysis. If no matching run-dir is "
            "found, the report marks it as missing."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output .txt report.",
    )
    return parser.parse_args()


def _extract_lambda_from_name(path: Path) -> str | None:
    match = re.search(r"px_([0-9]+(?:\.[0-9]+)?)", path.name)
    if not match:
        return None
    return match.group(1)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_metric(node: dict[str, Any], key: str) -> str:
    selection_metrics = node.get("selection_metrics") or {}
    value = selection_metrics.get(key)
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _iter_iteration_entries(summary_payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    iterations = summary_payload.get("iterations", [])
    if isinstance(iterations, list) and iterations:
        return iterations
    return []


def _safe_prf_value(payload: dict[str, Any] | None, key: str = "f1") -> str:
    if not isinstance(payload, dict):
        return "NA"
    value = payload.get(key)
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _numeric_metric_from_payload(payload: dict[str, Any], key: str) -> float | None:
    metrics = payload.get("selection_metrics") or {}
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _numeric_prf_value(payload: dict[str, Any] | None, key: str = "f1") -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    return float(value)


def _format_mean_std(values: list[float]) -> str:
    if not values:
        return "NA"
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = math.sqrt(variance)
    return f"{mean_value:.2f} ± {std_value:.2f}"


def _render_step(step: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    step_index = step.get("step_index")
    region_rank = step.get("region_rank")
    lines.append(f"  Step {step_index} | Region {region_rank}")
    lines.append(
        "  Parents="
        f"{step.get('num_parent_nodes')} | "
        "Candidates/Parent="
        f"{step.get('num_candidates_considered_per_parent')} | "
        "Expanded Unique Prompts="
        f"{step.get('num_expanded_unique_prompts')}"
    )

    expanded_nodes = step.get("expanded_unique_nodes", [])
    for node in expanded_nodes:
        retained = "YES" if node.get("retained_in_beam") else "NO"
        lines.append(
            "    "
            f"Prompt Beam={node.get('beam_index')} | "
            f"Parent Beam={node.get('parent_beam_index')} | "
            f"Candidate Pos={node.get('candidate_position')} | "
            f"Changed Spans={node.get('num_changed_spans')} | "
            f"Retained={retained} | "
            f"Loss={_safe_metric(node, 'mean_cross_entropy')} | "
            f"Perplexity={_safe_metric(node, 'mean_perplexity')} | "
            f"Combined={_safe_metric(node, 'combined_score')}"
        )

    retained_nodes = step.get("retained_beam_nodes", [])
    if retained_nodes:
        lines.append("  Retained Beam Nodes:")
        for node in retained_nodes:
            lines.append(
                "    "
                f"Beam={node.get('beam_index')} | "
                f"Parent Beam={node.get('parent_beam_index')} | "
                f"Changed Spans={node.get('num_changed_spans')} | "
                f"Loss={_safe_metric(node, 'mean_cross_entropy')} | "
                f"Perplexity={_safe_metric(node, 'mean_perplexity')} | "
                f"Combined={_safe_metric(node, 'combined_score')}"
            )
    return lines


def _render_iteration(iteration_payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    iteration_index = iteration_payload.get("iteration_index")
    beam_width = iteration_payload.get("beam_width")
    selected_prompt = iteration_payload.get("selected_prompt") or {}
    lines.append(f"Iteration {iteration_index} | Beam Width={beam_width}")
    lines.append(
        "Selected Prompt: "
        f"beam_index={selected_prompt.get('beam_index')} | "
        f"strategy={selected_prompt.get('selection_strategy')} | "
        f"loss={_metric_from_payload(selected_prompt, 'mean_cross_entropy')} | "
        f"perplexity={_metric_from_payload(selected_prompt, 'mean_perplexity')} | "
        f"combined={_metric_from_payload(selected_prompt, 'combined_score')} | "
        f"dev_f1={_safe_prf_value(selected_prompt.get('dev_prf'))}"
    )
    retained_generated_prompts = iteration_payload.get("generated_prompts", [])
    if retained_generated_prompts:
        lines.append("Retained Final Beam Prompts:")
        for retained_rank, prompt_payload in enumerate(retained_generated_prompts):
            lines.append(
                "  "
                f"Rank={retained_rank} | "
                f"loss={_metric_from_payload(prompt_payload, 'mean_cross_entropy')} | "
                f"perplexity={_metric_from_payload(prompt_payload, 'mean_perplexity')} | "
                f"combined={_metric_from_payload(prompt_payload, 'combined_score')} | "
                f"dev_f1={_safe_prf_value(prompt_payload.get('dev_prf'))}"
            )
    for step in iteration_payload.get("beam_search_steps", []):
        lines.extend(_render_step(step))
    return lines


def _render_final_iteration_retained_beam(iteration_payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    retained_generated_prompts = iteration_payload.get("generated_prompts", [])
    if not retained_generated_prompts:
        return lines
    lines.append("Final Iteration Retained Beam Prompts:")
    for retained_rank, prompt_payload in enumerate(retained_generated_prompts):
        lines.append(
            "  "
            f"Rank={retained_rank} | "
            f"loss={_metric_from_payload(prompt_payload, 'mean_cross_entropy')} | "
            f"perplexity={_metric_from_payload(prompt_payload, 'mean_perplexity')} | "
            f"combined={_metric_from_payload(prompt_payload, 'combined_score')} | "
            f"dev_f1={_safe_prf_value(prompt_payload.get('dev_prf'))}"
        )
    return lines


def _render_iteration_retained_beam_summary(iteration_payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    iteration_index = iteration_payload.get("iteration_index")
    retained_generated_prompts = iteration_payload.get("generated_prompts", [])
    beam_width = iteration_payload.get("beam_width")
    lines.append(
        f"  Iteration {iteration_index} | Beam Width={beam_width} | "
        f"Retained={len(retained_generated_prompts)}"
    )
    if not retained_generated_prompts:
        lines.append("    No retained prompts")
        return lines
    loss_values = [
        value
        for value in (
            _numeric_metric_from_payload(prompt_payload, "mean_cross_entropy")
            for prompt_payload in retained_generated_prompts
        )
        if value is not None
    ]
    perplexity_values = [
        value
        for value in (
            _numeric_metric_from_payload(prompt_payload, "mean_perplexity")
            for prompt_payload in retained_generated_prompts
        )
        if value is not None
    ]
    combined_values = [
        value
        for value in (
            _numeric_metric_from_payload(prompt_payload, "combined_score")
            for prompt_payload in retained_generated_prompts
        )
        if value is not None
    ]
    dev_f1_values = [
        value
        for value in (
            _numeric_prf_value(prompt_payload.get("dev_prf"))
            for prompt_payload in retained_generated_prompts
        )
        if value is not None
    ]
    for retained_rank, prompt_payload in enumerate(retained_generated_prompts):
        lines.append(
            "    "
            f"Rank={retained_rank} | "
            f"loss={_metric_from_payload(prompt_payload, 'mean_cross_entropy')} | "
            f"perplexity={_metric_from_payload(prompt_payload, 'mean_perplexity')} | "
            f"combined={_metric_from_payload(prompt_payload, 'combined_score')} | "
            f"dev_f1={_safe_prf_value(prompt_payload.get('dev_prf'))}"
        )
    lines.append(
        "    Avg ± Std | "
        f"loss={_format_mean_std(loss_values)} | "
        f"perplexity={_format_mean_std(perplexity_values)} | "
        f"combined={_format_mean_std(combined_values)} | "
        f"dev_f1={_format_mean_std(dev_f1_values)}"
    )
    return lines


def _build_iteration_avg_summary_row(iteration_payload: dict[str, Any]) -> str:
    retained_generated_prompts = iteration_payload.get("generated_prompts", [])
    if not retained_generated_prompts:
        return "NA\tNA\tNA\tNA"

    loss_values = [
        value
        for value in (
            _numeric_metric_from_payload(prompt_payload, "mean_cross_entropy")
            for prompt_payload in retained_generated_prompts
        )
        if value is not None
    ]
    perplexity_values = [
        value
        for value in (
            _numeric_metric_from_payload(prompt_payload, "mean_perplexity")
            for prompt_payload in retained_generated_prompts
        )
        if value is not None
    ]
    combined_values = [
        value
        for value in (
            _numeric_metric_from_payload(prompt_payload, "combined_score")
            for prompt_payload in retained_generated_prompts
        )
        if value is not None
    ]
    dev_f1_values = [
        value
        for value in (
            _numeric_prf_value(prompt_payload.get("dev_prf"))
            for prompt_payload in retained_generated_prompts
        )
        if value is not None
    ]
    return "\t".join(
        [
            _format_mean_std(loss_values),
            _format_mean_std(perplexity_values),
            _format_mean_std(combined_values),
            _format_mean_std(dev_f1_values),
        ]
    )


def _metric_from_payload(payload: dict[str, Any], key: str) -> str:
    metrics = payload.get("selection_metrics") or {}
    value = metrics.get(key)
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _render_report(
    run_dirs: list[Path],
    expected_lambdas: list[str],
) -> str:
    lambda_to_run: dict[str, Path] = {}
    for run_dir in run_dirs:
        lambda_value = _extract_lambda_from_name(run_dir)
        if lambda_value is None:
            continue
        lambda_to_run[lambda_value] = run_dir

    ordered_lambdas = list(dict.fromkeys(expected_lambdas + sorted(lambda_to_run.keys(), key=float)))
    lines: list[str] = []
    lines.append("Compact Iteration Summary")
    lines.append("=========================")
    lines.append("")

    lines.append("Iteration Average Rows")
    lines.append("======================")
    lines.append("")

    for lambda_value in ordered_lambdas:
        run_dir = lambda_to_run.get(lambda_value)
        lines.append(f"Lambda px_{lambda_value}")
        lines.append(f"Run Dir: {run_dir if run_dir else 'MISSING'}")
        if run_dir is None:
            lines.append("NA\tNA\tNA\tNA")
            lines.append("")
            continue

        summary_payload = _load_summary(run_dir)
        iterations = list(_iter_iteration_entries(summary_payload))
        for iteration_payload in iterations:
            lines.append(_build_iteration_avg_summary_row(iteration_payload))
        lines.append("")

    for lambda_value in ordered_lambdas:
        run_dir = lambda_to_run.get(lambda_value)
        lines.append(f"Lambda px_{lambda_value}")
        lines.append(f"Run Dir: {run_dir if run_dir else 'MISSING'}")
        if run_dir is None:
            lines.append("Status: missing run directory")
            lines.append("")
            continue

        summary_payload = _load_summary(run_dir)
        iterations = list(_iter_iteration_entries(summary_payload))
        lines.append(f"Num Iterations: {summary_payload.get('num_iterations')}")
        for iteration_payload in iterations:
            lines.extend(_render_iteration_retained_beam_summary(iteration_payload))
        lines.append("")

    lines.append("Lambda Beam Score Analysis")
    lines.append("==========================")
    lines.append("")

    for lambda_value in ordered_lambdas:
        run_dir = lambda_to_run.get(lambda_value)
        lines.append(f"Lambda px_{lambda_value}")
        lines.append(f"Run Dir: {run_dir if run_dir else 'MISSING'}")
        if run_dir is None:
            lines.append("Status: missing run directory")
            lines.append("")
            continue

        summary_payload = _load_summary(run_dir)
        iterations = list(_iter_iteration_entries(summary_payload))
        lines.append(f"Num Iterations: {summary_payload.get('num_iterations')}")
        lines.append("")
        for iteration_payload in iterations:
            lines.extend(_render_iteration(iteration_payload))
            lines.append("")
        if iterations:
            lines.extend(_render_final_iteration_retained_beam(iterations[-1]))
            lines.append("")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    run_dirs = [Path(run_dir).resolve() for run_dir in args.run_dir]
    report_text = _render_report(
        run_dirs=run_dirs,
        expected_lambdas=args.expected_lambda,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")
    print(f"saved report to {args.output}")


if __name__ == "__main__":
    main()
