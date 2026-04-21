#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read region-tune summary.json files and report the best retained beam "
            "prompt after the first iteration by highest dev-set F1."
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
        help="Optional expected lambda values. Missing runs will be marked in the report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output txt report.",
    )
    return parser.parse_args()


def _extract_lambda_from_name(path: Path) -> str | None:
    match = re.search(r"px_([0-9]+(?:\.[0-9]+)?)", path.name)
    if not match:
        if re.search(
            r"llm_cand_sugg_beam3real_20260301_030139_node11_region3s_qwen4_lambda_tune$",
            path.name,
        ):
            return "0.2"
        return None
    return match.group(1)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_first_iteration_generated_prompts(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    iterations = summary_payload.get("iterations")
    if isinstance(iterations, list) and iterations:
        first_iteration = iterations[0]
        generated_prompts = first_iteration.get("generated_prompts")
        if isinstance(generated_prompts, list):
            return generated_prompts
    generated_prompts = summary_payload.get("generated_prompts")
    if isinstance(generated_prompts, list):
        return generated_prompts
    return []


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _select_best_prompt(generated_prompts: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid_prompts = [
        prompt_payload
        for prompt_payload in generated_prompts
        if isinstance(prompt_payload, dict)
        and prompt_payload.get("prompt")
        and isinstance(prompt_payload.get("dev_prf"), dict)
        and prompt_payload["dev_prf"].get("f1") is not None
    ]
    if not valid_prompts:
        return None
    return max(
        valid_prompts,
        key=lambda prompt_payload: (
            _safe_float(prompt_payload["dev_prf"].get("f1"), float("-inf")),
            -_safe_float(
                (prompt_payload.get("selection_metrics") or {}).get("combined_score"),
                float("inf"),
            ),
            -_safe_float(
                (prompt_payload.get("selection_metrics") or {}).get("mean_cross_entropy"),
                float("inf"),
            ),
            -_safe_float(
                (prompt_payload.get("selection_metrics") or {}).get("mean_perplexity"),
                float("inf"),
            ),
        ),
    )


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.2f}"


def _render_best_prompt_block(lambda_value: str, run_dir: Path | None) -> list[str]:
    lines: list[str] = []
    lines.append(f"Lambda px_{lambda_value}")
    lines.append(f"Run Dir: {run_dir if run_dir else 'MISSING'}")
    if run_dir is None:
        lines.append("Status: missing run directory")
        return lines

    summary_payload = _load_summary(run_dir)
    generated_prompts = _get_first_iteration_generated_prompts(summary_payload)
    best_prompt = _select_best_prompt(generated_prompts)
    lines.append(f"Generated Prompts in Iteration 1: {len(generated_prompts)}")
    if best_prompt is None:
        lines.append("Status: no valid prompt with dev_f1 found")
        return lines

    selection_metrics = best_prompt.get("selection_metrics") or {}
    dev_prf = best_prompt.get("dev_prf") or {}
    lines.append("Best Prompt by Highest Dev F1 After Iteration 1:")
    lines.append(
        "  "
        f"loss={_format_metric(selection_metrics.get('mean_cross_entropy'))} | "
        f"perplexity={_format_metric(selection_metrics.get('mean_perplexity'))} | "
        f"combined={_format_metric(selection_metrics.get('combined_score'))} | "
        f"dev_f1={_format_metric(dev_prf.get('f1'))}"
    )
    lines.append("  Prompt:")
    lines.append(best_prompt["prompt"])
    return lines


def _render_report(run_dirs: list[Path], expected_lambdas: list[str]) -> str:
    lambda_to_run: dict[str, Path] = {}
    for run_dir in run_dirs:
        lambda_value = _extract_lambda_from_name(run_dir)
        if lambda_value is None:
            continue
        lambda_to_run[lambda_value] = run_dir

    ordered_lambdas = list(
        dict.fromkeys(expected_lambdas + sorted(lambda_to_run.keys(), key=float))
    )
    lines: list[str] = []
    lines.append("Best Region-Tune Prompt After Iteration 1")
    lines.append("=========================================")
    lines.append("")
    for lambda_value in ordered_lambdas:
        lines.extend(_render_best_prompt_block(lambda_value, lambda_to_run.get(lambda_value)))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    run_dirs = [Path(run_dir).resolve() for run_dir in args.run_dir]
    report_text = _render_report(run_dirs, args.expected_lambda)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")
    print(f"saved report to {args.output}")


if __name__ == "__main__":
    main()
