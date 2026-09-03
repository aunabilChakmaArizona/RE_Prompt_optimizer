"""Summarize QA second-stage stable validation performance."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from prompt_optimization.run_io import DEFAULT_OUTPUT_ROOT


SECOND_STAGE_METHODS = {
    "lpo",
    "greater",
    "greater_tg",
    "gradpo_gen",
    "gradpo_prob",
    "gradpo_gen_random",
}


def parse_args() -> argparse.Namespace:
    """Read the optimizer output root and destination report path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("qa_prompt_optimization_stats.txt"),
    )
    return parser.parse_args()


def load_summaries(output_root: Path) -> list[dict[str, Any]]:
    """Load completed second-stage summary files under the experiment root."""
    summaries = []
    if not output_root.exists():
        return summaries
    for path in output_root.rglob("summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("optimizer") not in SECOND_STAGE_METHODS:
            continue
        payload["summary_path"] = str(path.resolve())
        summaries.append(payload)
    return summaries


def average(values: Sequence[float]) -> float | None:
    """Return a mean or None for an empty numeric sequence."""
    return mean(values) if values else None


def format_value(value: float | None, digits: int = 3) -> str:
    """Format optional floating-point table values consistently."""
    return "--" if value is None else f"{value:.{digits}f}"


def model_short_name(model_id: str) -> str:
    """Shorten known target model IDs for compact report tables."""
    normalized = model_id.casefold()
    if "qwen" in normalized:
        return "Qwen3-4B"
    if "gemma" in normalized:
        return "Gemma3-4B"
    return model_id


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a plain Markdown table suitable for a paper-editing LLM."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def group_rows(summaries: Sequence[dict[str, Any]]) -> list[list[Any]]:
    """Aggregate raw and stable validation gains by mode, model, and method."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        key = (
            str(summary["qa_mode"]),
            model_short_name(str(summary["model"])),
            str(summary.get("backend", "transformers")),
            str(summary["optimizer"]),
        )
        grouped[key].append(summary)
    rows = []
    for (mode, model, backend, method), items in sorted(grouped.items()):
        changed = [item for item in items if item.get("changed")]
        gains = [float(item["validation"]["accuracy_gain"]) for item in items]
        stable_gains = [
            float(
                item["validation"].get(
                    "stable_accuracy_gain",
                    item["validation"]["accuracy_gain"],
                )
            )
            for item in items
        ]
        changed_gains = [
            float(item["validation"]["accuracy_gain"]) for item in changed
        ]
        changed_stable_gains = [
            float(
                item["validation"].get(
                    "stable_accuracy_gain",
                    item["validation"]["accuracy_gain"],
                )
            )
            for item in changed
        ]
        rows.append(
            [
                mode,
                model,
                backend,
                method,
                len(items),
                sum(gain > 0 for gain in stable_gains),
                len(changed),
                format_value(100.0 * average(gains) if gains else None),
                format_value(
                    100.0 * average(stable_gains) if stable_gains else None
                ),
                format_value(100.0 * average(changed_gains) if changed_gains else None),
                format_value(
                    100.0 * average(changed_stable_gains)
                    if changed_stable_gains
                    else None
                ),
            ]
        )
    return rows


def build_report(summaries: Sequence[dict[str, Any]], output_root: Path) -> str:
    """Build the complete human- and LLM-readable QA experiment report."""
    headers = [
        "QA mode",
        "Model",
        "Backend",
        "Method",
        "Attempts",
        "Dev improved",
        "Changed",
        "Avg raw dev gain all (pp)",
        "Avg stable dev gain all (pp)",
        "Avg raw dev gain changed (pp)",
        "Avg stable dev gain changed (pp)",
    ]
    sections = [
        "OpenBookQA two-stage prompt-optimization statistics",
        "=" * 56,
        "",
        f"Output root: {output_root.resolve()}",
        f"Completed second-stage attempts found: {len(summaries)}",
        "Expected full matrix: 120 attempts (20 first-stage starts x 6 refiners).",
        "",
        "Notes:",
        "- Every completed second-stage attempt is counted, including development failures.",
        "- A development failure retains the first-stage prompt.",
        "- Development improvement is decided by mean fold accuracy minus lambda times population standard deviation.",
        "- Test evaluation is intentionally run later with the separate final-test runner.",
        "- GradPO-Gen-Random changes region selection only; generation and beam search match GradPO-Gen.",
        "",
        "Aggregate results",
        "-----------------",
        markdown_table(headers, group_rows(summaries)),
        "",
        "Source summary files",
        "--------------------",
        *[f"- {item['summary_path']}" for item in sorted(summaries, key=lambda value: value["summary_path"])],
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    """Load completed runs and save their aggregate statistics report."""
    args = parse_args()
    summaries = load_summaries(args.output_root.expanduser())
    report = build_report(summaries, args.output_root.expanduser())
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report to: {args.report_file.resolve()}")


if __name__ == "__main__":
    main()
