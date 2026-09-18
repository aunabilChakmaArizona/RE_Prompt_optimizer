"""Combine the Qwen and Gemma fixed-prompt OpenBookQA final-test tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from prompt_optimization.qa_task import REPO_ROOT, resolve_repo_path
from prompt_optimization.run_io import save_text
from run_openbookqa_final_test_evaluation import (
    DEFAULT_OUTPUT_ROOT, FINAL_CODES, MODEL_IDS, build_text_report,
)


DEFAULT_SUMMARIES = [
    DEFAULT_OUTPUT_ROOT / "non_reasoning" / "selected_prompts" / code / "summary.json"
    for code in FINAL_CODES.values()
]
DEFAULT_REPORT = REPO_ROOT / "experiment_tracking/final_test/openbookqa_final_test_results.txt"


def parse_args() -> argparse.Namespace:
    """Read completed model summaries and the central TXT destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="+", default=[str(path) for path in DEFAULT_SUMMARIES],
                        help="Completed Qwen and Gemma summary files; defaults match the two shell scripts.")
    parser.add_argument("--output", default=str(DEFAULT_REPORT),
                        help="Combined plain-text table stored in central experiment tracking.")
    return parser.parse_args()


def combine_reports(summaries: Sequence[dict[str, Any]]) -> str:
    """Check matching test protocols and include every row without selecting by test scores."""
    if len(summaries) != 2 or {item["family"] for item in summaries} != set(MODEL_IDS):
        raise ValueError("Provide exactly one completed Qwen summary and one completed Gemma summary.")
    if summaries[0]["protocol"] != summaries[1]["protocol"]:
        raise ValueError("Cannot combine test results with different datasets, prompts, seeds, or decoding limits.")
    if any(len(item["rows"]) != 36 for item in summaries):
        raise ValueError("Each model must contain all 36 selected method rows.")
    lines = [
        "OpenBookQA final test results: development-selected prompts only",
        "===============================================================",
        "Official 500-question test; non-reasoning; one decoding run per unique instruction.",
        "No prompt/configuration is selected or changed using these test results.",
        "Model-specific default sampling settings are preserved and shown below.",
        "Initial baseline + 5 first-stage prompts + 30 second-stage rows per model = 72 rows.",
        "",
    ]
    for summary in sorted(summaries, key=lambda item: list(MODEL_IDS).index(item["family"])):
        lines.extend([build_text_report(summary), ""])
    return "\n".join(lines)


def main() -> None:
    """Write the combined tracking table after both model test runs finish."""
    args = parse_args()
    paths = [resolve_repo_path(path) for path in args.summaries]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Run both final-test shell scripts first; missing completed summary: {path}")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    output = resolve_repo_path(args.output)
    save_text(output, combine_reports(summaries))
    print(f"Saved combined 72-row final-test report to: {output}")


if __name__ == "__main__":
    main()
