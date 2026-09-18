"""Compare fresh OpenBookQA test scores with an earlier identical-prompt matrix."""

from __future__ import annotations

import argparse
import json
from statistics import fmean
from typing import Any

from prompt_optimization.qa_task import resolve_repo_path
from prompt_optimization.run_io import save_text
from run_openbookqa_final_test_evaluation import REFINERS, SOURCE_SLOTS


METHOD_NAMES = {
    "gradpo_gen": "GradPO-Gen", "gradpo_prob": "GradPO-Prob",
    "gradpo_gen_random": "GradPO-Gen-Random", "lpo": "LPO",
    "greater": "GreaTer", "greater_tg": "GreaTer-TG",
}


def parse_args() -> argparse.Namespace:
    """Read the completed old/new batch summaries, fresh solo check, and TXT path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-summary", required=True, help="Earlier 36-row batch summary.")
    parser.add_argument("--new-summary", required=True, help="Fresh 36-row batch summary.")
    parser.add_argument("--solo-summary", required=True, help="Fresh single-run initial-prompt summary.")
    parser.add_argument("--output", required=True, help="Separate TXT comparison destination.")
    return parser.parse_args()


def aligned_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Build a padded plain-text Markdown table without altering score values."""
    widths = [max(len(header), *(len(row[i]) for row in rows))
              for i, header in enumerate(headers)]
    lines = ["| " + " | ".join(value.ljust(width) for value, width in zip(headers, widths)) + " |",
             "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend("| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"
                 for row in rows)
    return lines


def checked_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Require the complete selected matrix and internally consistent 500-question scores."""
    expected = {"initial"}
    for source in SOURCE_SLOTS:
        expected.add(f"first_stage_{source}")
        expected.update(f"second_stage_{source}_{method}" for method in REFINERS)
    rows = summary["rows"]
    by_id = {row["row_id"]: row for row in rows}
    if len(rows) != 36 or set(by_id) != expected:
        raise ValueError("Each batch summary must contain the complete 36-row matrix.")
    for row in rows:
        if row["total"] != 500 or abs(row["accuracy"] - row["correct"] / 500) > 1e-10:
            raise ValueError(f"Invalid saved accuracy for {row['row_id']}.")
        parent = row["parent_row_id"]
        if row["row_id"] == "initial":
            expected_parent = ""
        elif row["stage"] == "first_stage":
            expected_parent = "initial"
        else:
            expected_parent = f"first_stage_{row['source']}"
        if parent != expected_parent:
            raise ValueError(f"Incorrect parent for {row['row_id']}.")
        if parent:
            gain = 100 * (row["accuracy"] - by_id[parent]["accuracy"])
            if abs(gain - row["gain_percentage_points"]) > 1e-8:
                raise ValueError(f"Invalid source-relative gain for {row['row_id']}.")
    return by_id


def build_comparison(previous: dict[str, Any], new: dict[str, Any],
                     solo: dict[str, Any]) -> str:
    """Compare exact selected prompts, preserving every attempted refinement and its own parent."""
    if previous["family"] != new["family"] or previous["model"] != new["model"]:
        raise ValueError("Compare the same target model only.")
    for field in ("task", "qa_mode", "test_sha256", "record_set_id", "test_size",
                  "run_count", "seed", "answer_instruction", "enable_thinking", "do_sample"):
        if previous["protocol"][field] != new["protocol"][field]:
            raise ValueError(f"Test identity/decoding mismatch: {field}.")
    if previous["decoding"] != new["decoding"]:
        raise ValueError("Sampling settings differ between the batches.")
    old_rows, new_rows = checked_rows(previous), checked_rows(new)
    for row_id, row in new_rows.items():
        for field in ("instruction_prompt", "prompt_sha256", "parent_row_id", "optimization_code"):
            if row[field] != old_rows[row_id][field]:
                raise ValueError(f"Selected prompt/source changed: {row_id}/{field}.")
    if (solo["model"] != new["model"] or solo["prompt"].strip() != new_rows["initial"]["instruction_prompt"]
            or solo["run_count"] != 1 or solo["test_size"] != 500
            or solo["runs"][0]["evaluation_seed"] != new_rows["initial"]["evaluation_seed"]):
        raise ValueError("The solo check does not match the new initial prompt/test seed.")
    solo_run = solo["runs"][0]
    lines = [f"OpenBookQA non-reasoning test rerun: {new['model']}",
             "==========================================================",
             f"Previous CODE: {previous['code']}", f"Fresh batch CODE: {new['code']}",
             f"Fresh solo CODE: {solo['code']}",
             f"Output limits: previous={previous['protocol']['max_new_tokens']}, fresh={new['protocol']['max_new_tokens']}",
             f"Sampling: {new['decoding']} | base seed={new['protocol']['seed']}",
             "Official 500-question test; one decoding run; strict tagged-answer accuracy.",
             "Accuracy change compares reruns; refinement gain compares each prompt with its own parent.",
             "All scores are percentages and changes/gains are percentage points (pp).",
             "Different output/context limits mean this is NOT a pure sampling-variance experiment.",
             "No stable test score, significance claim, or test-based prompt selection.", "",
             "Initial checks", "--------------"]
    checks = []
    for label, run in (("Previous batch", old_rows["initial"]), ("Fresh batch", new_rows["initial"])):
        checks.append([label, f"{100*run['accuracy']:.2f}", str(run["missing_answer_tags"]),
                       str(run["invalid_choice_labels"])])
    checks.append(["Fresh solo", f"{100*solo_run['accuracy']:.2f}",
                   str(solo_run["metrics"]["missing_answer_tags"]),
                   str(solo_run["metrics"]["invalid_choice_labels"])])
    lines.extend(aligned_table(["Evaluation", "Accuracy", "Missing tags", "Invalid labels"], checks))
    lines.extend(["", "Every selected first-/second-stage prompt", "----------------------------------------"])
    detailed = []
    for row in new["rows"]:
        if row["row_id"] == "initial":
            continue
        old = old_rows[row["row_id"]]
        method = "First stage" if row["stage"] == "first_stage" else METHOD_NAMES[row["method"]]
        detailed.append([row["source"], method, f"{100*old['accuracy']:.2f}",
                         f"{100*row['accuracy']:.2f}", f"{100*(row['accuracy']-old['accuracy']):+.2f}",
                         f"{old['gain_percentage_points']:+.2f}", f"{row['gain_percentage_points']:+.2f}",
                         str(old["missing_answer_tags"]), str(row["missing_answer_tags"])])
    lines.extend(aligned_table(["Source", "Method", "Old accuracy", "New accuracy", "Accuracy change",
                               "Old parent gain", "New parent gain", "Old missing", "New missing"], detailed))
    lines.extend(["", "Second-stage averages (all five sources, unchanged prompts included)",
                  "--------------------------------------------------------------------"])
    averages = []
    for method in REFINERS:
        ids = [f"second_stage_{source}_{method}" for source in SOURCE_SLOTS]
        old_accuracy = 100*fmean(old_rows[i]["accuracy"] for i in ids)
        new_accuracy = 100*fmean(new_rows[i]["accuracy"] for i in ids)
        same = sum(old_rows[i]["correct"] == new_rows[i]["correct"] for i in ids)
        higher = sum(old_rows[i]["correct"] < new_rows[i]["correct"] for i in ids)
        averages.append([METHOD_NAMES[method], "5", f"{old_accuracy:.2f}", f"{new_accuracy:.2f}",
                         f"{new_accuracy-old_accuracy:+.2f}",
                         f"{fmean(old_rows[i]['gain_percentage_points'] for i in ids):+.2f}",
                         f"{fmean(new_rows[i]['gain_percentage_points'] for i in ids):+.2f}",
                         f"{higher}/{same}/{5-higher-same}"])
    lines.extend(aligned_table(["Method", "N", "Old avg accuracy", "New avg accuracy", "Avg accuracy change",
                               "Old avg parent gain", "New avg parent gain", "Higher/same/lower"], averages))
    lines.extend(["", "Higher/same/lower refers to accuracy versus the previous test, NOT versus first stage.",
                  "Means use unrounded saved metrics. Initial-to-first-stage gains can include format compliance.",
                  "Experimental source slots keep their previously recorded actual iteration provenance.",
                  f"Fresh per-row prompt provenance and predictions: {new['run_dir']}/summary.json", ""])
    return "\n".join(lines)


def main() -> None:
    """Write a separate old/new score comparison after both fresh runs finish."""
    args = parse_args()
    paths = [resolve_repo_path(value) for value in (args.previous_summary, args.new_summary, args.solo_summary)]
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    output = resolve_repo_path(args.output)
    if output.exists():
        raise FileExistsError(f"Comparison already exists; choose a fresh output path: {output}")
    report = build_comparison(*summaries)
    report += "\nInput summaries:\n" + "\n".join(str(path) for path in paths) + "\n"
    save_text(output, report)
    print(f"Saved old/new test score comparison to: {output}", flush=True)


if __name__ == "__main__":
    main()
