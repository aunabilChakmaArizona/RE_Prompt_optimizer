"""Evaluate development-selected Qwen Math-500 prompts with one vLLM model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from agents.agent_decoding import model_default_sampling_parameters
from agents.agent_vllm_models import (
    load_vllm_model_and_tokenizer,
    shutdown_vllm_model,
    vllm_backend_metadata,
)
from agents.agent_vllm_prompting import run_prompts_vllm
from math_inference_common import ANSWER_INSTRUCTION_PROMPT, build_math_prompt, validate_math_records
from math_grading.graders import grader_metadata, validate_grading_dependencies
from run_math_test_inference import score_predictions, write_json, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiment_tracking/final_test/math500_qwen_selected_prompts.tsv"
DEFAULT_TEST_PATH = REPO_ROOT / "data/processed/math500/test.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/math_final_test/reasoning/selected_prompts"
MODEL_ID = "Qwen/Qwen3-4B"
EXPECTED_ROW_IDS = {
    "initial",
    "first_stage_rpo5",
    "second_stage_rpo5_gradpo_gen",
    "second_stage_rpo5_gradpo_prob",
    "second_stage_rpo5_gradpo_gen_random",
    "second_stage_rpo5_lpo",
    "second_stage_rpo5_greater",
    "first_stage_rpo10",
    "second_stage_rpo10_gradpo_gen",
    "second_stage_rpo10_gradpo_gen_random",
    "second_stage_rpo10_lpo",
    "second_stage_rpo10_greater",
    "second_stage_rpo10_greater_tg",
}


def parse_args() -> argparse.Namespace:
    """Read the frozen-manifest and five-seed Math-500 test settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, help="Unique final-test run identity.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="Frozen selected-prompt manifest with SHA-256 hashes.")
    parser.add_argument("--dataset", default=str(DEFAULT_TEST_PATH),
                        help="Prepared official MATH-500 test JSONL.")
    parser.add_argument("--model", default=MODEL_ID, help="Qwen target model ID.")
    parser.add_argument("--device", default="cuda:0", help="Logical CUDA device.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="Fraction of the assigned GPU reserved by vLLM.")
    parser.add_argument("--vllm-max-model-len", type=int, default=32768,
                        help="Maximum vLLM input-plus-output context length.")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Maximum reasoning and answer tokens per problem.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 100, 1000, 10000],
                        help="Independent base seeds evaluated and saved separately.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                        help="Parent directory for final-test outputs.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing run with the same CODE.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate paths, hashes, and records without loading Qwen.")
    args = parser.parse_args()
    if args.model != MODEL_ID:
        parser.error(f"This frozen manifest requires --model {MODEL_ID}.")
    if args.max_new_tokens <= 0 or args.vllm_max_model_len <= args.max_new_tokens:
        parser.error("Context length must be greater than the positive output limit.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1].")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        parser.error("--seeds must be distinct non-negative integers.")
    return args


def resolve_path(value: str | Path) -> Path:
    """Resolve repository-relative paths without depending on the launch directory."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON objects from one JSONL file."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_selected_prompts(path_value: str | Path) -> list[dict[str, str]]:
    """Verify the intended improved-only matrix and load exact prompt text."""
    path = resolve_path(path_value)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream, delimiter="\t")]
    if len(rows) != len(EXPECTED_ROW_IDS) or {row["row_id"] for row in rows} != EXPECTED_ROW_IDS:
        raise ValueError("Math final-test manifest does not contain the expected 13 unique rows.")
    by_id = {row["row_id"]: row for row in rows}
    for row in rows:
        if row["family"] != "qwen":
            raise ValueError("The Math final-test manifest must contain only Qwen rows.")
        prompt_path = resolve_path(row["prompt_file"])
        raw = prompt_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["prompt_sha256"]:
            raise ValueError(f"Selected prompt changed after freezing: {prompt_path}")
        row["instruction_prompt"] = raw.decode("utf-8").strip()
        if not row["instruction_prompt"]:
            raise ValueError(f"Selected prompt is empty: {prompt_path}")
        parent = row["parent_row_id"]
        expected_parent = (
            f"first_stage_{row['source']}" if row["stage"] == "second_stage"
            else "initial" if row["stage"] == "first_stage" else ""
        )
        if parent != expected_parent or (parent and parent not in by_id):
            raise ValueError(f"Invalid parent link for {row['row_id']}.")
    return rows


def unique_instructions(rows: Sequence[dict[str, str]]) -> list[str]:
    """Evaluate identical instructions once while preserving manifest order."""
    return list(dict.fromkeys(row["instruction_prompt"] for row in rows))


def prepare_run_directory(root: Path, code: str, overwrite: bool) -> Path:
    """Create one protected final-test directory for this CODE."""
    run_dir = root / code
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Results already exist in {run_dir}; use a new CODE.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_result_rows(
    selected: Sequence[dict[str, str]],
    evaluations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map unique prompt evaluations to manifest rows and calculate parent gains."""
    by_prompt = {item["instruction_prompt"]: item for item in evaluations}
    rows = []
    for selected_row in selected:
        evaluation = by_prompt[selected_row["instruction_prompt"]]
        stats = evaluation["statistics"]
        rows.append({
            **selected_row,
            "statistics": stats,
            "openai_accuracy": float(stats["openai"]["accuracy"]),
            "simple_accuracy": float(stats["simple"]["accuracy"]),
            "math_verify_accuracy": float(stats["math_verify"]["accuracy"]),
            "evaluation_name": evaluation["evaluation_name"],
        })
    by_id = {row["row_id"]: row for row in rows}
    for row in rows:
        parent = row["parent_row_id"]
        row["openai_gain_percentage_points"] = (
            100.0 * (row["openai_accuracy"] - by_id[parent]["openai_accuracy"])
            if parent else None
        )
    return rows


def build_seed_report(summary: dict[str, Any]) -> str:
    """Render one seed's primary and secondary grader results."""
    lines = [
        f"Qwen MATH-500 final test | seed={summary['seed']}",
        f"CODE: {summary['code']}",
        "No stable-score penalty is applied to test results.",
        "",
        "| Row | OpenAI | Math-Verify | Simple | Gain vs parent | Missing tags |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        gain = row["openai_gain_percentage_points"]
        lines.append(
            f"| {row['row_id']} | {100.0 * row['openai_accuracy']:.2f} | "
            f"{100.0 * row['math_verify_accuracy']:.2f} | "
            f"{100.0 * row['simple_accuracy']:.2f} | "
            f"{'-' if gain is None else f'{gain:+.2f}'} | "
            f"{row['statistics']['missing_answer_tags']} |"
        )
    return "\n".join(lines) + "\n"


def aggregate_runs(run_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Calculate mean and sample standard deviation across completed seeds."""
    row_ids = [row["row_id"] for row in run_summaries[0]["rows"]]
    aggregate_rows = []
    for row_id in row_ids:
        seed_rows = [next(row for row in run["rows"] if row["row_id"] == row_id)
                     for run in run_summaries]
        aggregate = {key: seed_rows[0][key] for key in (
            "row_id", "stage", "source", "actual_iteration", "method",
            "optimization_code", "prompt_file", "prompt_sha256", "parent_row_id",
        )}
        for grader in ("openai", "math_verify", "simple"):
            values = [100.0 * row[f"{grader}_accuracy"] for row in seed_rows]
            aggregate[f"{grader}_mean"] = statistics.fmean(values)
            aggregate[f"{grader}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        gains = [row["openai_gain_percentage_points"] for row in seed_rows
                 if row["openai_gain_percentage_points"] is not None]
        aggregate["openai_gain_mean"] = statistics.fmean(gains) if gains else None
        aggregate_rows.append(aggregate)
    return {"run_count": len(run_summaries), "rows": aggregate_rows}


def build_aggregate_report(summary: dict[str, Any]) -> str:
    """Render mean and sample standard deviation without a test stable score."""
    lines = [
        f"Qwen MATH-500 final test | {summary['run_count']} seeds",
        "Values are mean +/- sample standard deviation in percentage points.",
        "",
        "| Row | OpenAI | Math-Verify | Simple | Mean gain vs parent |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        gain = row["openai_gain_mean"]
        lines.append(
            f"| {row['row_id']} | {row['openai_mean']:.2f} +/- {row['openai_std']:.2f} | "
            f"{row['math_verify_mean']:.2f} +/- {row['math_verify_std']:.2f} | "
            f"{row['simple_mean']:.2f} +/- {row['simple_std']:.2f} | "
            f"{'-' if gain is None else f'{gain:+.2f}'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Load Qwen once, evaluate all selected prompts for each seed, and save results."""
    args = parse_args()
    validate_grading_dependencies()
    selected = load_selected_prompts(args.manifest)
    instructions = unique_instructions(selected)
    dataset_path = resolve_path(args.dataset)
    records = read_jsonl(dataset_path)
    task_type = validate_math_records(records)
    if len(records) != 500 or task_type != "math_symbolic_answer":
        raise ValueError("Use the prepared 500-problem MATH-500 test split.")
    print(
        f"[math-final] rows={len(selected)} | unique_prompts={len(instructions)} | "
        f"problems={len(records)} | requests_per_seed={len(instructions) * len(records)} | "
        f"seeds={args.seeds}",
        flush=True,
    )
    if args.dry_run:
        print("Dry run passed: manifest hashes, parent links, and test records are valid.")
        return

    run_dir = prepare_run_directory(resolve_path(args.output_root), args.code, args.overwrite)
    write_json(run_dir / "config.json", {
        **vars(args),
        "dataset": str(dataset_path),
        "manifest_sha256": hashlib.sha256(resolve_path(args.manifest).read_bytes()).hexdigest(),
        "selected_rows": selected,
        "unique_prompt_count": len(instructions),
        "answer_instruction_prompt": ANSWER_INSTRUCTION_PROMPT,
        "decoding": model_default_sampling_parameters(args.model),
    })
    model, tokenizer = load_vllm_model_and_tokenizer(
        args.model,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
    )
    run_summaries = []
    started_at = time.monotonic()
    try:
        for run_index, seed in enumerate(args.seeds, start=1):
            print(f"[math-final] run {run_index}/{len(args.seeds)} started | seed={seed}", flush=True)
            flat_prompts = [
                build_math_prompt(instruction, record["question"])
                for instruction in instructions
                for record in records
            ]
            request_seeds = [seed + index for _ in instructions for index in range(len(records))]
            responses, token_usages = run_prompts_vllm(
                flat_prompts,
                model_id=args.model,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=True,
                do_sample=True,
                do_log=True,
                log_label=f"math500_final_seed_{seed}",
                return_token_usage=True,
                seeds=request_seeds,
            )
            seed_dir = run_dir / f"seed_{seed}"
            (seed_dir / "evaluations").mkdir(parents=True, exist_ok=True)
            (seed_dir / "prompts").mkdir(parents=True, exist_ok=True)
            evaluations = []
            for prompt_index, instruction in enumerate(instructions, start=1):
                start = (prompt_index - 1) * len(records)
                end = start + len(records)
                predictions, stats = score_predictions(
                    records, responses[start:end], token_usages[start:end], task_type
                )
                name = f"test_prompt_{prompt_index:02d}"
                write_jsonl(seed_dir / "evaluations" / f"{name}_predictions.jsonl", predictions)
                write_json(seed_dir / "evaluations" / f"{name}_summary.json", {
                    "instruction_prompt": instruction,
                    "statistics": stats,
                })
                (seed_dir / "prompts" / f"{name}.txt").write_text(
                    instruction + "\n", encoding="utf-8"
                )
                evaluations.append({
                    "evaluation_name": name,
                    "instruction_prompt": instruction,
                    "statistics": stats,
                })
            seed_summary = {
                "code": f"{args.code}_seed{seed}",
                "seed": seed,
                "model": args.model,
                "dataset": str(dataset_path),
                "task_type": task_type,
                "graders": grader_metadata(),
                "backend": vllm_backend_metadata(),
                "rows": build_result_rows(selected, evaluations),
            }
            write_json(seed_dir / "summary.json", seed_summary)
            (seed_dir / "results.txt").write_text(
                build_seed_report(seed_summary), encoding="utf-8"
            )
            run_summaries.append(seed_summary)
            write_json(run_dir / "progress.json", {
                "status": "complete" if run_index == len(args.seeds) else "in_progress",
                "completed_runs": run_index,
                "requested_runs": len(args.seeds),
                "seeds": args.seeds,
                "elapsed_seconds": time.monotonic() - started_at,
            })
            print(f"[math-final] run {run_index}/{len(args.seeds)} completed | seed={seed}", flush=True)
    finally:
        shutdown_vllm_model(model)

    aggregate = aggregate_runs(run_summaries)
    aggregate.update({
        "code": args.code,
        "model": args.model,
        "dataset": str(dataset_path),
        "seeds": args.seeds,
        "primary_grader": "openai",
        "stable_score_used": False,
    })
    write_json(run_dir / "summary.json", aggregate)
    (run_dir / "results.txt").write_text(build_aggregate_report(aggregate), encoding="utf-8")
    print(f"Saved five-seed aggregate results to: {run_dir / 'results.txt'}", flush=True)


if __name__ == "__main__":
    main()
