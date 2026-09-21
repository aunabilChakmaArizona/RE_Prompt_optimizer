"""Test all development-selected OpenBookQA prompts with one loaded vLLM model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path
from typing import Any, Sequence

from agents.agent_decoding import model_default_sampling_parameters
from prompt_optimization.evaluation import QAEvaluator, record_set_id
from prompt_optimization.models import ModelPool
from prompt_optimization.qa_task import (
    DEFAULT_TEST_PATH, REPO_ROOT, load_qa_records, resolve_mode, resolve_repo_path,
)
from prompt_optimization.run_io import (
    RunLogger, create_run_directory, format_elapsed, save_json, save_text,
)


DEFAULT_MANIFEST = REPO_ROOT / "experiment_tracking/final_test/openbookqa_selected_prompts.tsv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/qa_final_test"
MODEL_IDS = {"qwen": "Qwen/Qwen3-4B", "gemma": "google/gemma-3-4b-it"}
FINAL_CODES = {
    family: f"openbookqa_non_reasoning_{family}_selected_prompts_test_once"
    for family in MODEL_IDS
}
SOURCE_SLOTS = ("rpo5", "rpo10", "evoprompt5", "evoprompt10", "etgpo1")
REFINERS = ("gradpo_gen", "gradpo_prob", "gradpo_gen_random", "lpo", "greater", "greater_tg")


def parse_args() -> argparse.Namespace:
    """Read test-only settings for one seed or several separately saved seed runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(MODEL_IDS), required=True,
                        help="Target model family: Qwen3-4B or Gemma3-4B-IT.")
    parser.add_argument("--code", required=True, help="Unique identity of the final-test run.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="Frozen list of selected prompt files and their SHA-256 hashes.")
    parser.add_argument(
        "--source-slots",
        nargs="+",
        choices=SOURCE_SLOTS,
        default=list(SOURCE_SLOTS),
        help="First-stage source prompts expected in the selected-prompt manifest.",
    )
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH),
                        help="Prepared official OpenBookQA test JSONL (500 questions).")
    parser.add_argument("--backend", choices=("vllm",), default="vllm",
                        help="Final tests use vLLM only; no HF or optimizer model is loaded.")
    parser.add_argument("--device", default="cuda:0", help="Logical GPU within CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="Fraction of the selected GPU reserved for the sole vLLM model.")
    parser.add_argument("--vllm-max-model-len", type=int, default=16384,
                        help="Maximum input plus output context length for vLLM.")
    parser.add_argument("--vllm-disable-images", action="store_true",
                        help="Disable unused image inputs for the text-only Gemma task.")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="Answer-generation limit; the non-reasoning protocol uses 10.")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=42,
                            help="One base seed; instructions share each question's seed.")
    seed_group.add_argument("--seeds", type=int, nargs="+",
                            help="Different base seeds; evaluate the full matrix separately for each, loading once.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                        help="Root for test predictions, summaries, token counts, and the TXT table.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace only this final-test run directory, never optimizer outputs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check all selected files and the test set without loading a model or writing outputs.")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.vllm_max_model_len <= args.max_new_tokens:
        parser.error("Token limits must be positive, with context longer than the output limit.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be greater than zero and at most one.")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.seeds is not None and (
        any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("--seeds must be distinct non-negative integers.")
    if len(set(args.source_slots)) != len(args.source_slots):
        parser.error("--source-slots must not contain duplicates.")
    return args


def load_selected_prompts(
    manifest_path: str | Path,
    family: str,
    source_slots: Sequence[str] = SOURCE_SLOTS,
) -> list[dict[str, str]]:
    """Validate the requested method matrix and load its exact selected instructions."""
    path = resolve_repo_path(manifest_path)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream, delimiter="\t")
                if row["family"] == family]
    expected_ids = {"initial"}
    for source in source_slots:
        expected_ids.add(f"first_stage_{source}")
        expected_ids.update(f"second_stage_{source}_{method}" for method in REFINERS)
    expected_count = len(expected_ids)
    if len(rows) != expected_count or {row["row_id"] for row in rows} != expected_ids:
        raise ValueError(
            f"Expected exactly {expected_count} complete, unique selected rows for "
            f"{family} and sources {list(source_slots)}."
        )
    by_id = {row["row_id"]: row for row in rows}
    for row in rows:
        prompt_path = resolve_repo_path(row["prompt_file"])
        raw = prompt_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["prompt_sha256"]:
            raise ValueError(f"Selected prompt changed after selection: {prompt_path}")
        row["instruction_prompt"] = raw.decode("utf-8").strip()
        if not row["instruction_prompt"]:
            raise ValueError(f"Empty selected instruction: {prompt_path}")
        parent = row["parent_row_id"]
        expected_parent = (f"first_stage_{row['source']}" if row["stage"] == "second_stage"
                           else "initial" if row["stage"] == "first_stage" else "")
        if parent != expected_parent or (parent and parent not in by_id):
            raise ValueError(f"Invalid source link for {row['row_id']}.")
    return rows


def unique_instructions(rows: Sequence[dict[str, str]]) -> list[str]:
    """Keep one evaluation per identical instruction while retaining every method row."""
    return list(dict.fromkeys(row["instruction_prompt"] for row in rows))


def build_result_rows(
    selected: Sequence[dict[str, str]],
    evaluations: Sequence[dict[str, Any]],
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Map shared evaluations back to all methods and compute source-relative test gains."""
    by_prompt = {item["instruction_prompt"]: (index, item)
                 for index, item in enumerate(evaluations, start=1)}
    if len(by_prompt) != len(unique_instructions(selected)):
        raise ValueError("Missing or duplicate unique-prompt test evaluations.")
    rows = []
    first_row_by_prompt = {}
    for selection in selected:
        prompt = selection["instruction_prompt"]
        index, evaluation = by_prompt[prompt]
        metrics = evaluation["metrics"]
        name = f"test_prompt_{index:02d}"
        reused_from = first_row_by_prompt.get(prompt)
        first_row_by_prompt.setdefault(prompt, selection["row_id"])
        rows.append({
            **selection,
            "correct": metrics["correct"], "total": metrics["total"],
            "accuracy": float(metrics["accuracy"]),
            "missing_answer_tags": metrics["missing_answer_tags"],
            "invalid_choice_labels": metrics["invalid_choice_labels"],
            "token_usage": metrics["token_usage"],
            "evaluation_seed": evaluation["evaluation_seed"],
            "reused_from": reused_from,
            "evaluation_file": str(run_dir / "evaluations" / f"{name}_summary.json"),
            "predictions_file": str(run_dir / "evaluations" / f"{name}_predictions.jsonl"),
            "tested_prompt_file": str(run_dir / "prompts" / f"{name}.txt"),
        })
    by_id = {row["row_id"]: row for row in rows}
    for row in rows:
        parent = row["parent_row_id"]
        row["gain_percentage_points"] = (
            100.0 * (row["accuracy"] - by_id[parent]["accuracy"]) if parent else None
        )
        row["changed_from_source"] = (
            row["instruction_prompt"] != by_id[parent]["instruction_prompt"] if parent else None
        )
    return rows


def build_text_report(summary: dict[str, Any]) -> str:
    """Render every selected method's accuracy, source-relative gain, and saved artifacts."""
    protocol = summary["protocol"]
    lines = [
        f"OpenBookQA non-reasoning final test: {summary['family']}",
        f"CODE: {summary['code']}", f"Model: {summary['model']}",
        f"Test: {summary['test_path']} | questions={protocol['test_size']} | decoding runs=1",
        f"vLLM | max new tokens={protocol['max_new_tokens']} | seed={protocol['seed']}",
        f"Sampling: {summary['decoding']} | thinking disabled",
        f"Rows={len(summary['rows'])} | unique instructions={summary['unique_prompt_count']}",
        "Gain: first stage vs initial; second stage vs its own first-stage source (percentage points).",
        "Identical instructions share predictions; all unchanged/failed refinements remain listed.",
        "One sampled run: no across-run standard deviation or significance estimate.", "",
        "| Source | Stage/method | Correct/total | Accuracy (%) | Gain (pp) | Avg output tokens | Missing tags | Reused from |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        gain = row["gain_percentage_points"]
        gain_text = "-" if gain is None else f"{gain:+.2f}"
        lines.append(
            f"| {row['source']} | {row['stage']}/{row['method']} | "
            f"{row['correct']}/{row['total']} | {100.0 * row['accuracy']:.2f} | "
            f"{gain_text} | {row['token_usage']['averages']['output_tokens']:.2f} | "
            f"{row['missing_answer_tags']} | {row['reused_from'] or '-'} |"
        )
    lines.extend(["", "Selected prompt provenance and prediction files", "----------------------------------------------"])
    for row in summary["rows"]:
        lines.extend([
            f"{row['row_id']} | first-stage winning iteration={row['actual_iteration']} | CODE={row['optimization_code']}",
            f"  Selected instruction: {row['prompt_file']}",
            f"  SHA-256: {row['prompt_sha256']}",
            f"  Predictions (raw output and per-question token counts): {row['predictions_file']}",
        ])
    return "\n".join(lines) + "\n"


def evaluate_seed(
    args: argparse.Namespace,
    selected: Sequence[dict[str, str]],
    records: Sequence[dict[str, Any]],
    pool: ModelPool,
    run_dir: Path,
    protocol: dict[str, Any],
    code: str,
    started_at: float,
) -> dict[str, Any]:
    """Evaluate one complete prompt matrix and save its predictions without averaging runs."""
    seed_started_at = time.monotonic()
    prompts = unique_instructions(selected)
    evaluator = QAEvaluator(model_pool=pool, mode=resolve_mode("non_reasoning", "openbookqa"),
                            batch_size=4, max_new_tokens=args.max_new_tokens,
                            seed=protocol["seed"], run_started_at=started_at)
    evaluations = evaluator.evaluate_many(
        prompts, records, split_name="test",
        log_label=f"openbookqa_final_test_{args.family}_seed_{protocol['seed']}",
    )
    if len(evaluations) != len(prompts) or any(
        item["metrics"]["total"] != len(records) for item in evaluations
    ):
        raise ValueError("Incomplete test responses; no final score table will be saved.")
    logger = RunLogger(run_dir)
    for index, evaluation in enumerate(evaluations, start=1):
        name = f"test_prompt_{index:02d}"
        logger.evaluation(name, evaluation)
        save_text(run_dir / "prompts" / f"{name}.txt", evaluation["instruction_prompt"])
    model_id = MODEL_IDS[args.family]
    summary = {
        "code": code, "family": args.family, "model": model_id,
        "protocol": protocol, "test_path": str(resolve_repo_path(args.test_path)),
        "run_dir": str(run_dir), "decoding": model_default_sampling_parameters(model_id),
        "unique_prompt_count": len(prompts),
        "elapsed_seconds": time.monotonic() - seed_started_at,
        "rows": build_result_rows(selected, evaluations, run_dir),
    }
    save_json(run_dir / "summary.json", summary)
    save_text(run_dir / "results.txt", build_text_report(summary))
    print(f"Saved all {len(selected)} test rows to: {run_dir / 'results.txt'} | "
          f"seed_elapsed={format_elapsed(summary['elapsed_seconds'])}", flush=True)
    return summary


def main() -> None:
    """Keep one target model loaded while saving each requested seed's full matrix separately."""
    args = parse_args()
    started_at = time.monotonic()
    mode = resolve_mode("non_reasoning", "openbookqa")
    selected = load_selected_prompts(args.manifest, args.family, args.source_slots)
    prompts = unique_instructions(selected)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    multiple_runs = len(seeds) > 1
    test_path = resolve_repo_path(args.test_path)
    records = load_qa_records(test_path, expected_task="openbookqa")
    if len(records) != 500 or any(record.get("split") != "test" for record in records):
        raise ValueError("Use the prepared official OpenBookQA test split: 500 test questions.")
    print(f"[final-test:{args.family}] selected rows={len(selected)} | unique prompts={len(prompts)} | "
          f"test questions=500 | requests per run={len(prompts) * len(records)} | "
          f"decoding runs={len(seeds)} | seeds={seeds}", flush=True)
    if args.dry_run:
        print("Dry run passed: prompt hashes and source links checked; no model or outputs created.")
        return
    model_id = MODEL_IDS[args.family]
    protocol = {
        "task": "openbookqa", "qa_mode": "non_reasoning", "backend": "vllm",
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(resolve_repo_path(args.manifest).read_bytes()).hexdigest(),
        "record_set_id": record_set_id(records), "test_size": len(records),
        "run_count": 1, "max_new_tokens": args.max_new_tokens, "seed": seeds[0],
        "enable_thinking": False, "do_sample": True,
        "answer_instruction": mode.answer_instruction,
        "request_seed_policy": "QAEvaluator shared test seed plus question index; same across instructions",
    }
    run_dir = create_run_directory(args.output_root, "selected_prompts", mode.name, args.code, args.overwrite)
    config_protocol = {**protocol, "run_count": len(seeds), "seeds": seeds} if multiple_runs else protocol
    save_json(run_dir / "config.json", {**vars(args), "model": model_id, "protocol": config_protocol,
                                      "selected_rows": selected, "unique_prompt_count": len(prompts)})
    pool = ModelPool(
        target_model_id=model_id, optimizer_model_id=None, target_device=args.device,
        optimizer_device=None, keep_models_loaded=True, seed=seeds[0], backend="vllm",
        gpu_memory_utilization=args.gpu_memory_utilization, vllm_max_model_len=args.vllm_max_model_len,
        vllm_disable_images=args.vllm_disable_images,
    )
    progress = {
        "code": args.code, "family": args.family, "model": model_id,
        "protocol": config_protocol, "requested_runs": len(seeds), "completed_runs": 0,
        "seeds": seeds, "status": "in_progress", "runs": [],
        "aggregation": "Not computed; report mean and sample std after all runs finish.",
    }
    if multiple_runs:
        save_json(run_dir / "summary.json", progress)
    try:
        for run_index, seed in enumerate(seeds, start=1):
            print(f"[final-test:{args.family}] run {run_index}/{len(seeds)} started | "
                  f"seed={seed} | rows={len(selected)} | "
                  f"total_elapsed={format_elapsed(time.monotonic()-started_at)}",
                  flush=True)
            seed_dir = run_dir / f"seed_{seed}" if multiple_runs else run_dir
            seed_code = f"{args.code}_seed{seed}" if multiple_runs else args.code
            seed_protocol = {**protocol, "seed": seed}
            summary = evaluate_seed(args, selected, records, pool, seed_dir, seed_protocol,
                                    seed_code, started_at)
            if multiple_runs:
                progress["runs"].append({
                    "run_index": run_index, "seed": seed, "code": seed_code,
                    "summary_file": str(seed_dir / "summary.json"),
                    "results_file": str(seed_dir / "results.txt"),
                    "elapsed_seconds": summary["elapsed_seconds"],
                })
                progress["completed_runs"] = run_index
                progress["status"] = "complete" if run_index == len(seeds) else "in_progress"
                progress["elapsed_seconds"] = time.monotonic() - started_at
                save_json(run_dir / "summary.json", progress)
                index_lines = [f"OpenBookQA {args.family}: separately saved seed runs",
                               f"CODE: {args.code}", f"Completed: {run_index}/{len(seeds)}",
                               "No mean/std or stable test score computed.", ""]
                index_lines.extend(f"seed={item['seed']} | {item['results_file']}"
                                   for item in progress["runs"])
                save_text(run_dir / "results.txt", "\n".join(index_lines))
            print(f"[final-test:{args.family}] run {run_index}/{len(seeds)} completed | "
                  f"seed={seed} | total_elapsed={format_elapsed(time.monotonic()-started_at)}", flush=True)
    finally:
        pool.unload_all()


if __name__ == "__main__":
    main()
