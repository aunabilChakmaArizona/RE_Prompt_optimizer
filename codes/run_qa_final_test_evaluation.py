"""Evaluate one fixed QA instruction on the test set over five decoding runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence

from prompt_optimization.evaluation import QAEvaluator, metric_accuracy
from prompt_optimization.models import ModelPool
from prompt_optimization.qa_task import (
    DEFAULT_TEST_PATH,
    REPO_ROOT,
    load_qa_records,
    resolve_mode,
)
from prompt_optimization.run_io import (
    RunLogger,
    create_run_directory,
    load_initial_prompt,
    save_json,
    save_text,
)


DEFAULT_FINAL_TEST_OUTPUT_ROOT = REPO_ROOT / "outputs" / "qa_final_test"


def parse_args() -> argparse.Namespace:
    """Read the fixed-prompt final-test evaluation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, help="Unique identity for this test run.")
    parser.add_argument(
        "--qa-mode",
        choices=("reasoning", "non_reasoning"),
        required=True,
        help="Whether the target model reasons before returning the option label.",
    )
    parser.add_argument("--model", required=True, help="Target Qwen3 or Gemma3 model.")
    parser.add_argument("--device", default="cuda:0", help="Target model device map.")
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default="vllm",
        help="Generation backend; vLLM is recommended for final inference.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of selected GPU memory reserved by vLLM.",
    )
    parser.add_argument(
        "--test-path",
        default=str(DEFAULT_TEST_PATH),
        help="Prepared test JSONL evaluated in every run.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Instruction text to evaluate; otherwise use a file or mode default.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Text file containing a saved first- or second-stage instruction.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of independent test generations; the final protocol uses five.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="First decoding seed; later runs use consecutive seeds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Transformers batch size; vLLM schedules submitted prompts dynamically.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Defaults to 4096 for reasoning and 16 for non-reasoning.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_FINAL_TEST_OUTPUT_ROOT),
        help="Root directory where final-test artifacts are saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing final-test directory with the same identity.",
    )
    args = parser.parse_args()
    if args.prompt and args.prompt_file:
        parser.error("Use only one of --prompt and --prompt-file.")
    if args.runs <= 0 or args.batch_size <= 0:
        parser.error("--runs and --batch-size must be positive.")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive when provided.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be greater than zero and at most one.")
    return args


def build_text_report(
    args: argparse.Namespace,
    prompt: str,
    run_rows: Sequence[dict[str, Any]],
    accuracy_mean: float,
    accuracy_std: float,
) -> str:
    """Render a compact plain-text report for one five-run test evaluation."""
    lines = [
        "OpenBookQA final test evaluation",
        "================================",
        f"Code: {args.code}",
        f"Mode: {args.qa_mode}",
        f"Model: {args.model}",
        f"Backend: {args.backend}",
        f"Runs: {len(run_rows)}",
        f"Accuracy mean: {accuracy_mean:.6f} ({100.0 * accuracy_mean:.2f}%)",
        f"Accuracy std: {accuracy_std:.6f} ({100.0 * accuracy_std:.2f} pp)",
        "",
        "| Run | Base seed | Evaluation seed | Correct | Total | Accuracy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        "| {run_index} | {base_seed} | {evaluation_seed} | {correct} | "
        "{total} | {accuracy:.6f} |".format(**row)
        for row in run_rows
    )
    lines.extend(["", "Instruction prompt", "------------------", prompt, ""])
    return "\n".join(lines)


def main() -> None:
    """Run five fixed-seed test evaluations and save individual and aggregate results."""
    args = parse_args()
    mode = resolve_mode(args.qa_mode)
    max_new_tokens = args.max_new_tokens or mode.default_max_new_tokens
    prompt = load_initial_prompt(mode, args.prompt, args.prompt_file)
    test_records = load_qa_records(args.test_path)
    run_dir = create_run_directory(
        args.output_root,
        "final_test",
        args.qa_mode,
        args.code,
        args.overwrite,
    )
    logger = RunLogger(run_dir)
    model_pool = ModelPool(
        target_model_id=args.model,
        optimizer_model_id=None,
        target_device=args.device,
        optimizer_device=None,
        keep_models_loaded=True,
        seed=args.seed,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    save_text(run_dir / "prompt.txt", prompt)
    save_json(
        run_dir / "config.json",
        {
            **vars(args),
            "resolved_max_new_tokens": max_new_tokens,
            "test_size": len(test_records),
            "run_seeds": [args.seed + index for index in range(args.runs)],
        },
    )
    try:
        run_rows = []
        for run_index in range(1, args.runs + 1):
            base_seed = args.seed + run_index - 1
            evaluator = QAEvaluator(
                model_pool=model_pool,
                mode=mode,
                batch_size=args.batch_size,
                max_new_tokens=max_new_tokens,
                seed=base_seed,
            )
            evaluation = evaluator.evaluate(
                prompt,
                test_records,
                split_name="test",
                log_label=f"qa_final_test_{args.code}_run_{run_index}",
            )
            logger.evaluation(f"test_run_{run_index}", evaluation)
            metrics = evaluation["metrics"]
            run_rows.append(
                {
                    "run_index": run_index,
                    "base_seed": base_seed,
                    "evaluation_seed": evaluation["evaluation_seed"],
                    "correct": metrics["correct"],
                    "total": metrics["total"],
                    "accuracy": metric_accuracy(evaluation),
                    "metrics": metrics,
                }
            )
        accuracies = [float(row["accuracy"]) for row in run_rows]
        accuracy_mean = fmean(accuracies)
        accuracy_std = pstdev(accuracies)
        summary = {
            "code": args.code,
            "qa_mode": args.qa_mode,
            "model": args.model,
            "backend": args.backend,
            "prompt": prompt,
            "test_path": str(Path(args.test_path).expanduser()),
            "test_size": len(test_records),
            "run_count": len(run_rows),
            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,
            "accuracy_mean_percent": 100.0 * accuracy_mean,
            "accuracy_std_percentage_points": 100.0 * accuracy_std,
            "runs": run_rows,
        }
        save_json(run_dir / "summary.json", summary)
        save_text(
            run_dir / "results.txt",
            build_text_report(
                args,
                prompt,
                run_rows,
                accuracy_mean,
                accuracy_std,
            ),
        )
        print(
            f"Test accuracy: {100.0 * accuracy_mean:.2f} "
            f"+/- {100.0 * accuracy_std:.2f} pp over {len(run_rows)} runs"
        )
        print(f"Saved final-test results to: {run_dir}")
    finally:
        model_pool.close()


if __name__ == "__main__":
    main()
