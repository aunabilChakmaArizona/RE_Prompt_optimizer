"""Run generative inference on a prepared math test set.

Example:
    python -u codes/run_math_test_inference.py \
        --code aime_qwen_base \
        --model Qwen/Qwen3-4B \
        --device cuda:0 \
        --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

from agents.agent_decoding import model_default_sampling_parameters
from agents.agent_token_usage import TokenUsage, summarize_token_usage
from math_grading.graders import (
    grade_math_answer,
    grader_metadata,
    normalize_numeric_answer,
    normalize_symbolic_answer,
    validate_grading_dependencies,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "data/processed/aime/test.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/math_test"

DEFAULT_INSTRUCTION_PROMPT = (
    "Solve the following math problem. Think step by step carefully before answering."
)
ANSWER_INSTRUCTION_PROMPT = (
    "After you finish reasoning, output the final answer exactly once between the tags <answer> and </answer>. " 
    "Put only the final answer inside the tags, using LaTeX notation when needed."
)
ANSWER_PATTERN = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    """Read command-line settings for one math test run."""
    parser = argparse.ArgumentParser(description="Run math test inference with triple answer grading.")
    parser.add_argument("--code", required=True, help="Unique identity used as the output folder name.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Hugging Face model name or path.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Prepared math JSONL test file.")
    parser.add_argument("--device", "--cuda", dest="device", default=None, help="Model device, such as cuda:0.")
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="Inference backend. Transformers remains the default.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Number of prompts generated per batch.")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="Fraction of selected GPU memory available to the vLLM engine.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum generated tokens per problem.")
    parser.add_argument("--start", "--ep_start", dest="start", type=int, default=0, help="First test index.")
    parser.add_argument("--end", "--ep_end", dest="end", type=int, default=None, help="Exclusive final test index.")
    parser.add_argument("--prompt", default=DEFAULT_INSTRUCTION_PROMPT, help="Instruction placed before the fixed answer instruction.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Parent directory for CODE folders.")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable model-specific thinking mode.")
    parser.add_argument("--overwrite", action="store_true", help="Replace files from an existing run with the same CODE.")
    return parser.parse_args()


def resolve_repo_path(path_value: str) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def safe_name(value: str) -> str:
    """Convert a run identity into a safe folder name."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if not name:
        raise ValueError("--code must contain at least one letter or number.")
    return name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON objects from a JSONL file."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_math_prompt(instruction_prompt: str, question: str) -> str:
    """Place the instruction, fixed answer instruction, and question in order."""
    return (
        f"{instruction_prompt.strip()}\n\n"
        f"{ANSWER_INSTRUCTION_PROMPT}\n\n"
        f"Question:\n{question.strip()}"
    )


def extract_tagged_answer(response: str) -> str | None:
    """Extract the last complete answer enclosed by answer tags."""
    matches = ANSWER_PATTERN.findall(response)
    if not matches:
        return None
    return matches[-1].strip()


def extract_last_boxed_answer(response: str) -> str | None:
    """Extract the content of the last balanced LaTeX boxed expression."""
    box_starts = list(re.finditer(r"\\boxed\s*\{", response))
    for box_start in reversed(box_starts):
        opening_brace = response.find("{", box_start.start())
        depth = 0
        for index in range(opening_brace, len(response)):
            character = response[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    answer = response[opening_brace + 1 : index].strip()
                    if answer:
                        return answer
                    break
    return None


def extract_math_answer(response: str) -> tuple[str | None, str]:
    """Use an answer tag first and the last boxed answer as a fallback."""
    tagged_answer = extract_tagged_answer(response)
    if tagged_answer:
        return tagged_answer, "answer_tag"

    boxed_answer = extract_last_boxed_answer(response)
    if boxed_answer:
        return boxed_answer, "boxed_fallback"
    return None, "missing"


def validate_answer_processing() -> None:
    """Check representative extraction and simple-normalization cases."""
    assert extract_tagged_answer("work <answer>0042</answer>") == "0042"
    assert extract_tagged_answer("<answer>1</answer> then <answer>2</answer>") == "2"
    assert extract_tagged_answer("answer 42") is None
    assert extract_last_boxed_answer("work \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_last_boxed_answer("\\boxed{1} then \\boxed{2}") == "2"
    assert extract_last_boxed_answer("unfinished \\boxed{3") is None
    assert extract_math_answer("\\boxed{1} <answer>2</answer>") == ("2", "answer_tag")
    assert extract_math_answer("final \\boxed{3}") == ("3", "boxed_fallback")
    assert extract_math_answer("no final answer") == (None, "missing")
    assert normalize_numeric_answer("0042") == "42"
    assert normalize_numeric_answer("$\\boxed{000.500}$") == "0.5"
    assert normalize_numeric_answer("000") == "0"
    assert normalize_numeric_answer("1.2.3") is None
    assert normalize_symbolic_answer("$\\boxed{\\dfrac{14}{3}}$") == "\\frac{14}{3}"
    assert normalize_symbolic_answer("\\left( 3, 4 \\right)") == "(3,4)"
    assert normalize_symbolic_answer("\\text{Evelyn}") == "evelyn"


def validate_records(records: Sequence[dict[str, Any]]) -> str:
    """Check selected records and return their shared mathematical task type."""
    if not records:
        raise ValueError("The selected test range is empty.")

    task_types = {str(record.get("task_type", "")).strip() for record in records}
    if len(task_types) != 1 or "" in task_types:
        raise ValueError("All selected records must have one non-empty task_type.")
    task_type = next(iter(task_types))

    for record in records:
        if not str(record.get("question", "")).strip():
            raise ValueError(f"Record {record.get('id', '<unknown>')} has no question.")
        answer = str(record.get("answer", "")).strip()
        if not answer:
            raise ValueError(f"Record {record.get('id', '<unknown>')} has no answer.")
        if task_type == "math_short_answer" and normalize_numeric_answer(answer) is None:
            raise ValueError(f"Record {record.get('id', '<unknown>')} has an invalid numeric answer.")
    return task_type


def update_group_statistics(group: dict[str, Any], grades: dict[str, Any]) -> None:
    """Add one prediction's three grading outcomes to a grouped summary."""
    group["total"] += 1
    for grader_name in ("simple", "openai", "math_verify"):
        group[f"{grader_name}_correct"] += int(grades[f"{grader_name}_correct"])


def finalize_group_statistics(groups: dict[str, dict[str, Any]]) -> None:
    """Calculate grader accuracies in each non-empty metadata group."""
    for group in groups.values():
        total = group["total"]
        for grader_name in ("simple", "openai", "math_verify"):
            group[f"{grader_name}_accuracy"] = group[f"{grader_name}_correct"] / total


def empty_group_statistics() -> dict[str, int]:
    """Create zeroed counters for one metadata group."""
    return {
        "total": 0,
        "simple_correct": 0,
        "openai_correct": 0,
        "math_verify_correct": 0,
    }


def accuracy_statistics(correct: int, total: int) -> dict[str, int | float]:
    """Create correct, incorrect, and accuracy values for one grader."""
    return {
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
        "accuracy_percent": 100.0 * correct / total,
    }


def score_predictions(
    records: Sequence[dict[str, Any]],
    responses: Sequence[str],
    token_usages: Sequence[TokenUsage],
    task_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract answers, run three graders, and collect summary counts."""
    if len(records) != len(responses):
        raise ValueError("The number of model responses does not match the number of records.")
    if len(records) != len(token_usages):
        raise ValueError("The number of token-usage records does not match the test records.")

    results = []
    correct_counts = {"simple": 0, "openai": 0, "math_verify": 0}
    tag_only_correct_counts = {"simple": 0, "openai": 0, "math_verify": 0}
    missing_tag_count = 0
    extraction_counts = {"answer_tag": 0, "boxed_fallback": 0, "missing": 0}
    simple_normalization_failure_count = 0
    grading_error_counts = {"openai": 0, "math_verify": 0}
    grouped_statistics: dict[str, dict[str, dict[str, Any]]] = {
        "by_year": {},
        "by_subject": {},
        "by_level": {},
    }

    for record, response, token_usage in zip(records, responses, token_usages):
        tagged_answer = extract_tagged_answer(response)
        extracted_answer, extraction_source = extract_math_answer(response)
        grades = grade_math_answer(extracted_answer, str(record["answer"]), task_type)

        if tagged_answer is None:
            missing_tag_count += 1
        extraction_counts[extraction_source] += 1
        if extracted_answer is not None and grades["normalized_prediction"] is None:
            simple_normalization_failure_count += 1

        for grader_name in correct_counts:
            correct_counts[grader_name] += int(grades[f"{grader_name}_correct"])
            if extraction_source == "answer_tag":
                tag_only_correct_counts[grader_name] += int(
                    grades[f"{grader_name}_correct"]
                )
        for grader_name in grading_error_counts:
            grading_error_counts[grader_name] += int(grades[f"{grader_name}_error"] is not None)

        for field_name, summary_name in (
            ("year", "by_year"),
            ("subject", "by_subject"),
            ("level", "by_level"),
        ):
            if record.get(field_name) is not None:
                group_key = str(record[field_name])
                group = grouped_statistics[summary_name].setdefault(
                    group_key, empty_group_statistics()
                )
                update_group_statistics(group, grades)

        result = {
            "id": record.get("id"),
            "dataset": record.get("dataset"),
            "task_type": record.get("task_type"),
            "question": record["question"],
            "gold_answer": record["answer"],
            "normalized_gold_answer": grades["normalized_gold_answer"],
            "raw_response": response,
            "token_usage": dict(token_usage),
            "extracted_answer": extracted_answer,
            "answer_extraction_source": extraction_source,
            "normalized_prediction": grades["normalized_prediction"],
            "simple_correct": grades["simple_correct"],
            "openai_correct": grades["openai_correct"],
            "math_verify_correct": grades["math_verify_correct"],
            "openai_error": grades["openai_error"],
            "math_verify_error": grades["math_verify_error"],
        }
        for metadata_field in (
            "year",
            "exam",
            "problem_number",
            "subject",
            "level",
            "source_id",
        ):
            if metadata_field in record:
                result[metadata_field] = record[metadata_field]
        results.append(result)

    for groups in grouped_statistics.values():
        finalize_group_statistics(groups)

    total = len(records)
    statistics = {
        "total": total,
        "simple": accuracy_statistics(correct_counts["simple"], total),
        "openai": accuracy_statistics(correct_counts["openai"], total),
        "math_verify": accuracy_statistics(correct_counts["math_verify"], total),
        "tag_only": {
            grader_name: accuracy_statistics(tag_only_correct_counts[grader_name], total)
            for grader_name in tag_only_correct_counts
        },
        "missing_answer_tags": missing_tag_count,
        "answer_extraction": extraction_counts,
        "simple_normalization_failures": simple_normalization_failure_count,
        "grading_errors": grading_error_counts,
        "token_usage": summarize_token_usage(token_usages),
        **grouped_statistics,
    }
    return results, statistics


def run_inference_backend(
    args: argparse.Namespace, prompts: Sequence[str]
) -> tuple[list[str], list[TokenUsage], float, dict[str, Any]]:
    """Generate responses with the selected Transformers or vLLM backend."""
    enable_thinking = not args.disable_thinking
    log_label = f"math_test_{safe_name(args.code)}"
    decoding_parameters = model_default_sampling_parameters(args.model)

    if args.backend == "vllm":
        from agents.agent_vllm_models import (
            load_vllm_model_and_tokenizer,
            vllm_backend_metadata,
        )
        from agents.agent_vllm_prompting import run_prompts_vllm

        model, tokenizer = load_vllm_model_and_tokenizer(
            args.model,
            device=args.device,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        started_at = time.time()
        responses, token_usages = run_prompts_vllm(
            prompts,
            model_id=args.model,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=enable_thinking,
            do_log=True,
            log_label=log_label,
            return_token_usage=True,
        )
        return (
            responses,
            token_usages,
            time.time() - started_at,
            vllm_backend_metadata(),
        )

    from agents.agent_llm_prompting import run_prompts
    from agents.agent_models import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(args.model, device_map=args.device)
    started_at = time.time()
    responses, token_usages = run_prompts(
        prompts,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        enable_thinking=enable_thinking,
        do_log=True,
        log_label=log_label,
        do_sample=True,
        return_token_usage=True,
        **decoding_parameters,
    )
    metadata = {"name": "transformers", "batching": "fixed"}
    return responses, token_usages, time.time() - started_at, metadata


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write records as one JSON object per line."""
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a JSON object with readable indentation."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_run_directory(output_dir: Path, code: str, overwrite: bool) -> Path:
    """Create one output folder and protect existing run results by default."""
    run_dir = output_dir / safe_name(code)
    result_files = [run_dir / "predictions.jsonl", run_dir / "summary.json"]
    if not overwrite and any(path.exists() for path in result_files):
        raise FileExistsError(
            f"Results already exist in {run_dir}. Use a new --code or pass --overwrite."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    """Run math inference, evaluate answers, and save all run artifacts."""
    args = parse_args()
    validate_answer_processing()
    validate_grading_dependencies()

    if args.start < 0:
        raise ValueError("--start must be non-negative.")
    if args.end is not None and args.end <= args.start:
        raise ValueError("--end must be greater than --start.")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--batch_size and --max_new_tokens must be positive.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu_memory_utilization must be greater than 0 and at most 1.")

    dataset_path = resolve_repo_path(args.dataset)
    output_dir = resolve_repo_path(args.output_dir)
    all_records = read_jsonl(dataset_path)
    records = all_records[args.start : args.end]
    task_type = validate_records(records)

    instruction_prompt = args.prompt.strip()
    if not instruction_prompt:
        raise ValueError("--prompt must not be empty.")
    decoding_parameters = model_default_sampling_parameters(args.model)
    prompts = [build_math_prompt(instruction_prompt, record["question"]) for record in records]
    run_dir = prepare_run_directory(output_dir, args.code, args.overwrite)
    (run_dir / "prompt.txt").write_text(
        build_math_prompt(instruction_prompt, "{question}") + "\n", encoding="utf-8"
    )

    print(f"CODE: {args.code}")
    print(f"Dataset: {dataset_path}")
    print(f"Task type: {task_type}")
    print(f"Examples: {len(records)} ({args.start}:{args.end})")
    print(f"Backend: {args.backend}")
    print(f"Decoding: sampling {decoding_parameters}")
    if args.backend == "vllm":
        print("Batching: continuous (--batch_size is not used by vLLM)")
    print(f"Output: {run_dir}")

    responses, token_usages, elapsed_seconds, backend_metadata = run_inference_backend(
        args, prompts
    )

    results, statistics = score_predictions(records, responses, token_usages, task_type)
    summary = {
        "code": args.code,
        "model": args.model,
        "dataset": str(dataset_path),
        "task_type": task_type,
        "test_range": {"start": args.start, "end": args.end},
        "instruction_prompt": instruction_prompt,
        "answer_instruction_prompt": ANSWER_INSTRUCTION_PROMPT,
        "settings": {
            "device": args.device,
            "batch_size": args.batch_size,
            "batch_size_applies": args.backend == "transformers",
            "max_new_tokens": args.max_new_tokens,
            "thinking_enabled": not args.disable_thinking,
            "do_sample": True,
            **decoding_parameters,
            "backend": args.backend,
            "gpu_memory_utilization": (
                args.gpu_memory_utilization if args.backend == "vllm" else None
            ),
        },
        "backend": backend_metadata,
        "elapsed_seconds": elapsed_seconds,
        "graders": grader_metadata(),
        "statistics": statistics,
        "files": {"predictions": "predictions.jsonl", "prompt": "prompt.txt"},
    }

    write_jsonl(run_dir / "predictions.jsonl", results)
    write_json(run_dir / "summary.json", summary)

    for grader_name in ("simple", "openai", "math_verify"):
        grader_stats = statistics[grader_name]
        print(
            f"{grader_name} accuracy: {grader_stats['correct']}/{statistics['total']} "
            f"({grader_stats['accuracy_percent']:.2f}%)"
        )
    tag_only_math_verify = statistics["tag_only"]["math_verify"]
    print(
        f"tag-only math_verify accuracy: {tag_only_math_verify['correct']}/"
        f"{statistics['total']} ({tag_only_math_verify['accuracy_percent']:.2f}%)"
    )
    print(f"Missing answer tags: {statistics['missing_answer_tags']}")
    print(f"Answer extraction: {statistics['answer_extraction']}")
    print(f"Token usage: {statistics['token_usage']}")
    print(f"Simple normalization failures: {statistics['simple_normalization_failures']}")
    print(f"Grading errors: {statistics['grading_errors']}")
    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()
