"""Shared inference and evaluation code for multiple-choice QA test sets."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

from agents.agent_decoding import model_default_sampling_parameters
from agents.agent_token_usage import TokenUsage, summarize_token_usage


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "openbookqa" / "test.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "qa_test"
ANSWER_PATTERN = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL)

REASONING_INITIAL_PROMPT = (
    "Answer the following multiple-choice question. Think step by step carefully "
    "and select the best answer."
)
NON_REASONING_INITIAL_PROMPT = (
    "Answer the following multiple-choice question. Select the best answer directly "
    "without reasoning or explanation."
)
REASONING_ANSWER_INSTRUCTION = (
    "After you finish reasoning, output only the option label exactly once between the tags <answer> and </answer>, for example: <answer>B</answer>."
)
NON_REASONING_ANSWER_INSTRUCTION = (
    "Output only the option label exactly once between the tags <answer> and </answer>, for example: <answer>B</answer>. Do not output anything else."
)


def parse_args(description: str, default_instruction: str) -> argparse.Namespace:
    """Read command-line settings for one QA test run."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--code", required=True, help="Unique identity used as the output folder name.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Hugging Face model name or path.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Prepared QA JSONL test file.")
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
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum generated tokens per question.")
    parser.add_argument("--start", "--ep_start", dest="start", type=int, default=0, help="First test index.")
    parser.add_argument("--end", "--ep_end", dest="end", type=int, default=None, help="Exclusive final test index.")
    parser.add_argument("--prompt", default=default_instruction, help="Instruction placed before the fixed answer instruction.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Parent directory for mode-specific folders.")
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


def format_choices(choices: Sequence[dict[str, Any]]) -> str:
    """Format labeled answer choices as one choice per line."""
    return "\n".join(
        f"{str(choice['label']).strip().upper()}. {str(choice['text']).strip()}"
        for choice in choices
    )


def build_qa_prompt(
    instruction_prompt: str,
    answer_instruction: str,
    question: str,
    choices: Sequence[dict[str, Any]],
) -> str:
    """Place the instruction, answer format, question, and choices in order."""
    return (
        f"{instruction_prompt.strip()}\n\n"
        f"{answer_instruction}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Choices:\n{format_choices(choices)}"
    )


def extract_tagged_answer(response: str) -> str | None:
    """Extract the last complete answer enclosed by answer tags."""
    matches = ANSWER_PATTERN.findall(response)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_choice_label(value: str | None, valid_labels: set[str]) -> str | None:
    """Normalize a short answer such as B, (B), or B. to a valid label."""
    if value is None:
        return None
    candidate = value.strip()
    boxed_match = re.fullmatch(r"\\boxed\s*\{\s*([A-Za-z])\s*\}", candidate)
    if boxed_match:
        candidate = boxed_match.group(1)
    label_match = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?\s*[.):.]?", candidate)
    if not label_match:
        return None
    label = label_match.group(1).upper()
    if label not in valid_labels:
        return None
    return label


def validate_answer_processing() -> None:
    """Check representative answer extraction and normalization cases."""
    labels = {"A", "B", "C", "D"}
    assert extract_tagged_answer("work <answer>b</answer>") == "b"
    assert extract_tagged_answer("<answer>A</answer> then <answer>C</answer>") == "C"
    assert extract_tagged_answer("answer B") is None
    assert normalize_choice_label("b", labels) == "B"
    assert normalize_choice_label("(C)", labels) == "C"
    assert normalize_choice_label("D.", labels) == "D"
    assert normalize_choice_label("\\boxed{A}", labels) == "A"
    assert normalize_choice_label("B because it is correct", labels) is None
    assert normalize_choice_label("E", labels) is None


def validate_records(records: Sequence[dict[str, Any]]) -> None:
    """Check that selected records contain valid multiple-choice questions."""
    if not records:
        raise ValueError("The selected test range is empty.")

    seen_ids = set()
    for index, record in enumerate(records):
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            raise ValueError(f"Record at selected index {index} has no ID.")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate record ID in selected range: {record_id}")
        seen_ids.add(record_id)

        if record.get("task_type") != "multiple_choice_qa":
            raise ValueError(f"Record {record_id} is not a multiple_choice_qa example.")
        if not str(record.get("question", "")).strip():
            raise ValueError(f"Record {record_id} has no question.")

        choices = record.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"Record {record_id} must have at least two choices.")
        labels = [str(choice.get("label", "")).strip().upper() for choice in choices]
        if any(not label for label in labels) or len(labels) != len(set(labels)):
            raise ValueError(f"Record {record_id} has missing or duplicate choice labels.")
        if any(not str(choice.get("text", "")).strip() for choice in choices):
            raise ValueError(f"Record {record_id} has an empty choice text.")

        gold_label = str(record.get("answer", "")).strip().upper()
        if gold_label not in labels:
            raise ValueError(f"Record {record_id} has an invalid gold answer: {gold_label!r}")


def score_predictions(
    records: Sequence[dict[str, Any]],
    responses: Sequence[str],
    token_usages: Sequence[TokenUsage],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract option labels and calculate multiple-choice accuracy."""
    if len(records) != len(responses):
        raise ValueError("The number of model responses does not match the number of records.")
    if len(records) != len(token_usages):
        raise ValueError("The number of token-usage records does not match the test records.")

    results = []
    correct_count = 0
    missing_tag_count = 0
    invalid_label_count = 0

    for record, response, token_usage in zip(records, responses, token_usages):
        extracted_answer = extract_tagged_answer(response)
        valid_labels = {
            str(choice["label"]).strip().upper() for choice in record["choices"]
        }
        predicted_label = normalize_choice_label(extracted_answer, valid_labels)
        gold_label = str(record["answer"]).strip().upper()
        is_correct = predicted_label == gold_label

        if extracted_answer is None:
            missing_tag_count += 1
        elif predicted_label is None:
            invalid_label_count += 1
        correct_count += int(is_correct)

        results.append(
            {
                "id": record["id"],
                "dataset": record.get("dataset"),
                "task_type": record.get("task_type"),
                "question": record["question"],
                "choices": record["choices"],
                "gold_answer": gold_label,
                "gold_answer_text": record.get("answer_text"),
                "raw_response": response,
                "token_usage": dict(token_usage),
                "extracted_answer": extracted_answer,
                "predicted_answer": predicted_label,
                "correct": is_correct,
            }
        )

    total = len(records)
    statistics = {
        "total": total,
        "correct": correct_count,
        "incorrect": total - correct_count,
        "accuracy": correct_count / total,
        "accuracy_percent": 100.0 * correct_count / total,
        "missing_answer_tags": missing_tag_count,
        "invalid_choice_labels": invalid_label_count,
        "token_usage": summarize_token_usage(token_usages),
    }
    return results, statistics


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write records as one JSON object per line."""
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a JSON object with readable indentation."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_run_directory(
    output_dir: Path, mode_name: str, code: str, overwrite: bool
) -> Path:
    """Create a mode-specific output folder and protect existing results."""
    run_dir = output_dir / safe_name(mode_name) / safe_name(code)
    result_files = [run_dir / "predictions.jsonl", run_dir / "summary.json"]
    if not overwrite and any(path.exists() for path in result_files):
        raise FileExistsError(
            f"Results already exist in {run_dir}. Use a new --code or pass --overwrite."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_inference_backend(
    args: argparse.Namespace,
    prompts: Sequence[str],
    mode_name: str,
    enable_thinking: bool,
) -> tuple[list[str], list[TokenUsage], float, dict[str, Any]]:
    """Generate QA responses with the selected Transformers or vLLM backend."""
    log_label = f"qa_test_{safe_name(mode_name)}_{safe_name(args.code)}"
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


def run_qa_test_inference(
    *,
    mode_name: str,
    default_instruction: str,
    answer_instruction: str,
    enable_thinking: bool,
) -> None:
    """Run one reasoning or non-reasoning QA evaluation."""
    args = parse_args(
        f"Run {mode_name.replace('_', '-')} multiple-choice QA test inference.",
        default_instruction,
    )
    validate_answer_processing()

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
    validate_records(records)

    instruction_prompt = args.prompt.strip()
    if not instruction_prompt:
        raise ValueError("--prompt must not be empty.")
    decoding_parameters = model_default_sampling_parameters(args.model)
    prompts = [
        build_qa_prompt(
            instruction_prompt,
            answer_instruction,
            str(record["question"]),
            record["choices"],
        )
        for record in records
    ]
    run_dir = prepare_run_directory(output_dir, mode_name, args.code, args.overwrite)
    (run_dir / "prompt.txt").write_text(
        build_qa_prompt(
            instruction_prompt,
            answer_instruction,
            "{question}",
            [
                {"label": "A", "text": "{choice_a}"},
                {"label": "B", "text": "{choice_b}"},
                {"label": "C", "text": "{choice_c}"},
                {"label": "D", "text": "{choice_d}"},
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"CODE: {args.code}")
    print(f"Mode: {mode_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Examples: {len(records)} ({args.start}:{args.end})")
    print(f"Thinking enabled: {enable_thinking}")
    print(f"Backend: {args.backend}")
    print(f"Decoding: sampling {decoding_parameters}")
    if args.backend == "vllm":
        print("Batching: continuous (--batch_size is not used by vLLM)")
    print(f"Output: {run_dir}")

    responses, token_usages, elapsed_seconds, backend_metadata = run_inference_backend(
        args,
        prompts,
        mode_name,
        enable_thinking,
    )

    results, statistics = score_predictions(records, responses, token_usages)
    summary = {
        "code": args.code,
        "mode": mode_name,
        "model": args.model,
        "dataset": str(dataset_path),
        "task_type": "multiple_choice_qa",
        "test_range": {"start": args.start, "end": args.end},
        "instruction_prompt": instruction_prompt,
        "answer_instruction_prompt": answer_instruction,
        "settings": {
            "device": args.device,
            "batch_size": args.batch_size,
            "batch_size_applies": args.backend == "transformers",
            "max_new_tokens": args.max_new_tokens,
            "thinking_enabled": enable_thinking,
            "do_sample": True,
            **decoding_parameters,
            "backend": args.backend,
            "gpu_memory_utilization": (
                args.gpu_memory_utilization if args.backend == "vllm" else None
            ),
        },
        "backend": backend_metadata,
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": len(records) / elapsed_seconds if elapsed_seconds else None,
        "statistics": statistics,
        "files": {"predictions": "predictions.jsonl", "prompt": "prompt.txt"},
    }

    write_jsonl(run_dir / "predictions.jsonl", results)
    write_json(run_dir / "summary.json", summary)

    print(
        f"Accuracy: {statistics['correct']}/{statistics['total']} "
        f"({statistics['accuracy_percent']:.2f}%)"
    )
    print(f"Missing answer tags: {statistics['missing_answer_tags']}")
    print(f"Invalid choice labels: {statistics['invalid_choice_labels']}")
    print(f"Token usage: {statistics['token_usage']}")
    print(f"Elapsed inference time: {elapsed_seconds:.2f}s")
    print(f"Saved results to: {run_dir}")
