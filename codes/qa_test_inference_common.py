"""Shared inference and evaluation code for multiple-choice and open QA tests."""

from __future__ import annotations

import argparse
import json
import re
import string
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from agents.agent_decoding import model_default_sampling_parameters
from agents.agent_token_usage import TokenUsage, summarize_token_usage


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "openbookqa" / "test.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "qa_test"
ANSWER_PATTERN = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL)
DATE_MDY_PATTERN = re.compile(r"(\d+)/(\d+)/(\d+)")
DATE_YMD_PATTERN = re.compile(r"(\d+)-(\d+)-(\d+)")

REASONING_INITIAL_PROMPT = (
    "You are given a multiple-choice question and a set of answer choices. "
    "Think step by step carefully and select the best answer."
)
NON_REASONING_INITIAL_PROMPT = (
    "You are given a multiple-choice question and a set of answer choices. "
    "Select the best answer."
)
REASONING_ANSWER_INSTRUCTION = (
    "After you finish reasoning, output only the option label exactly once between the tags <answer> and </answer>, for example: <answer>B</answer>."
)
NON_REASONING_ANSWER_INSTRUCTION = (
    "Do not think or provide any reasoning. Just output the option label exactly once "
    "between the tags <answer> and </answer>, for example: <answer>B</answer>. "
    "Do not output anything else."
)
OPEN_QA_REASONING_INITIAL_PROMPT = (
    "You are given a question. Think step by step carefully and provide the best answer."
)
OPEN_QA_NON_REASONING_INITIAL_PROMPT = (
    "You are given a question. Provide the best answer."
)
OPEN_QA_REASONING_ANSWER_INSTRUCTION = (
    "After you finish reasoning, output only the final answer between the tags "
    "<answer> and </answer>. If there is more than one answer, separate the answers "
    "with semicolons."
)
OPEN_QA_NON_REASONING_ANSWER_INSTRUCTION = (
    "Do not think or provide any reasoning. Just output the answer between the tags "
    "<answer> and </answer>. If there is more than one answer, separate the answers "
    "with semicolons. Do not output anything else."
)


def parse_args(
    description: str,
    default_dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> argparse.Namespace:
    """Read command-line settings for one QA test run."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--code", required=True, help="Unique identity used as the output folder name.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Hugging Face model name or path.")
    parser.add_argument(
        "--dataset",
        default=str(default_dataset_path),
        help="Prepared QA JSONL test file.",
    )
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
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum generated tokens per question; otherwise use the task and mode default.",
    )
    parser.add_argument("--start", "--ep_start", dest="start", type=int, default=0, help="First test index.")
    parser.add_argument("--end", "--ep_end", dest="end", type=int, default=None, help="Exclusive final test index.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Instruction placed before the fixed answer instruction; otherwise use the task default.",
    )
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


def build_open_qa_prompt(
    instruction_prompt: str,
    answer_instruction: str,
    question: str,
) -> str:
    """Place an open-answer instruction, answer format, and question in order."""
    return (
        f"{instruction_prompt.strip()}\n\n"
        f"{answer_instruction}\n\n"
        f"Question:\n{question.strip()}"
    )


def format_context_passages(context: Sequence[dict[str, Any]]) -> str:
    """Format titled context paragraphs in their original dataset order."""
    passages = []
    for index, paragraph in enumerate(context, start=1):
        title = str(paragraph["title"]).strip()
        sentences = " ".join(
            str(sentence).strip() for sentence in paragraph["sentences"]
        ).strip()
        passages.append(f"[{index}] {title}\n{sentences}")
    return "\n\n".join(passages)


def build_context_open_qa_prompt(
    instruction_prompt: str,
    answer_instruction: str,
    context: Sequence[dict[str, Any]],
    question: str,
) -> str:
    """Place an instruction, context passages, and an open question in order."""
    return (
        f"{instruction_prompt.strip()}\n\n"
        f"{answer_instruction}\n\n"
        f"Context:\n{format_context_passages(context)}\n\n"
        f"Question:\n{question.strip()}"
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


def normalize_open_answer(value: str) -> str:
    """Normalize one textual answer for case-insensitive entity matching."""
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in normalized
    )
    words = [word for word in normalized.split() if word not in {"a", "an", "the"}]
    return " ".join(words)


def split_open_answers(value: str | None) -> list[str]:
    """Split a tagged response into semicolon-separated textual answers."""
    if value is None:
        return []
    answers = []
    for part in re.split(r"[;\n]+", value):
        normalized = normalize_open_answer(part)
        if normalized and normalized not in answers:
            answers.append(normalized)
    return answers


def split_raw_open_answers(value: str | None) -> list[str]:
    """Split a tagged response while preserving benchmark answer spelling."""
    if value is None:
        return []
    answers = []
    for part in re.split(r"[;\n]+", value):
        answer = part.strip()
        if answer and answer not in answers:
            answers.append(answer)
    return answers


def official_webquestions_value(value: str) -> str | tuple[int, int, int]:
    """Convert supported date strings as done by the WebQuestions evaluator."""
    match = DATE_MDY_PATTERN.match(value)
    if match:
        return int(match.group(3)), int(match.group(1)), int(match.group(2))
    match = DATE_YMD_PATTERN.match(value)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return value


def answer_set_scores(
    gold_answers: Sequence[str | tuple[int, int, int]],
    predicted_answers: Sequence[str | tuple[int, int, int]],
) -> tuple[float, float, float]:
    """Calculate recall, precision, and F1 for one answer set."""
    if not predicted_answers:
        return 0.0, 1.0, 0.0
    matched_predictions = sum(answer in gold_answers for answer in predicted_answers)
    matched_gold = sum(answer in predicted_answers for answer in gold_answers)
    precision = matched_predictions / len(predicted_answers)
    recall = matched_gold / len(gold_answers)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return recall, precision, f1


def normalize_hotpot_answer(value: str) -> str:
    """Normalize an answer exactly as the official HotpotQA evaluator does."""
    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def hotpot_answer_scores( #todo: check if this f1 and exact match formula is correct or not
    prediction: str,
    gold_answer: str,
) -> tuple[float, float, float, float]:
    """Calculate official HotpotQA answer EM, precision, recall, and token F1."""
    normalized_prediction = normalize_hotpot_answer(prediction)
    normalized_gold = normalize_hotpot_answer(gold_answer)
    exact_match = float(normalized_prediction == normalized_gold)
    special_answers = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special_answers
        or normalized_gold in special_answers
    ) and normalized_prediction != normalized_gold:
        return exact_match, 0.0, 0.0, 0.0

    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    matching_tokens = sum(common.values())
    if matching_tokens == 0:
        return exact_match, 0.0, 0.0, 0.0
    precision = matching_tokens / len(prediction_tokens)
    recall = matching_tokens / len(gold_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)
    return exact_match, precision, recall, f1


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
    assert normalize_open_answer("The Beatles!") == "beatles"
    assert split_open_answers("Paris; New York") == ["paris", "new york"]
    assert official_webquestions_value("1836-02-23") == (1836, 2, 23)
    assert normalize_hotpot_answer("The Eiffel Tower!") == "eiffel tower"
    assert hotpot_answer_scores("the red fox", "Red fox") == (1.0, 1.0, 1.0, 1.0)


def validate_records(records: Sequence[dict[str, Any]]) -> str:
    """Check selected records and return their common QA task type."""
    if not records:
        raise ValueError("The selected test range is empty.")

    task_types = {str(record.get("task_type", "")).strip() for record in records}
    supported_task_types = {
        "multiple_choice_qa",
        "open_qa",
        "hotpotqa_open_qa",
    }
    if len(task_types) != 1 or task_types.pop() not in supported_task_types:
        raise ValueError("Records must share one supported task_type.")
    task_type = str(records[0]["task_type"])
    seen_ids = set()
    for index, record in enumerate(records):
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            raise ValueError(f"Record at selected index {index} has no ID.")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate record ID in selected range: {record_id}")
        seen_ids.add(record_id)

        if not str(record.get("question", "")).strip():
            raise ValueError(f"Record {record_id} has no question.")

        if task_type in {"open_qa", "hotpotqa_open_qa"}:
            answers = record.get("answers")
            if not isinstance(answers, list) or not any(
                str(answer).strip() for answer in answers
            ):
                raise ValueError(f"Open-QA record {record_id} has no gold answers.")
            if task_type == "hotpotqa_open_qa":
                context = record.get("context")
                if not isinstance(context, list) or not context:
                    raise ValueError(f"HotpotQA record {record_id} has no context.")
                for paragraph in context:
                    if not str(paragraph.get("title", "")).strip():
                        raise ValueError(
                            f"HotpotQA record {record_id} has an untitled paragraph."
                        )
                    sentences = paragraph.get("sentences")
                    if not isinstance(sentences, list) or not sentences:
                        raise ValueError(
                            f"HotpotQA record {record_id} has an empty paragraph."
                        )
            continue

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
            raise ValueError(
                f"Record {record_id} has no usable gold answer. "
                "The official CommonsenseQA test labels are hidden; use "
                "data/processed/commonsenseqa/local_test.jsonl for local scoring."
            )
    return task_type


def score_multiple_choice_predictions(
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


def score_open_qa_predictions(
    records: Sequence[dict[str, Any]],
    responses: Sequence[str],
    token_usages: Sequence[TokenUsage],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Calculate macro answer-set precision, recall, F1, and exact match."""
    if len(records) != len(responses) or len(records) != len(token_usages):
        raise ValueError("Records, responses, and token usage must have equal lengths.")
    results = []
    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0
    normalized_precision_total = 0.0
    normalized_recall_total = 0.0
    normalized_f1_total = 0.0
    exact_match_count = 0
    normalized_exact_match_count = 0
    missing_tag_count = 0
    for record, response, token_usage in zip(records, responses, token_usages):
        extracted_answer = extract_tagged_answer(response)
        raw_predicted_answers = split_raw_open_answers(extracted_answer)
        raw_gold_answers = [str(answer).strip() for answer in record["answers"]]
        official_predicted_answers = [
            official_webquestions_value(answer) for answer in raw_predicted_answers
        ]
        official_gold_answers = [
            official_webquestions_value(answer) for answer in raw_gold_answers
        ]
        recall, precision, f1 = answer_set_scores(
            official_gold_answers,
            official_predicted_answers,
        )
        predicted_answers = set(split_open_answers(extracted_answer))
        gold_answers = {
            normalize_open_answer(str(answer))
            for answer in record["answers"]
            if normalize_open_answer(str(answer))
        }
        normalized_recall, normalized_precision, normalized_f1 = answer_set_scores(
            list(gold_answers),
            list(predicted_answers),
        )
        exact_match = f1 == 1.0
        normalized_exact_match = normalized_f1 == 1.0
        precision_total += precision
        recall_total += recall
        f1_total += f1
        normalized_precision_total += normalized_precision
        normalized_recall_total += normalized_recall
        normalized_f1_total += normalized_f1
        exact_match_count += int(exact_match)
        normalized_exact_match_count += int(normalized_exact_match)
        missing_tag_count += int(extracted_answer is None)
        results.append(
            {
                "id": record["id"],
                "dataset": record.get("dataset"),
                "task_type": record.get("task_type"),
                "question": record["question"],
                "gold_answers": record["answers"],
                "source_url": record.get("source_url"),
                "raw_response": response,
                "token_usage": dict(token_usage),
                "extracted_answer": extracted_answer,
                "raw_predicted_answers": raw_predicted_answers,
                "predicted_answers": sorted(predicted_answers),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "exact_match": exact_match,
                "normalized_precision": normalized_precision,
                "normalized_recall": normalized_recall,
                "normalized_f1": normalized_f1,
                "normalized_exact_match": normalized_exact_match,
            }
        )
    total = len(records)
    statistics = {
        "total": total,
        "macro_precision": precision_total / total,
        "macro_recall": recall_total / total,
        "macro_f1": f1_total / total,
        "macro_f1_percent": 100.0 * f1_total / total,
        "exact_match_count": exact_match_count,
        "exact_match_accuracy": exact_match_count / total,
        "exact_match_accuracy_percent": 100.0 * exact_match_count / total,
        "normalized_macro_precision": normalized_precision_total / total,
        "normalized_macro_recall": normalized_recall_total / total,
        "normalized_macro_f1": normalized_f1_total / total,
        "normalized_macro_f1_percent": 100.0 * normalized_f1_total / total,
        "normalized_exact_match_count": normalized_exact_match_count,
        "normalized_exact_match_accuracy": normalized_exact_match_count / total,
        "normalized_exact_match_accuracy_percent": (
            100.0 * normalized_exact_match_count / total
        ),
        "missing_answer_tags": missing_tag_count,
        "token_usage": summarize_token_usage(token_usages),
    }
    return results, statistics


def score_hotpotqa_predictions(
    records: Sequence[dict[str, Any]],
    responses: Sequence[str],
    token_usages: Sequence[TokenUsage],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score answer text with the official HotpotQA answer EM and token F1 rules."""
    if len(records) != len(responses) or len(records) != len(token_usages):
        raise ValueError("Records, responses, and token usage must have equal lengths.")

    results = []
    exact_match_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0
    missing_tag_count = 0
    for record, response, token_usage in zip(records, responses, token_usages):
        extracted_answer = extract_tagged_answer(response)
        predicted_answer = extracted_answer or ""
        gold_answer = str(record["answer"])
        exact_match, precision, recall, f1 = hotpot_answer_scores(
            predicted_answer,
            gold_answer,
        )
        exact_match_total += exact_match
        precision_total += precision
        recall_total += recall
        f1_total += f1
        missing_tag_count += int(extracted_answer is None)
        results.append(
            {
                "id": record["id"],
                "dataset": record.get("dataset"),
                "task_type": record.get("task_type"),
                "question": record["question"],
                "question_type": record.get("question_type"),
                "level": record.get("level"),
                "gold_answer": gold_answer,
                "raw_response": response,
                "token_usage": dict(token_usage),
                "extracted_answer": extracted_answer,
                "predicted_answer": predicted_answer,
                "answer_exact_match": exact_match,
                "answer_precision": precision,
                "answer_recall": recall,
                "answer_f1": f1,
            }
        )

    total = len(records)
    statistics = {
        "total": total,
        "answer_exact_match": exact_match_total / total,
        "answer_exact_match_percent": 100.0 * exact_match_total / total,
        "answer_precision": precision_total / total,
        "answer_recall": recall_total / total,
        "answer_f1": f1_total / total,
        "answer_f1_percent": 100.0 * f1_total / total,
        "missing_answer_tags": missing_tag_count,
        "token_usage": summarize_token_usage(token_usages),
    }
    return results, statistics


def score_predictions(
    records: Sequence[dict[str, Any]],
    responses: Sequence[str],
    token_usages: Sequence[TokenUsage],
    task_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score model responses with the metric for the selected QA task."""
    if task_type == "hotpotqa_open_qa":
        return score_hotpotqa_predictions(records, responses, token_usages)
    if task_type == "open_qa":
        return score_open_qa_predictions(records, responses, token_usages)
    return score_multiple_choice_predictions(records, responses, token_usages)


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
    default_max_new_tokens: int,
    default_dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> None:
    """Run one reasoning or non-reasoning QA evaluation."""
    args = parse_args(
        f"Run {mode_name.replace('_', '-')} QA test inference.",
        default_dataset_path=default_dataset_path,
    )
    validate_answer_processing()

    if args.start < 0:
        raise ValueError("--start must be non-negative.")
    if args.end is not None and args.end <= args.start:
        raise ValueError("--end must be greater than --start.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive when provided.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu_memory_utilization must be greater than 0 and at most 1.")

    dataset_path = resolve_repo_path(args.dataset)
    output_dir = resolve_repo_path(args.output_dir)
    all_records = read_jsonl(dataset_path)
    records = all_records[args.start : args.end]
    task_type = validate_records(records)
    if task_type == "hotpotqa_open_qa":
        task_default_instruction = default_instruction
        resolved_answer_instruction = answer_instruction
        task_default_max_new_tokens = default_max_new_tokens
    elif task_type == "open_qa":
        task_default_instruction = (
            OPEN_QA_REASONING_INITIAL_PROMPT
            if enable_thinking
            else OPEN_QA_NON_REASONING_INITIAL_PROMPT
        )
        resolved_answer_instruction = (
            OPEN_QA_REASONING_ANSWER_INSTRUCTION
            if enable_thinking
            else OPEN_QA_NON_REASONING_ANSWER_INSTRUCTION
        )
        task_default_max_new_tokens = 4096 if enable_thinking else 128
    else:
        task_default_instruction = default_instruction
        resolved_answer_instruction = answer_instruction
        task_default_max_new_tokens = default_max_new_tokens
    args.max_new_tokens = args.max_new_tokens or task_default_max_new_tokens
    instruction_prompt = (args.prompt or task_default_instruction).strip()
    if not instruction_prompt:
        raise ValueError("--prompt must not be empty.")
    decoding_parameters = model_default_sampling_parameters(args.model)
    if task_type == "hotpotqa_open_qa":
        prompts = [
            build_context_open_qa_prompt(
                instruction_prompt,
                resolved_answer_instruction,
                record["context"],
                str(record["question"]),
            )
            for record in records
        ]
        prompt_template = build_context_open_qa_prompt(
            instruction_prompt,
            resolved_answer_instruction,
            [{"title": "{title}", "sentences": ["{context_sentences}"]}],
            "{question}",
        )
    elif task_type == "open_qa":
        prompts = [
            build_open_qa_prompt(
                instruction_prompt,
                resolved_answer_instruction,
                str(record["question"]),
            )
            for record in records
        ]
        prompt_template = build_open_qa_prompt(
            instruction_prompt,
            resolved_answer_instruction,
            "{question}",
        )
    else:
        prompts = [
            build_qa_prompt(
                instruction_prompt,
                resolved_answer_instruction,
                str(record["question"]),
                record["choices"],
            )
            for record in records
        ]
        template_choices = [
            {"label": choice["label"], "text": f"{{choice_{choice['label'].lower()}}}"}
            for choice in records[0]["choices"]
        ]
        prompt_template = build_qa_prompt(
            instruction_prompt,
            resolved_answer_instruction,
            "{question}",
            template_choices,
        )
    run_dir = prepare_run_directory(output_dir, mode_name, args.code, args.overwrite)
    (run_dir / "prompt.txt").write_text(
        prompt_template + "\n",
        encoding="utf-8",
    )

    print(f"CODE: {args.code}")
    print(f"Mode: {mode_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Task type: {task_type}")
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

    results, statistics = score_predictions(
        records,
        responses,
        token_usages,
        task_type,
    )
    summary = {
        "code": args.code,
        "mode": mode_name,
        "model": args.model,
        "dataset": str(dataset_path),
        "task_type": task_type,
        "test_range": {"start": args.start, "end": args.end},
        "instruction_prompt": instruction_prompt,
        "answer_instruction_prompt": resolved_answer_instruction,
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

    if task_type == "hotpotqa_open_qa":
        print(
            "HotpotQA answer EM: "
            f"{statistics['answer_exact_match_percent']:.2f}%"
        )
        print(
            "HotpotQA answer F1: "
            f"{statistics['answer_f1_percent']:.2f}%"
        )
    elif task_type == "open_qa":
        print(f"Official answer-set F1: {statistics['macro_f1_percent']:.2f}%")
        print(
            "Normalized answer-set F1: "
            f"{statistics['normalized_macro_f1_percent']:.2f}%"
        )
        print(
            f"Exact-set accuracy: {statistics['exact_match_count']}/"
            f"{statistics['total']} "
            f"({statistics['exact_match_accuracy_percent']:.2f}%)"
        )
    else:
        print(
            f"Accuracy: {statistics['correct']}/{statistics['total']} "
            f"({statistics['accuracy_percent']:.2f}%)"
        )
    print(f"Missing answer tags: {statistics['missing_answer_tags']}")
    if task_type == "multiple_choice_qa":
        print(f"Invalid choice labels: {statistics['invalid_choice_labels']}")
    print(f"Token usage: {statistics['token_usage']}")
    print(f"Elapsed inference time: {elapsed_seconds:.2f}s")
    print(f"Saved results to: {run_dir}")
