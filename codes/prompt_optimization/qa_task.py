"""QA task definitions shared by every prompt optimizer."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from qa_test_inference_common import (
    NON_REASONING_ANSWER_INSTRUCTION,
    NON_REASONING_INITIAL_PROMPT,
    REASONING_ANSWER_INSTRUCTION,
    REASONING_INITIAL_PROMPT,
    build_qa_prompt,
    build_context_open_qa_prompt,
    extract_tagged_answer,
    hotpot_answer_scores,
    normalize_choice_label,
    validate_records,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "openbookqa" / "train.jsonl"
DEFAULT_VALIDATION_PATH = (
    REPO_ROOT / "data" / "processed" / "openbookqa" / "validation.jsonl"
)
DEFAULT_TEST_PATH = REPO_ROOT / "data" / "processed" / "openbookqa" / "test.jsonl"
# The HotpotQA experimental setup follows "Fine-Tuning and Prompt Optimization:
# Two Great Steps that Work Better Together."
HOTPOTQA_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "hotpotqa" / "train.jsonl"
HOTPOTQA_VALIDATION_PATH = (
    REPO_ROOT / "data" / "processed" / "hotpotqa" / "validation.jsonl"
)

HOTPOTQA_REASONING_INITIAL_PROMPT = (
    "You are given context passages and a question that may require combining "
    "information from multiple passages. Think step by step carefully and provide "
    "the best answer using the context."
)
HOTPOTQA_REASONING_ANSWER_INSTRUCTION = (
    "After you finish reasoning, output only the final answer between the tags "
    "<answer> and </answer>."
)

@dataclass(frozen=True)
class QAMode:
    """Store the fixed behavior for one QA prompting mode."""

    name: str
    task_name: str
    initial_prompt: str
    answer_instruction: str
    enable_thinking: bool
    default_max_new_tokens: int


OPENBOOKQA_MODES = {
    "reasoning": QAMode(
        name="reasoning",
        task_name="openbookqa",
        initial_prompt=REASONING_INITIAL_PROMPT,
        answer_instruction=REASONING_ANSWER_INSTRUCTION,
        enable_thinking=True,
        default_max_new_tokens=4096,
    ),
    "non_reasoning": QAMode(
        name="non_reasoning",
        task_name="openbookqa",
        initial_prompt=NON_REASONING_INITIAL_PROMPT,
        answer_instruction=NON_REASONING_ANSWER_INSTRUCTION,
        enable_thinking=False,
        default_max_new_tokens=10,
    ),
}

HOTPOTQA_MODES = {
    "reasoning": QAMode(
        name="reasoning",
        task_name="hotpotqa",
        initial_prompt=HOTPOTQA_REASONING_INITIAL_PROMPT,
        answer_instruction=HOTPOTQA_REASONING_ANSWER_INSTRUCTION,
        enable_thinking=True,
        default_max_new_tokens=4096,
    ),
}

# Preserve the original public name for code that assumes OpenBookQA.
QA_MODES = OPENBOOKQA_MODES


def resolve_mode(mode_name: str, task_name: str = "openbookqa") -> QAMode:
    """Return the fixed configuration for one QA task and reasoning mode."""
    task_modes = {
        "openbookqa": OPENBOOKQA_MODES,
        "hotpotqa": HOTPOTQA_MODES,
    }
    try:
        return task_modes[task_name][mode_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported QA task/mode combination: {task_name!r}/{mode_name!r}"
        ) from exc


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_qa_records(
    path_value: str | Path,
    expected_task: str | None = None,
) -> list[dict[str, Any]]:
    """Load and validate one prepared QA JSONL split."""
    path = resolve_repo_path(path_value)
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    task_type = validate_records(records)
    expected_type = {
        "openbookqa": "multiple_choice_qa",
        "hotpotqa": "hotpotqa_open_qa",
    }.get(expected_task)
    if expected_type is not None and task_type != expected_type:
        raise ValueError(
            f"Expected {expected_task} records ({expected_type}), found {task_type}."
        )
    return records


def render_qa_prompt(
    instruction_prompt: str,
    record: dict[str, Any],
    mode: QAMode,
) -> str:
    """Render an optimized instruction with the fixed answer format and question."""
    instruction_prompt = instruction_prompt.strip()
    if not instruction_prompt:
        raise ValueError("The instruction prompt must not be empty.")
    if mode.task_name == "hotpotqa":
        return build_context_open_qa_prompt(
            instruction_prompt,
            mode.answer_instruction,
            record["context"],
            str(record["question"]),
        )
    return build_qa_prompt(
        instruction_prompt,
        mode.answer_instruction,
        str(record["question"]),
        record["choices"],
    )


def score_qa_response(
    record: dict[str, Any],
    response: str,
    mode: QAMode,
) -> dict[str, Any]:
    """Extract and score one tagged answer for the selected QA task."""
    extracted = extract_tagged_answer(response)
    if mode.task_name == "hotpotqa":
        prediction = extracted or ""
        gold = str(record["answer"])
        exact_match, precision, recall, f1 = hotpot_answer_scores(prediction, gold)
        return {
            "id": record["id"],
            "gold_answer": gold,
            "predicted_answer": prediction,
            "extracted_answer": extracted,
            "correct": bool(exact_match),
            "answer_exact_match": exact_match,
            "answer_precision": precision,
            "answer_recall": recall,
            "answer_f1": f1,
            "missing_answer_tag": extracted is None,
            "invalid_choice_label": False,
            "raw_response": response,
        }
    valid_labels = {
        str(choice["label"]).strip().upper() for choice in record["choices"]
    }
    prediction = normalize_choice_label(extracted, valid_labels)
    gold = str(record["answer"]).strip().upper()
    return {
        "id": record["id"],
        "gold_answer": gold,
        "predicted_answer": prediction,
        "extracted_answer": extracted,
        "correct": prediction == gold,
        "missing_answer_tag": extracted is None,
        "invalid_choice_label": extracted is not None and prediction is None,
        "raw_response": response,
    }


def summarize_qa_predictions(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Calculate exact-match accuracy and task-specific diagnostic metrics."""
    if not predictions:
        raise ValueError("Cannot summarize an empty prediction list.")
    correct = sum(bool(item["correct"]) for item in predictions)
    total = len(predictions)
    metrics = {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
        "accuracy_percent": 100.0 * correct / total,
        "missing_answer_tags": sum(
            bool(item["missing_answer_tag"]) for item in predictions
        ),
        "invalid_choice_labels": sum(
            bool(item["invalid_choice_label"]) for item in predictions
        ),
    }
    if "answer_f1" in predictions[0]:
        metrics.update(
            {
                "answer_exact_match": metrics["accuracy"],
                "answer_exact_match_percent": metrics["accuracy_percent"],
                "answer_precision": sum(
                    float(item["answer_precision"]) for item in predictions
                ) / total,
                "answer_recall": sum(
                    float(item["answer_recall"]) for item in predictions
                ) / total,
                "answer_f1": sum(float(item["answer_f1"]) for item in predictions) / total,
                "answer_f1_percent": 100.0
                * sum(float(item["answer_f1"]) for item in predictions)
                / total,
            }
        )
    return metrics


def sample_records(
    records: Sequence[dict[str, Any]],
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample records without replacement using the experiment RNG."""
    if sample_size <= 0 or sample_size >= len(records):
        return list(records)
    return rng.sample(list(records), sample_size)


def sample_label_balanced_records(
    records: Sequence[dict[str, Any]],
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample approximately equal numbers of A, B, C, and D answers."""
    if sample_size <= 0 or sample_size >= len(records):
        return list(records)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["answer"]).strip().upper(), []).append(record)
    labels = sorted(groups)
    if not labels:
        return []
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    per_label = sample_size // len(labels)
    for label in labels:
        selected.extend(groups[label][:per_label])
    selected_ids = {str(record["id"]) for record in selected}
    remaining = [
        record for record in records if str(record["id"]) not in selected_ids
    ]
    rng.shuffle(remaining)
    selected.extend(remaining[: sample_size - len(selected)])
    rng.shuffle(selected)
    return selected


def choices_as_text(record: dict[str, Any]) -> str:
    """Format choices compactly for optimizer feedback meta-prompts."""
    return " | ".join(
        f"{str(choice['label']).strip().upper()}. {str(choice['text']).strip()}"
        for choice in record["choices"]
    )


def context_as_text(record: dict[str, Any]) -> str:
    """Format HotpotQA context passages for optimizer feedback."""
    lines = []
    for index, paragraph in enumerate(record.get("context", []), start=1):
        sentences = " ".join(str(item).strip() for item in paragraph["sentences"])
        lines.append(f"[{index}] {paragraph['title']}: {sentences}")
    return "\n".join(lines)


def feedback_example(
    record: dict[str, Any],
    prediction: dict[str, Any],
    index: int,
) -> str:
    """Describe one prediction without exposing the dataset science fact."""
    predicted = prediction.get("predicted_answer") or "INVALID"
    outcome = "correct" if prediction.get("correct") else "incorrect"
    if record.get("task_type") == "hotpotqa_open_qa":
        return "\n".join(
            [
                f"Example {index}",
                f"Context: {context_as_text(record)}",
                f"Question: {record['question']}",
                f"Gold answer: {record['answer']}",
                f"Predicted answer: {predicted}",
                f"Outcome: {outcome}",
            ]
        )
    return "\n".join(
        [
            f"Example {index}",
            f"Question: {record['question']}",
            f"Choices: {choices_as_text(record)}",
            f"Gold option: {record['answer']}",
            f"Predicted option: {predicted}",
            f"Outcome: {outcome}",
        ]
    )


def reasoning_without_tagged_answer(response: str) -> str:
    """Remove the tagged final answer while preserving the model's reasoning text."""
    reasoning = re.sub(
        r"<answer\s*>.*?</answer\s*>",
        "",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    reasoning = re.sub(
        r"<answer\s*>.*$",
        "",
        reasoning,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(r"\n{3,}", "\n\n", reasoning).strip()


def rpo_feedback_example(
    record: dict[str, Any],
    prediction: dict[str, Any],
    index: int,
    mode: QAMode,
) -> str:
    """Format RPO feedback with reasoning only when the QA mode produces it."""
    predicted = prediction.get("predicted_answer") or "INVALID"
    outcome = "correct" if prediction.get("correct") else "incorrect"
    lines = [f"Task {index}"]
    if mode.task_name == "hotpotqa":
        lines.extend(
            [
                f"Context: {context_as_text(record)}",
                f"Question: {record['question']}",
                f"Ground-Truth Answer: {record['answer']}",
            ]
        )
    else:
        lines.extend(
            [
                f"Question: {record['question']}",
                f"Choices: {choices_as_text(record)}",
                f"Ground-Truth Answer: {record['answer']}",
            ]
        )
    if mode.name == "reasoning":
        reasoning = reasoning_without_tagged_answer(
            str(prediction.get("raw_response", ""))
        )
        lines.extend(
            [
                "LLM Reasoning:",
                reasoning or "No reasoning text was provided.",
            ]
        )
    lines.extend(
        [
            f"LLM Selected Answer: {predicted}",
            f"Outcome: {outcome}",
        ]
    )
    return "\n".join(lines)


def etgpo_failure_example(
    record: dict[str, Any],
    prediction: dict[str, Any],
    index: int,
    mode: QAMode,
    posthoc_feedback: str | None = None,
) -> str:
    """Format ETGPO failures without exposing the fixed answer format."""
    predicted = prediction.get("predicted_answer") or "INVALID"
    reasoning_section: list[str] = []
    if posthoc_feedback is not None:
        reasoning_section = [
            "### Most Likely Reasoning Behind the Incorrect Prediction",
            posthoc_feedback.strip() or "No feedback generated.",
            "",
        ]
    elif mode.name == "reasoning":
        reasoning = reasoning_without_tagged_answer(
            str(prediction.get("raw_response", ""))
        )
        reasoning_section = [
            "### Model's Reasoning",
            reasoning or "No reasoning text was provided.",
            "",
        ]
    task_input = ["### Question", str(record["question"]), ""]
    if mode.task_name == "hotpotqa":
        task_input = [
            "### Context",
            context_as_text(record),
            "",
            *task_input,
        ]
    else:
        task_input.extend(["### Choices", choices_as_text(record), ""])
    return "\n".join(
        [
            f"## Failure {index}",
            f"Problem ID: {record['id']}",
            "",
            *task_input,
            "### Correct Answer",
            str(record["answer"]),
            "",
            *reasoning_section,
            "### Model's Selected Answer",
            str(predicted),
            "",
            "---",
        ]
    )


def corpus_texts(records: Sequence[dict[str, Any]]) -> list[str]:
    """Collect question, choice, and fact text for copied-content checks."""
    texts: list[str] = []
    for record in records:
        texts.append(str(record.get("question", "")))
        texts.extend(str(choice.get("text", "")) for choice in record.get("choices", []))
        if record.get("fact"):
            texts.append(str(record["fact"]))
    return [text for text in texts if text.strip()]
