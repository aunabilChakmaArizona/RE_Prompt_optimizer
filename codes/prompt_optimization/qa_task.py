"""OpenBookQA task definitions shared by every prompt optimizer."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from qa_test_inference_common import (
    NON_REASONING_ANSWER_INSTRUCTION,
    NON_REASONING_INITIAL_PROMPT,
    REASONING_ANSWER_INSTRUCTION,
    REASONING_INITIAL_PROMPT,
    build_qa_prompt,
    extract_tagged_answer,
    normalize_choice_label,
    validate_records,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "openbookqa" / "train.jsonl"
DEFAULT_VALIDATION_PATH = (
    REPO_ROOT / "data" / "processed" / "openbookqa" / "validation.jsonl"
)
DEFAULT_TEST_PATH = REPO_ROOT / "data" / "processed" / "openbookqa" / "test.jsonl"

@dataclass(frozen=True)
class QAMode:
    """Store the fixed behavior for one QA prompting mode."""

    name: str
    initial_prompt: str
    answer_instruction: str
    enable_thinking: bool
    default_max_new_tokens: int


QA_MODES = {
    "reasoning": QAMode(
        name="reasoning",
        initial_prompt=REASONING_INITIAL_PROMPT,
        answer_instruction=REASONING_ANSWER_INSTRUCTION,
        enable_thinking=True,
        default_max_new_tokens=4096,
    ),
    "non_reasoning": QAMode(
        name="non_reasoning",
        initial_prompt=NON_REASONING_INITIAL_PROMPT,
        answer_instruction=NON_REASONING_ANSWER_INSTRUCTION,
        enable_thinking=False,
        default_max_new_tokens=16,
    ),
}


def resolve_mode(mode_name: str) -> QAMode:
    """Return the fixed configuration for a reasoning mode."""
    try:
        return QA_MODES[mode_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported QA mode: {mode_name!r}") from exc


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_qa_records(path_value: str | Path) -> list[dict[str, Any]]:
    """Load and validate one prepared OpenBookQA JSONL split."""
    path = resolve_repo_path(path_value)
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    validate_records(records)
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
    return build_qa_prompt(
        instruction_prompt,
        mode.answer_instruction,
        str(record["question"]),
        record["choices"],
    )


def score_qa_response(record: dict[str, Any], response: str) -> dict[str, Any]:
    """Extract one tagged option label and compare it with the gold label."""
    extracted = extract_tagged_answer(response)
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
    """Calculate exact multiple-choice accuracy and format-failure counts."""
    if not predictions:
        raise ValueError("Cannot summarize an empty prediction list.")
    correct = sum(bool(item["correct"]) for item in predictions)
    total = len(predictions)
    return {
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


def feedback_example(
    record: dict[str, Any],
    prediction: dict[str, Any],
    index: int,
) -> str:
    """Describe one prediction without exposing the dataset science fact."""
    predicted = prediction.get("predicted_answer") or "INVALID"
    outcome = "correct" if prediction.get("correct") else "incorrect"
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


def corpus_texts(records: Sequence[dict[str, Any]]) -> list[str]:
    """Collect question, choice, and fact text for copied-content checks."""
    texts: list[str] = []
    for record in records:
        texts.append(str(record.get("question", "")))
        texts.extend(str(choice.get("text", "")) for choice in record.get("choices", []))
        if record.get("fact"):
            texts.append(str(record["fact"]))
    return [text for text in texts if text.strip()]
