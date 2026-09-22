"""QA task definitions shared by every prompt optimizer."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from anli_task_common import (
    ANLI_ANSWER_INSTRUCTION,
    ANLI_INITIAL_PROMPT,
    ANLI_REASONING_ANSWER_INSTRUCTION,
    ANLI_REASONING_INITIAL_PROMPT,
    anli_gold_relation,
    build_anli_prompt,
    normalize_anli_relation,
)
from math_grading.graders import grade_math_answer
from math_inference_common import (
    ANSWER_INSTRUCTION_PROMPT as MATH_ANSWER_INSTRUCTION_PROMPT,
    DEFAULT_INSTRUCTION_PROMPT as MATH_INITIAL_PROMPT,
    build_math_prompt,
    extract_math_answer,
    extract_tagged_answer as extract_math_tagged_answer,
    validate_math_records,
)
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
ANLI_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "anli" / "train.jsonl"
ANLI_VALIDATION_PATH = (
    REPO_ROOT / "data" / "processed" / "anli" / "validation_promptopt.jsonl"
)
# The HotpotQA experimental setup follows "Fine-Tuning and Prompt Optimization:
# Two Great Steps that Work Better Together."
HOTPOTQA_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "hotpotqa" / "train.jsonl"
HOTPOTQA_VALIDATION_PATH = (
    REPO_ROOT / "data" / "processed" / "hotpotqa" / "validation.jsonl"
)
MATH500_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "math500" / "train.jsonl"
MATH500_VALIDATION_PATH = (
    REPO_ROOT / "data" / "processed" / "math500" / "validation.jsonl"
)
MATH_OPTIMIZER_REASONING_MAX_TOKENS = 2_000

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

ANLI_MODES = {
    "reasoning": QAMode(
        name="reasoning",
        task_name="anli",
        initial_prompt=ANLI_REASONING_INITIAL_PROMPT,
        answer_instruction=ANLI_REASONING_ANSWER_INSTRUCTION,
        enable_thinking=True,
        default_max_new_tokens=1024,
    ),
    "non_reasoning": QAMode(
        name="non_reasoning",
        task_name="anli",
        initial_prompt=ANLI_INITIAL_PROMPT,
        answer_instruction=ANLI_ANSWER_INSTRUCTION,
        enable_thinking=False,
        default_max_new_tokens=16,
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

MATH500_MODES = {
    "reasoning": QAMode(
        name="reasoning",
        task_name="math500",
        initial_prompt=MATH_INITIAL_PROMPT,
        answer_instruction=MATH_ANSWER_INSTRUCTION_PROMPT,
        enable_thinking=True,
        default_max_new_tokens=8192,
    ),
}

# Preserve the original public name for code that assumes OpenBookQA.
QA_MODES = OPENBOOKQA_MODES


def resolve_mode(mode_name: str, task_name: str = "openbookqa") -> QAMode:
    """Return the fixed configuration for one QA task and reasoning mode."""
    task_modes = {
        "openbookqa": OPENBOOKQA_MODES,
        "anli": ANLI_MODES,
        "hotpotqa": HOTPOTQA_MODES,
        "math500": MATH500_MODES,
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
    if expected_task == "math500":
        task_type = validate_math_records(records)
    else:
        task_type = validate_records(records)
    expected_type = {
        "openbookqa": "multiple_choice_qa",
        "anli": "multiple_choice_qa",
        "hotpotqa": "hotpotqa_open_qa",
        "math500": "math_symbolic_answer",
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
    if mode.task_name == "math500":
        return build_math_prompt(instruction_prompt, str(record["question"]))
    if mode.task_name == "anli":
        return build_anli_prompt(
            instruction_prompt,
            mode.answer_instruction,
            str(record["premise"]),
            str(record["hypothesis"]),
        )
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
    if mode.task_name == "math500":
        tagged_answer = extract_math_tagged_answer(response)
        extracted, extraction_source = extract_math_answer(response)
        gold = str(record["answer"])
        grades = grade_math_answer(extracted, gold, str(record["task_type"]))
        return {
            "id": record["id"],
            "gold_answer": gold,
            "predicted_answer": extracted,
            "extracted_answer": extracted,
            "answer_extraction_source": extraction_source,
            "normalized_prediction": grades["normalized_prediction"],
            "normalized_gold_answer": grades["normalized_gold_answer"],
            "correct": bool(grades["openai_correct"]),
            "simple_correct": bool(grades["simple_correct"]),
            "openai_correct": bool(grades["openai_correct"]),
            "math_verify_correct": bool(grades["math_verify_correct"]),
            "openai_error": grades["openai_error"],
            "math_verify_error": grades["math_verify_error"],
            "missing_answer_tag": tagged_answer is None,
            "invalid_choice_label": False,
            "raw_response": response,
        }
    extracted = extract_tagged_answer(response)
    if mode.task_name == "anli":
        prediction = normalize_anli_relation(extracted)
        gold = anli_gold_relation(record)
        invalid = extracted is not None and prediction is None
        return {
            "id": record["id"],
            "gold_answer": gold,
            "predicted_answer": prediction,
            "extracted_answer": extracted,
            "correct": prediction == gold,
            "missing_answer_tag": extracted is None,
            "invalid_choice_label": invalid,
            "invalid_relationship": invalid,
            "raw_response": response,
        }
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
    if "openai_correct" in predictions[0]:
        simple_correct = sum(bool(item["simple_correct"]) for item in predictions)
        math_verify_correct = sum(
            bool(item["math_verify_correct"]) for item in predictions
        )
        extraction_counts = {"answer_tag": 0, "boxed_fallback": 0, "missing": 0}
        for item in predictions:
            extraction_counts[item["answer_extraction_source"]] += 1
        metrics.update(
            {
                "primary_grader": "openai_prm800k",
                "simple_correct": simple_correct,
                "simple_accuracy": simple_correct / total,
                "openai_correct": correct,
                "openai_accuracy": correct / total,
                "math_verify_correct": math_verify_correct,
                "math_verify_accuracy": math_verify_correct / total,
                "answer_extraction": extraction_counts,
                "grading_errors": {
                    "openai": sum(item["openai_error"] is not None for item in predictions),
                    "math_verify": sum(
                        item["math_verify_error"] is not None for item in predictions
                    ),
                },
            }
        )
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


def select_validation_fold_subset(
    records: Sequence[dict[str, Any]],
    fold_size: int | None,
) -> list[dict[str, Any]]:
    """Keep a deterministic equal-size prefix from each validation fold."""
    if fold_size is None:
        return list(records)
    if fold_size <= 0:
        raise ValueError("Validation fold size must be positive.")

    available: dict[str, int] = {}
    for record in records:
        if record.get("validation_fold") is None:
            raise ValueError(
                "Validation subsampling requires validation_fold on every record."
            )
        fold = str(record["validation_fold"])
        available[fold] = available.get(fold, 0) + 1
    if len(available) != 3:
        raise ValueError(
            "Validation subsampling requires exactly three folds; "
            f"found {len(available)}."
        )
    undersized = {
        fold: count for fold, count in available.items() if count < fold_size
    }
    if undersized:
        raise ValueError(
            f"Requested {fold_size} records per fold, but folds are too small: "
            f"{undersized}."
        )

    retained: dict[str, int] = {fold: 0 for fold in available}
    selected: list[dict[str, Any]] = []
    for record in records:
        fold = str(record["validation_fold"])
        if retained[fold] >= fold_size:
            continue
        selected.append(record)
        retained[fold] += 1
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
    mode: QAMode,
) -> str:
    """Format LPO feedback and retain the complete reasoning trace when used."""
    return rpo_feedback_example(record, prediction, index, mode)


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


def compact_math_reasoning( #aunabil:remember we are compacting math long reasoning for feedback
    response: str,
    tokenizer: Any | None = None,
    max_tokens: int = MATH_OPTIMIZER_REASONING_MAX_TOKENS,
) -> str:
    """Remove the tagged answer and retain both ends of a long math trace."""
    reasoning = reasoning_without_tagged_answer(response)
    if tokenizer is None:
        return reasoning
    token_ids = tokenizer.encode(reasoning, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return reasoning
    half = max_tokens // 2
    beginning = tokenizer.decode(token_ids[:half], skip_special_tokens=True)
    ending = tokenizer.decode(token_ids[-half:], skip_special_tokens=True)
    return (
        f"{beginning.rstrip()}\n\n"
        "[... middle of long reasoning omitted ...]\n\n"
        f"{ending.lstrip()}"
    )


def rpo_feedback_example(
    record: dict[str, Any],
    prediction: dict[str, Any],
    index: int,
    mode: QAMode,
    optimizer_tokenizer: Any | None = None,
    reasoning_max_tokens: int = MATH_OPTIMIZER_REASONING_MAX_TOKENS,
    output_token_limit: int | None = None,
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
    elif mode.task_name == "anli":
        lines.extend(
            [
                f"Premise: {record['premise']}",
                f"Hypothesis: {record['hypothesis']}",
                f"Ground-Truth Relationship: {anli_gold_relation(record)}",
            ]
        )
    elif mode.task_name == "math500":
        lines.extend(
            [
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
        raw_response = str(prediction.get("raw_response", ""))
        reasoning = (
            compact_math_reasoning( #aunabil: we are compacting this - remember
                raw_response,
                optimizer_tokenizer,
                reasoning_max_tokens,
            )
            if mode.task_name == "math500"
            else reasoning_without_tagged_answer(raw_response)
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
    if mode.task_name == "math500":
        output_tokens = prediction.get("token_usage", {}).get("output_tokens")
        if output_tokens is not None:
            lines.append(f"Generated output tokens: {output_tokens}")
        if output_token_limit is not None:
            lines.append(f"Output-token limit: {output_token_limit}")
        if not prediction.get("correct"):
            if prediction.get("predicted_answer") not in (None, ""):
                failure_type = "incorrect final answer"
            elif (
                output_tokens is not None
                and output_token_limit is not None
                and output_tokens >= output_token_limit
            ):
                failure_type = "reasoning token limit exceeded"
            else:
                failure_type = "missing final answer"
            lines.append(f"Failure type: {failure_type}")
    return "\n".join(lines)


def etgpo_failure_example(
    record: dict[str, Any],
    prediction: dict[str, Any],
    index: int,
    mode: QAMode,
    posthoc_feedback: str | None = None,
    optimizer_tokenizer: Any | None = None,
    reasoning_max_tokens: int = MATH_OPTIMIZER_REASONING_MAX_TOKENS,
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
        raw_response = str(prediction.get("raw_response", ""))
        reasoning = (
            compact_math_reasoning(
                raw_response,
                optimizer_tokenizer,
                reasoning_max_tokens,
            )
            if mode.task_name == "math500"
            else reasoning_without_tagged_answer(raw_response)
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
    elif mode.task_name != "math500":
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
