"""Generated-answer evaluation shared by all OpenBookQA optimizers."""

from __future__ import annotations

import hashlib
from statistics import fmean, pstdev
import time
from typing import Any, Sequence

from agents.agent_token_usage import summarize_token_usage
from prompt_optimization.models import ModelPool, TARGET_ROLE, seed_everything
from prompt_optimization.run_io import format_elapsed
from prompt_optimization.qa_task import (
    QAMode,
    render_qa_prompt,
    score_qa_response,
    summarize_qa_predictions,
)


def record_set_id(records: Sequence[dict[str, Any]]) -> str:
    """Create a stable short identifier for an ordered evaluation subset."""
    joined = "\n".join(str(record["id"]) for record in records)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


class QAEvaluator:
    """Run exact tagged-answer accuracy for arbitrary instruction prompts."""

    def __init__(
        self,
        *,
        model_pool: ModelPool,
        mode: QAMode,
        batch_size: int,
        max_new_tokens: int,
        seed: int,
        validation_std_penalty: float = 2.0,
        run_started_at: float | None = None,
    ):
        """Store target generation settings shared across evaluations."""
        if batch_size <= 0 or max_new_tokens <= 0:
            raise ValueError("Evaluation batch size and max tokens must be positive.")
        if validation_std_penalty < 0:
            raise ValueError("Validation standard-deviation penalty must be non-negative.")
        self.model_pool = model_pool
        self.mode = mode
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.validation_std_penalty = validation_std_penalty
        self.run_started_at = run_started_at or time.monotonic()

    def evaluate(
        self,
        instruction_prompt: str,
        records: Sequence[dict[str, Any]],
        *,
        split_name: str,
        log_label: str,
    ) -> dict[str, Any]:
        """Generate answers and return prompt-level and per-example results."""
        if not records:
            raise ValueError("Cannot evaluate an empty record subset.")
        evaluation_started_at = time.monotonic()
        print(
            f"[qa:{log_label}] evaluation started | split={split_name} | "
            f"examples={len(records)} | "
            f"total_elapsed={format_elapsed(evaluation_started_at - self.run_started_at)}",
            flush=True,
        )
        rendered_prompts = [
            render_qa_prompt(instruction_prompt, record, self.mode)
            for record in records
        ]
        subset_id = record_set_id(records)
        seed_material = f"{self.seed}\n{self.mode.name}\n{split_name}\n{subset_id}"
        evaluation_seed = self.seed + int(
            hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8],
            16,
        )
        self.model_pool.ensure(TARGET_ROLE)
        seed_everything(evaluation_seed)
        responses, token_usages = self.model_pool.generate(
            TARGET_ROLE,
            rendered_prompts,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            enable_thinking=self.mode.enable_thinking,
            do_sample=True,
            log_label=log_label,
            return_token_usage=True,
            seed=evaluation_seed,
        )
        predictions = []
        for record, response, token_usage in zip(records, responses, token_usages):
            prediction = score_qa_response(record, response)
            prediction.update(
                {
                    "question": record["question"],
                    "choices": record["choices"],
                    "answer_text": record.get("answer_text"),
                    "token_usage": dict(token_usage),
                }
            )
            if record.get("validation_fold") is not None:
                prediction["validation_fold"] = record["validation_fold"]
            predictions.append(prediction)
        metrics = summarize_qa_predictions(predictions)
        if split_name == "validation":
            metrics.update(
                validation_stability_metrics(
                    records,
                    predictions,
                    std_penalty=self.validation_std_penalty,
                )
            )
        metrics["token_usage"] = summarize_token_usage(token_usages)
        score_text = f"accuracy={100.0 * float(metrics['accuracy']):.2f}%"
        if "stable_accuracy" in metrics:
            score_text += (
                f" | stable={100.0 * float(metrics['stable_accuracy']):.2f}%"
                f" | fold_std={100.0 * float(metrics['accuracy_std']):.2f}pp"
            )
        completed_at = time.monotonic()
        print(
            f"[qa:{log_label}] evaluation completed | {score_text} | "
            f"phase_elapsed={format_elapsed(completed_at - evaluation_started_at)} | "
            f"total_elapsed={format_elapsed(completed_at - self.run_started_at)}",
            flush=True,
        )
        return {
            "split": split_name,
            "record_set_id": subset_id,
            "evaluation_seed": evaluation_seed,
            "instruction_prompt": instruction_prompt,
            "metrics": metrics,
            "predictions": predictions,
        }


def metric_accuracy(evaluation: dict[str, Any]) -> float:
    """Read exact accuracy from a QA evaluation result."""
    return float(evaluation["metrics"]["accuracy"])


def metric_selection_score(evaluation: dict[str, Any]) -> float:
    """Read stable validation accuracy, or plain accuracy for training subsets."""
    metrics = evaluation["metrics"]
    return float(metrics.get("stable_accuracy", metrics["accuracy"]))


def validation_stability_metrics(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    *,
    std_penalty: float,
) -> dict[str, Any]:
    """Calculate mean-minus-lambda-std accuracy over three fixed validation folds."""
    if len(records) != len(predictions):
        raise ValueError("Validation records and predictions must have equal lengths.")
    fold_totals: dict[str, int] = {}
    fold_correct: dict[str, int] = {}
    for record, prediction in zip(records, predictions):
        if record.get("validation_fold") is None:
            raise ValueError(
                "Every validation record must contain a validation_fold value."
            )
        fold = str(record["validation_fold"])
        fold_totals[fold] = fold_totals.get(fold, 0) + 1
        fold_correct[fold] = fold_correct.get(fold, 0) + int(
            bool(prediction["correct"])
        )
    if len(fold_totals) != 3:
        raise ValueError(
            "Stable validation scoring requires exactly three validation folds; "
            f"found {len(fold_totals)}."
        )
    if len(set(fold_totals.values())) != 1:
        raise ValueError(
            "Stable validation scoring requires equally sized validation folds."
        )
    fold_metrics = {
        fold: {
            "total": fold_totals[fold],
            "correct": fold_correct[fold],
            "accuracy": fold_correct[fold] / fold_totals[fold],
        }
        for fold in sorted(fold_totals, key=lambda value: int(value))
    }
    fold_accuracies = [item["accuracy"] for item in fold_metrics.values()]
    accuracy_mean = fmean(fold_accuracies)
    accuracy_std = pstdev(fold_accuracies)
    stable_accuracy = accuracy_mean - std_penalty * accuracy_std
    return {
        "validation_folds": fold_metrics,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "stable_accuracy": stable_accuracy,
        "stable_accuracy_percent": 100.0 * stable_accuracy,
        "validation_std_penalty": std_penalty,
    }


def select_mixed_feedback(
    records: Sequence[dict[str, Any]],
    evaluation: dict[str, Any],
    count: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Select feedback with at least one success and failure when available."""
    if count <= 0:
        return []
    pairs = list(zip(records, evaluation["predictions"]))
    correct = [pair for pair in pairs if pair[1]["correct"]]
    incorrect = [pair for pair in pairs if not pair[1]["correct"]]
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if correct:
        selected.append(correct[0])
    if incorrect and len(selected) < count:
        selected.append(incorrect[0])
    used_ids = {str(record["id"]) for record, _ in selected}
    selected.extend(
        pair
        for pair in pairs
        if str(pair[0]["id"]) not in used_ids
        and len(selected) < count
    )
    return selected[:count]


def select_incorrect_feedback(
    records: Sequence[dict[str, Any]],
    evaluation: dict[str, Any],
    count: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Select up to the requested number of incorrect predictions."""
    if count <= 0:
        return []
    return [
        (record, prediction)
        for record, prediction in zip(records, evaluation["predictions"])
        if not prediction["correct"]
    ][:count]
