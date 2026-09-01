"""Generated-answer evaluation shared by all OpenBookQA optimizers."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from agents.agent_token_usage import summarize_token_usage
from prompt_optimization.models import ModelPool, TARGET_ROLE, seed_everything
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
    ):
        """Store target generation settings shared across evaluations."""
        if batch_size <= 0 or max_new_tokens <= 0:
            raise ValueError("Evaluation batch size and max tokens must be positive.")
        self.model_pool = model_pool
        self.mode = mode
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.seed = seed

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
            predictions.append(prediction)
        metrics = summarize_qa_predictions(predictions)
        metrics["token_usage"] = summarize_token_usage(token_usages)
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
