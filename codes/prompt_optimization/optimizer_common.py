"""Selection, generation, and final reporting helpers for QA optimizers."""

from __future__ import annotations

import re
from typing import Any, Sequence

from prompt_optimization.cli_common import QAOptimizationContext
from prompt_optimization.evaluation import metric_accuracy, metric_selection_score
from prompt_optimization.meta_prompts import extract_tagged_prompts, unique_nonempty
from prompt_optimization.models import OPTIMIZER_ROLE
from prompt_optimization.run_io import save_json, save_text


FEEDBACK_PATTERN = re.compile(
    r"<feedback\s*>(.*?)</feedback\s*>", re.IGNORECASE | re.DOTALL
)


def extract_feedback(text: str) -> str:
    """Extract tagged feedback, falling back to the full response."""
    matches = FEEDBACK_PATTERN.findall(text)
    return (matches[-1] if matches else text).strip()


def generate_optimizer_texts(
    context: QAOptimizationContext,
    prompts: Sequence[str],
    *,
    log_label: str,
    enable_thinking: bool = True,
) -> list[str]:
    """Run meta-prompts through the configured reasoning-based optimizer model."""
    return context.model_pool.generate(
        OPTIMIZER_ROLE,
        list(prompts),
        max_new_tokens=context.args.optimizer_max_new_tokens,
        batch_size=context.args.optimizer_batch_size,
        enable_thinking=enable_thinking,
        do_sample=True,
        log_label=log_label,
        return_token_usage=False,
    )


def generate_tagged_candidates(
    context: QAOptimizationContext,
    meta_prompts: Sequence[str],
    *,
    log_label: str,
    include_prompts: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """Generate and parse tagged candidate instructions from meta-prompts."""
    raw_outputs = generate_optimizer_texts(
        context,
        meta_prompts,
        log_label=log_label,
    )
    candidates = unique_nonempty(
        [*include_prompts, *extract_tagged_prompts(raw_outputs)]
    )
    return candidates, raw_outputs


def evaluate_candidates(
    context: QAOptimizationContext,
    candidates: Sequence[str],
    records: Sequence[dict[str, Any]],
    *,
    split_name: str,
    phase: str,
    iteration: int,
) -> list[dict[str, Any]]:
    """Evaluate candidates and store their accuracy and applicable selection score."""
    scored: list[dict[str, Any]] = []
    for candidate_index, prompt in enumerate(unique_nonempty(candidates)):
        evaluation = context.evaluator.evaluate(
            prompt,
            records,
            split_name=split_name,
            log_label=f"qa_promptopt_{context.optimizer_name}_{phase}",
        )
        item = {
            "phase": phase,
            "iteration": iteration,
            "candidate_index": candidate_index,
            "prompt": prompt,
            "accuracy": metric_accuracy(evaluation),
            "selection_score": metric_selection_score(evaluation),
            "metrics": evaluation["metrics"],
            "evaluation": evaluation,
        }
        scored.append(item)
        context.logger.candidate(
            phase=phase,
            iteration=iteration,
            candidate_index=candidate_index,
            prompt=prompt,
            metrics=evaluation["metrics"],
        )
    return scored


def best_scored_candidate(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Choose the highest selection score, then accuracy and earlier order."""
    if not scored:
        raise ValueError("No scored candidates were supplied.")
    return max(
        scored,
        key=lambda item: (
            float(item.get("selection_score", item["accuracy"])),
            float(item["accuracy"]),
            -int(item["candidate_index"]),
        ),
    )


def finalize_run(
    context: QAOptimizationContext,
    *,
    initial_evaluation: dict[str, Any],
    final_prompt: str,
    final_evaluation: dict[str, Any],
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save final prompts, validation results, and the tuning-run summary."""
    save_text(context.run_dir / "final_prompt.txt", final_prompt)
    context.logger.evaluation("initial_validation", initial_evaluation)
    context.logger.evaluation("final_validation", final_evaluation)

    initial_selection_score = metric_selection_score(initial_evaluation)
    final_selection_score = metric_selection_score(final_evaluation)
    summary = {
        "code": context.args.code,
        "optimizer": context.optimizer_name,
        "qa_mode": context.mode.name,
        "backend": context.args.backend,
        "model": context.args.model,
        "optimizer_model": context.args.optimizer_model,
        "initial_prompt": context.initial_prompt,
        "final_prompt": final_prompt,
        "changed": final_prompt != context.initial_prompt,
        "validation": {
            "initial": initial_evaluation["metrics"],
            "final": final_evaluation["metrics"],
            "accuracy_gain": metric_accuracy(final_evaluation)
            - metric_accuracy(initial_evaluation),
            "stable_accuracy_gain": final_selection_score
            - initial_selection_score,
        },
        "test": None,
        **(extra_summary or {}),
    }
    save_json(context.run_dir / "summary.json", summary)
    context.logger.event(
        "run_completed",
        changed=summary["changed"],
        validation_accuracy_gain=summary["validation"]["accuracy_gain"],
        validation_stable_accuracy_gain=summary["validation"][
            "stable_accuracy_gain"
        ],
    )
    return summary
