"""Selection, generation, and final reporting helpers for QA optimizers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from prompt_optimization.cli_common import QAOptimizationContext
from prompt_optimization.evaluation import metric_accuracy
from prompt_optimization.evaluation import record_set_id
from prompt_optimization.meta_prompts import extract_tagged_prompts, unique_nonempty
from prompt_optimization.models import OPTIMIZER_ROLE
from prompt_optimization.run_io import safe_name, save_json, save_text


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
    """Evaluate candidate instructions and log comparable exact accuracies."""
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
    """Choose highest accuracy, resolving ties by earlier candidate order."""
    if not scored:
        raise ValueError("No scored candidates were supplied.")
    return max(
        scored,
        key=lambda item: (float(item["accuracy"]), -int(item["candidate_index"])),
    )


def _test_cache_path(
    context: QAOptimizationContext,
    instruction_prompt: str,
) -> Path:
    """Build a shared test-evaluation cache path for one exact prompt and setup."""
    output_root = Path(context.args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = output_root.resolve()
    payload = "\n".join(
        [
            context.args.model,
            context.args.backend,
            context.mode.name,
            str(context.evaluator.max_new_tokens),
            str(context.args.seed),
            record_set_id(context.test_records),
            instruction_prompt,
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (
        output_root
        / "_test_cache"
        / safe_name(context.mode.name)
        / safe_name(context.args.model)
        / f"{digest}.json"
    )


def _evaluate_test_with_cache(
    context: QAOptimizationContext,
    instruction_prompt: str,
    log_label: str,
) -> dict[str, Any]:
    """Reuse an exact shared test evaluation across the six refiners of one source."""
    cache_path = _test_cache_path(context, instruction_prompt)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    evaluation = context.evaluator.evaluate(
        instruction_prompt,
        context.test_records,
        split_name="test",
        log_label=log_label,
    )
    save_json(cache_path, evaluation)
    return evaluation


def finalize_run(
    context: QAOptimizationContext,
    *,
    initial_evaluation: dict[str, Any],
    final_prompt: str,
    final_evaluation: dict[str, Any],
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save final prompts, validation results, optional test results, and summary."""
    save_text(context.run_dir / "final_prompt.txt", final_prompt)
    context.logger.evaluation("initial_validation", initial_evaluation)
    context.logger.evaluation("final_validation", final_evaluation)

    test_payload = None
    if context.args.evaluate_test:
        initial_test = _evaluate_test_with_cache(
            context,
            context.initial_prompt,
            log_label=f"qa_promptopt_{context.optimizer_name}_initial_test",
        )
        context.logger.evaluation("initial_test", initial_test)
        if final_prompt == context.initial_prompt:
            final_test = initial_test
        else:
            final_test = _evaluate_test_with_cache(
                context,
                final_prompt,
                log_label=f"qa_promptopt_{context.optimizer_name}_final_test",
            )
            context.logger.evaluation("final_test", final_test)
        test_payload = {
            "initial": initial_test["metrics"],
            "final": final_test["metrics"],
            "accuracy_gain": metric_accuracy(final_test) - metric_accuracy(initial_test),
        }

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
        },
        "test": test_payload,
        **(extra_summary or {}),
    }
    save_json(context.run_dir / "summary.json", summary)
    context.logger.event(
        "run_completed",
        changed=summary["changed"],
        validation_accuracy_gain=summary["validation"]["accuracy_gain"],
    )
    return summary
