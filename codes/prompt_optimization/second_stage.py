"""QA implementations of LPO, GreaTer, and the three GradPO variants."""

from __future__ import annotations

import random
import re
import time
from typing import Any, Sequence

try:
    import torch
except ImportError:
    torch = None

from agents.agent_prompts import GRADIENT_REGION_CANDIDATE_SYNTHESIS_TAGGED_BODY_V1
from prompt_optimization.cli_common import QAOptimizationContext
from prompt_optimization.evaluation import (
    metric_accuracy,
    metric_selection_score,
    select_incorrect_feedback,
)
from prompt_optimization.gradient_cache import (
    balanced_subset_cache_key,
    build_gradient_pool_identity,
    gradient_pool_cache_path,
    load_gradient_cache,
    lock_gradient_cache,
    prediction_outcome_hash,
    save_gradient_cache,
)
from prompt_optimization.meta_prompts import (
    extract_json_object,
    gradpo_candidate_prompt,
    lpo_location_prompt,
    lpo_rewrite_prompt,
    qa_task_description,
    qa_task_label,
    unique_nonempty,
)
from prompt_optimization.models import TARGET_ROLE
from prompt_optimization.optimizer_common import (
    best_scored_candidate,
    evaluate_candidates,
    finalize_run,
    generate_optimizer_texts,
    log_progress,
    scored_item_text,
)
from prompt_optimization.qa_task import (
    QAMode,
    feedback_example,
    render_qa_prompt,
    sample_records,
    score_qa_response,
)
from prompt_optimization.run_io import format_elapsed, save_json
from prompt_optimization.source_validation_cache import (
    DEFAULT_SOURCE_VALIDATION_CACHE_ROOT,
    build_source_validation_identity,
    load_source_validation_cache,
    lock_source_validation_cache,
    save_source_validation_cache,
    source_validation_cache_path,
)
from prompt_optimization.sequence_gradients import (
    build_gradient_region_pool,
    collect_instruction_gradients,
    mark_selected_regions,
    model_device,
    proposal_token_candidates,
    rank_fixed_token_candidates,
    replace_selected_regions,
    score_combined_objectives,
    select_gradient_regions,
    same_length_single_token_replacement_prompt,
    tensor_free_gradient_summary,
)


def _source_scored_item(
    context: QAOptimizationContext,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Represent the initial first-stage prompt as a scored candidate."""
    return {
        "phase": "source_validation",
        "iteration": 0,
        "candidate_index": 0,
        "prompt": context.initial_prompt,
        "accuracy": metric_accuracy(evaluation),
        "selection_score": metric_selection_score(evaluation),
        "metrics": evaluation["metrics"],
        "evaluation": evaluation,
    }


def _sample_second_stage_records(
    context: QAOptimizationContext,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Randomly sample training records for non-gradient second-stage methods."""
    return sample_records(context.train_records, sample_size, context.rng)


def _balance_gradient_pairs(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    rng: random.Random,
    sample_size: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    """Select equal random counts of correct and incorrect model results."""
    if len(records) != len(predictions):
        raise ValueError("Gradient records and predictions must have equal lengths.")
    if sample_size <= 0 or sample_size % 2 != 0:
        raise ValueError("Gradient sample size must be a positive even number.")
    pairs = list(zip(records, predictions))
    correct = [pair for pair in pairs if bool(pair[1]["correct"])]
    incorrect = [pair for pair in pairs if not bool(pair[1]["correct"])]
    requested_per_outcome = sample_size // 2
    selected_per_outcome = min(
        requested_per_outcome,
        len(correct),
        len(incorrect),
    )
    selected = [
        *rng.sample(correct, selected_per_outcome),
        *rng.sample(incorrect, selected_per_outcome),
    ]
    rng.shuffle(selected)
    counts = {
        "available_correct": len(correct),
        "available_incorrect": len(incorrect),
        "requested_total": sample_size,
        "requested_per_outcome": requested_per_outcome,
        "selected_correct": selected_per_outcome,
        "selected_incorrect": selected_per_outcome,
        "selected_total": 2 * selected_per_outcome,
        "shortfall": sample_size - 2 * selected_per_outcome,
    }
    return selected, counts


def _gradient_pool_cache_rows(
    records: Sequence[dict[str, Any]],
    responses: Sequence[str],
    predictions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build inspectable cache rows containing every raw target-model response."""
    if not (len(records) == len(responses) == len(predictions)):
        raise ValueError("Cached records, responses, and predictions must align.")
    return [
        {
            "record_id": str(record["id"]),
            "raw_response": response,
            "gold_answer": prediction.get("gold_answer"),
            "predicted_answer": prediction.get("predicted_answer"),
            "correct": bool(prediction["correct"]),
        }
        for record, response, prediction in zip(records, responses, predictions)
    ]


def _responses_from_gradient_cache(
    cache_payload: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> list[str]:
    """Restore raw responses while verifying the cached record order."""
    rows = cache_payload["pool_results"]
    cached_ids = [str(row["record_id"]) for row in rows]
    expected_ids = [str(record["id"]) for record in records]
    if cached_ids != expected_ids:
        raise ValueError("Cached gradient-pool record order does not match its metadata.")
    return [str(row["raw_response"]) for row in rows]


def _balanced_subset_entry(
    selected_pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    counts: dict[str, int],
    *,
    gradient_sample_size: int,
    balance_seed: int,
    outcome_hash: str,
) -> dict[str, Any]:
    """Serialize one exact seeded correct/incorrect gradient subset."""
    return {
        "gradient_sample_size": int(gradient_sample_size),
        "balance_seed": int(balance_seed),
        "outcome_hash": outcome_hash,
        "selected_record_ids": [
            str(record["id"]) for record, _prediction in selected_pairs
        ],
        "correct_record_ids": [
            str(record["id"])
            for record, prediction in selected_pairs
            if bool(prediction["correct"])
        ],
        "incorrect_record_ids": [
            str(record["id"])
            for record, prediction in selected_pairs
            if not bool(prediction["correct"])
        ],
        "counts": dict(counts),
    }


def _restore_balanced_pairs(
    entry: dict[str, Any],
    records: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    """Restore the cached balanced subset in its original shuffled order."""
    pairs_by_id = {
        str(record["id"]): (record, prediction)
        for record, prediction in zip(records, predictions)
    }
    if len(pairs_by_id) != len(records):
        raise ValueError("Gradient-pool records must have unique IDs for caching.")
    selected_ids = [str(record_id) for record_id in entry["selected_record_ids"]]
    if any(record_id not in pairs_by_id for record_id in selected_ids):
        raise ValueError("Cached balanced subset references an unavailable record ID.")
    selected_pairs = [pairs_by_id[record_id] for record_id in selected_ids]
    counts = {key: int(value) for key, value in entry["counts"].items()}
    return selected_pairs, counts


def _build_balanced_gradient_subset(
    context: QAOptimizationContext,
    instruction_prompt: str,
    sample_size: int,
    gradient_sample_size: int,
    *,
    log_label: str,
) -> tuple[list[dict[str, Any]], list[str] | None, dict[str, Any]]:
    """Reuse or create a shared pool, then restore one seeded balanced subset."""
    pool_seed = int(context.args.seed)
    balance_seed = int(context.args.seed)
    pool_rng = random.Random(pool_seed)
    pool_records = sample_records(context.train_records, sample_size, pool_rng)
    rendered_prompts = [
        render_qa_prompt(instruction_prompt, record, context.mode)
        for record in pool_records
    ]
    pool_started_at = time.monotonic()
    generation_backend = (
        "vllm" if context.model_pool.uses_vllm_generation else "transformers"
    )
    cache_identity = build_gradient_pool_identity(
        model_id=context.args.model,
        generation_backend=generation_backend,
        task_name=context.mode.task_name,
        mode_name=context.mode.name,
        source_prompt=instruction_prompt,
        pool_records=pool_records,
        rendered_prompts=rendered_prompts,
        pool_seed=pool_seed,
        requested_pool_size=sample_size,
        max_new_tokens=context.evaluator.max_new_tokens,
        enable_thinking=context.mode.enable_thinking,
    )
    cache_path, cache_key = gradient_pool_cache_path(
        context.args.gradient_cache_root,
        cache_identity,
    )
    cache_enabled = not context.args.disable_gradient_cache
    pool_cache_hit = False
    balanced_subset_cache_hit = False

    def generate_pool_responses() -> list[str]:
        """Run the one expensive greedy inference used by every gradient refiner."""
        log_progress(
            context,
            f"gradient pool inference started | examples={len(pool_records)}",
        )
        return context.model_pool.generate(
            TARGET_ROLE,
            rendered_prompts,
            max_new_tokens=context.evaluator.max_new_tokens,
            batch_size=context.evaluator.batch_size,
            enable_thinking=context.mode.enable_thinking,
            do_sample=False,
            log_label=log_label,
            return_token_usage=False,
            seed=pool_seed,
        )

    if cache_enabled:
        log_progress(context, f"gradient cache lookup | path={cache_path}")
        with lock_gradient_cache(cache_path):
            cache_payload = None
            if not context.args.refresh_gradient_cache:
                cache_payload = load_gradient_cache(cache_path, cache_identity)
            if cache_payload is None:
                responses = generate_pool_responses()
                cache_payload = {
                    "metadata": cache_identity,
                    "pool_results": [],
                    "balanced_subsets": {},
                }
                log_progress(context, "gradient pool cache miss | generated responses")
            else:
                responses = _responses_from_gradient_cache(
                    cache_payload,
                    pool_records,
                )
                context.model_pool.record_cached_generation()
                pool_cache_hit = True
                log_progress(context, "gradient pool cache hit | restored raw responses")

            predictions = [
                score_qa_response(record, response, context.mode)
                for record, response in zip(pool_records, responses)
            ]
            cache_payload["pool_results"] = _gradient_pool_cache_rows(
                pool_records,
                responses,
                predictions,
            )
            outcome_hash = prediction_outcome_hash(predictions)
            subset_key = balanced_subset_cache_key(
                gradient_sample_size=gradient_sample_size,
                balance_seed=balance_seed,
                outcome_hash=outcome_hash,
            )
            subset_entry = cache_payload["balanced_subsets"].get(subset_key)
            if subset_entry is None:
                selected_pairs, counts = _balance_gradient_pairs(
                    pool_records,
                    predictions,
                    random.Random(balance_seed),
                    gradient_sample_size,
                )
                subset_entry = _balanced_subset_entry(
                    selected_pairs,
                    counts,
                    gradient_sample_size=gradient_sample_size,
                    balance_seed=balance_seed,
                    outcome_hash=outcome_hash,
                )
                cache_payload["balanced_subsets"][subset_key] = subset_entry
                log_progress(context, "balanced gradient subset cache miss | selected")
            else:
                selected_pairs, counts = _restore_balanced_pairs(
                    subset_entry,
                    pool_records,
                    predictions,
                )
                balanced_subset_cache_hit = True
                log_progress(context, "balanced gradient subset cache hit | restored")
            save_gradient_cache(cache_path, cache_payload)
    else:
        log_progress(context, "gradient cache disabled")
        responses = generate_pool_responses()
        predictions = [
            score_qa_response(record, response, context.mode)
            for record, response in zip(pool_records, responses)
        ]
        outcome_hash = prediction_outcome_hash(predictions)
        subset_key = balanced_subset_cache_key(
            gradient_sample_size=gradient_sample_size,
            balance_seed=balance_seed,
            outcome_hash=outcome_hash,
        )
        selected_pairs, counts = _balance_gradient_pairs(
            pool_records,
            predictions,
            random.Random(balance_seed),
            gradient_sample_size,
        )

    if counts["shortfall"] > 0:
        log_progress(
            context,
            "WARNING: requested gradient subset unavailable; using largest "
            f"balanced subset | requested_total={counts['requested_total']} | "
            f"selected_total={counts['selected_total']} | "
            f"available_correct={counts['available_correct']} | "
            f"available_incorrect={counts['available_incorrect']}",
        )
    selected_records = [record for record, _prediction in selected_pairs]
    reasoning_traces = (
        [prediction["raw_response"] for _record, prediction in selected_pairs]
        if context.mode.enable_thinking
        else None
    )
    metadata = {
        "strategy": "random_pool_then_equal_correct_incorrect",
        "pool_seed": pool_seed,
        "balance_seed": balance_seed,
        "initial_pool_size": len(pool_records),
        **counts,
        "cache": {
            "enabled": cache_enabled,
            "path": str(cache_path) if cache_enabled else None,
            "cache_key": cache_key if cache_enabled else None,
            "balanced_subset_key": subset_key if cache_enabled else None,
            "pool_cache_hit": pool_cache_hit,
            "balanced_subset_cache_hit": balanced_subset_cache_hit,
        },
        "initial_pool_record_ids": [str(record["id"]) for record in pool_records],
        "selected_examples": [
            {
                "id": str(record["id"]),
                "correct": bool(prediction["correct"]),
                "gold_answer": prediction.get("gold_answer"),
                "predicted_answer": prediction.get("predicted_answer"),
            }
            for record, prediction in selected_pairs
        ],
    }
    log_progress(
        context,
        "gradient pool balancing completed | "
        f"available_correct={counts['available_correct']} | "
        f"available_incorrect={counts['available_incorrect']} | "
        f"selected_correct={counts['selected_correct']} | "
        f"selected_incorrect={counts['selected_incorrect']}",
        phase_started_at=pool_started_at,
    )
    return selected_records, reasoning_traces, metadata


def _strictly_select_against_source(
    source: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Select the best validation candidate only if it beats the source prompt."""
    if not candidates:
        return source, False
    best_candidate = best_scored_candidate(candidates)
    improved = float(best_candidate["selection_score"]) > float(
        source["selection_score"]
    )
    return (best_candidate if improved else source), improved


def _score_objective_candidates(
    context: QAOptimizationContext,
    prompts: Sequence[str],
    records: Sequence[dict[str, Any]],
    *,
    model,
    tokenizer,
    batch_size: int,
    fluency_lambda: float,
    reasoning_traces: Sequence[str] | None,
) -> list[dict[str, float]]:
    """Batch candidate task-loss and fluency calculations with HF."""
    return score_combined_objectives(
        prompts,
        records,
        mode=context.mode,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        fluency_lambda=fluency_lambda,
        reasoning_traces=reasoning_traces,
    )


def _prepare_final_evaluation_backend(
    context: QAOptimizationContext,
    args,
) -> str:
    """Switch regular gradient runs from HF to vLLM for final validation."""
    requested = args.final_evaluation_backend
    if requested == "vllm" and context.model_pool.backend == "transformers":
        log_progress(
            context,
            "switching from HF to vLLM for batched final validation",
        )
        context.model_pool.activate_vllm_only(args.gpu_memory_utilization)
    return "vllm" if context.model_pool.uses_vllm_generation else "transformers"


def _evaluate_with_cached_source(
    context: QAOptimizationContext,
    candidate_prompts: Sequence[str],
    *,
    phase: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Reuse one exact source evaluation while scoring all new candidates."""
    candidates = unique_nonempty(
        [prompt for prompt in candidate_prompts if prompt != context.initial_prompt]
    )
    cache_disabled = bool(
        getattr(context.args, "disable_source_validation_cache", False)
    )
    if cache_disabled:
        scored = evaluate_candidates(
            context,
            [context.initial_prompt, *candidates],
            context.validation_records,
            split_name="validation",
            phase=phase,
            iteration=1,
        )
        source_candidate = next(
            item for item in scored if item["prompt"] == context.initial_prompt
        )
        initial_evaluation = source_candidate["evaluation"]
        return (
            initial_evaluation,
            _source_scored_item(context, initial_evaluation),
            [item for item in scored if item["prompt"] != context.initial_prompt],
        )

    generation_backend = (
        "vllm" if context.model_pool.uses_vllm_generation else "transformers"
    )
    rendered_source_prompts = [
        render_qa_prompt(context.initial_prompt, record, context.mode)
        for record in context.validation_records
    ]
    identity = build_source_validation_identity(
        model_id=context.args.model,
        generation_backend=generation_backend,
        task_name=context.mode.task_name,
        mode_name=context.mode.name,
        source_prompt=context.initial_prompt,
        validation_records=context.validation_records,
        rendered_prompts=rendered_source_prompts,
        seed=context.evaluator.seed,
        max_new_tokens=context.evaluator.max_new_tokens,
        enable_thinking=context.mode.enable_thinking,
        validation_std_penalty=context.evaluator.validation_std_penalty,
    )
    cache_root = getattr(
        context.args,
        "source_validation_cache_root",
        str(DEFAULT_SOURCE_VALIDATION_CACHE_ROOT),
    )
    cache_path, cache_key = source_validation_cache_path(cache_root, identity)
    refresh_cache = bool(
        getattr(context.args, "refresh_source_validation_cache", False)
    )
    log_progress(context, f"source validation cache lookup | path={cache_path}")

    with lock_source_validation_cache(cache_path):
        cached = None
        if not refresh_cache:
            cached = load_source_validation_cache(cache_path, identity)
        if cached is None:
            scored = evaluate_candidates(
                context,
                [context.initial_prompt, *candidates],
                context.validation_records,
                split_name="validation",
                phase=phase,
                iteration=1,
            )
            source_candidate = next(
                item for item in scored if item["prompt"] == context.initial_prompt
            )
            initial_evaluation = source_candidate["evaluation"]
            save_source_validation_cache(
                cache_path,
                {
                    "metadata": identity,
                    "cache_key": cache_key,
                    "evaluation": initial_evaluation,
                },
            )
            context.logger.event(
                "source_validation_cache_miss",
                cache_path=str(cache_path),
                cache_key=cache_key,
            )
            log_progress(context, "source validation cache miss | evaluated and saved")
            return (
                initial_evaluation,
                _source_scored_item(context, initial_evaluation),
                [item for item in scored if item["prompt"] != context.initial_prompt],
            )

    initial_evaluation = cached["evaluation"]
    context.logger.event(
        "source_validation_cache_hit",
        cache_path=str(cache_path),
        cache_key=cache_key,
    )
    log_progress(context, "source validation cache hit | source generation skipped")
    dev_scored = []
    if candidates:
        dev_scored = evaluate_candidates(
            context,
            candidates,
            context.validation_records,
            split_name="validation",
            phase=phase,
            iteration=1,
        )
    return (
        initial_evaluation,
        _source_scored_item(context, initial_evaluation),
        dev_scored,
    )


def _evaluate_source_and_candidates(
    context: QAOptimizationContext,
    args,
    candidate_prompts: Sequence[str],
    *,
    phase: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    """Evaluate the source and all retained prompts together on validation."""
    final_backend = _prepare_final_evaluation_backend(context, args)
    initial_evaluation, source, candidates = _evaluate_with_cached_source(
        context,
        candidate_prompts,
        phase=phase,
    )
    return initial_evaluation, source, candidates, final_backend


def _extract_lpo_prompt(raw_output: str) -> str:
    """Extract one full LPO prompt while leaving its edit tags untouched."""
    prompt = raw_output.strip()
    fenced = re.search(
        r"```(?:text)?\s*(.*?)\s*```",
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        prompt = fenced.group(1).strip()
    for pattern in (r"\[P\](.*?)\[/P\]", r"<p>(.*?)</p>"):
        wrapped = re.search(pattern, prompt, flags=re.DOTALL | re.IGNORECASE)
        if wrapped:
            return wrapped.group(1).strip()
    return prompt


def _clean_lpo_candidate_prompt(text: str) -> str:
    """Remove LPO's output wrappers and local edit tags from one candidate."""
    cleaned = _extract_lpo_prompt(text)
    cleaned = re.sub(r"</?edit>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _lpo_candidate_prompts_from_outputs(
    raw_outputs: Sequence[str],
    fallback_prompt: str,
) -> list[str]:
    """Extract each rewritten LPO prompt and remove its edit tags."""
    return unique_nonempty(
        [
            _clean_lpo_candidate_prompt(candidate)
            for candidate in [fallback_prompt, *raw_outputs]
        ]
    )


def run_lpo(context: QAOptimizationContext, args) -> dict[str, Any]:
    """Run one reasoning-based local prompt optimization step for QA."""
    log_progress(context, "run started | iterations=1")
    train_records = _sample_second_stage_records(context, args.train_sample_size)
    train_evaluation = context.evaluator.evaluate(
        context.initial_prompt,
        train_records,
        split_name="train_selection",
        log_label="qa_lpo_train_selection",
    )
    feedback_pairs = select_incorrect_feedback(
        train_records,
        train_evaluation,
        args.feedback_examples,
    )
    log_progress(
        context,
        f"feedback selection completed | incorrect_examples={len(feedback_pairs)}",
    )
    feedback_texts = [
        feedback_example(record, prediction, index, context.mode)
        for index, (record, prediction) in enumerate(feedback_pairs, start=1)
    ]
    location_meta_prompt = lpo_location_prompt(
        context.initial_prompt,
        context.mode,
        feedback_texts,
        args.max_locations,
        args.max_words_per_location,
    )
    location_output = generate_optimizer_texts(
        context,
        [location_meta_prompt],
        log_label="qa_lpo_location_tagging",
    )[0]
    tagged_prompt = _extract_lpo_prompt(location_output)
    locations = [
        {
            "location_rank": index,
            "text": match.group(1),
        }
        for index, match in enumerate(
            re.finditer(
                r"<edit>(.*?)</edit>",
                tagged_prompt,
                flags=re.DOTALL | re.IGNORECASE,
            ),
            start=1,
        )
    ]
    log_progress(
        context,
        f"location tagging completed | selected_locations={len(locations)}",
    )
    save_json(
        context.run_dir / "selected_spans.json",
        {
            "method": "lpo",
            "raw_location_output": location_output,
            "tagged_prompt": tagged_prompt,
            "locations": locations,
        },
    )

    candidate_prompts: list[str] = []
    rewrite_outputs: list[str] = []
    rewrite_meta_prompt = None
    if locations:
        rewrite_meta_prompt = lpo_rewrite_prompt(
            tagged_prompt,
            feedback_texts,
            context.mode,
        )
        rewrite_outputs = generate_optimizer_texts(
            context,
            [rewrite_meta_prompt] * args.num_candidates,
            log_label="qa_lpo_local_rewrite",
        )
        candidate_prompts = [
            prompt
            for prompt in _lpo_candidate_prompts_from_outputs(
                rewrite_outputs,
                context.initial_prompt,
            )
            if prompt != context.initial_prompt
        ]
    log_progress(
        context,
        f"local rewriting completed | parsed_candidates={len(candidate_prompts)}",
    )
    save_json(
        context.run_dir / "optimizer_trace.json",
        {
            "location_meta_prompt": location_meta_prompt,
            "location_raw_output": location_output,
            "tagged_prompt": tagged_prompt,
            "parsed_locations": locations,
            "feedback_examples": feedback_texts,
            "rewrite_meta_prompt": rewrite_meta_prompt,
            "rewrite_raw_outputs": rewrite_outputs,
            "parsed_candidate_prompts": candidate_prompts,
        },
    )

    initial_evaluation, source, dev_scored = _evaluate_with_cached_source(
        context,
        candidate_prompts,
        phase="lpo_candidate_validation",
    )
    selected, improved = _strictly_select_against_source(source, dev_scored)
    log_progress(
        context,
        f"iteration 1/1 completed | improved={improved} | "
        f"current {scored_item_text(selected)} | source {scored_item_text(source)} | "
        f"best {scored_item_text(selected)}",
    )
    return finalize_run(
        context,
        initial_evaluation=initial_evaluation,
        final_prompt=selected["prompt"],
        final_evaluation=selected["evaluation"],
        extra_summary={
            "algorithm": "lpo",
            "iterations": 1,
            "improved_on_validation": improved,
            "selected_location_count": len(locations),
            "feedback_record_ids": [
                record["id"] for record, _ in feedback_pairs
            ],
            "raw_rewrite_outputs": rewrite_outputs,
        },
    )


def _sequential_greater_region(
    gradient_analysis: dict[str, Any],
    start_position: int,
    check_isalnum: bool,
) -> dict[str, Any]:
    """Choose a sequential token with an optional alphanumeric-content check."""
    prompt = str(gradient_analysis["instruction_prompt"])
    records = list(gradient_analysis["token_gradients"])
    for record in records:
        token_index = int(record["token_index"])
        start = int(record["char_start"])
        end = int(record["char_end"])
        text = prompt[start:end]
        if token_index >= start_position and (
            not check_isalnum
            or any(character.isalnum() for character in text)
        ):
            return {
                "region_rank": 1,
                "peak_token_index": token_index,
                "start_token": token_index,
                "end_token": token_index,
                "token_indices": [token_index],
                "token_count": 1,
                "start_char": start,
                "end_char": end,
                "region_text": text,
                "text": text,
                "gradient_score": float(record["gradient_norm"]),
                "selection_mode": "sequential",
                "check_isalnum": check_isalnum,
            }
    raise ValueError("No editable token exists at or after --start-position.")


def run_greater(context: QAOptimizationContext, args) -> dict[str, Any]: #aunabil3rd: where is the initial results of caching goin on? we talked about this where the GreaTer and GradPO have a common initial trainset run and we pick the balanced subset which is supposed to be cached
    """Run one GreaTer sequential or top-gradient single-token refinement."""
    log_progress(context, "run started | iterations=1")
    train_records, reasoning_traces, gradient_sample = (
        _build_balanced_gradient_subset(
            context,
            context.initial_prompt,
            args.train_sample_size,
            args.gradient_sample_size,
            log_label=f"qa_{args.variant}_gradient_pool_inference",
        )
    )
    save_json(context.run_dir / "gradient_sample.json", gradient_sample)
    model, tokenizer = context.model_pool.ensure(TARGET_ROLE)
    gradient_started_at = time.monotonic()
    log_progress(
        context,
        f"gradient collection started | examples={len(train_records)} | "
        f"batch_size={args.gradient_batch_size}",
    )
    gradient_analysis = collect_instruction_gradients(
        context.initial_prompt,
        train_records,
        mode=context.mode,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.gradient_batch_size,
        reasoning_traces=reasoning_traces,
    )
    log_progress(
        context,
        "gradient collection completed",
        phase_started_at=gradient_started_at,
    )
    if args.variant == "greater":
        region = _sequential_greater_region(
            gradient_analysis,
            args.start_position,
            args.check_isalnum,
        )
    else:
        pool = build_gradient_region_pool(
            gradient_analysis,
            max_region_tokens=1,
            expansion_threshold_ratio=args.region_expansion_threshold,
            check_isalnum=args.check_isalnum,
        )
        region = select_gradient_regions(
            pool,
            count=1,
            selection_mode="top_gradient",
            rng=context.rng,
        )[0]
        region["text"] = region["region_text"]
        region["check_isalnum"] = args.check_isalnum
    log_progress(
        context,
        f"region selection completed | token_index={region['peak_token_index']} | "
        f"text={region['region_text']!r} | gradient={float(region['gradient_score']):.6f}",
    )
    proposal_records = train_records[: min(args.proposal_example_size, len(train_records))]
    proposal_started_at = time.monotonic()
    log_progress(
        context,
        f"token proposal started | examples={len(proposal_records)}",
    )
    proposed_token_ids, proposal_metadata = proposal_token_candidates(
        gradient_analysis,
        int(region["peak_token_index"]),
        proposal_records,
        mode=context.mode,
        model=model,
        tokenizer=tokenizer,
        top_k=args.proposal_top_k,
        min_candidates=args.proposal_min_candidates,
        check_isalnum=args.check_isalnum,
    )
    log_progress(
        context,
        f"token proposal completed | proposed_tokens={len(proposed_token_ids)}",
        phase_started_at=proposal_started_at,
    )
    ranking_started_at = time.monotonic()
    log_progress(
        context,
        f"gradient candidate ranking started | candidates={len(proposed_token_ids)}",
    )
    ranked_candidate_tokens = rank_fixed_token_candidates(
        gradient_analysis,
        int(region["peak_token_index"]),
        proposed_token_ids,
        train_records,
        mode=context.mode,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.gradient_batch_size,
        fluency_lambda=args.fluency_lambda,
        reasoning_traces=reasoning_traces,
    )
    log_progress(
        context,
        f"gradient candidate ranking completed | ranked={len(ranked_candidate_tokens)}",
        phase_started_at=ranking_started_at,
    )
    candidate_tokens = ranked_candidate_tokens[: args.selection_top_mu]
    if not any(item["is_original"] for item in candidate_tokens):
        original = next(
            item
            for item in ranked_candidate_tokens
            if item["is_original"]
        )
        candidate_tokens.append(original)
    source_token_ids = list(gradient_analysis["token_ids"])
    token_index = int(region["peak_token_index"])
    stable_candidate_tokens = []
    candidate_prompts = [context.initial_prompt]
    for item in candidate_tokens:
        candidate_token_ids = list(source_token_ids)
        candidate_token_ids[token_index] = int(item["token_id"])
        candidate_prompt = (
            context.initial_prompt
            if bool(item["is_original"])
            else same_length_single_token_replacement_prompt(
                tokenizer,
                source_token_ids,
                token_index,
                int(item["token_id"]),
            )
        )
        if candidate_prompt is None:
            continue
        stable_candidate_tokens.append(
            {
                **item,
                "prompt": candidate_prompt,
                "token_index": token_index,
                "candidate_token_ids": candidate_token_ids,
            }
        )
        candidate_prompts.append(candidate_prompt)
    candidate_tokens = stable_candidate_tokens
    candidate_prompts = unique_nonempty(candidate_prompts)
    objective_started_at = time.monotonic()
    log_progress(
        context,
        f"batched objective scoring started | candidates={len(candidate_prompts)} | "
        f"examples={len(train_records)}",
    )
    candidate_objectives = _score_objective_candidates(
        context,
        candidate_prompts,
        train_records,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.selection_batch_size,
        fluency_lambda=args.fluency_lambda,
        reasoning_traces=reasoning_traces,
    )
    objective_scores = [
        {"candidate_index": index, "prompt": prompt, **score}
        for index, (prompt, score) in enumerate(
            zip(candidate_prompts, candidate_objectives)
        )
    ]
    best_objective = min(
        objective_scores,
        key=lambda item: (
            float(item["combined_score"]),
            int(item["candidate_index"]),
        ),
    )
    log_progress(
        context,
        "batched objective scoring completed | "
        f"best={float(best_objective['combined_score']):.6f}",
        phase_started_at=objective_started_at,
    )
    top_objective = sorted(
        objective_scores,
        key=lambda item: (float(item["combined_score"]), int(item["candidate_index"])),
    )[: args.top_u]
    dev_prompts = [
        item["prompt"] for item in top_objective if item["prompt"] != context.initial_prompt
    ]
    save_json(
        context.run_dir / "gradient_analysis.json",
        tensor_free_gradient_summary(gradient_analysis),
    )
    save_json(
        context.run_dir / "selected_spans.json",
        {"method": args.variant, "regions": [region]},
    )
    save_json(
        context.run_dir / "token_candidates.json",
        {
            "proposal_metadata": proposal_metadata,
            "gradient_ranking_method": "candidate_one_hot_combined_loss",
            "gradient_ranking_record_count": len(train_records),
            "gradient_ranking_fluency_lambda": args.fluency_lambda,
            "gradient_ranked_candidates": ranked_candidate_tokens,
            "candidates": candidate_tokens,
            "objective_scores": objective_scores,
        },
    )
    del gradient_analysis, model, tokenizer
    initial_evaluation, source, dev_scored, final_backend = (
        _evaluate_source_and_candidates(
            context,
            args,
            dev_prompts,
            phase=f"{args.variant}_candidate_validation",
        )
    )
    selected, improved = _strictly_select_against_source(source, dev_scored)
    log_progress(
        context,
        f"iteration 1/1 completed | improved={improved} | "
        f"current {scored_item_text(selected)} | source {scored_item_text(source)} | "
        f"best {scored_item_text(selected)}",
    )
    return finalize_run(
        context,
        initial_evaluation=initial_evaluation,
        final_prompt=selected["prompt"],
        final_evaluation=selected["evaluation"],
        extra_summary={
            "algorithm": args.variant,
            "iterations": 1,
            "improved_on_validation": improved,
            "objective_task_loss_backend": "transformers",
            "final_evaluation_backend": final_backend,
            "gradient_sampling": {
                key: value
                for key, value in gradient_sample.items()
                if key not in {"initial_pool_record_ids", "selected_examples"}
            },
            "selected_region": region,
            "proposal_metadata": proposal_metadata,
            "top_objective_candidates": top_objective,
        },
    )


def _parse_gradpo_gen_candidates(
    raw_output: str,
    regions: Sequence[dict[str, Any]],
    candidate_count: int,
    mode: QAMode,
) -> list[dict[str, Any]]:
    """Parse per-region GradPO-Gen replacements and retain each source span."""
    parsed = extract_json_object(raw_output) or {}
    output = []
    for region in regions:
        rank = int(region["region_rank"])
        region_payload = parsed.get(f"span_{rank}", {})
        values = region_payload.get("candidates", []) if isinstance(region_payload, dict) else []
        if not isinstance(values, list):
            values = []
        candidates, rejected_candidates = _filter_gradpo_region_candidates(
            str(region["region_text"]),
            [str(value) for value in values],
            candidate_count,
            mode,
        )
        output.append(
            {
                "region_rank": rank,
                "region_text": region["region_text"],
                "candidate_source": "target_model_generation",
                "candidates": candidates,
                "rejected_candidates": rejected_candidates,
            }
        )
    return output


def _normalized_gradpo_candidate(text: str) -> str:
    """Normalize a replacement for comparison with task-specific restrictions."""
    normalized = text.strip().casefold().replace("–", "-").replace("—", "-")
    return re.sub(r"\s*-\s*", "-", normalized)


def _gradpo_candidate_restriction_reason(
    candidate: str,
    source_text: str,
    mode: QAMode,
) -> str | None:
    """Explain why one new GradPO replacement is forbidden for the current task."""
    if candidate == source_text:
        return None
    if (
        mode.task_name == "openbookqa"
        and _normalized_gradpo_candidate(candidate) == "true-false"
    ):
        return "restricted OpenBookQA replacement: true-false"
    return None


def _filter_gradpo_region_candidates(
    source_text: str,
    proposed_candidates: Sequence[str],
    candidate_count: int,
    mode: QAMode,
) -> tuple[list[str], list[dict[str, str]]]:
    """Keep the source and allowed unique replacements up to the requested count."""
    candidates = [source_text]
    rejected = []
    for candidate in unique_nonempty([str(value) for value in proposed_candidates]):
        if candidate == source_text or candidate in candidates:
            continue
        reason = _gradpo_candidate_restriction_reason(candidate, source_text, mode)
        if reason is not None:
            rejected.append({"candidate": candidate, "reason": reason})
            continue
        candidates.append(candidate)
        if len(candidates) >= candidate_count + 1:
            break
    return candidates, rejected


def _top_p_first_token_ids(
    logits,
    *,
    tokenizer,
    maximum_count: int,
    top_p: float,
    original_token_id: int | None,
) -> list[int]:
    """Collect a nucleus of valid first-token candidates in probability order."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    search_count = min(probabilities.numel(), max(maximum_count * 8, maximum_count))
    special_ids = set(tokenizer.all_special_ids)
    while True:
        values, indices = torch.topk(probabilities, k=search_count)
        collected = []
        cumulative = 0.0
        non_original = 0
        for probability, token_id in zip(values.tolist(), indices.tolist()):
            cumulative += float(probability)
            token_id = int(token_id)
            if token_id in special_ids:
                continue
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not text:
                continue
            collected.append(token_id)
            if original_token_id is None or token_id != original_token_id:
                non_original += 1
            if len(collected) >= maximum_count:
                break
            if cumulative >= top_p and non_original >= 1:
                break
        if collected and (
            len(collected) >= maximum_count
            or (cumulative >= top_p and non_original >= 1)
            or search_count == probabilities.numel()
        ):
            return collected
        search_count = min(probabilities.numel(), search_count * 2)


def _probability_rewrite_context(
    instruction_prompt: str,
    prompt_prefix: str,
) -> str:
    """Match GradPO-Prob's full-original-prompt rewrite context."""
    return (
        "Original Prompt:\n"
        f"```{instruction_prompt}```\n\n"
        "Task:\n"
        "Write a natural revised version of the original prompt while preserving "
        "meaning, structure, and tone. Prefer paraphrase or other clear, robust, "
        "and effective wording.\n\n"
        "Revised prompt:\n"
        f"```{prompt_prefix.strip()}"
    )


def _gradpo_probability_candidates(
    instruction_prompt: str,
    regions: Sequence[dict[str, Any]],
    *,
    mode: QAMode,
    candidate_count: int,
    model,
    tokenizer,
    max_new_tokens: int,
    generation_batch_size: int,
) -> list[dict[str, Any]]:
    """Generate fixed-token-length span candidates from target-LM probabilities."""
    if torch is None:
        raise ImportError("GradPO-Prob requires PyTorch in the active environment.")
    from agents.agent_llm_prompting import run_prompts

    device = model_device(model)
    probability_top_p = 0.95
    output = []
    special_ids = set(tokenizer.all_special_ids)
    for region in regions:
        prefix = instruction_prompt[: int(region["start_char"])]
        probability_context = _probability_rewrite_context(
            instruction_prompt,
            prefix,
        )
        prefix_ids = tokenizer(
            probability_context,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)
        if prefix_ids.size(1) == 0:
            fallback_id = tokenizer.bos_token_id
            if fallback_id is None:
                fallback_id = tokenizer.eos_token_id
            if fallback_id is None:
                raise ValueError("Tokenizer has no BOS or EOS token for an empty prefix.")
            prefix_ids = torch.tensor([[fallback_id]], device=device)
        with torch.inference_mode():
            next_logits = model(input_ids=prefix_ids, use_cache=False).logits[:, -1, :]
        original_region_ids = tokenizer.encode(
            str(region["region_text"]),
            add_special_tokens=False,
        )
        original_first_id = original_region_ids[0] if original_region_ids else None
        top_ids = _top_p_first_token_ids(
            next_logits[0],
            tokenizer=tokenizer,
            maximum_count=max(candidate_count, candidate_count * candidate_count),
            top_p=probability_top_p,
            original_token_id=original_first_id,
        )[:candidate_count]
        candidates = [str(region["region_text"])]
        candidate_details = []
        rejected_candidates = []
        first_token_ids = [
            int(first_id) for first_id in top_ids if int(first_id) not in special_ids
        ]
        prefix_token_ids = prefix_ids[0].detach().cpu().tolist()
        first_token_texts = [
            tokenizer.decode(
                [first_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for first_id in first_token_ids
        ]
        continuation_prompts = [
            tokenizer.decode(
                [*prefix_token_ids, first_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for first_id in first_token_ids
        ]
        remaining_region_tokens = max(0, int(region["token_count"]) - 1)
        if remaining_region_tokens > 0:
            continuation_outputs = run_prompts(
                continuation_prompts,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=min(max_new_tokens, remaining_region_tokens),
                batch_size=generation_batch_size,
                use_chat_template=False,
                add_generation_prompt=False,
                enable_thinking=False,
                do_sample=True,
                do_log=True,
                log_label=f"qa_gradpo_prob_region_candidates_r{region['region_rank']}",
            )
        else:
            continuation_outputs = [""] * len(continuation_prompts)
        for first_id, first_token_text, continuation_output in zip(
            first_token_ids,
            first_token_texts,
            continuation_outputs,
        ):
            text = first_token_text + continuation_output
            if (
                not text.strip()
                or not text.isascii()
                or "\n" in text
                or not any(character.isalnum() for character in text)
            ):
                continue
            restriction_reason = _gradpo_candidate_restriction_reason(
                text,
                str(region["region_text"]),
                mode,
            )
            if restriction_reason is not None:
                rejected_candidates.append(
                    {"candidate": text, "reason": restriction_reason}
                )
                continue
            candidates = unique_nonempty([*candidates, text])
            candidate_details.append(
                {
                    "first_token_id": first_id,
                    "first_token_text": first_token_text,
                    "continuation_output": continuation_output,
                    "text": text,
                }
            )
            if len(candidates) >= candidate_count + 1:
                break
        output.append(
            {
                "region_rank": int(region["region_rank"]),
                "region_text": region["region_text"],
                "candidate_source": "target_lm_probability",
                "lm_probability_context": "full_prompt_as_context",
                "lm_probability_top_p": probability_top_p,
                "candidates": candidates,
                "candidate_details": candidate_details,
                "rejected_candidates": rejected_candidates,
            }
        )
    return output


def _gradpo_synthesis_prompt(
    instruction_prompt: str,
    regions: Sequence[dict[str, Any]],
    replacements: dict[int, str],
    mode: QAMode,
) -> str:
    """Adapt the relation-extraction GradPO synthesis prompt to one QA mode."""
    marked_prompt = mark_selected_regions(instruction_prompt, regions)
    replacement_blocks = []
    for region in regions:
        rank = int(region["region_rank"])
        replacement_blocks.append(
            "\n".join(
                [
                    f"Span {rank}",
                    f"Text: ```{region['region_text']}```",
                    "Replace with: "
                    f"```{replacements.get(rank, str(region['region_text']))}```",
                ]
            )
        )
    qa_prompt = "\n\n".join(
        [
            f"You are an expert prompt generator for a {qa_task_label(mode)} task.",
            qa_task_description(mode),
            GRADIENT_REGION_CANDIDATE_SYNTHESIS_TAGGED_BODY_V1,
        ]
    )
    return (
        qa_prompt
        .replace("#ALL_MARKED_PROMPT#", marked_prompt)
        .replace("#SELECTED_REPLACEMENTS#", "\n\n".join(replacement_blocks))
    )


def _normalize_synthesized_prompt(raw_output: str) -> str:
    """Extract the last complete prompt block and remove leaked span tags."""
    text = raw_output.strip()
    fenced = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    prompt_blocks = re.findall(
        r"<prompt\s*>(.*?)</prompt\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not prompt_blocks:
        return ""
    text = prompt_blocks[-1].strip()
    text = re.sub(r"</?span_\d+>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _beam_search_replacements(
    context: QAOptimizationContext,
    regions: Sequence[dict[str, Any]],
    region_candidates: Sequence[dict[str, Any]],
    train_records: Sequence[dict[str, Any]],
    *,
    model,
    tokenizer,
    beam_width: int,
    selection_batch_size: int,
    fluency_lambda: float,
    replacement_mode: str,
    synthesis_max_new_tokens: int,
    synthesis_batch_size: int,
    reasoning_traces: Sequence[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Combine local replacements and keep the lowest-loss prompt beam."""
    candidate_index = {
        int(item["region_rank"]): item for item in region_candidates
    }
    beam = [
        {
            "prompt": context.initial_prompt,
            "replacements": {},
            "history": [],
        }
    ]
    trace = []
    synthesis_cache: dict[tuple[tuple[int, str], ...], dict[str, Any]] = {
        (): {
            "prompt": context.initial_prompt,
            "meta_prompt": None,
            "raw_output": None,
            "rejected": False,
            "rejection_reason": None,
            "no_op": True,
        }
    }
    beam_started_at = time.monotonic()
    for region_index, region in enumerate(regions, start=1):
        region_started_at = time.monotonic()
        rank = int(region["region_rank"])
        log_progress(
            context,
            f"beam region {region_index}/{len(regions)} started | "
            f"region_rank={rank} | text={region['region_text']!r} | "
            f"incoming_beam={len(beam)}",
        )
        expansion_specs = []
        for beam_item in beam:
            for replacement in candidate_index[rank]["candidates"]:
                replacements = dict(beam_item["replacements"])
                history = list(beam_item["history"])
                if str(replacement) == str(region["region_text"]):
                    replacements.pop(rank, None)
                else:
                    replacements[rank] = str(replacement)
                    history.append(rank)
                expansion_specs.append(
                    {
                        "replacements": replacements,
                        "history": history,
                    }
                )
        if replacement_mode == "llm_synthesis":
            missing_keys = []
            missing_prompts = []
            for spec in expansion_specs:
                key = tuple(sorted(spec["replacements"].items()))
                if key in synthesis_cache or key in missing_keys:
                    continue
                complete_replacements = {
                    int(item["region_rank"]): spec["replacements"].get(
                        int(item["region_rank"]),
                        str(item["region_text"]),
                    )
                    for item in regions
                }
                missing_keys.append(key)
                missing_prompts.append(
                    _gradpo_synthesis_prompt(
                        context.initial_prompt,
                        regions,
                        complete_replacements,
                        context.mode,
                    )
                )
            if missing_prompts:
                raw_outputs = context.model_pool.generate(
                    TARGET_ROLE,
                    missing_prompts,
                    max_new_tokens=synthesis_max_new_tokens,
                    batch_size=synthesis_batch_size,
                    enable_thinking=False,
                    do_sample=False,
                    log_label="qa_gradpo_beam_synthesis",
                    return_token_usage=False,
                )
                for key, meta_prompt, raw_output in zip(
                    missing_keys,
                    missing_prompts,
                    raw_outputs,
                ):
                    revised = _normalize_synthesized_prompt(raw_output)
                    rejection_reason = None
                    if not revised:
                        rejection_reason = "missing complete <prompt> block"
                        context.logger.event(
                            "gradpo_synthesis_rejected",
                            replacements=dict(key),
                            reason=rejection_reason,
                        )
                    synthesis_cache[key] = {
                        "prompt": revised,
                        "meta_prompt": meta_prompt,
                        "raw_output": raw_output,
                        "rejected": rejection_reason is not None,
                        "rejection_reason": rejection_reason,
                        "no_op": False,
                    }
        elif replacement_mode != "direct":
            raise ValueError(f"Unsupported beam replacement mode: {replacement_mode!r}")

        expansions: dict[str, dict[str, Any]] = {}
        for spec in expansion_specs:
            key = tuple(sorted(spec["replacements"].items()))
            if replacement_mode == "llm_synthesis":
                synthesis = synthesis_cache[key]
                if synthesis["rejected"]:
                    continue
                prompt = synthesis["prompt"]
                replacement_metadata = {
                    "replacement_mode": replacement_mode,
                    "meta_prompt": synthesis["meta_prompt"],
                    "raw_output": synthesis["raw_output"],
                    "rejected": synthesis["rejected"],
                    "rejection_reason": synthesis["rejection_reason"],
                    "no_op": synthesis["no_op"],
                }
            else:
                prompt = replace_selected_regions(
                    context.initial_prompt,
                    regions,
                    spec["replacements"],
                )
                replacement_metadata = {
                    "replacement_mode": replacement_mode,
                    "meta_prompt": None,
                    "raw_output": None,
                    "rejected": False,
                    "rejection_reason": None,
                    "no_op": not bool(spec["replacements"]),
                }
            expansions.setdefault(
                prompt,
                {
                    "prompt": prompt,
                    "replacements": spec["replacements"],
                    "history": spec["history"],
                    "replacement_metadata": replacement_metadata,
                },
            )
        expansion_list = list(expansions.values())
        objectives = _score_objective_candidates(
            context,
            [item["prompt"] for item in expansion_list],
            train_records,
            model=model,
            tokenizer=tokenizer,
            batch_size=selection_batch_size,
            fluency_lambda=fluency_lambda,
            reasoning_traces=reasoning_traces,
        )
        scored_expansions = [
            {"candidate_index": index, **expansion, **objective}
            for index, (expansion, objective) in enumerate(
                zip(expansion_list, objectives)
            )
        ]
        beam = sorted(
            scored_expansions,
            key=lambda item: (
                float(item["combined_score"]),
                int(item["candidate_index"]),
            ),
        )[:beam_width]
        trace.append(
            {
                "region_rank": rank,
                "expansion_count": len(scored_expansions),
                "retained_beam": beam,
            }
        )
        context.logger.event(
            "gradpo_beam_region_completed",
            region_rank=rank,
            expansion_count=len(scored_expansions),
            retained_beam=beam,
        )
        best_objective = float(beam[0]["combined_score"]) if beam else float("inf")
        log_progress(
            context,
            f"beam region {region_index}/{len(regions)} completed | "
            f"expansions={len(scored_expansions)} | retained={len(beam)} | "
            f"current_best_objective={best_objective:.6f} | "
            f"beam_elapsed={format_elapsed(time.monotonic() - beam_started_at)}",
            phase_started_at=region_started_at,
        )
    return beam, trace


def _resolve_gradpo_shape(args, model_id: str) -> tuple[int, int]:
    """Apply the paper's Qwen or Gemma span-count and span-length defaults."""
    normalized = model_id.casefold()
    default_count = 5 if "qwen" in normalized else 3
    default_length = 2 if "qwen" in normalized else 3
    return (
        args.num_edit_regions or default_count,
        args.max_region_tokens or default_length,
    )


def run_gradpo(context: QAOptimizationContext, args) -> dict[str, Any]:
    """Run GradPO-Gen, GradPO-Prob, or random-region GradPO-Gen for QA."""
    log_progress(context, "run started | iterations=1")
    train_records, reasoning_traces, gradient_sample = (
        _build_balanced_gradient_subset(
            context,
            context.initial_prompt,
            args.train_sample_size,
            args.gradient_sample_size,
            log_label=f"qa_gradpo_{args.variant}_gradient_pool_inference",
        )
    )
    save_json(context.run_dir / "gradient_sample.json", gradient_sample)
    model, tokenizer = context.model_pool.ensure(TARGET_ROLE)
    gradient_started_at = time.monotonic()
    log_progress(
        context,
        f"gradient collection started | examples={len(train_records)} | "
        f"batch_size={args.gradient_batch_size}",
    )
    gradient_analysis = collect_instruction_gradients(
        context.initial_prompt,
        train_records,
        mode=context.mode,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.gradient_batch_size,
        reasoning_traces=reasoning_traces,
    )
    log_progress(
        context,
        "gradient collection completed",
        phase_started_at=gradient_started_at,
    )
    num_edit_regions, max_region_tokens = _resolve_gradpo_shape(args, args.model)
    region_pool = build_gradient_region_pool(
        gradient_analysis,
        max_region_tokens=max_region_tokens,
        expansion_threshold_ratio=args.region_expansion_threshold,
    )
    selection_mode = "random" if args.variant == "gen_random" else "top_gradient"
    selected_regions = select_gradient_regions(
        region_pool,
        count=num_edit_regions,
        selection_mode=selection_mode,
        rng=context.rng,
    )
    for region in selected_regions:
        region["text"] = region["region_text"]
    log_progress(
        context,
        f"region selection completed | mode={selection_mode} | "
        f"selected={len(selected_regions)} | "
        f"texts={[region['region_text'] for region in selected_regions]!r}",
    )
    raw_candidate_output = None
    candidate_meta_prompt = None
    candidate_started_at = time.monotonic()
    log_progress(
        context,
        f"region-candidate generation started | variant={args.variant} | "
        f"regions={len(selected_regions)}",
    )
    if args.variant in {"gen", "gen_random"}:
        marked_prompt = mark_selected_regions(
            context.initial_prompt,
            selected_regions,
        )
        candidate_meta_prompt = gradpo_candidate_prompt(
            marked_prompt,
            selected_regions,
            args.num_region_candidates,
        )
        raw_candidate_output = context.model_pool.generate(
            TARGET_ROLE,
            [candidate_meta_prompt],
            max_new_tokens=args.candidate_max_new_tokens,
            batch_size=1,
            enable_thinking=False,
            do_sample=True,
            log_label=f"qa_gradpo_{args.variant}_candidate_generation",
            return_token_usage=False,
        )[0]
        region_candidates = _parse_gradpo_gen_candidates(
            raw_candidate_output,
            selected_regions,
            args.num_region_candidates,
            context.mode,
        )
    else:
        region_candidates = _gradpo_probability_candidates(
            context.initial_prompt,
            selected_regions,
            mode=context.mode,
            candidate_count=args.num_region_candidates,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.candidate_max_new_tokens,
            generation_batch_size=args.synthesis_batch_size,
        )
    rejected_candidates = [
        {
            "region_rank": int(region_candidates_item["region_rank"]),
            **rejected,
        }
        for region_candidates_item in region_candidates
        for rejected in region_candidates_item.get("rejected_candidates", [])
    ]
    if rejected_candidates:
        log_progress(
            context,
            "restricted GradPO candidates removed | "
            f"count={len(rejected_candidates)} | candidates={rejected_candidates!r}",
        )
        context.logger.event(
            "gradpo_candidates_rejected",
            rejected_candidates=rejected_candidates,
        )
    log_progress(
        context,
        f"region-candidate generation completed | region_sets={len(region_candidates)}",
        phase_started_at=candidate_started_at,
    )
    beam_started_at = time.monotonic()
    log_progress(
        context,
        f"beam search started | regions={len(selected_regions)} | "
        f"beam_width={args.beam_width}",
    )
    beam, beam_trace = _beam_search_replacements(
        context,
        selected_regions,
        region_candidates,
        train_records,
        model=model,
        tokenizer=tokenizer,
        beam_width=args.beam_width,
        selection_batch_size=args.selection_batch_size,
        fluency_lambda=args.fluency_lambda,
        replacement_mode=args.beam_replacement_mode,
        synthesis_max_new_tokens=args.synthesis_max_new_tokens,
        synthesis_batch_size=args.synthesis_batch_size,
        reasoning_traces=reasoning_traces,
    )
    log_progress(
        context,
        f"beam search completed | retained={len(beam)}",
        phase_started_at=beam_started_at,
    )
    dev_prompts = unique_nonempty(
        [item["prompt"] for item in beam if item["prompt"] != context.initial_prompt]
    )
    save_json(
        context.run_dir / "gradient_analysis.json",
        tensor_free_gradient_summary(gradient_analysis),
    )
    save_json(
        context.run_dir / "selected_spans.json",
        {
            "method": f"gradpo_{args.variant}",
            "selection_mode": selection_mode,
            "beam_replacement_mode": args.beam_replacement_mode,
            "candidate_region_pool_size": len(region_pool),
            "regions": selected_regions,
        },
    )
    save_json(
        context.run_dir / "region_candidates.json",
        {
            "candidate_meta_prompt": candidate_meta_prompt,
            "raw_candidate_output": raw_candidate_output,
            "region_candidates": region_candidates,
        },
    )
    save_json(context.run_dir / "beam_trace.json", beam_trace)
    del gradient_analysis, model, tokenizer
    initial_evaluation, source, dev_scored, final_backend = (
        _evaluate_source_and_candidates(
            context,
            args,
            dev_prompts,
            phase=f"gradpo_{args.variant}_candidate_validation",
        )
    )
    selected, improved = _strictly_select_against_source(source, dev_scored)
    log_progress(
        context,
        f"iteration 1/1 completed | improved={improved} | "
        f"current {scored_item_text(selected)} | source {scored_item_text(source)} | "
        f"best {scored_item_text(selected)}",
    )
    return finalize_run(
        context,
        initial_evaluation=initial_evaluation,
        final_prompt=selected["prompt"],
        final_evaluation=selected["evaluation"],
        extra_summary={
            "algorithm": f"gradpo_{args.variant}",
            "iterations": 1,
            "improved_on_validation": improved,
            "objective_task_loss_backend": "transformers",
            "final_evaluation_backend": final_backend,
            "gradient_sampling": {
                key: value
                for key, value in gradient_sample.items()
                if key not in {"initial_pool_record_ids", "selected_examples"}
            },
            "selection_mode": selection_mode,
            "num_edit_regions": num_edit_regions,
            "max_region_tokens": max_region_tokens,
            "region_pool_size": len(region_pool),
            "selected_regions": selected_regions,
            "region_candidates": region_candidates,
            "final_beam": beam,
        },
    )
