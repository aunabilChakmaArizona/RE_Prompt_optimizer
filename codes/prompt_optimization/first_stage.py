"""QA implementations of RPO, EvoPrompt-DE, and ETGPO first-stage search."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

from prompt_optimization.cli_common import QAOptimizationContext
from prompt_optimization.evaluation import (
    metric_accuracy,
    select_mixed_feedback,
)
from prompt_optimization.meta_prompts import (
    etgpo_guidance_prompt,
    etgpo_taxonomy_prompt,
    evoprompt_de_prompt,
    evoprompt_seed_prompt,
    extract_json_object,
    extract_tagged_prompts,
    rpo_feedback_prompt,
    rpo_rewrite_prompt,
    unique_nonempty,
)
from prompt_optimization.optimizer_common import (
    best_scored_candidate,
    evaluate_candidates,
    extract_feedback,
    finalize_run,
    generate_optimizer_texts,
    generate_tagged_candidates,
)
from prompt_optimization.qa_task import feedback_example, sample_records
from prompt_optimization.run_io import save_json, save_text


def _softmax_parent(
    population: Sequence[dict[str, Any]],
    temperature: float,
    rng: random.Random,
) -> dict[str, Any]:
    """Sample a population parent from validation accuracy softmax weights."""
    if temperature <= 0:
        return max(population, key=lambda item: float(item["accuracy"]))
    scores = [float(item["accuracy"]) / temperature for item in population]
    shift = max(scores)
    weights = [math.exp(score - shift) for score in scores]
    return rng.choices(list(population), weights=weights, k=1)[0]


def _deduplicate_population(
    population: Sequence[dict[str, Any]],
    maximum_size: int,
) -> list[dict[str, Any]]:
    """Keep the best record for each prompt and truncate the active population."""
    best_by_prompt: dict[str, dict[str, Any]] = {}
    for item in population:
        current = best_by_prompt.get(item["prompt"])
        if current is None or float(item["accuracy"]) > float(current["accuracy"]):
            best_by_prompt[item["prompt"]] = item
    return sorted(
        best_by_prompt.values(),
        key=lambda item: (float(item["accuracy"]), -int(item["node_id"])),
        reverse=True,
    )[:maximum_size]


def run_rpo(context: QAOptimizationContext, args) -> dict[str, Any]:
    """Run feedback-driven RPO and save iteration-5 and iteration-10 snapshots."""
    initial_evaluation = context.evaluator.evaluate(
        context.initial_prompt,
        context.validation_records,
        split_name="validation",
        log_label="qa_rpo_initial_validation",
    )
    population = [
        {
            "node_id": 0,
            "prompt": context.initial_prompt,
            "accuracy": metric_accuracy(initial_evaluation),
            "evaluation": initial_evaluation,
            "parent_node_id": None,
            "iteration": 0,
        }
    ]
    best = population[0]
    next_node_id = 1
    snapshots: dict[str, dict[str, Any]] = {}

    for iteration in range(1, args.iterations + 1):
        parent = _softmax_parent(
            population,
            args.population_sampling_temperature,
            context.rng,
        )
        feedback_records = sample_records(
            context.train_records,
            args.feedback_sample_size,
            context.rng,
        )
        feedback_evaluation = context.evaluator.evaluate(
            parent["prompt"],
            feedback_records,
            split_name="train_feedback",
            log_label="qa_rpo_feedback_inference",
        )
        selected = select_mixed_feedback(
            feedback_records,
            feedback_evaluation,
            args.feedback_examples,
        )
        example_texts = [
            feedback_example(record, prediction, index)
            for index, (record, prediction) in enumerate(selected, start=1)
        ]
        diagnosis_meta_prompt = rpo_feedback_prompt(
            parent["prompt"],
            context.mode,
            example_texts,
        )
        diagnosis_outputs = generate_optimizer_texts(
            context,
            [diagnosis_meta_prompt],
            log_label="qa_rpo_feedback_generation",
        )
        diagnosis = extract_feedback(diagnosis_outputs[0])
        rewrite_meta_prompt = rpo_rewrite_prompt(
            parent["prompt"],
            diagnosis,
            context.mode,
        )
        candidates, rewrite_outputs = generate_tagged_candidates(
            context,
            [rewrite_meta_prompt],
            log_label="qa_rpo_prompt_rewrite",
        )
        context.logger.event(
            "rpo_optimizer_trace",
            iteration=iteration,
            parent_node_id=parent["node_id"],
            feedback_examples=example_texts,
            diagnosis_meta_prompt=diagnosis_meta_prompt,
            diagnosis_raw_outputs=diagnosis_outputs,
            diagnosis=diagnosis,
            rewrite_meta_prompt=rewrite_meta_prompt,
            rewrite_raw_outputs=rewrite_outputs,
            parsed_candidates=candidates,
        )
        if not candidates:
            context.logger.event(
                "rpo_generation_failed",
                iteration=iteration,
                parent_node_id=parent["node_id"],
                raw_output=rewrite_outputs,
            )
        else:
            scored = evaluate_candidates(
                context,
                candidates,
                context.validation_records,
                split_name="validation",
                phase="rpo_child_validation",
                iteration=iteration,
            )
            child_score = best_scored_candidate(scored)
            child = {
                "node_id": next_node_id,
                "prompt": child_score["prompt"],
                "accuracy": child_score["accuracy"],
                "evaluation": child_score["evaluation"],
                "parent_node_id": parent["node_id"],
                "iteration": iteration,
                "diagnosis": diagnosis,
                "feedback_record_ids": [record["id"] for record, _ in selected],
            }
            next_node_id += 1
            population = _deduplicate_population(
                [*population, child],
                args.population_size,
            )
            if float(child["accuracy"]) > float(best["accuracy"]):
                best = child

        context.logger.event(
            "rpo_iteration_completed",
            iteration=iteration,
            parent_node_id=parent["node_id"],
            population=[
                {
                    "node_id": item["node_id"],
                    "prompt": item["prompt"],
                    "accuracy": item["accuracy"],
                }
                for item in population
            ],
            best_node_id=best["node_id"],
        )
        save_json(
            context.run_dir / "population.json",
            [
                {key: value for key, value in item.items() if key != "evaluation"}
                for item in population
            ],
        )
        if iteration in args.snapshot_iterations:
            snapshot = {
                "iteration": iteration,
                "node_id": best["node_id"],
                "prompt": best["prompt"],
                "metrics": best["evaluation"]["metrics"],
            }
            snapshots[str(iteration)] = snapshot
            save_text(context.run_dir / f"prompt_iteration_{iteration}.txt", best["prompt"])
            save_json(context.run_dir / f"snapshot_iteration_{iteration}.json", snapshot)

    return finalize_run(
        context,
        initial_evaluation=initial_evaluation,
        final_prompt=best["prompt"],
        final_evaluation=best["evaluation"],
        extra_summary={
            "algorithm": "rpo",
            "iterations": args.iterations,
            "snapshots": snapshots,
            "final_node_id": best["node_id"],
        },
    )


def _evoprompt_human_seed(context: QAOptimizationContext) -> str:
    """Return the second fixed human seed for the selected QA mode."""
    if context.mode.name == "reasoning":
        return (
            "Solve the multiple-choice question using relevant scientific knowledge, "
            "check each option, and choose the best supported answer."
        )
    return (
        "Use relevant scientific knowledge to compare the choices and select the "
        "single best answer to the multiple-choice question."
    )


def _build_evoprompt_population(
    context: QAOptimizationContext,
    population_size: int,
) -> list[str]:
    """Create two human and remaining AI-generated EvoPrompt seeds."""
    if population_size < 5:
        raise ValueError("EvoPrompt-DE requires a population size of at least five.")
    ai_count = population_size - 2
    meta_prompt = evoprompt_seed_prompt(
        context.initial_prompt,
        context.mode,
        ai_count,
    )
    ai_seeds: list[str] = []
    for attempt in range(1, 4):
        outputs = generate_optimizer_texts(
            context,
            [meta_prompt],
            log_label="qa_evoprompt_initial_population",
        )
        ai_seeds = unique_nonempty([*ai_seeds, *extract_tagged_prompts(outputs)])
        context.logger.event(
            "evoprompt_seed_generation",
            attempt=attempt,
            meta_prompt=meta_prompt,
            raw_outputs=outputs,
            parsed_ai_seeds=ai_seeds,
        )
        if len(ai_seeds) >= ai_count:
            break
        context.logger.event(
            "evoprompt_seed_retry",
            attempt=attempt,
            distinct_ai_seeds=len(ai_seeds),
            required_ai_seeds=ai_count,
        )
    population = unique_nonempty(
        [context.initial_prompt, _evoprompt_human_seed(context), *ai_seeds]
    )
    if len(population) < population_size:
        raise RuntimeError(
            "The optimizer did not produce enough distinct EvoPrompt seed prompts. "
            f"Expected {population_size}, received {len(population)}."
        )
    return population[:population_size]


def _sample_de_donors(
    population: Sequence[str],
    target_index: int,
    rng: random.Random,
) -> tuple[str, str, str]:
    """Sample three distinct donor prompts other than the DE target."""
    candidates = [prompt for index, prompt in enumerate(population) if index != target_index]
    if len(candidates) < 3:
        raise ValueError("Differential evolution needs three non-target donors.")
    donor_a, donor_b, donor_c = rng.sample(candidates, 3)
    return donor_a, donor_b, donor_c


def _evaluate_prompt_sequence(
    context: QAOptimizationContext,
    prompts: Sequence[str],
    records: Sequence[dict[str, Any]],
    *,
    split_name: str,
    phase: str,
    iteration: int,
) -> list[dict[str, Any]]:
    """Evaluate unique prompts once and map scores back to population positions."""
    unique_scores = evaluate_candidates(
        context,
        prompts,
        records,
        split_name=split_name,
        phase=phase,
        iteration=iteration,
    )
    score_by_prompt = {item["prompt"]: item for item in unique_scores}
    return [
        {**score_by_prompt[prompt], "candidate_index": index}
        for index, prompt in enumerate(prompts)
    ]


def run_evoprompt_de(context: QAOptimizationContext, args) -> dict[str, Any]:
    """Run population-based EvoPrompt differential evolution for QA instructions."""
    initial_evaluation = context.evaluator.evaluate(
        context.initial_prompt,
        context.validation_records,
        split_name="validation",
        log_label="qa_evoprompt_initial_validation",
    )
    population = _build_evoprompt_population(context, args.population_size)
    dev_best = {
        "prompt": context.initial_prompt,
        "accuracy": metric_accuracy(initial_evaluation),
        "evaluation": initial_evaluation,
        "iteration": 0,
    }
    snapshots: dict[str, dict[str, Any]] = {}

    for iteration in range(1, args.iterations + 1):
        train_records = sample_records(
            context.train_records,
            args.train_sample_size,
            context.rng,
        )
        parent_scores = _evaluate_prompt_sequence(
            context,
            population,
            train_records,
            split_name="train_fitness",
            phase="evoprompt_parent_fitness",
            iteration=iteration,
        )
        de_meta_prompts = []
        for target_index, target_prompt in enumerate(population):
            donor_a, donor_b, donor_c = _sample_de_donors(
                population,
                target_index,
                context.rng,
            )
            de_meta_prompts.append(
                evoprompt_de_prompt(
                    context.initial_prompt,
                    target_prompt,
                    donor_a,
                    donor_b,
                    donor_c,
                    context.mode,
                )
            )
        raw_outputs = generate_optimizer_texts(
            context,
            de_meta_prompts,
            log_label="qa_evoprompt_de_generation",
        )
        children = []
        for target_prompt, raw_output in zip(population, raw_outputs):
            parsed = extract_tagged_prompts([raw_output])
            children.append(parsed[0] if parsed else target_prompt)
        context.logger.event(
            "evoprompt_de_generation",
            iteration=iteration,
            meta_prompts=de_meta_prompts,
            raw_outputs=raw_outputs,
            parsed_children=children,
        )
        child_scores = _evaluate_prompt_sequence(
            context,
            children,
            train_records,
            split_name="train_fitness",
            phase="evoprompt_child_fitness",
            iteration=iteration,
        )
        survivors = []
        survivor_scores = []
        for parent_score, child_score in zip(parent_scores, child_scores):
            if float(child_score["accuracy"]) > float(parent_score["accuracy"]):
                survivors.append(child_score["prompt"])
                survivor_scores.append(child_score)
            else:
                survivors.append(parent_score["prompt"])
                survivor_scores.append(parent_score)
        population = survivors
        train_best = best_scored_candidate(survivor_scores)
        dev_evaluation = context.evaluator.evaluate(
            train_best["prompt"],
            context.validation_records,
            split_name="validation",
            log_label="qa_evoprompt_iteration_validation",
        )
        dev_accuracy = metric_accuracy(dev_evaluation)
        if dev_accuracy > float(dev_best["accuracy"]):
            dev_best = {
                "prompt": train_best["prompt"],
                "accuracy": dev_accuracy,
                "evaluation": dev_evaluation,
                "iteration": iteration,
            }
        context.logger.event(
            "evoprompt_iteration_completed",
            iteration=iteration,
            train_best_prompt=train_best["prompt"],
            train_best_accuracy=train_best["accuracy"],
            validation_accuracy=dev_accuracy,
            best_validation_accuracy=dev_best["accuracy"],
            population=population,
        )
        save_json(
            context.run_dir / "population.json",
            {"iteration": iteration, "prompts": population},
        )
        if iteration in args.snapshot_iterations:
            snapshot = {
                "iteration": iteration,
                "prompt": dev_best["prompt"],
                "metrics": dev_best["evaluation"]["metrics"],
                "found_at_iteration": dev_best["iteration"],
            }
            snapshots[str(iteration)] = snapshot
            save_text(
                context.run_dir / f"prompt_iteration_{iteration}.txt",
                dev_best["prompt"],
            )
            save_json(context.run_dir / f"snapshot_iteration_{iteration}.json", snapshot)

    return finalize_run(
        context,
        initial_evaluation=initial_evaluation,
        final_prompt=dev_best["prompt"],
        final_evaluation=dev_best["evaluation"],
        extra_summary={
            "algorithm": "evoprompt_de",
            "iterations": args.iterations,
            "snapshots": snapshots,
            "best_found_at_iteration": dev_best["iteration"],
            "final_population": population,
        },
    )


def _merge_taxonomy(
    taxonomy: Sequence[dict[str, Any]],
    additions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge ETGPO categories by normalized name and accumulate counts."""
    merged: dict[str, dict[str, Any]] = {}
    for category in [*taxonomy, *additions]:
        name = str(category.get("name", "")).strip()
        if not name:
            continue
        key = name.casefold()
        count = int(category.get("count", 1) or 1)
        if key not in merged:
            merged[key] = {
                "name": name,
                "description": str(category.get("description", "")).strip(),
                "count": count,
            }
        else:
            merged[key]["count"] += count
            if not merged[key]["description"]:
                merged[key]["description"] = str(
                    category.get("description", "")
                ).strip()
    return sorted(merged.values(), key=lambda item: int(item["count"]), reverse=True)


def run_etgpo(context: QAOptimizationContext, args) -> dict[str, Any]:
    """Run one taxonomy-guided ETGPO refinement over sampled QA errors."""
    initial_evaluation = context.evaluator.evaluate(
        context.initial_prompt,
        context.validation_records,
        split_name="validation",
        log_label="qa_etgpo_initial_validation",
    )
    train_records = sample_records(
        context.train_records,
        args.train_sample_size,
        context.rng,
    )
    train_evaluation = context.evaluator.evaluate(
        context.initial_prompt,
        train_records,
        split_name="train_errors",
        log_label="qa_etgpo_train_errors",
    )
    errors = [
        (record, prediction)
        for record, prediction in zip(train_records, train_evaluation["predictions"])
        if not prediction["correct"]
    ]
    context.rng.shuffle(errors)
    coverage_count = math.ceil(len(errors) * args.error_coverage)
    taxonomy: list[dict[str, Any]] = []
    processed = 0
    batch_index = 0
    while processed < coverage_count:
        batch = errors[processed : processed + args.error_batch_size]
        if not batch:
            break
        batch_index += 1
        examples = [
            feedback_example(record, prediction, index)
            for index, (record, prediction) in enumerate(batch, start=1)
        ]
        meta_prompt = etgpo_taxonomy_prompt(
            taxonomy,
            examples,
            args.min_categories,
            args.max_categories,
        )
        raw_output = generate_optimizer_texts(
            context,
            [meta_prompt],
            log_label="qa_etgpo_taxonomy",
        )[0]
        parsed = extract_json_object(raw_output) or {}
        categories = parsed.get("categories", [])
        if isinstance(categories, list):
            taxonomy = _merge_taxonomy(
                [],
                [item for item in categories if isinstance(item, dict)],
            )
        processed += len(batch)
        context.logger.event(
            "etgpo_taxonomy_batch",
            batch_index=batch_index,
            processed_errors=processed,
            target_coverage_count=coverage_count,
            taxonomy=taxonomy,
            meta_prompt=meta_prompt,
            raw_output=raw_output,
        )
    if not taxonomy and errors:
        taxonomy = [
            {
                "name": "general instruction failure",
                "description": "The instruction did not reliably guide selection of the correct option.",
                "count": len(errors),
            }
        ]
    taxonomy = taxonomy[: args.max_categories]
    guidance_meta_prompt = etgpo_guidance_prompt(
        context.initial_prompt,
        context.mode,
        taxonomy,
        args.num_candidates,
    )
    candidates, raw_outputs = generate_tagged_candidates(
        context,
        [guidance_meta_prompt],
        log_label="qa_etgpo_guidance",
        include_prompts=[context.initial_prompt],
    )
    candidate_prompts = [
        prompt for prompt in candidates if prompt != context.initial_prompt
    ]
    scored = evaluate_candidates(
        context,
        candidate_prompts,
        context.validation_records,
        split_name="validation",
        phase="etgpo_candidate_validation",
        iteration=1,
    )
    context.logger.event(
        "etgpo_guidance_generation",
        meta_prompt=guidance_meta_prompt,
        raw_outputs=raw_outputs,
        parsed_candidates=candidates,
    )
    source = {
        "prompt": context.initial_prompt,
        "accuracy": metric_accuracy(initial_evaluation),
        "evaluation": initial_evaluation,
    }
    selected = best_scored_candidate(scored) if scored else source
    if float(selected["accuracy"]) <= float(source["accuracy"]):
        selected = source
    save_json(context.run_dir / "taxonomy.json", taxonomy)
    return finalize_run(
        context,
        initial_evaluation=initial_evaluation,
        final_prompt=selected["prompt"],
        final_evaluation=selected["evaluation"],
        extra_summary={
            "algorithm": "etgpo",
            "iterations": 1,
            "train_error_count": len(errors),
            "processed_error_count": processed,
            "error_coverage": args.error_coverage,
            "taxonomy": taxonomy,
            "raw_guidance_outputs": raw_outputs,
        },
    )
