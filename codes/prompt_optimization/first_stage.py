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
    etgpo_first_taxonomy_prompt,
    etgpo_guidance_prompt,
    etgpo_update_taxonomy_prompt,
    evoprompt_de_prompt,
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
from prompt_optimization.qa_evoprompt_seeds import QA_EVOPROMPT_SEEDS
from prompt_optimization.qa_task import (
    etgpo_failure_example,
    rpo_feedback_example,
    sample_records,
)
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
        feedback_example_texts = [
            rpo_feedback_example(record, prediction, index)
            for index, (record, prediction) in enumerate(selected, start=1)
        ]
        feedback_meta_prompts = [
            rpo_feedback_prompt(context.mode, example_text)
            for example_text in feedback_example_texts
        ]
        feedback_raw_outputs = generate_optimizer_texts(
            context,
            feedback_meta_prompts,
            log_label="qa_rpo_feedback_generation",
        )
        feedback_texts = [
            extract_feedback(raw_output) for raw_output in feedback_raw_outputs
        ]
        rewrite_feedback_examples = [
            f"{example_text}\nFeedback: {feedback_text}"
            for example_text, feedback_text in zip(
                feedback_example_texts,
                feedback_texts,
            )
        ]
        rewrite_meta_prompt = rpo_rewrite_prompt(
            parent["prompt"],
            rewrite_feedback_examples,
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
            feedback_examples=feedback_example_texts,
            feedback_meta_prompts=feedback_meta_prompts,
            feedback_raw_outputs=feedback_raw_outputs,
            feedback_texts=feedback_texts,
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
                "feedback_texts": feedback_texts,
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


def _build_evoprompt_population(
    context: QAOptimizationContext,
    population_size: int,
) -> list[str]:
    """Load the fixed five-prompt EvoPrompt population for one QA mode."""
    fixed_seeds = QA_EVOPROMPT_SEEDS[context.mode.name]
    population_records = [
        {"label": "source_prompt", "prompt": context.initial_prompt},
        *fixed_seeds,
    ]
    if population_size != len(population_records):
        raise ValueError(
            "The fixed QA EvoPrompt population contains exactly "
            f"{len(population_records)} prompts; received --population-size "
            f"{population_size}."
        )
    population = unique_nonempty(
        [str(item["prompt"]) for item in population_records]
    )
    if len(population) < population_size:
        raise ValueError(
            "The fixed EvoPrompt population contains duplicate or empty prompts. "
            f"Expected {population_size} distinct prompts, received {len(population)}."
        )
    context.logger.event(
        "evoprompt_fixed_initial_population",
        population=population_records,
    )
    return population


def _sample_de_donors(
    population: Sequence[str],
    target_index: int,
    current_best_prompt: str,
    rng: random.Random,
) -> tuple[str, str, str]:
    """Sample two random donors and use the current best as EvoPrompt's third donor."""
    candidates = [prompt for index, prompt in enumerate(population) if index != target_index]
    if len(candidates) < 3:
        raise ValueError("Differential evolution needs three non-target donors.")
    donor_a, donor_b, _ = rng.sample(candidates, 3)
    return donor_a, donor_b, current_best_prompt


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
        current_best_prompt = best_scored_candidate(parent_scores)["prompt"]
        de_meta_prompts = []
        de_parent_records = []
        for target_index, target_prompt in enumerate(population):
            donor_a, donor_b, donor_c = _sample_de_donors(
                population,
                target_index,
                current_best_prompt,
                context.rng,
            )
            de_meta_prompts.append(
                evoprompt_de_prompt(
                    target_prompt,
                    donor_a,
                    donor_b,
                    donor_c,
                    context.mode,
                )
            )
            de_parent_records.append(
                {
                    "target_index": target_index,
                    "target_prompt": target_prompt,
                    "donor_a": donor_a,
                    "donor_b": donor_b,
                    "donor_c_current_best": donor_c,
                }
            )
        raw_outputs = generate_optimizer_texts(
            context,
            de_meta_prompts,
            log_label="qa_evoprompt_de_generation",
            enable_thinking=False,
        )
        children = []
        for target_prompt, raw_output in zip(population, raw_outputs):
            parsed = extract_tagged_prompts([raw_output])
            children.append(parsed[0] if parsed else target_prompt)
        context.logger.event(
            "evoprompt_de_generation",
            iteration=iteration,
            current_best_prompt=current_best_prompt,
            parents=de_parent_records,
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


def _add_taxonomy_categories(
    taxonomy: Sequence[dict[str, Any]],
    additions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add genuinely new ETGPO categories while preserving accumulated statistics."""
    merged = [dict(category) for category in taxonomy]
    by_name = {
        str(category["category_name"]).strip().casefold(): category
        for category in merged
    }
    descriptive_fields = (
        "summary",
        "description",
        "example",
        "error_type",
        "why_leads_to_wrong_answer",
    )
    for addition in additions:
        name = str(
            addition.get("category_name", addition.get("name", ""))
        ).strip()
        if not name:
            continue
        key = name.casefold()
        if key not in by_name:
            category = {
                "category_name": name,
                **{
                    field: str(addition.get(field, "")).strip()
                    for field in descriptive_fields
                },
                "trace_count": 0,
                "problem_ids": [],
            }
            merged.append(category)
            by_name[key] = category
            continue
        existing = by_name[key]
        for field in descriptive_fields:
            if not existing.get(field) and addition.get(field):
                existing[field] = str(addition[field]).strip()
    return merged


def _resolve_failure_number(value: Any, batch_size: int) -> int | None:
    """Convert an ETGPO failure identifier into a valid one-based batch position."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= batch_size else None


def _record_taxonomy_assignments(
    taxonomy: Sequence[dict[str, Any]],
    prior_assignments: Sequence[dict[str, Any]],
    raw_assignments: Sequence[dict[str, Any]],
    batch: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    batch_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Record one category assignment per failure and update category counts."""
    updated_taxonomy = [dict(category) for category in taxonomy]
    for category in updated_taxonomy:
        category["problem_ids"] = list(category.get("problem_ids", []))
    assignments = [dict(assignment) for assignment in prior_assignments]
    assignment_by_failure: dict[int, dict[str, Any]] = {}
    for assignment in raw_assignments:
        failure_number = _resolve_failure_number(
            assignment.get("failure_id"),
            len(batch),
        )
        if failure_number is not None and failure_number not in assignment_by_failure:
            assignment_by_failure[failure_number] = assignment

    for failure_number, (record, _) in enumerate(batch, start=1):
        raw_assignment = assignment_by_failure.get(failure_number, {})
        category_name = str(raw_assignment.get("category_name", "")).strip()
        if not category_name:
            category_name = "Uncategorized model error"
        updated_taxonomy = _add_taxonomy_categories(
            updated_taxonomy,
            [
                {
                    "category_name": category_name,
                    "summary": "A failure that the taxonomy response did not classify.",
                    "description": (
                        "The model selected an incorrect answer, but the taxonomy "
                        "response did not provide a usable category assignment."
                    ),
                    "error_type": "unclassified error",
                }
            ],
        )
        category_by_name = {
            str(category["category_name"]).strip().casefold(): category
            for category in updated_taxonomy
        }
        category = category_by_name[category_name.casefold()]
        canonical_name = str(category["category_name"])
        problem_id = str(record["id"])
        category["trace_count"] = int(category.get("trace_count", 0)) + 1
        if problem_id not in category["problem_ids"]:
            category["problem_ids"].append(problem_id)
        trace_details = raw_assignment.get("trace_details", {})
        if not isinstance(trace_details, dict):
            trace_details = {"trace_specific_details": str(trace_details)}
        assignments.append(
            {
                "batch_index": batch_index,
                "failure_id": failure_number,
                "problem_id": problem_id,
                "category_name": canonical_name,
                "trace_details": trace_details,
            }
        )
    return updated_taxonomy, assignments


def _select_taxonomy_categories(
    taxonomy: Sequence[dict[str, Any]],
    total_failures: int,
    coverage_threshold: float,
    max_categories: int,
    min_problems: int,
) -> tuple[list[dict[str, Any]], float]:
    """Select frequent categories until the requested failure coverage is reached."""
    ranked = sorted(
        taxonomy,
        key=lambda category: (
            -int(category.get("trace_count", 0)),
            str(category.get("category_name", "")).casefold(),
        ),
    )
    eligible = [
        category
        for category in ranked
        if len(category.get("problem_ids", [])) >= min_problems
    ]
    if not eligible:
        eligible = ranked
    selected: list[dict[str, Any]] = []
    covered_failures = 0
    target_failures = coverage_threshold * total_failures
    for category in eligible:
        if len(selected) >= max_categories or covered_failures >= target_failures:
            break
        selected.append(dict(category))
        covered_failures += int(category.get("trace_count", 0))
    achieved_coverage = covered_failures / total_failures if total_failures else 0.0
    return selected, achieved_coverage


def _etgpo_candidate_from_output(
    raw_output: str,
    source_prompt: str,
) -> str | None:
    """Extract one complete ETGPO prompt from one independent guidance response."""
    parsed = extract_json_object(raw_output)
    if parsed:
        full_prompt = str(parsed.get("full_prompt", "")).strip()
        if full_prompt:
            return full_prompt
        preamble = str(parsed.get("preamble", "")).strip()
        guidance_items = parsed.get("guidance_items", [])
        guidance_texts = []
        if isinstance(guidance_items, list):
            guidance_texts = [
                str(item.get("guidance_text", "")).strip()
                for item in guidance_items
                if isinstance(item, dict) and item.get("guidance_text")
            ]
        pieces = [source_prompt, preamble, *guidance_texts]
        assembled = "\n\n".join(piece for piece in pieces if piece)
        if assembled != source_prompt:
            return assembled
    tagged = extract_tagged_prompts([raw_output])
    return tagged[0] if tagged else None


def run_etgpo(context: QAOptimizationContext, args) -> dict[str, Any]:
    """Run ETGPO taxonomy construction and repeated guidance generation for QA."""
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
    taxonomy: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    processed = 0
    batch_index = 0
    while processed < len(errors):
        batch = errors[processed : processed + args.error_batch_size]
        if not batch:
            break
        batch_index += 1
        examples = [
            etgpo_failure_example(record, prediction, index, context.mode)
            for index, (record, prediction) in enumerate(batch, start=1)
        ]
        if batch_index == 1:
            meta_prompt = etgpo_first_taxonomy_prompt(examples, context.mode)
        else:
            meta_prompt = etgpo_update_taxonomy_prompt(
                taxonomy,
                examples,
                context.mode,
            )
        raw_output = generate_optimizer_texts(
            context,
            [meta_prompt],
            log_label="qa_etgpo_taxonomy",
        )[0]
        parsed = extract_json_object(raw_output) or {}
        category_key = "categories" if batch_index == 1 else "new_categories"
        categories = parsed.get(category_key, [])
        if isinstance(categories, list):
            taxonomy = _add_taxonomy_categories(
                taxonomy,
                [item for item in categories if isinstance(item, dict)],
            )
        raw_assignments = parsed.get("failure_assignments", [])
        if not isinstance(raw_assignments, list):
            raw_assignments = []
        taxonomy, assignments = _record_taxonomy_assignments(
            taxonomy,
            assignments,
            [item for item in raw_assignments if isinstance(item, dict)],
            batch,
            batch_index,
        )
        processed += len(batch)
        context.logger.event(
            "etgpo_taxonomy_batch",
            batch_index=batch_index,
            processed_errors=processed,
            total_errors=len(errors),
            taxonomy=taxonomy,
            assignments=assignments[-len(batch) :],
            meta_prompt=meta_prompt,
            raw_output=raw_output,
            parsed_output=parsed,
        )
    selected_taxonomy, achieved_coverage = _select_taxonomy_categories(
        taxonomy,
        len(errors),
        args.error_coverage,
        args.max_categories,
        args.min_problems,
    )
    raw_outputs: list[str] = []
    candidate_prompts: list[str] = []
    guidance_meta_prompt = ""
    if selected_taxonomy:
        guidance_meta_prompt = etgpo_guidance_prompt(
            context.initial_prompt,
            context.mode,
            selected_taxonomy,
            len(errors),
        )
        raw_outputs = generate_optimizer_texts(
            context,
            [guidance_meta_prompt] * args.num_candidates,
            log_label="qa_etgpo_guidance",
        )
        candidate_prompts = unique_nonempty(
            [
                candidate
                for raw_output in raw_outputs
                if (
                    candidate := _etgpo_candidate_from_output(
                        raw_output,
                        context.initial_prompt,
                    )
                )
            ]
        )
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
        repeated_generation_count=args.num_candidates if selected_taxonomy else 0,
        raw_outputs=raw_outputs,
        parsed_candidates=candidate_prompts,
    )
    source = {
        "prompt": context.initial_prompt,
        "accuracy": metric_accuracy(initial_evaluation),
        "evaluation": initial_evaluation,
    }
    selected = best_scored_candidate(scored) if scored else source
    if float(selected["accuracy"]) <= float(source["accuracy"]):
        selected = source
    taxonomy_payload = {
        "total_failures": len(errors),
        "processed_failures": processed,
        "categories": taxonomy,
        "assignments": assignments,
        "selection": {
            "coverage_threshold": args.error_coverage,
            "achieved_coverage": achieved_coverage,
            "minimum_problems": args.min_problems,
            "maximum_categories": args.max_categories,
            "selected_categories": selected_taxonomy,
        },
    }
    save_json(context.run_dir / "taxonomy.json", taxonomy_payload)
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
            "achieved_error_coverage": achieved_coverage,
            "taxonomy": taxonomy,
            "selected_taxonomy": selected_taxonomy,
            "taxonomy_assignments": assignments,
            "guidance_generation_count": (
                args.num_candidates if selected_taxonomy else 0
            ),
            "raw_guidance_outputs": raw_outputs,
        },
    )
