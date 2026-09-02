"""QA implementations of LPO, GreaTer, and the three GradPO variants."""

from __future__ import annotations

import re
from typing import Any, Sequence

try:
    import torch
except ImportError:
    torch = None

from agents.agent_decoding import model_default_sampling_parameters
from agents.agent_prompts import GRADIENT_REGION_CANDIDATE_SYNTHESIS_BODY_V1
from prompt_optimization.cli_common import QAOptimizationContext
from prompt_optimization.evaluation import metric_accuracy, select_mixed_feedback
from prompt_optimization.meta_prompts import (
    QA_TASK_DESCRIPTIONS,
    extract_json_object,
    gradpo_candidate_prompt,
    lpo_location_prompt,
    lpo_rewrite_prompt,
    unique_nonempty,
)
from prompt_optimization.models import TARGET_ROLE
from prompt_optimization.optimizer_common import (
    best_scored_candidate,
    evaluate_candidates,
    finalize_run,
    generate_optimizer_texts,
)
from prompt_optimization.qa_task import (
    QAMode,
    feedback_example,
    sample_label_balanced_records,
)
from prompt_optimization.run_io import save_json
from prompt_optimization.sequence_gradients import (
    align_replacement_whitespace,
    build_gradient_region_pool,
    collect_instruction_gradients,
    mark_selected_regions,
    model_device,
    proposal_token_candidates,
    rank_fixed_token_candidates,
    replace_selected_regions,
    score_combined_objective,
    select_gradient_regions,
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
        "metrics": evaluation["metrics"],
        "evaluation": evaluation,
    }


def _strictly_select_against_source(
    source: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Select the best validation candidate only if it beats the source prompt."""
    if not candidates:
        return source, False
    best_candidate = best_scored_candidate(candidates)
    improved = float(best_candidate["accuracy"]) > float(source["accuracy"])
    return (best_candidate if improved else source), improved


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
    initial_evaluation = context.evaluator.evaluate(
        context.initial_prompt,
        context.validation_records,
        split_name="validation",
        log_label="qa_lpo_initial_validation",
    )
    source = _source_scored_item(context, initial_evaluation)
    train_records = sample_label_balanced_records(
        context.train_records,
        args.train_sample_size,
        context.rng,
    )
    train_evaluation = context.evaluator.evaluate(
        context.initial_prompt,
        train_records,
        split_name="train_selection",
        log_label="qa_lpo_train_selection",
    )
    feedback_pairs = select_mixed_feedback(
        train_records,
        train_evaluation,
        args.feedback_examples,
    )
    feedback_texts = [
        feedback_example(record, prediction, index)
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
    save_json(
        context.run_dir / "selected_spans.json",
        {
            "method": "lpo",
            "raw_location_output": location_output,
            "tagged_prompt": tagged_prompt,
            "locations": locations,
        },
    )

    candidate_prompts = [context.initial_prompt]
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
        candidate_prompts = _lpo_candidate_prompts_from_outputs(
            rewrite_outputs,
            context.initial_prompt,
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

    train_scored = evaluate_candidates(
        context,
        candidate_prompts,
        train_records,
        split_name="train_selection",
        phase="lpo_candidate_train",
        iteration=1,
    )
    top_train = sorted(
        train_scored,
        key=lambda item: (float(item["accuracy"]), -int(item["candidate_index"])),
        reverse=True,
    )[: args.top_z]
    dev_prompts = [item["prompt"] for item in top_train if item["prompt"] != context.initial_prompt]
    dev_scored = evaluate_candidates(
        context,
        dev_prompts,
        context.validation_records,
        split_name="validation",
        phase="lpo_candidate_validation",
        iteration=1,
    )
    selected, improved = _strictly_select_against_source(source, dev_scored)
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
) -> dict[str, Any]:
    """Choose the first editable prompt token at or after a sequential position."""
    prompt = str(gradient_analysis["instruction_prompt"])
    records = list(gradient_analysis["token_gradients"])
    for record in records:
        token_index = int(record["token_index"])
        start = int(record["char_start"])
        end = int(record["char_end"])
        text = prompt[start:end]
        if token_index >= start_position and any(character.isalnum() for character in text):
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
            }
    raise ValueError("No editable token exists at or after --start-position.")


def _prompt_for_single_token_candidate(
    instruction_prompt: str,
    region: dict[str, Any],
    candidate_text: str,
) -> str:
    """Replace one aligned GreaTer token while preserving boundary whitespace."""
    start = int(region["start_char"])
    end = int(region["end_char"])
    replacement = align_replacement_whitespace(
        str(region["region_text"]),
        candidate_text,
    )
    return (instruction_prompt[:start] + replacement + instruction_prompt[end:]).strip()


def _is_stable_single_token_replacement(
    source_token_ids: Sequence[int],
    candidate_token_ids: Sequence[int],
) -> bool:
    """Check that retokenization changes at most one source and target token."""
    prefix = 0
    shared_length = min(len(source_token_ids), len(candidate_token_ids))
    while (
        prefix < shared_length
        and source_token_ids[prefix] == candidate_token_ids[prefix]
    ):
        prefix += 1

    source_suffix = len(source_token_ids)
    candidate_suffix = len(candidate_token_ids)
    while (
        source_suffix > prefix
        and candidate_suffix > prefix
        and source_token_ids[source_suffix - 1]
        == candidate_token_ids[candidate_suffix - 1]
    ):
        source_suffix -= 1
        candidate_suffix -= 1

    return source_suffix - prefix <= 1 and candidate_suffix - prefix <= 1


def run_greater(context: QAOptimizationContext, args) -> dict[str, Any]:
    """Run one GreaTer sequential or top-gradient single-token refinement."""
    initial_evaluation = context.evaluator.evaluate(
        context.initial_prompt,
        context.validation_records,
        split_name="validation",
        log_label=f"qa_{args.variant}_initial_validation",
    )
    source = _source_scored_item(context, initial_evaluation)
    train_records = sample_label_balanced_records(
        context.train_records,
        args.train_sample_size,
        context.rng,
    )
    model, tokenizer = context.model_pool.ensure(TARGET_ROLE)
    gradient_analysis = collect_instruction_gradients(
        context.initial_prompt,
        train_records,
        mode=context.mode,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.gradient_batch_size,
    )
    if args.variant == "greater":
        region = _sequential_greater_region(gradient_analysis, args.start_position)
    else:
        pool = build_gradient_region_pool(
            gradient_analysis,
            max_region_tokens=1,
            expansion_threshold_ratio=args.region_expansion_threshold,
        )
        region = select_gradient_regions(
            pool,
            count=1,
            selection_mode="top_gradient",
            rng=context.rng,
        )[0]
        region["text"] = region["region_text"]
    proposal_records = train_records[: min(args.proposal_example_size, len(train_records))]
    proposed_token_ids, proposal_metadata = proposal_token_candidates(
        gradient_analysis,
        int(region["peak_token_index"]),
        proposal_records,
        mode=context.mode,
        model=model,
        tokenizer=tokenizer,
        top_k=args.proposal_top_k,
        min_candidates=args.proposal_min_candidates,
    )
    candidate_tokens = rank_fixed_token_candidates(
        gradient_analysis,
        int(region["peak_token_index"]),
        proposed_token_ids,
        model=model,
        tokenizer=tokenizer,
    )
    candidate_tokens = candidate_tokens[: args.selection_top_mu]
    if not any(item["is_original"] for item in candidate_tokens):
        original = next(
            item
            for item in rank_fixed_token_candidates(
                gradient_analysis,
                int(region["peak_token_index"]),
                proposed_token_ids,
                model=model,
                tokenizer=tokenizer,
            )
            if item["is_original"]
        )
        candidate_tokens.append(original)
    source_token_ids = tokenizer.encode(
        context.initial_prompt,
        add_special_tokens=False,
    )
    stable_candidate_tokens = []
    candidate_prompts = [context.initial_prompt]
    for item in candidate_tokens:
        candidate_prompt = _prompt_for_single_token_candidate(
            context.initial_prompt,
            region,
            item["token_text"],
        )
        candidate_token_ids = tokenizer.encode(
            candidate_prompt,
            add_special_tokens=False,
        )
        if not _is_stable_single_token_replacement(
            source_token_ids,
            candidate_token_ids,
        ):
            continue
        stable_candidate_tokens.append(
            {
                **item,
                "prompt": candidate_prompt,
            }
        )
        candidate_prompts.append(candidate_prompt)
    candidate_tokens = stable_candidate_tokens
    candidate_prompts = unique_nonempty(candidate_prompts)
    objective_scores = []
    for candidate_index, prompt in enumerate(candidate_prompts):
        score = score_combined_objective(
            prompt,
            train_records,
            mode=context.mode,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.selection_batch_size,
            fluency_lambda=args.fluency_lambda,
        )
        objective_scores.append(
            {"candidate_index": candidate_index, "prompt": prompt, **score}
        )
    top_objective = sorted(
        objective_scores,
        key=lambda item: (float(item["combined_score"]), int(item["candidate_index"])),
    )[: args.top_u]
    dev_prompts = [
        item["prompt"] for item in top_objective if item["prompt"] != context.initial_prompt
    ]
    dev_scored = evaluate_candidates(
        context,
        dev_prompts,
        context.validation_records,
        split_name="validation",
        phase=f"{args.variant}_candidate_validation",
        iteration=1,
    )
    selected, improved = _strictly_select_against_source(source, dev_scored)
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
            "candidates": candidate_tokens,
            "objective_scores": objective_scores,
        },
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
            "selected_region": region,
            "proposal_metadata": proposal_metadata,
            "top_objective_candidates": top_objective,
        },
    )


def _parse_gradpo_gen_candidates(
    raw_output: str,
    regions: Sequence[dict[str, Any]],
    candidate_count: int,
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
        candidates = unique_nonempty(
            [str(region["region_text"]), *[str(value) for value in values]]
        )[: candidate_count + 1]
        output.append(
            {
                "region_rank": rank,
                "region_text": region["region_text"],
                "candidate_source": "target_model_generation",
                "candidates": candidates,
            }
        )
    return output


def _sample_next_token_id(
    logits,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    """Sample one token with the fixed target-model temperature, top-p, and top-k."""
    scaled = logits.float() / max(temperature, 1e-6)
    if top_k > 0 and top_k < scaled.numel():
        threshold = torch.topk(scaled, k=top_k).values[-1]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    probabilities = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities > top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum()
        sampled_index = int(torch.multinomial(sorted_probabilities, 1).item())
        return int(sorted_indices[sampled_index].item())
    return int(torch.multinomial(probabilities, 1).item())


def _probability_continuation(
    prefix_ids: torch.Tensor,
    first_token_id: int,
    continuation_length: int,
    model,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[int]:
    """Sample a fixed-length replacement continuation after one proposed token."""
    if torch is None:
        raise ImportError("GradPO-Prob requires PyTorch in the active environment.")
    device = model_device(model)
    generated = [first_token_id]
    current = torch.cat(
        [prefix_ids.to(device), torch.tensor([[first_token_id]], device=device)],
        dim=1,
    )
    with torch.inference_mode():
        for _ in range(max(0, continuation_length - 1)):
            logits = model(input_ids=current, use_cache=False).logits[:, -1, :]
            next_id = _sample_next_token_id(
                logits[0],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            generated.append(next_id)
            current = torch.cat(
                [current, torch.tensor([[next_id]], device=device)],
                dim=1,
            )
    return generated


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
    candidate_count: int,
    model,
    tokenizer,
    model_id: str,
) -> list[dict[str, Any]]:
    """Generate fixed-token-length span candidates from target-LM probabilities."""
    if torch is None:
        raise ImportError("GradPO-Prob requires PyTorch in the active environment.")
    device = model_device(model)
    sampling = model_default_sampling_parameters(model_id)
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
        for first_id in top_ids:
            if first_id in special_ids:
                continue
            token_ids = _probability_continuation(
                prefix_ids,
                int(first_id),
                int(region["token_count"]),
                model,
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
            )
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            if (
                not text.strip()
                or not text.isascii()
                or "\n" in text
                or not any(character.isalnum() for character in text)
            ):
                continue
            candidates = unique_nonempty([*candidates, text])
            candidate_details.append(
                {"first_token_id": int(first_id), "token_ids": token_ids, "text": text}
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
            "You are an expert prompt generator for a multiple-choice question-answering task.",
            QA_TASK_DESCRIPTIONS[mode.name],
            GRADIENT_REGION_CANDIDATE_SYNTHESIS_BODY_V1,
        ]
    )
    return (
        qa_prompt
        .replace("#ALL_MARKED_PROMPT#", marked_prompt)
        .replace("#SELECTED_REPLACEMENTS#", "\n\n".join(replacement_blocks))
    )


def _normalize_synthesized_prompt(raw_output: str) -> str:
    """Remove a surrounding fence and any leaked numbered span tags."""
    text = raw_output.strip()
    fenced = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
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
    synthesis_cache: dict[tuple[tuple[int, str], ...], dict[str, Any]] = {}
    for region in regions:
        rank = int(region["region_rank"])
        expansion_specs = []
        for beam_item in beam:
            for replacement in candidate_index[rank]["candidates"]:
                replacements = {
                    **beam_item["replacements"],
                    rank: str(replacement),
                }
                expansion_specs.append(
                    {
                        "replacements": replacements,
                        "history": [*beam_item["history"], rank],
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
                    do_sample=True,
                    log_label="qa_gradpo_beam_synthesis",
                    return_token_usage=False,
                )
                for key, meta_prompt, raw_output in zip(
                    missing_keys,
                    missing_prompts,
                    raw_outputs,
                ):
                    revised = _normalize_synthesized_prompt(raw_output)
                    used_fallback = not bool(revised)
                    if used_fallback:
                        revised = replace_selected_regions(
                            context.initial_prompt,
                            regions,
                            dict(key),
                        )
                    synthesis_cache[key] = {
                        "prompt": revised,
                        "meta_prompt": meta_prompt,
                        "raw_output": raw_output,
                        "used_fallback": used_fallback,
                    }
        elif replacement_mode != "direct":
            raise ValueError(f"Unsupported beam replacement mode: {replacement_mode!r}")

        expansions: dict[str, dict[str, Any]] = {}
        for spec in expansion_specs:
            key = tuple(sorted(spec["replacements"].items()))
            if replacement_mode == "llm_synthesis":
                synthesis = synthesis_cache[key]
                prompt = synthesis["prompt"]
                replacement_metadata = {
                    "replacement_mode": replacement_mode,
                    "meta_prompt": synthesis["meta_prompt"],
                    "raw_output": synthesis["raw_output"],
                    "used_fallback": synthesis["used_fallback"],
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
                    "used_fallback": False,
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
        scored_expansions = []
        for candidate_index_value, expansion in enumerate(expansions.values()):
            objective = score_combined_objective(
                expansion["prompt"],
                train_records,
                mode=context.mode,
                model=model,
                tokenizer=tokenizer,
                batch_size=selection_batch_size,
                fluency_lambda=fluency_lambda,
            )
            scored_expansions.append(
                {"candidate_index": candidate_index_value, **expansion, **objective}
            )
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
    initial_evaluation = context.evaluator.evaluate(
        context.initial_prompt,
        context.validation_records,
        split_name="validation",
        log_label=f"qa_gradpo_{args.variant}_initial_validation",
    )
    source = _source_scored_item(context, initial_evaluation)
    train_records = sample_label_balanced_records(
        context.train_records,
        args.train_sample_size,
        context.rng,
    )
    model, tokenizer = context.model_pool.ensure(TARGET_ROLE)
    gradient_analysis = collect_instruction_gradients(
        context.initial_prompt,
        train_records,
        mode=context.mode,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.gradient_batch_size,
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
    raw_candidate_output = None
    candidate_meta_prompt = None
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
        )
    else:
        region_candidates = _gradpo_probability_candidates(
            context.initial_prompt,
            selected_regions,
            candidate_count=args.num_region_candidates,
            model=model,
            tokenizer=tokenizer,
            model_id=args.model,
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
    )
    dev_prompts = unique_nonempty(
        [item["prompt"] for item in beam if item["prompt"] != context.initial_prompt]
    )
    dev_scored = evaluate_candidates(
        context,
        dev_prompts,
        context.validation_records,
        split_name="validation",
        phase=f"gradpo_{args.variant}_candidate_validation",
        iteration=1,
    )
    selected, improved = _strictly_select_against_source(source, dev_scored)
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
    return finalize_run(
        context,
        initial_evaluation=initial_evaluation,
        final_prompt=selected["prompt"],
        final_evaluation=selected["evaluation"],
        extra_summary={
            "algorithm": f"gradpo_{args.variant}",
            "iterations": 1,
            "improved_on_validation": improved,
            "selection_mode": selection_mode,
            "num_edit_regions": num_edit_regions,
            "max_region_tokens": max_region_tokens,
            "region_pool_size": len(region_pool),
            "selected_regions": selected_regions,
            "region_candidates": region_candidates,
            "final_beam": beam,
        },
    )
