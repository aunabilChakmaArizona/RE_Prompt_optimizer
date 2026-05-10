from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from typing import Any, Iterable, List, Optional

from agents.agent_feedback_samples import FeedbackSamples
from agents.agent_graph_node import GraphNode
from agents.agent_memory import clear_iteration_memory
from agents.agent_prompts import (
    EXAMPLE_GENERATION_PROMPT_V1,
    FEEDBACK_PROMPT_MAP,
    INFERENCE_MODE_NON_SEPARATE,
    apply_tag_overrides,
    resolve_mutation_prompt,
)
from agents.agent_train_config import build_parser, resolve_data_dir
from agents.agent_train_io import (
    create_run_dir,
    load_initial_prompt_node,
    restore_logging,
    save_population,
    save_resume_state,
    save_json,
    save_summary,
    setup_logging,
    write_args,
)
from agents.agent_train_pipeline import build_root_node, load_model_and_data
from agents.agent_scorer import NO_RELATION
from agents.agent_utils import stable_f1_score_or_neg_inf


EVOPROMPT_DE_TEMPLATE_RE = """Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between Prompt 1 and Prompt 2:
Prompt 1: Determine whether the named relation holds between the tagged subject and tagged object in the query sentence. Use the support sentence only as an example of the relation.
Prompt 2: Decide if the query sentence expresses the requested relation from Subject to Object. Check the relation definition, entity roles, direction, and whether the evidence is explicit or clearly implied.
2. Randomly mutate the different parts.
3. Combine the different parts with Prompt 3, selectively replace it with the different parts from step 2, and generate a new prompt.
Prompt 3: Carefully compare the relation definition with the query sentence, verify that the tagged Subject and Object have the right semantic types and direction, and avoid using irrelevant facts from the support example.
4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with #PROMPT_OPEN_TAG# and #PROMPT_CLOSE_TAG#:
Basic Prompt: You are given a relation name, a relation description, a support sentence that exemplifies the relation, and a query sentence. Answer only yes or no depending on whether the relation holds between the tagged Subject and Object in the query sentence.

1. Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Determine whether the named relation holds between the tagged subject and tagged object in the query sentence. Use the support sentence only as an example of the relation.
Prompt 2: Decide if the query sentence expresses the requested relation from Subject to Object. Check the relation definition, entity roles, direction, and whether the evidence is explicit or clearly implied.
Different parts:
"named relation holds" vs "query sentence expresses the requested relation"
"tagged subject and tagged object" vs "Subject to Object"
"Use the support sentence only as an example" vs "Check the relation definition, entity roles, direction"

2. Randomly mutate the different parts:
"named relation holds" -> "specified relation is valid"
"query sentence expresses the requested relation" -> "query provides enough evidence for the target relation"
"tagged subject and tagged object" -> "marked Subject and Object entities"
"Check the relation definition, entity roles, direction" -> "verify the definition, argument roles, semantic types, and direction"

3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt:
Prompt 3: Carefully compare the relation definition with the query sentence, verify that the tagged Subject and Object have the right semantic types and direction, and avoid using irrelevant facts from the support example.
New Prompt: Carefully compare the relation definition with the query sentence and decide whether the specified relation is valid for the marked Subject and Object entities. Verify the definition, argument roles, semantic types, and direction, and use the support sentence only as a guide to the relation type.

4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with #PROMPT_OPEN_TAG# and #PROMPT_CLOSE_TAG#:
Basic Prompt: You are given a relation name, a relation description, a support sentence that exemplifies the relation, and a query sentence. Answer only yes or no depending on whether the relation holds between the tagged Subject and Object in the query sentence.
Final Prompt: #PROMPT_OPEN_TAG#Carefully compare the relation definition with the query sentence and decide whether the specified relation is valid for the tagged Subject and Object entities. Use the support sentence only as an example of the relation type. Verify the argument roles, semantic types, direction of the relation, and whether the query gives explicit or clearly implied evidence. If the relation holds between the Subject and Object in the query sentence, answer "yes"; otherwise, answer "no".#PROMPT_CLOSE_TAG#

Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts.
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2, and generate a new prompt.
Prompt 3: <prompt3>
4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with #PROMPT_OPEN_TAG# and #PROMPT_CLOSE_TAG#:
Basic Prompt: <prompt0>

1. """


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    parser.description = "Run EvoPrompt-DE prompt optimization for relation extraction."
    parser.set_defaults(population_size=5, update_mode="no_feedback")
    parser.add_argument(
        "--evoprompt-de-donor-random",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Match EvoPrompt DE donor behavior. If false, prompt3 is the current "
            "best prompt; if true, prompt3 is sampled randomly from the population."
        ),
    )
    parser.add_argument(
        "--evoprompt-train-episodes-per-iteration",
        type=int,
        default=1000,
        help="Number of train-sampled episodes used as the per-iteration fitness set.",
    )
    return parser.parse_args()


def _clean_prompt(prompt: str) -> str:
    return "\n".join(line.rstrip() for line in prompt.strip().splitlines()).strip()


def _as_non_separate_prompt(instruction: str) -> str:
    return _clean_prompt(
        f"""{instruction}

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)

#SUPPORT_SENTENCE_BLOCK#

Query Sentence: #QUERY_SENTENCE#

If the relation holds between the Subject and Object in the query sentence, say "yes"; otherwise, say "no". Output only "yes" or "no", and nothing else."""
    )


def _initial_prompt_variants(
    base_prompt: str,
    *,
    inference_mode: str,
) -> list[tuple[str, str]]:
    base_prompt = _clean_prompt(base_prompt)
    variants = [
        ("original", base_prompt),
        (
            "human1_simplest",
            _clean_prompt(
                """Decide whether the given relation holds between the tagged Subject and Object entities in the query sentence. The support sentence is an example of the relation. Answer "yes" if the relation holds in the query; otherwise answer "no"."""
            ),
        ),
        (
            "chatgpt_large",
            _clean_prompt(
                """You are given a relation name, its description, a support sentence that exemplifies the relation, and a query sentence. Determine whether the query sentence expresses that exact relation between the tagged Subject and Object entities. Check the relation definition, the semantic type of each entity, the direction from Subject to Object, and whether the evidence is explicit or strongly implied. Treat the support sentence as an example of the relation, not as evidence about the query. If the relation holds in the query sentence, answer "yes"; otherwise answer "no"."""
            ),
        ),
        (
            "chatgpt_easy",
            _clean_prompt(
                """Read the relation description, then read the query sentence. The Subject and Object are already marked with tags. Decide if the query says that this relation is true for that Subject and Object. The support sentence is only an example to help you understand the relation. Answer only "yes" or "no"."""
            ),
        ),
        (
            "chatgpt_complex",
            _clean_prompt(
                """Perform binary relation verification for the query sentence. First identify the tagged Subject and Object, then compare their roles to the relation name and definition. Confirm that the relation direction is Subject-to-Object, that the Object has the required type or value for the relation, and that the query sentence itself supplies sufficient explicit or contextually implied evidence. Ignore distracting entities and do not transfer facts from the support sentence beyond its illustration of the relation schema. Output "yes" exactly when the relation holds; otherwise output "no"."""
            ),
        ),
    ]
    if inference_mode == INFERENCE_MODE_NON_SEPARATE:
        return [
            (mark, prompt if mark == "original" else _as_non_separate_prompt(prompt))
            for mark, prompt in variants
        ]
    return variants


def _make_prompt_node(
    *,
    root: GraphNode,
    prompt: str,
    mark: str,
    node_id: int,
    feedback_prompt: str,
    mutation_prompt: str,
) -> GraphNode:
    inference_prompt = prompt
    instruction_prompt = root.inference_instruction_prompt
    if root.inference_mode != INFERENCE_MODE_NON_SEPARATE:
        instruction_prompt = prompt
    node = GraphNode(
        inference_prompt=inference_prompt,
        inference_mode=root.inference_mode,
        inference_instruction_prompt=instruction_prompt,
        inference_answer_instruction_prompt=root.inference_answer_instruction_prompt,
        inference_example_prompt=root.inference_example_prompt,
        inference_input_prompt=root.inference_input_prompt,
        feedback_prompt=feedback_prompt,
        mutation_prompt=mutation_prompt,
        example_generation_prompt=EXAMPLE_GENERATION_PROMPT_V1,
        node_id=node_id,
    )
    node.differentiation = f"initial_mark={mark}"
    return node


def _score_attr(node: GraphNode, attr_name: str, std_penalty: float) -> float:
    return stable_f1_score_or_neg_inf(
        getattr(node, attr_name, None),
        std_multiplier=std_penalty,
    )


def _metric_mean_attr(node: GraphNode, attr_name: str) -> float:
    score = getattr(node, attr_name, None)
    if isinstance(score, dict) and score.get("f1_mean") is not None:
        return float(score["f1_mean"])
    return float("-inf")


def _metric_std_attr(node: GraphNode, attr_name: str) -> float:
    score = getattr(node, attr_name, None)
    if isinstance(score, dict):
        return float(score.get("f1_std", 0.0) or 0.0)
    return float("inf")


def _sort_population(
    nodes: Iterable[GraphNode],
    *,
    std_penalty: float,
    score_attr: str = "val_score",
) -> List[GraphNode]:
    return sorted(
        nodes,
        key=lambda node: (
            _score_attr(node, score_attr, std_penalty),
            _metric_mean_attr(node, score_attr),
            -_metric_std_attr(node, score_attr),
            node.node_id if node.node_id is not None else -1,
        ),
        reverse=True,
    )


def _build_train_relation_index(feedback_pool: dict[str, list[dict]]) -> dict[str, list[dict]]:
    relation_index: dict[str, list[dict]] = defaultdict(list)
    for instances in feedback_pool.values():
        for instance in instances:
            relation = instance.get("relation")
            if relation:
                relation_index[relation].append(instance)
    return dict(relation_index)


def _sample_from_relation(
    relation_index: dict[str, list[dict]],
    relation: str,
    rng: random.Random,
    *,
    exclude_ids: set[str] | None = None,
) -> dict:
    candidates = relation_index[relation]
    if exclude_ids:
        filtered = [
            instance for instance in candidates if str(instance.get("id")) not in exclude_ids
        ]
        if filtered:
            candidates = filtered
    return rng.choice(candidates)


def _sample_train_episode_bundle(
    *,
    relation_index: dict[str, list[dict]],
    rng: random.Random,
    num_episodes: int,
    num_ways: int,
    positive_probability: float = 0.05,
) -> dict[str, Any]:
    relation_names = sorted(
        relation
        for relation, instances in relation_index.items()
        if relation != NO_RELATION and instances
    )
    if len(relation_names) < num_ways:
        raise ValueError(
            f"Need at least {num_ways} train relations, found {len(relation_names)}."
        )

    shots: dict[str, dict] = {}
    queries: dict[str, dict] = {}
    episodes: list[dict[str, list]] = []
    negative_relations = list(relation_names)
    if NO_RELATION in relation_index and relation_index[NO_RELATION]:
        negative_relations.append(NO_RELATION)

    for episode_index in range(num_episodes):
        way_relations = rng.sample(relation_names, num_ways)
        meta_train = []
        support_ids: set[str] = set()
        for relation in way_relations:
            support = _sample_from_relation(relation_index, relation, rng)
            support_id = str(support["id"])
            shots[support_id] = support
            support_ids.add(support_id)
            meta_train.append([support_id])

        if rng.random() < positive_probability:
            query_relation = rng.choice(way_relations)
        else:
            outside_relations = [
                relation for relation in negative_relations if relation not in way_relations
            ]
            query_relation = rng.choice(outside_relations or negative_relations)

        query = _sample_from_relation(
            relation_index,
            query_relation,
            rng,
            exclude_ids=support_ids,
        )
        query_id = f"train_iter_query_{episode_index}_{query['id']}"
        queries[query_id] = dict(query)
        episodes.append({"meta_train": meta_train, "meta_test": [query_id]})

    return {
        "split": "train_sampled",
        "episodes": episodes,
        "shots": shots,
        "queries": queries,
        "umbc_shots": {},
    }


def _score_payload(score: Any, std_penalty: float) -> dict[str, Any]:
    payload = dict(score) if isinstance(score, dict) else {"raw_score": score}
    payload["stable_f1"] = stable_f1_score_or_neg_inf(
        score if isinstance(score, dict) else None,
        std_multiplier=std_penalty,
    )
    return payload


def _sample_de_donors(
    *,
    population: List[GraphNode],
    target: GraphNode,
    best: GraphNode,
    rng: random.Random,
    donor_random: bool,
) -> tuple[GraphNode, GraphNode, GraphNode]:
    candidates = [node for node in population if node is not target]
    if len(candidates) < 3:
        raise RuntimeError("EvoPrompt-DE requires at least four active population nodes.")

    donor_a, donor_b, donor_c = rng.sample(candidates, 3)
    if donor_random:
        return donor_a, donor_b, donor_c

    return donor_a, donor_b, best


def _build_de_prompt(
    *,
    prompt0: str,
    prompt1: str,
    prompt2: str,
    prompt3: str,
    prompt_open_tag: str,
    prompt_close_tag: str,
) -> str:
    return (
        EVOPROMPT_DE_TEMPLATE_RE.replace("<prompt0>", prompt0)
        .replace("<prompt1>", prompt1)
        .replace("<prompt2>", prompt2)
        .replace("<prompt3>", prompt3)
        .replace("#PROMPT_OPEN_TAG#", prompt_open_tag)
        .replace("#PROMPT_CLOSE_TAG#", prompt_close_tag)
    )


def _generate_de_child(
    *,
    target: GraphNode,
    donor_a: GraphNode,
    donor_b: GraphNode,
    donor_c: GraphNode,
    mutate_prompt_fn,
    args,
    iteration: int,
) -> Optional[tuple[str, str, str]]:
    de_prompt = _build_de_prompt(
        prompt0=target.inference_prompt,
        prompt1=donor_a.inference_prompt,
        prompt2=donor_b.inference_prompt,
        prompt3=donor_c.inference_prompt,
        prompt_open_tag=args.prompt_open_tag,
        prompt_close_tag=args.prompt_close_tag,
    )
    print("[core_trainer_evoprompt_de] DE meta-prompt:")
    print(de_prompt)
    optimizer_node = GraphNode(
        inference_prompt=target.inference_prompt,
        inference_mode=target.inference_mode,
        inference_instruction_prompt=target.inference_instruction_prompt,
        inference_answer_instruction_prompt=target.inference_answer_instruction_prompt,
        inference_example_prompt=target.inference_example_prompt,
        inference_input_prompt=target.inference_input_prompt,
        mutation_prompt=de_prompt,
        node_id=target.node_id,
    )
    mutation_result = mutate_prompt_fn(
        optimizer_node,
        FeedbackSamples(),
        mutation_prompt_override=de_prompt,
        mutation_prompt_key_override="evoprompt_de",
        enable_thinking=False,
    )
    if mutation_result is None:
        print(
            "[core_trainer_evoprompt_de] DE generation failed "
            f"for target node_id={target.node_id}"
        )
        return None
    child_prompt, raw_response, mutation_prompt_used, *_ = mutation_result
    print("[core_trainer_evoprompt_de] DE child prompt:")
    print(child_prompt)
    return child_prompt, raw_response, mutation_prompt_used


def main() -> None:
    overall_start = time.monotonic()
    args = parse_args()
    if args.validation_f1_std_penalty < 0:
        raise ValueError("--validation-f1-std-penalty must be non-negative.")
    if args.population_size != 5:
        raise ValueError(
            "This EvoPrompt-DE runner is intentionally fixed to --population-size 5 for now."
        )
    if args.evoprompt_train_episodes_per_iteration <= 0:
        raise ValueError("--evoprompt-train-episodes-per-iteration must be > 0.")
    if args.load_population:
        raise ValueError("EvoPrompt-DE does not support --load-population yet.")
    if args.resume_from_nohup_log:
        raise ValueError("EvoPrompt-DE does not support --resume-from-nohup-log yet.")

    data_dir = resolve_data_dir(args.data_dir)
    run_dir = create_run_dir(args.trainings_dir, args.model)
    log_file, original_stdout, original_stderr = setup_logging(run_dir)

    try:
        print(f"[core_trainer_evoprompt_de] data directory: {data_dir}")
        print("[core_trainer_evoprompt_de] run_dir:", run_dir)
        print("[core_trainer_evoprompt_de] model:", args.model)
        print("[core_trainer_evoprompt_de] optimizer_model:", args.optimizer_model or args.model)
        print("[core_trainer_evoprompt_de] dataset_type:", args.dataset_type)
        print("[core_trainer_evoprompt_de] algorithm: EvoPrompt-DE")

        write_args(run_dir, args)
        eval_output_dir = os.path.join(run_dir, "eval_outputs")
        os.makedirs(eval_output_dir, exist_ok=True)

        _, _, rng, feedback_pool, _, dev_data, _, funcs = load_model_and_data(
            args, data_dir, eval_output_dir, args.seed
        )
        _, _, _, mutate_prompt, evaluate = funcs
        relation_index = _build_train_relation_index(feedback_pool)
        num_train_episode_ways = (
            len(dev_data["episodes"][0]["meta_train"])
            if dev_data.get("episodes")
            else 5
        )
        print(
            "[core_trainer_evoprompt_de] train sampled fitness episodes per iteration:",
            args.evoprompt_train_episodes_per_iteration,
            "ways:",
            num_train_episode_ways,
        )

        feedback_prompt = apply_tag_overrides(
            FEEDBACK_PROMPT_MAP[args.feedback_prompt],
            feedback_open_tag=args.feedback_open_tag,
            feedback_close_tag=args.feedback_close_tag,
            prompt_open_tag=args.prompt_open_tag,
            prompt_close_tag=args.prompt_close_tag,
        )
        mutation_prompt = apply_tag_overrides(
            resolve_mutation_prompt(args.mutation_prompt, args.inference_mode),
            feedback_open_tag=args.feedback_open_tag,
            feedback_close_tag=args.feedback_close_tag,
            prompt_open_tag=args.prompt_open_tag,
            prompt_close_tag=args.prompt_close_tag,
        )

        initial_prompt_node = None
        if args.initial_prompt_source_path:
            initial_prompt_node, initial_prompt_population_path = load_initial_prompt_node(
                args.initial_prompt_source_path,
                args.initial_prompt_node_id,
                args.inference_mode,
            )
            print(
                "[core_trainer_evoprompt_de] initial prompt source:",
                initial_prompt_population_path,
            )
            print(
                "[core_trainer_evoprompt_de] initial prompt node_id:",
                initial_prompt_node.get("node_id"),
            )

        root = build_root_node(
            feedback_prompt=feedback_prompt,
            mutation_prompt=mutation_prompt,
            inference_mode=args.inference_mode,
            example_generation_prompt=EXAMPLE_GENERATION_PROMPT_V1,
            initial_prompt_node=initial_prompt_node,
        )

        population: List[GraphNode] = []
        population_history: List[GraphNode] = []
        initial_variants = _initial_prompt_variants(
            root.inference_prompt,
            inference_mode=args.inference_mode,
        )
        for node_id, (mark, prompt) in enumerate(initial_variants):
            node = _make_prompt_node(
                root=root,
                prompt=prompt,
                mark=mark,
                node_id=node_id,
                feedback_prompt=feedback_prompt,
                mutation_prompt=mutation_prompt,
            )
            print(
                "[core_trainer_evoprompt_de] initial population "
                f"node_id={node.node_id} mark={mark}"
            )
            print(node.inference_prompt)
            population.append(node)
            population_history.append(node)

        dev_best_node: GraphNode | None = None
        dev_best_history: list[dict[str, Any]] = []
        next_node_id = len(population_history)

        for iteration in range(args.max_iterations):
            iteration_start = time.monotonic()
            print(
                f"[core_trainer_evoprompt_de] iteration {iteration + 1}/"
                f"{args.max_iterations} start"
            )
            train_eval_bundle = _sample_train_episode_bundle(
                relation_index=relation_index,
                rng=rng,
                num_episodes=args.evoprompt_train_episodes_per_iteration,
                num_ways=num_train_episode_ways,
            )
            print(
                "[core_trainer_evoprompt_de] evaluating existing population on "
                f"{len(train_eval_bundle['episodes'])} sampled train episodes"
            )
            for target in population:
                target.fitness_score = evaluate(
                    target,
                    train_eval_bundle,
                    eval_id_override=f"iter_{iteration + 1}_fitness_existing_{target.node_id}",
                    evolution_iteration=iteration + 1,
                    evolution_max_iterations=args.max_iterations,
                )
            population = _sort_population(
                population,
                std_penalty=args.validation_f1_std_penalty,
                score_attr="fitness_score",
            )[: args.population_size]
            best_node = _sort_population(
                population,
                std_penalty=args.validation_f1_std_penalty,
                score_attr="fitness_score",
            )[0]
            new_population: List[GraphNode] = []

            for index, target in enumerate(list(population)):
                print(
                    "[core_trainer_evoprompt_de] target "
                    f"pop_index={index} node_id={target.node_id}"
                )
                donor_a, donor_b, donor_c = _sample_de_donors(
                    population=population,
                    target=target,
                    best=best_node,
                    rng=rng,
                    donor_random=args.evoprompt_de_donor_random,
                )
                print(
                    "[core_trainer_evoprompt_de] donors "
                    f"a={donor_a.node_id} b={donor_b.node_id} c={donor_c.node_id}"
                )
                generated = _generate_de_child(
                    target=target,
                    donor_a=donor_a,
                    donor_b=donor_b,
                    donor_c=donor_c,
                    mutate_prompt_fn=mutate_prompt,
                    args=args,
                    iteration=iteration,
                )
                if generated is None:
                    new_population.append(target)
                    continue
                child_prompt, raw_response, mutation_prompt_used = generated
                child = GraphNode(
                    inference_prompt=child_prompt,
                    inference_mode=target.inference_mode,
                    inference_instruction_prompt=(
                        child_prompt
                        if target.inference_mode != INFERENCE_MODE_NON_SEPARATE
                        else target.inference_instruction_prompt
                    ),
                    inference_answer_instruction_prompt=target.inference_answer_instruction_prompt,
                    inference_example_prompt=target.inference_example_prompt,
                    inference_input_prompt=target.inference_input_prompt,
                    parent=target,
                    feedback="",
                    feedback_prompt=feedback_prompt,
                    mutation_prompt=mutation_prompt,
                    example_generation_prompt=EXAMPLE_GENERATION_PROMPT_V1,
                    mutation_prompt_used=mutation_prompt_used,
                    raw_mutation_response=raw_response,
                    differentiation_prompt_used="evoprompt_de",
                    differentiation=(
                        "evoprompt_de "
                        f"target={target.node_id} donor_a={donor_a.node_id} "
                        f"donor_b={donor_b.node_id} donor_c={donor_c.node_id}"
                    ),
                    node_id=next_node_id,
                )
                next_node_id += 1
                child.fitness_score = evaluate(
                    child,
                    train_eval_bundle,
                    eval_id_override=f"iter_{iteration + 1}_fitness_child_{child.node_id}",
                    evolution_iteration=iteration + 1,
                    evolution_max_iterations=args.max_iterations,
                )
                target.add_child_from_feedback(FeedbackSamples(), child)
                population_history.append(child)

                selected = max(
                    [target, child],
                    key=lambda node: (
                        _score_attr(
                            node,
                            "fitness_score",
                            args.validation_f1_std_penalty,
                        ),
                        _metric_mean_attr(node, "fitness_score"),
                        -_metric_std_attr(node, "fitness_score"),
                        node.node_id if node.node_id is not None else -1,
                    ),
                )
                new_population.append(selected)
                print(
                    "[core_trainer_evoprompt_de] selected "
                    f"node_id={selected.node_id} train_stable_f1="
                    f"{_score_attr(selected, 'fitness_score', args.validation_f1_std_penalty):.6f}"
                )

            population = _sort_population(
                new_population,
                std_penalty=args.validation_f1_std_penalty,
                score_attr="fitness_score",
            )[: args.population_size]

            iteration_best_by_train = population[0]
            print(
                "[core_trainer_evoprompt_de] evaluating current train-fitness best "
                f"on actual dev: node_id={iteration_best_by_train.node_id}"
            )
            iteration_best_by_train.val_score = evaluate(
                iteration_best_by_train,
                "validation",
                eval_id_override=f"iter_{iteration + 1}_dev_candidate_{iteration_best_by_train.node_id}",
                evolution_iteration=iteration + 1,
                evolution_max_iterations=args.max_iterations,
            )
            if dev_best_node is None or (
                _score_attr(
                    iteration_best_by_train,
                    "val_score",
                    args.validation_f1_std_penalty,
                )
                > _score_attr(dev_best_node, "val_score", args.validation_f1_std_penalty)
            ):
                dev_best_node = iteration_best_by_train

            dev_best_history.append(
                {
                    "iteration": iteration + 1,
                    "train_fitness_best_node_id": iteration_best_by_train.node_id,
                    "train_fitness_best_score": _score_payload(
                        getattr(iteration_best_by_train, "fitness_score", None),
                        args.validation_f1_std_penalty,
                    ),
                    "dev_candidate_node_id": iteration_best_by_train.node_id,
                    "dev_candidate_score": _score_payload(
                        iteration_best_by_train.val_score,
                        args.validation_f1_std_penalty,
                    ),
                    "best_dev_so_far_node_id": (
                        dev_best_node.node_id if dev_best_node is not None else None
                    ),
                    "best_dev_so_far_score": _score_payload(
                        dev_best_node.val_score if dev_best_node is not None else None,
                        args.validation_f1_std_penalty,
                    ),
                    "best_dev_so_far_prompt": (
                        dev_best_node.inference_prompt if dev_best_node is not None else None
                    ),
                }
            )
            save_json(
                run_dir,
                "dev_best_history.json",
                {"history": dev_best_history},
            )
            clear_iteration_memory(iteration, best_node, population[0])
            print(
                f"[core_trainer_evoprompt_de] iteration {iteration + 1} end "
                f"(elapsed={time.monotonic() - iteration_start:.2f}s, "
                f"overall={time.monotonic() - overall_start:.2f}s)"
            )

        best_node = dev_best_node or _sort_population(
            population,
            std_penalty=args.validation_f1_std_penalty,
            score_attr="fitness_score",
        )[0]
        print("[core_trainer_evoprompt_de] recording best validation metrics")
        best_test_metrics = best_node.val_score

        save_population(
            run_dir,
            population_history,
            best_node,
            final_population=population,
        )
        save_resume_state(run_dir, rng, next_node_id)
        save_summary(
            run_dir,
            {
                "run_dir": run_dir,
                "algorithm": "evoprompt_de",
                "best_node_id": best_node.node_id,
                "best_val_score": best_node.val_score,
                "best_test_metrics": best_test_metrics,
                "population_size": len(population),
                "population_history_size": len(population_history),
                "initial_population_marks": [mark for mark, _ in initial_variants],
                "validation_f1_std_penalty": args.validation_f1_std_penalty,
                "train_episodes_per_iteration": args.evoprompt_train_episodes_per_iteration,
                "dev_best_history": dev_best_history,
            },
        )
        print(f"[core_trainer_evoprompt_de] done (elapsed={time.monotonic() - overall_start:.2f}s)")
    finally:
        restore_logging(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
