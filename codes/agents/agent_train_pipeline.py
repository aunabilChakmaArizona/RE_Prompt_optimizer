from __future__ import annotations

import random
from typing import Dict, List, Tuple

from agents.agent_data_loader import load_split_episodes, load_train_samples
from agents.agent_evaluate import evaluate_fn as _evaluate_fn
from agents.agent_generate_feedback import generate_feedback_fn as _generate_feedback_fn
from agents.agent_graph_node import GraphNode
from agents.agent_models import load_model_and_tokenizer
from agents.agent_mutate_prompt import mutate_prompt_fn as _mutate_prompt_fn
from agents.agent_prompts import (
    EXAMPLE_GENERATION_PROMPT_V1,
    INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1,
    INFERENCE_INPUT_PROMPT_V1,
    INFERENCE_EXAMPLE_PROMPT_V1,
    INFERENCE_INSTRUCTION_PROMPT_V1,
    INFERENCE_MODE_NON_SEPARATE,
    INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    INFERENCE_MODE_SEPARATE_WITH_EXAMPLES,
    INFERENCE_PROMPT_V1,
)
from agents.agent_run_inference import run_inference_fn as _run_inference_fn
from agents.agent_sample_feedback import sample_feedback_fn as _sample_feedback_fn
from agents.agent_utils import get_sentence_with_tags


def _resolve_binary_token_id(tokenizer, base_token: str) -> int:
    token_ids = tokenizer.encode(base_token, add_special_tokens=False)
    if len(token_ids) == 1:
        return token_ids[0]
    spaced_token_ids = tokenizer.encode(f" {base_token}", add_special_tokens=False)
    if len(spaced_token_ids) == 1:
        return spaced_token_ids[0]
    raise ValueError(
        f"Token '{base_token}' is not a single token for this tokenizer."
    )


def build_feedback_pool(shots: Dict) -> Dict[str, List[dict]]:
    pool: Dict[str, List[dict]] = {}
    for shot in shots.values():
        relation = shot["relation"]
        pool.setdefault(relation, []).append(shot)
    return pool


def build_shot_index(shots: Dict) -> Dict[str, dict]:
    if not shots:
        return {}
    first_value = next(iter(shots.values()))
    if isinstance(first_value, list):
        index: Dict[str, dict] = {}
        for instances in shots.values():
            for inst in instances:
                index[inst["id"]] = inst
        return index
    return shots


def enrich_feedback_samples(feedback_samples, shots_by_id: Dict[str, dict]) -> None:
    for sample in feedback_samples.all_samples:
        support = shots_by_id[sample.id_1shot]
        query = shots_by_id[sample.id_query]
        sample.relation = support["relation"]
        sample.support_sentence = get_sentence_with_tags(support).strip()
        sample.query_sentence = get_sentence_with_tags(query).strip()


def load_data_assets(args, data_dir: str) -> Tuple[Dict, Dict, Dict, Dict]:

    print("[agent_train_pipeline] loading dataset")

    train_samples = load_train_samples(data_dir=data_dir, filename=args.train_samples)
    feedback_pool = train_samples
    train_shot_index = build_shot_index(feedback_pool)
    total_train_instances = sum(
        len(instances) for instances in feedback_pool.values()
        if isinstance(instances, list)
    )

    print(
        f"[agent_train_pipeline] train_samples={args.train_samples} "
        f"relations={len(feedback_pool)} instances={total_train_instances}"
    )
    print(
        f"[agent_train_pipeline] dataset_type={args.dataset_type} "
        f"dev_split={args.dev_split} test_split={args.test_split}"
    )

    dev_data = load_split_episodes(
        split=args.dev_split,
        data_dir=data_dir,
        dataset_type=args.dataset_type,
        ep_start=args.dev_ep_start,
        ep_end=args.dev_ep_end,
    )
    test_data = load_split_episodes(
        split=args.test_split,
        data_dir=data_dir,
        dataset_type=args.dataset_type,
        ep_start=args.test_ep_start,
        ep_end=args.test_ep_end,
    )
    
    print(
        f"[agent_train_pipeline] dev_episodes={len(dev_data.get('episodes', []))} "
        f"test_episodes={len(test_data.get('episodes', []))}"
    )
    print("[agent_train_pipeline] dataset loaded")

    return feedback_pool, train_shot_index, dev_data, test_data


def build_training_functions(
    *,
    args,
    model,
    tokenizer,
    feedback_pool: Dict[str, List[dict]],
    train_shot_index: Dict[str, dict],
    dev_data: Dict,
    test_data: Dict,
    eval_output_dir: str,
    rng: random.Random,
):
    yes_token_id = _resolve_binary_token_id(tokenizer, "yes")
    no_token_id = _resolve_binary_token_id(tokenizer, "no")

    def sample_feedback(k: int):
        samples = _sample_feedback_fn(feedback_pool, k=k, rng=rng)
        enrich_feedback_samples(samples, train_shot_index)
        return samples

    def run_inference(
        node: GraphNode,
        feedback_samples,
        *,
        evolution_iteration: int | None = None,
        evolution_max_iterations: int | None = None,
    ):
        return _run_inference_fn(
            node,
            feedback_samples,
            dataset_type=args.dataset_type,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.inference_batch_size,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            log_every=args.log_every,
            evolution_iteration=evolution_iteration,
            evolution_max_iterations=evolution_max_iterations,
        )

    def generate_feedback(node: GraphNode, feedback_samples):
        return _generate_feedback_fn(
            node,
            feedback_samples,
            dataset_type=args.dataset_type,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.feedback_batch_size,
            max_new_tokens=args.feedback_max_new_tokens,
            do_sample=args.do_sample,
            feedback_open_tag=args.feedback_open_tag,
            feedback_close_tag=args.feedback_close_tag,
        )

    def mutate_prompt(
        node: GraphNode,
        feedback_samples,
        *,
        mutation_prompt_override: str | None = None,
        mutation_prompt_key_override: str | None = None,
    ):
        return _mutate_prompt_fn(
            node,
            feedback_samples,
            dataset_type=args.dataset_type,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            prompt_open_tag=args.prompt_open_tag,
            prompt_close_tag=args.prompt_close_tag,
            mutation_prompt_override=mutation_prompt_override,
            mutation_prompt_key_override=mutation_prompt_key_override,
        )

    def evaluate(
        node: GraphNode,
        split: str,
        *,
        eval_id_override: str | int | None = None,
        evolution_iteration: int | None = None,
        evolution_max_iterations: int | None = None,
    ):
        eval_id = eval_id_override if eval_id_override is not None else (
            node.node_id if node.node_id is not None else 0
        )
        if split == "validation":
            episodes = dev_data["episodes"]
            shots = dev_data
        else:
            episodes = test_data["episodes"]
            shots = test_data
        return _evaluate_fn(
            node,
            split,
            dataset_type=args.dataset_type,
            model=model,
            tokenizer=tokenizer,
            episodes=episodes,
            shots=shots,
            query_index=args.query_index,
            batch_size=args.eval_batch_size,
            n_chunks=args.eval_n_chunks,
            eval_id=eval_id,
            output_dir=eval_output_dir,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            log_every=args.log_every,
            evolution_iteration=evolution_iteration,
            evolution_max_iterations=evolution_max_iterations,
        )

    return sample_feedback, run_inference, generate_feedback, mutate_prompt, evaluate


def build_root_node(
    feedback_prompt: str,
    mutation_prompt: str,
    inference_mode: str = INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    example_generation_prompt: str = EXAMPLE_GENERATION_PROMPT_V1,
) -> GraphNode:
    if inference_mode == INFERENCE_MODE_NON_SEPARATE:
        inference_prompt = INFERENCE_PROMPT_V1
        instruction_prompt = ""
        answer_instruction_prompt = ""
        example_prompt = ""
        input_prompt = ""
    elif inference_mode == INFERENCE_MODE_SEPARATE_NO_EXAMPLES:
        inference_prompt = INFERENCE_INSTRUCTION_PROMPT_V1
        instruction_prompt = INFERENCE_INSTRUCTION_PROMPT_V1
        answer_instruction_prompt = INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1
        example_prompt = ""
        input_prompt = INFERENCE_INPUT_PROMPT_V1
    elif inference_mode == INFERENCE_MODE_SEPARATE_WITH_EXAMPLES:
        inference_prompt = INFERENCE_INSTRUCTION_PROMPT_V1
        instruction_prompt = INFERENCE_INSTRUCTION_PROMPT_V1
        answer_instruction_prompt = INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1
        example_prompt = INFERENCE_EXAMPLE_PROMPT_V1
        input_prompt = INFERENCE_INPUT_PROMPT_V1
    else:
        raise ValueError(f"Unsupported inference_mode: {inference_mode}")

    return GraphNode(
        inference_prompt=inference_prompt,
        inference_mode=inference_mode,
        inference_instruction_prompt=instruction_prompt,
        inference_answer_instruction_prompt=answer_instruction_prompt,
        inference_example_prompt=example_prompt,
        inference_input_prompt=input_prompt,
        feedback_prompt=feedback_prompt,
        mutation_prompt=mutation_prompt,
        example_generation_prompt=example_generation_prompt,
        node_id=0,
    )


def load_model_and_data(args, data_dir: str, eval_output_dir: str, rng_seed: int):
    rng = random.Random(rng_seed)
    model, tokenizer = load_model_and_tokenizer(
        args.model, device_map=args.device_map
    )
    feedback_pool, train_shot_index, dev_data, test_data = load_data_assets(
        args, data_dir
    )
    functions = build_training_functions(
        args=args,
        model=model,
        tokenizer=tokenizer,
        feedback_pool=feedback_pool,
        train_shot_index=train_shot_index,
        dev_data=dev_data,
        test_data=test_data,
        eval_output_dir=eval_output_dir,
        rng=rng,
    )
    return model, tokenizer, rng, feedback_pool, train_shot_index, dev_data, test_data, functions
