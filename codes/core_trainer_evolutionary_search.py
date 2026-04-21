from __future__ import annotations

import os
import time

from agents.agent_evolutionary_search import EvolutionarySearch
from agents.agent_memory import clear_iteration_memory
from agents.agent_prompts import (
    EXAMPLE_GENERATION_PROMPT_V1,
    FEEDBACK_PROMPT_MAP,
    MUTATION_PROMPT_GROUP_MAP,
    apply_tag_overrides,
    resolve_mutation_prompt,
)
from agents.agent_train_config import parse_args, resolve_data_dir
from agents.agent_train_io import (
    create_run_dir,
    load_initial_prompt_node,
    restore_logging,
    save_population,
    save_summary,
    setup_logging,
    write_args,
)
from agents.agent_train_pipeline import build_root_node, load_model_and_data

# TODO: the "\r" and model and other inline loading must not do to log file (it too much printing in the log file) need to track so that I can see upto which step it is running but it can't go to log file.

def main() -> None:
    overall_start = time.monotonic()
    args = parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    run_dir = create_run_dir(args.trainings_dir, args.model)
    log_file, original_stdout, original_stderr = setup_logging(run_dir)

    try:
        print(f"[core_trainer] data directory: {data_dir}")
        print("[core_trainer] run_dir:", run_dir)
        print("[core_trainer] model:", args.model)
        print("[core_trainer] dataset_type:", args.dataset_type)
        print(f"[core_trainer] elapsed={time.monotonic() - overall_start:.2f}s (startup)")

        write_args(run_dir, args)
        eval_output_dir = os.path.join(run_dir, "eval_outputs")
        os.makedirs(eval_output_dir, exist_ok=True)

        _, _, rng, _, _, _, _, funcs = load_model_and_data(
            args, data_dir, eval_output_dir, args.seed
        )
        sample_feedback, run_inference, generate_feedback, mutate_prompt, evaluate = funcs 

        feedback_prompt = apply_tag_overrides(
            FEEDBACK_PROMPT_MAP[args.feedback_prompt],
            feedback_open_tag=args.feedback_open_tag,
            feedback_close_tag=args.feedback_close_tag,
            prompt_open_tag=args.prompt_open_tag,
            prompt_close_tag=args.prompt_close_tag,
        )
        if args.mutation_group_id is not None:
            mutation_prompt_keys = MUTATION_PROMPT_GROUP_MAP[args.mutation_group_id]
            mutation_prompts = [
                apply_tag_overrides(
                    resolve_mutation_prompt(key, args.inference_mode),
                    feedback_open_tag=args.feedback_open_tag,
                    feedback_close_tag=args.feedback_close_tag,
                    prompt_open_tag=args.prompt_open_tag,
                    prompt_close_tag=args.prompt_close_tag,
                )
                for key in mutation_prompt_keys
            ]
            print(
                "[core_trainer] mutation_group_id:",
                args.mutation_group_id,
                "round_robin_len:",
                len(mutation_prompts),
            )
        else:
            mutation_prompt_keys = [args.mutation_prompt]
            mutation_prompts = [
                apply_tag_overrides(
                    resolve_mutation_prompt(args.mutation_prompt, args.inference_mode),
                    feedback_open_tag=args.feedback_open_tag,
                    feedback_close_tag=args.feedback_close_tag,
                    prompt_open_tag=args.prompt_open_tag,
                    prompt_close_tag=args.prompt_close_tag,
                )
            ]
        mutation_prompt = mutation_prompts[0]

        initial_prompt_node = None
        if args.initial_prompt_source_path:
            initial_prompt_node, initial_prompt_population_path = load_initial_prompt_node(
                args.initial_prompt_source_path,
                args.initial_prompt_node_id,
                args.inference_mode,
            )
            print(
                "[core_trainer] initial prompt source:",
                initial_prompt_population_path,
            )
            print(
                "[core_trainer] initial prompt node_id:",
                initial_prompt_node.get("node_id"),
            )

        root = build_root_node(
            feedback_prompt=feedback_prompt,
            mutation_prompt=mutation_prompt,
            inference_mode=args.inference_mode,
            example_generation_prompt=EXAMPLE_GENERATION_PROMPT_V1,
            initial_prompt_node=initial_prompt_node,
        )

        search = EvolutionarySearch(
            root=root,
            max_iterations=args.max_iterations,
            population_size=args.population_size,
            feedback_sample_size=args.feedback_sample_size,
            population_sampling_temperature=args.population_sampling_temperature,
            feedback_prompt=feedback_prompt,
            mutation_prompt=mutation_prompt,
            mutation_prompts=mutation_prompts,
            mutation_prompt_keys=mutation_prompt_keys,
            example_generation_prompt=EXAMPLE_GENERATION_PROMPT_V1,
            rng=rng,
        )

        print(f"[core_trainer] elapsed={time.monotonic() - overall_start:.2f}s (before search)")

        best_node, population_history = search.run(
            sample_feedback_fn=sample_feedback,
            run_inference_fn=run_inference,
            generate_feedback_fn=generate_feedback,
            mutate_prompt_fn=mutate_prompt,
            evaluate_fn=evaluate,
            selection_mode=args.selection_mode,
            update_mode=args.update_mode,
            parent_selection_mode=args.parent_selection_mode,
            overall_start_time=overall_start,
            on_iteration_end=clear_iteration_memory,
        )

        print("[core_trainer] evaluating best on test")
        best_test_metrics = best_node.val_score #evaluate(best_node, "test")

        save_population(
            run_dir,
            population_history,
            best_node,
            final_population=search.population,
        )

        summary = {
            "run_dir": run_dir,
            "best_val_score": best_node.val_score,
            "best_test_metrics": best_test_metrics,
            "population_size": len(search.population),
            "population_history_size": len(population_history),
        }
        save_summary(run_dir, summary)

        print(f"[core_trainer] done (elapsed={time.monotonic() - overall_start:.2f}s)")
    finally:
        restore_logging(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
