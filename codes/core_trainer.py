from __future__ import annotations

import os
import time

from agents.agent_evolutionary_search import EvolutionarySearch
from agents.agent_prompts import (
    EXAMPLE_GENERATION_PROMPT_V1,
    FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1,
    MUTATION_PROMPT_V1,
)
from agents.agent_train_config import parse_args, resolve_data_dir
from agents.agent_train_io import (
    create_run_dir,
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
        print(f"[train] data directory: {data_dir}")
        print("[train] run_dir:", run_dir)
        print("[train] model:", args.model)
        print("[train] dataset_type:", args.dataset_type)
        print(f"[train] elapsed={time.monotonic() - overall_start:.2f}s (startup)")

        write_args(run_dir, args)
        eval_output_dir = os.path.join(run_dir, "eval_outputs")
        os.makedirs(eval_output_dir, exist_ok=True)

        _, _, rng, _, _, _, _, funcs = load_model_and_data(
            args, data_dir, eval_output_dir, args.seed
        )
        sample_feedback, run_inference, generate_feedback, mutate_prompt, evaluate = funcs 

        root = build_root_node()

        # search = EvolutionarySearch(
        #     root=root,
        #     max_iterations=args.max_iterations,
        #     feedback_sample_size=args.feedback_sample_size,
        #     temperature=args.temperature,
        #     feedback_prompt=FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1,
        #     mutation_prompt=MUTATION_PROMPT_V1,
        #     example_generation_prompt=EXAMPLE_GENERATION_PROMPT_V1,
        #     dataset_type=args.dataset_type,
        #     rng=rng,
        # )

        # print(f"[train] elapsed={time.monotonic() - overall_start:.2f}s (before search)")
        # best_node, population = search.run(
        #     sample_feedback_fn=sample_feedback,
        #     run_inference_fn=run_inference,
        #     generate_feedback_fn=generate_feedback,
        #     mutate_prompt_fn=mutate_prompt,
        #     evaluate_fn=evaluate,
        #     selection_mode=args.selection_mode,
        #     overall_start_time=overall_start,
        # )

        # print("[train] evaluating best on test")
        # best_test_metrics = evaluate(best_node, "test")

        # save_population(run_dir, population, best_node)

        # summary = {
        #     "run_dir": run_dir,
        #     "best_val_score": best_node.val_score,
        #     "best_test_metrics": best_test_metrics,
        #     "population_size": len(population),
        # }
        # save_summary(run_dir, summary)

        print(f"[train] done (elapsed={time.monotonic() - overall_start:.2f}s)")
    finally:
        restore_logging(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
