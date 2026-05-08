from __future__ import annotations

import os
import time

from agents.agent_prompts import (
    EXAMPLE_GENERATION_PROMPT_V1,
    FEEDBACK_PROMPT_MAP,
    apply_tag_overrides,
)
from agents.agent_taxonomy_search import TaxonomySearch
from agents.agent_train_config import parse_args, resolve_data_dir
from agents.agent_train_io import (
    create_run_dir,
    restore_logging,
    save_json,
    save_summary,
    setup_logging,
    write_args,
)
from agents.agent_train_pipeline import build_root_node, load_model_and_data


def main() -> None:
    overall_start = time.monotonic()
    args = parse_args()
    if args.optimizer_model and args.optimizer_model != args.model:
        raise ValueError(
            "--optimizer-model is currently supported by "
            "core_trainer_evolutionary_search.py, not core_trainer_taxonomy.py"
        )

    data_dir = resolve_data_dir(args.data_dir)
    run_dir = create_run_dir(args.trainings_dir, f"{args.model}_taxonomy")
    log_file, original_stdout, original_stderr = setup_logging(run_dir)

    try:
        print(f"[core_trainer_taxonomy] data directory: {data_dir}")
        print("[core_trainer_taxonomy] run_dir:", run_dir)
        print("[core_trainer_taxonomy] model:", args.model)
        print("[core_trainer_taxonomy] dataset_type:", args.dataset_type)
        print(
            f"[core_trainer_taxonomy] elapsed={time.monotonic() - overall_start:.2f}s (startup)"
        )

        write_args(run_dir, args)
        eval_output_dir = os.path.join(run_dir, "eval_outputs")
        os.makedirs(eval_output_dir, exist_ok=True)

        model, tokenizer, _, _, _, _, _, funcs = load_model_and_data(
            args, data_dir, eval_output_dir, args.seed
        )
        sample_feedback, run_inference, generate_feedback, _, evaluate = funcs

        feedback_prompt = apply_tag_overrides(
            FEEDBACK_PROMPT_MAP[args.feedback_prompt],
            feedback_open_tag=args.feedback_open_tag,
            feedback_close_tag=args.feedback_close_tag,
            prompt_open_tag=args.prompt_open_tag,
            prompt_close_tag=args.prompt_close_tag,
        )

        root = build_root_node(
            feedback_prompt=feedback_prompt,
            mutation_prompt="taxonomy_search_v1",
            inference_mode=args.inference_mode,
            example_generation_prompt=EXAMPLE_GENERATION_PROMPT_V1,
        )

        search = TaxonomySearch(
            root=root,
            model=model,
            tokenizer=tokenizer,
            feedback_sample_size=args.cluster_feedback_sample_size,
            category_min_count=args.cluster_category_min_count,
            max_categories=args.cluster_max_categories,
            num_candidates=args.taxonomy_num_candidates,
            top_k=args.taxonomy_top_k,
            feedback_examples_per_category=args.category_feedback_examples_per_category,
            prompt_open_tag=args.prompt_open_tag,
            prompt_close_tag=args.prompt_close_tag,
            feedback_batch_size=args.feedback_batch_size,
            feedback_max_new_tokens=args.feedback_max_new_tokens,
            mutation_max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )

        print(
            f"[core_trainer_taxonomy] elapsed={time.monotonic() - overall_start:.2f}s "
            "(before taxonomy search)"
        )

        results = search.run(
            sample_feedback_fn=sample_feedback,
            run_inference_fn=run_inference,
            generate_feedback_fn=generate_feedback,
            evaluate_fn=evaluate,
        )

        summary = {
            "run_dir": run_dir,
            "root_val_score": results["root_node"].val_score,
            "sampled_feedback_count": results.get("sampled_feedback_count", 0),
            "mistake_count": results.get("mistake_count", 0),
            "kept_categories": len(results.get("categories", [])),
            "candidate_prompts": len(results.get("all_candidates", [])),
            "top_prompts": len(results.get("top_prompts", [])),
        }
        save_summary(run_dir, summary)

        save_json(
            run_dir,
            "taxonomy_search_results.json",
            {
                "summary": summary,
                "root_node_id": results["root_node"].node_id,
                "root_val_score": results["root_node"].val_score,
                "categories": results.get("categories", []),
                "candidate_results": results.get("candidate_results", []),
                "top_prompts": results.get("top_prompts", []),
            },
        )

        print(
            f"[core_trainer_taxonomy] done (elapsed={time.monotonic() - overall_start:.2f}s)"
        )
    finally:
        restore_logging(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
