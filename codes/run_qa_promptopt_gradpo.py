"""Run GradPO-Gen, GradPO-Prob, or the random-region baseline on QA."""

from __future__ import annotations

import argparse

from prompt_optimization.cli_common import add_shared_arguments, build_context
from prompt_optimization.second_stage import run_gradpo


def parse_args() -> argparse.Namespace:
    """Read the full-scale QA GradPO configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser, require_optimizer_model=False)
    parser.add_argument(
        "--variant",
        choices=("gen", "prob", "gen_random"),
        required=True,
        help="Candidate source: generated, probability-based, or random-span generated.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=3000,
        help="Label-balanced training questions used for optimization.",
    )
    parser.add_argument(
        "--gradient-batch-size",
        type=int,
        default=2,
        help="Examples per batch when accumulating prompt-token gradients.",
    )
    parser.add_argument(
        "--selection-batch-size",
        type=int,
        default=4,
        help="Examples per batch when scoring candidate prompts.",
    )
    parser.add_argument(
        "--num-edit-regions",
        type=int,
        default=None,
        help="Defaults to 5 for Qwen and 3 for Gemma.",
    )
    parser.add_argument(
        "--max-region-tokens",
        type=int,
        default=None,
        help="Defaults to 2 for Qwen and 3 for Gemma.",
    )
    parser.add_argument(
        "--region-expansion-threshold",
        type=float,
        default=0.6,
        help="Relative gradient threshold for expanding around a selected token.",
    )
    parser.add_argument(
        "--num-region-candidates",
        type=int,
        default=5,
        help="Replacement candidates generated for each selected region.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Maximum partial prompts retained after each edited region.",
    )
    parser.add_argument(
        "--beam-replacement-mode",
        choices=("llm_synthesis", "direct"),
        default="llm_synthesis",
        help="Combine replacements with model synthesis or direct substitution.",
    )
    parser.add_argument(
        "--fluency-lambda",
        type=float,
        default=0.5,
        help="Weight of the prompt-fluency penalty in beam scoring.",
    )
    parser.add_argument(
        "--candidate-max-new-tokens",
        type=int,
        default=10000,
        help="Maximum output tokens for each region-candidate request.",
    )
    parser.add_argument(
        "--synthesis-max-new-tokens",
        type=int,
        default=10000,
        help="Maximum output tokens for each beam-synthesis request.",
    )
    parser.add_argument(
        "--synthesis-batch-size",
        type=int,
        default=4,
        help="Number of probability continuations or beam syntheses processed together.",
    )
    args = parser.parse_args()
    if args.backend == "vllm":
        parser.error("GradPO requires --backend transformers to compute gradients.")
    positive_values = (
        args.train_sample_size,
        args.gradient_batch_size,
        args.selection_batch_size,
        args.num_region_candidates,
        args.beam_width,
        args.candidate_max_new_tokens,
        args.synthesis_max_new_tokens,
        args.synthesis_batch_size,
    )
    if any(value <= 0 for value in positive_values):
        parser.error("GradPO sample, batch, beam, and candidate counts must be positive.")
    if args.num_edit_regions is not None and args.num_edit_regions <= 0:
        parser.error("--num-edit-regions must be positive when provided.")
    if args.max_region_tokens is not None and args.max_region_tokens <= 0:
        parser.error("--max-region-tokens must be positive when provided.")
    if args.fluency_lambda < 0:
        parser.error("--fluency-lambda must be non-negative.")
    if not 0.0 <= args.region_expansion_threshold <= 1.0:
        parser.error("Region expansion threshold must be between zero and one.")
    return args


def main() -> None:
    """Build the QA context, run one GradPO variant, and release memory."""
    args = parse_args()
    optimizer_name = f"gradpo_{args.variant}"
    context = build_context(args, optimizer_name)
    try:
        summary = run_gradpo(context, args)
        print(f"Saved QA {optimizer_name} run to: {context.run_dir}")
        print(f"Validation accuracy gain: {summary['validation']['accuracy_gain']:+.4f}")
        print(
            "Validation stable-score gain: "
            f"{summary['validation']['stable_accuracy_gain']:+.4f}"
        )
    finally:
        context.model_pool.close()


if __name__ == "__main__":
    main()
