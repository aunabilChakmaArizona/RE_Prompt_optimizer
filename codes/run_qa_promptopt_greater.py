"""Run GreaTer or top-gradient GreaTer-TG on an OpenBookQA prompt."""

from __future__ import annotations

import argparse

from prompt_optimization.cli_common import add_shared_arguments, build_context
from prompt_optimization.second_stage import run_greater


def parse_args() -> argparse.Namespace:
    """Read the QA GreaTer single-token optimization configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser, require_optimizer_model=False)
    parser.add_argument(
        "--variant",
        choices=("greater", "greater_tg"),
        required=True,
        help="Use sequential-position GreaTer or top-gradient GreaTer-TG.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=512,
        help="Label-balanced training questions used for optimization.",
    )
    parser.add_argument(
        "--gradient-batch-size",
        type=int,
        default=2,
        help="Examples per batch when accumulating token gradients.",
    )
    parser.add_argument(
        "--selection-batch-size",
        type=int,
        default=8,
        help="Examples per batch when ranking candidate prompts.",
    )
    parser.add_argument(
        "--start-position",
        type=int,
        default=0,
        help="First prompt-token position tested by sequential GreaTer.",
    )
    parser.add_argument(
        "--proposal-top-k",
        type=int,
        default=25,
        help="Top replacement tokens collected from each proposal example.",
    )
    parser.add_argument(
        "--proposal-example-size",
        type=int,
        default=50,
        help="Training examples used to collect replacement proposals.",
    )
    parser.add_argument(
        "--proposal-min-candidates",
        type=int,
        default=10,
        help="Minimum replacement candidates retained after aggregation.",
    )
    parser.add_argument(
        "--selection-top-mu",
        type=int,
        default=10,
        help="Top gradient-ranked replacements scored with the full objective.",
    )
    parser.add_argument(
        "--top-u",
        type=int,
        default=5,
        help="Top objective-ranked prompts evaluated on validation.",
    )
    parser.add_argument(
        "--fluency-lambda",
        type=float,
        default=0.2,
        help="Weight of the prompt-fluency penalty in candidate scoring.",
    )
    parser.add_argument(
        "--region-expansion-threshold",
        type=float,
        default=0.6,
        help="Relative gradient threshold used to form the one-token edit region.",
    )
    args = parser.parse_args()
    if args.backend == "vllm":
        parser.error("GreaTer requires --backend transformers to compute gradients.")
    if min(
        args.train_sample_size,
        args.gradient_batch_size,
        args.selection_batch_size,
        args.proposal_top_k,
        args.proposal_example_size,
        args.proposal_min_candidates,
        args.selection_top_mu,
        args.top_u,
    ) <= 0:
        parser.error("GreaTer sample, batch, and candidate counts must be positive.")
    if args.start_position < 0 or args.fluency_lambda < 0:
        parser.error("GreaTer position and fluency weight must be non-negative.")
    if not 0.0 <= args.region_expansion_threshold <= 1.0:
        parser.error("Region expansion threshold must be between zero and one.")
    return args


def main() -> None:
    """Build the QA context, run one GreaTer variant, and release memory."""
    args = parse_args()
    context = build_context(args, args.variant)
    try:
        summary = run_greater(context, args)
        print(f"Saved QA {args.variant} run to: {context.run_dir}")
        print(f"Validation accuracy gain: {summary['validation']['accuracy_gain']:+.4f}")
    finally:
        context.model_pool.close()


if __name__ == "__main__":
    main()
