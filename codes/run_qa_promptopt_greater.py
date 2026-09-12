"""Run GreaTer or top-gradient GreaTer-TG on a QA or Math prompt."""

from __future__ import annotations

import argparse

from prompt_optimization.cli_common import (
    add_gradient_runtime_arguments,
    add_shared_arguments,
    build_context,
)
from prompt_optimization.second_stage import run_greater


def parse_args() -> argparse.Namespace:
    """Read the QA GreaTer single-token optimization configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser, require_optimizer_model=False)
    add_gradient_runtime_arguments(parser)
    parser.add_argument(
        "--variant",
        choices=("greater", "greater_tg"),
        required=True,
        help="Use sequential-position GreaTer or top-gradient GreaTer-TG.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=3000, #aunabil2nd: for reasoning we shuold make this smaller believe, it will take a lot of time for this large set to generate reasoning; please look for every 1st or 2nd stage methods on reasoning modes for this kind of train size and see if its is too much or not
        help=(
            "Initial random training pool; gradients use an equal number of its "
            "correct and incorrect results."
        ),
    )
    parser.add_argument(
        "--gradient-batch-size",
        type=int,
        default=4,
        help="Examples per batch when accumulating token gradients.",
    )
    parser.add_argument(
        "--gradient-sample-size",
        type=int,
        default=200,
        help="Final even-sized subset split equally between correct and incorrect results.",
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
        "--check-isalnum",
        action="store_true",
        help=(
            "Require selected and proposed tokens to contain a letter or number; "
            "disabled by default to match original GreaTer."
        ),
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
        parser.error("GreaTer requires --backend transformers or dual for gradients.")
    if args.backend == "dual" and args.final_evaluation_backend != "vllm":
        parser.error("Dual mode uses its resident vLLM engine for final evaluation.")
    if min(
        args.train_sample_size,
        args.gradient_sample_size,
        args.gradient_batch_size,
        args.selection_batch_size,
        args.proposal_top_k,
        args.proposal_example_size,
        args.proposal_min_candidates,
        args.selection_top_mu,
        args.top_u,
    ) <= 0:
        parser.error("GreaTer sample, batch, and candidate counts must be positive.")
    if args.gradient_sample_size % 2 != 0:
        parser.error("--gradient-sample-size must be even.")
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
        print(
            "Validation stable-score gain: "
            f"{summary['validation']['stable_accuracy_gain']:+.4f}"
        )
    finally:
        context.model_pool.close()


if __name__ == "__main__":
    main()
