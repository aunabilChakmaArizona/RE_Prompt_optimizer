"""Run one first-stage ETGPO taxonomy-guided OpenBookQA refinement."""

from __future__ import annotations

import argparse

from prompt_optimization.cli_common import add_shared_arguments, build_context
from prompt_optimization.first_stage import run_etgpo


def parse_args() -> argparse.Namespace:
    """Read the full-scale QA ETGPO configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser, require_optimizer_model=True)
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=1000,
        help="Training questions sampled for error analysis.",
    )
    parser.add_argument(
        "--error-batch-size",
        type=int,
        default=6,
        help="Errors shown together when building the error taxonomy.",
    )
    parser.add_argument(
        "--error-coverage",
        type=float,
        default=0.7,
        help="Fraction of observed errors the selected categories should cover.",
    )
    parser.add_argument(
        "--min-categories",
        type=int,
        default=2,
        help="Minimum number of error categories used for refinement.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=5,
        help="Maximum number of error categories used for refinement.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Number of refined prompt candidates generated from the taxonomy.",
    )
    args = parser.parse_args()
    if args.train_sample_size <= 0 or args.error_batch_size <= 0:
        parser.error("ETGPO sample and batch sizes must be positive.")
    if not 0.0 < args.error_coverage <= 1.0:
        parser.error("--error-coverage must be greater than zero and at most one.")
    if not 1 <= args.min_categories <= args.max_categories:
        parser.error("ETGPO category limits are invalid.")
    if args.num_candidates <= 0:
        parser.error("--num-candidates must be positive.")
    return args


def main() -> None:
    """Build the QA context, run ETGPO, and release model memory."""
    args = parse_args()
    context = build_context(args, "etgpo")
    try:
        summary = run_etgpo(context, args)
        print(f"Saved QA ETGPO run to: {context.run_dir}")
        print(f"Validation accuracy gain: {summary['validation']['accuracy_gain']:+.4f}")
    finally:
        context.model_pool.close()


if __name__ == "__main__":
    main()
