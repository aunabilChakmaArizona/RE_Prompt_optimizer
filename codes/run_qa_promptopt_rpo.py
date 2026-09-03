"""Run first-stage RPO on reasoning or non-reasoning OpenBookQA prompts."""

from __future__ import annotations

import argparse

from prompt_optimization.cli_common import add_shared_arguments, build_context
from prompt_optimization.first_stage import run_rpo


def parse_args() -> argparse.Namespace:
    """Read the full-scale QA RPO configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser, require_optimizer_model=True)
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of RPO optimization iterations.",
    )
    parser.add_argument(
        "--snapshot-iterations",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Iterations whose best prompts are saved as stage-one outputs.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Number of prompts maintained in the RPO population.",
    )
    parser.add_argument(
        "--feedback-sample-size",
        type=int,
        default=100,
        help="Training questions sampled to evaluate feedback each iteration.",
    )
    parser.add_argument(
        "--feedback-examples",
        type=int,
        default=3,
        help=(
            "Number of mixed correct/incorrect examples that receive separate "
            "feedback each iteration."
        ),
    )
    parser.add_argument(
        "--population-sampling-temperature",
        type=float,
        default=1.0,
        help="Temperature used to sample parent prompts from the population.",
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.population_size <= 0:
        parser.error("--iterations and --population-size must be positive.")
    if args.feedback_sample_size <= 0 or args.feedback_examples <= 0:
        parser.error("RPO feedback sizes must be positive.")
    return args


def main() -> None:
    """Build the QA context, run RPO, and release model memory."""
    args = parse_args()
    context = build_context(args, "rpo")
    try:
        summary = run_rpo(context, args)
        print(f"Saved QA RPO run to: {context.run_dir}")
        print(f"Validation accuracy gain: {summary['validation']['accuracy_gain']:+.4f}")
        print(
            "Validation stable-score gain: "
            f"{summary['validation']['stable_accuracy_gain']:+.4f}"
        )
    finally:
        context.model_pool.close()


if __name__ == "__main__":
    main()
