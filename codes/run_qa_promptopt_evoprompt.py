"""Run first-stage EvoPrompt-DE on OpenBookQA instruction prompts."""

from __future__ import annotations

import argparse

from prompt_optimization.cli_common import add_shared_arguments, build_context
from prompt_optimization.first_stage import run_evoprompt_de


def parse_args() -> argparse.Namespace:
    """Read the full-scale QA EvoPrompt-DE configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser, require_optimizer_model=True)
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of differential-evolution iterations.",
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
        default=5,
        help="Number of prompts in the population; at least five are required.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=1000,
        help="Training questions sampled to score prompts each iteration.",
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.population_size < 5:
        parser.error("EvoPrompt needs positive iterations and population size at least 5.")
    if args.train_sample_size <= 0:
        parser.error("--train-sample-size must be positive.")
    return args


def main() -> None:
    """Build the QA context, run EvoPrompt-DE, and release model memory."""
    args = parse_args()
    context = build_context(args, "evoprompt_de")
    try:
        summary = run_evoprompt_de(context, args)
        print(f"Saved QA EvoPrompt-DE run to: {context.run_dir}")
        print(f"Validation accuracy gain: {summary['validation']['accuracy_gain']:+.4f}")
    finally:
        context.model_pool.close()


if __name__ == "__main__":
    main()
