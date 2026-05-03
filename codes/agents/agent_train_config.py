from __future__ import annotations

import argparse
import os
from typing import Optional

from agents.agent_prompts import (
    INFERENCE_MODE_CHOICES,
    INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    MUTATION_PROMPT_GROUP_MAP,
)
from agents.agent_utils import DEFAULT_F1_STABILITY_STD_MULTIPLIER
from agents.agent_evolutionary_search import PARENT_SELECTION_MODE_CHOICES
from agents.agent_cluster_search import CLUSTER_SELECTION_MODE_CHOICES


def default_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data"))


def default_trainings_dir() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "trainings")
    )

def resolve_data_dir(data_dir: Optional[str]) -> str:
    return data_dir or default_data_dir()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evolutionary prompt optimization.")
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--dataset-type", default="fs_tacred", help="Dataset type")
    parser.add_argument("--data-dir", default=None, help="Path to data dir")
    parser.add_argument(
        "--train-samples",
        default="fs_tacred_train_samples.pkl",
        help="Train samples pkl filename",
    )
    parser.add_argument("--dev-split", default="dev", help="Dev split name")
    parser.add_argument("--test-split", default="test", help="Test split name")
    parser.add_argument("--dev-ep-start", type=int, default=0)
    parser.add_argument("--dev-ep-end", type=int, default=None)
    parser.add_argument("--test-ep-start", type=int, default=0)
    parser.add_argument("--test-ep-end", type=int, default=None)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help=(
            "Maximum number of active prompts kept in the evolutionary population. "
            "If omitted, the population grows without a limit."
        ),
    )
    parser.add_argument("--feedback-sample-size", type=int, default=100)
    parser.add_argument("--num-shots", type=int, default=1, help="Number of examples in prompt")
    parser.add_argument(
        "--initial-prompt-source-path",
        default=None,
        help=(
            "Optional previous run directory or population.json path used to seed the "
            "initial prompt before evolutionary search starts."
        ),
    )
    parser.add_argument(
        "--initial-prompt-node-id",
        type=int,
        default=None,
        help=(
            "Optional node_id inside the source population.json. If omitted, the "
            "best node from that run is used."
        ),
    )
    parser.add_argument(
        "--load-population",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load the full final survivor population from --initial-prompt-source-path "
            "instead of seeding the search with a single prompt."
        ),
    )
    parser.add_argument(
        "--update-mode",
        type=str,
        default="feedback",
        choices=["feedback", "no_feedback"],
        help="Prompt update mode",
    )
    parser.add_argument("--selection-mode", type=str, default="mixed")
    parser.add_argument(
        "--parent-selection-mode",
        type=str,
        default="score_weighted",
        choices=list(PARENT_SELECTION_MODE_CHOICES),
        help=(
            "Parent selection policy for mutation: score_weighted samples from the "
            "active population using validation scores; initial_prompt_only always "
            "mutates the initial/root prompt."
        ),
    )
    parser.add_argument("--population-sampling-temperature", type=float, default=1.0)
    parser.add_argument(
        "--validation-f1-std-penalty",
        type=float,
        default=DEFAULT_F1_STABILITY_STD_MULTIPLIER,
        help=(
            "Rank validation prompts by f1_mean minus this multiplier times f1_std "
            "for parent sampling, population pruning, and best-node selection. "
            "Defaults to 2.5."
        ),
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable sampling during LLM text generation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feedback-prompt",
        default="correct_and_mistakes_v1",
        choices=["correct_and_mistakes_v1", "correct_v1", "mistakes_v1"],
        help="Feedback inference prompt variant",
    )
    parser.add_argument(
        "--mutation-prompt",
        default="v1",
        choices=[
            "v1",
            "random_v1",
            "random_v2",
            "no_feedback_v1",
            "traces_v1",
            "traces_differences_v1",
        ],
        help="Mutation prompt variant",
    )
    parser.add_argument(
        "--mutation-group-id",
        type=str,
        default=None,
        choices=list(MUTATION_PROMPT_GROUP_MAP.keys()),
        help=(
            "Optional mutation prompt group. If set, this overrides --mutation-prompt "
            "and applies group prompts in round-robin per iteration."
        ),
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        default=INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
        choices=list(INFERENCE_MODE_CHOICES),
        help=(
            "Inference prompt mode: non_separate uses INFERENCE_PROMPT_V1; "
            "separate_no_examples uses instruction+input; "
            "separate_with_examples uses instruction+example+input."
        ),
    )
    parser.add_argument(
        "--feedback-open-tag",
        default="<f>",
        help="Opening tag to wrap feedback output (default: <f>)",
    )
    parser.add_argument(
        "--feedback-close-tag",
        default="</f>",
        help="Closing tag to wrap feedback output (default: </f>)",
    )
    parser.add_argument(
        "--prompt-open-tag",
        default="<p>",
        help="Opening tag to wrap mutated prompt output (default: <p>)",
    )
    parser.add_argument(
        "--prompt-close-tag",
        default="</p>",
        help="Closing tag to wrap mutated prompt output (default: </p>)",
    )
    parser.add_argument(
        "--device-map",
        default="cuda:0",
        help="Device map override (e.g., cuda:0, cuda:1, cpu, auto)",
    )
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--feedback-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-n-chunks", type=int, default=3)
    parser.add_argument(
        "--feedback-max-new-tokens",
        type=int,
        default=10000,
        help="Max new tokens for feedback generation prompts",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10000,
        help="Max new tokens for non-feedback prompting (e.g., mutation)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log/progress update frequency in batches",
    )
    parser.add_argument(
        "--trainings-dir",
        default=default_trainings_dir(),
        help="Output root for training runs",
    )
    parser.add_argument(
        "--cluster-feedback-sample-size",
        type=int,
        default=500,
        help="Number of feedback samples used to build feedback categories.",
    )
    parser.add_argument(
        "--cluster-category-min-count",
        type=int,
        default=2,
        help="Minimum category count required to keep a category.",
    )
    parser.add_argument(
        "--cluster-max-categories",
        type=int,
        default=10,
        help="Maximum number of feedback categories to keep after sorting by count.",
    )
    parser.add_argument(
        "--cluster-num-clusters",
        type=int,
        default=5,
        help="Number of overlapping feedback clusters to build.",
    )
    parser.add_argument(
        "--cluster-candidates-per-cluster",
        type=int,
        default=5,
        help="Number of prompt candidates to sample per cluster.",
    )
    parser.add_argument(
        "--cluster-coverage-ratio",
        type=float,
        default=0.5,
        help="Fraction of kept categories covered by each cluster.",
    )
    parser.add_argument(
        "--cluster-selection-mode",
        type=str,
        default="usage_decay",
        choices=list(CLUSTER_SELECTION_MODE_CHOICES),
        help=(
            "Cluster category sampling mode: "
            "'usage_decay' spreads coverage by lowering reuse probability after selection; "
            "'error_count_weighted' samples categories with probability proportional to error count."
        ),
    )
    parser.add_argument(
        "--category-feedback-examples-per-category",
        type=int,
        default=3,
        help="Number of representative feedback texts to include per category in mutation prompts.",
    )
    parser.add_argument(
        "--taxonomy-num-candidates",
        type=int,
        default=25,
        help="Number of candidate prompts to generate from the full taxonomy.",
    )
    parser.add_argument(
        "--taxonomy-top-k",
        type=int,
        default=5,
        help="Number of top-performing prompts to keep from taxonomy search.",
    )
    return parser


def get_arg_defaults() -> dict:
    parser = build_parser()
    defaults = {}
    for action in parser._actions:
        if action.dest in ("help",) or action.default is argparse.SUPPRESS:
            continue
        defaults[action.dest] = action.default
    return defaults


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
