from __future__ import annotations

import argparse
import os
from typing import Optional


def default_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data"))


def default_trainings_dir() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "trainings")
    )

def resolve_data_dir(data_dir: Optional[str]) -> str:
    return data_dir or default_data_dir()

def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--feedback-sample-size", type=int, default=100)
    parser.add_argument("--num-shots", type=int, default=1, help="Number of examples in prompt")
    parser.add_argument("--selection-mode", default="mixed")
    parser.add_argument("--population-sampling-temperature", type=float, default=1.0)
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
        choices=["v1"],
        help="Mutation prompt variant",
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
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    parser.add_argument(
        "--trainings-dir",
        default=default_trainings_dir(),
        help="Output root for training runs",
    )
    return parser.parse_args()
