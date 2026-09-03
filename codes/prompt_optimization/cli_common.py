"""Shared command-line configuration and runtime context for QA optimizers."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from prompt_optimization.evaluation import QAEvaluator
from prompt_optimization.models import ModelPool, seed_everything
from prompt_optimization.qa_task import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALIDATION_PATH,
    QAMode,
    load_qa_records,
    resolve_mode,
)
from prompt_optimization.run_io import (
    DEFAULT_OUTPUT_ROOT,
    RunLogger,
    create_run_directory,
    load_initial_prompt,
    save_json,
    save_text,
)


@dataclass
class QAOptimizationContext:
    """Bundle shared data, models, evaluation, and output state for one run."""

    args: argparse.Namespace
    optimizer_name: str
    mode: QAMode
    train_records: list[dict[str, Any]]
    validation_records: list[dict[str, Any]]
    initial_prompt: str
    run_dir: Path
    logger: RunLogger
    model_pool: ModelPool
    evaluator: QAEvaluator
    rng: random.Random
    started_at: float


def add_shared_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_optimizer_model: bool,
) -> None:
    """Add dataset, model, prompt, decoding, and output arguments."""
    parser.add_argument("--code", required=True, help="Unique identity for this run.")
    parser.add_argument(
        "--qa-mode",
        choices=("reasoning", "non_reasoning"),
        required=True,
        help="Whether the target model reasons before returning the option label.",
    )
    parser.add_argument("--model", required=True, help="Target Qwen3 or Gemma3 model.")
    parser.add_argument(
        "--optimizer-model",
        required=require_optimizer_model,
        default=None,
        help="Larger model used for reasoning-based prompt proposals.",
    )
    parser.add_argument("--device", default="cuda:0", help="Target model device map.")
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="Generation backend; vLLM uses continuous batching.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of selected GPU memory reserved by vLLM.",
    )
    parser.add_argument(
        "--optimizer-device",
        default=None,
        help="Optimizer model device map; defaults to --device.",
    )
    parser.add_argument(
        "--keep-models-loaded",
        action="store_true",
        help="Keep target and optimizer models resident when memory or separate GPUs allow.",
    )
    parser.add_argument(
        "--train-path",
        default=str(DEFAULT_TRAIN_PATH),
        help="Training JSONL used to build feedback or gradients.",
    )
    parser.add_argument(
        "--validation-path",
        default=str(DEFAULT_VALIDATION_PATH),
        help="The 900-example JSONL with three folds used to select prompts.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Starting instruction text; otherwise use the mode's default prompt.",
    )
    parser.add_argument(
        "--initial-prompt-file",
        default=None,
        help="File containing the starting instruction instead of inline text.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling, generation, and evaluation.",
    )
    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=4,
        help="Number of QA examples evaluated together by the target model.",
    )
    parser.add_argument(
        "--target-max-new-tokens",
        type=int,
        default=None,
        help="Defaults to 4096 for reasoning and 16 for non-reasoning.",
    )
    parser.add_argument(
        "--optimizer-batch-size",
        type=int,
        default=1,
        help="Number of prompt-generation requests processed together.",
    )
    parser.add_argument(
        "--optimizer-max-new-tokens",
        type=int,
        default=10000,
        help="Maximum tokens generated for each optimizer-model response.",
    )
    parser.add_argument(
        "--validation-std-penalty",
        type=float,
        default=2.0,
        help="Lambda in validation mean accuracy minus lambda times fold std.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where run artifacts are saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory with the same run identity.",
    )


def build_context(
    args: argparse.Namespace,
    optimizer_name: str,
) -> QAOptimizationContext:
    """Load one QA experiment and initialize its shared runtime services."""
    started_at = time.monotonic()
    mode = resolve_mode(args.qa_mode)
    max_new_tokens = args.target_max_new_tokens or mode.default_max_new_tokens
    if max_new_tokens <= 0:
        raise ValueError("--target-max-new-tokens must be positive.")
    if args.target_batch_size <= 0 or args.optimizer_batch_size <= 0:
        raise ValueError("Target and optimizer batch sizes must be positive.")
    if args.optimizer_max_new_tokens <= 0:
        raise ValueError("--optimizer-max-new-tokens must be positive.")
    if args.validation_std_penalty < 0:
        raise ValueError("--validation-std-penalty must be non-negative.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be greater than 0 and at most 1.")
    if (
        args.backend == "vllm"
        and args.optimizer_model
        and args.optimizer_device
        and args.optimizer_device != args.device
    ):
        raise ValueError(
            "Offline vLLM uses one visible device per process; --optimizer-device "
            "must match --device."
        )
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    train_records = load_qa_records(args.train_path)
    validation_records = load_qa_records(args.validation_path)
    initial_prompt = load_initial_prompt(
        mode,
        args.initial_prompt,
        args.initial_prompt_file,
    )
    run_dir = create_run_directory(
        args.output_root,
        optimizer_name,
        args.qa_mode,
        args.code,
        args.overwrite,
    )
    logger = RunLogger(run_dir)
    model_pool = ModelPool(
        target_model_id=args.model,
        optimizer_model_id=args.optimizer_model,
        target_device=args.device,
        optimizer_device=args.optimizer_device,
        keep_models_loaded=args.keep_models_loaded,
        seed=args.seed,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    evaluator = QAEvaluator(
        model_pool=model_pool,
        mode=mode,
        batch_size=args.target_batch_size,
        max_new_tokens=max_new_tokens,
        seed=args.seed,
        validation_std_penalty=args.validation_std_penalty,
        run_started_at=started_at,
    )
    context = QAOptimizationContext(
        args=args,
        optimizer_name=optimizer_name,
        mode=mode,
        train_records=train_records,
        validation_records=validation_records,
        initial_prompt=initial_prompt,
        run_dir=run_dir,
        logger=logger,
        model_pool=model_pool,
        evaluator=evaluator,
        rng=rng,
        started_at=started_at,
    )
    save_text(run_dir / "initial_prompt.txt", initial_prompt)
    save_json(
        run_dir / "config.json",
        {
            **vars(args),
            "optimizer_name": optimizer_name,
            "qa_mode_config": asdict(mode),
            "resolved_target_max_new_tokens": max_new_tokens,
            "dataset_sizes": {
                "train": len(train_records),
                "validation": len(validation_records),
            },
        },
    )
    logger.event(
        "run_started",
        optimizer=optimizer_name,
        qa_mode=args.qa_mode,
        backend=args.backend,
        initial_prompt=initial_prompt,
    )
    return context
