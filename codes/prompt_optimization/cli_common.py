"""Shared command-line configuration and runtime context for QA optimizers."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from prompt_optimization.evaluation import QAEvaluator
from prompt_optimization.gradient_cache import (
    DEFAULT_GRADIENT_CACHE_ROOT,
    resolve_gradient_cache_root,
)
from prompt_optimization.models import ModelPool, seed_everything
from prompt_optimization.qa_task import (
    ANLI_TRAIN_PATH,
    ANLI_VALIDATION_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALIDATION_PATH,
    HOTPOTQA_TRAIN_PATH,
    HOTPOTQA_VALIDATION_PATH,
    MATH500_TRAIN_PATH,
    MATH500_VALIDATION_PATH,
    QAMode,
    load_qa_records,
    resolve_mode,
    select_validation_fold_subset,
)
from prompt_optimization.run_io import (
    DEFAULT_OUTPUT_ROOT,
    RunLogger,
    create_run_directory,
    load_initial_prompt,
    save_json,
    save_text,
)
from prompt_optimization.source_validation_cache import (
    DEFAULT_SOURCE_VALIDATION_CACHE_ROOT,
    resolve_source_validation_cache_root,
)


DEFAULT_MATH_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT.parent / "math_prompt_optimization"
DEFAULT_DUAL_VLLM_MAX_MODEL_LEN = 16_384


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
        "--qa-task",
        choices=("openbookqa", "anli", "hotpotqa", "math500"),
        default="openbookqa",
        help="Task; OpenBookQA remains the backward-compatible default.",
    )
    parser.add_argument(
        "--qa-mode",
        choices=("reasoning", "non_reasoning"),
        required=True,
        help="Whether the target model reasons before returning its answer.",
    )
    parser.add_argument("--model", required=True, help="Target Qwen3 or Gemma3 model.")
    parser.add_argument(
        "--optimizer-model",
        required=require_optimizer_model,
        default=None,
        help="Larger model used for reasoning-based prompt proposals.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Target device; in dual mode this selects the vLLM generation GPU.",
    )
    parser.add_argument(
        "--hf-device",
        default=None,
        help=(
            "HF target-model device for gradients and scoring; defaults to --device. "
            "With two visible GPUs, use cuda:1 to separate HF from vLLM cuda:0."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm", "dual"),
        default="transformers",
        help=(
            "Generation backend; dual keeps an HF model for gradients and a vLLM "
            "copy for generation and batched prediction."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of selected GPU memory reserved by vLLM.",
    )
    parser.add_argument(
        "--dual-vllm-gpu-memory-utilization",
        type=float,
        default=0.50,
        help="vLLM memory fraction in dual HF+vLLM mode; start at 0.50.",
    )
    parser.add_argument(
        "--vllm-conservative-settings",
        action="store_true",
        help=(
            "Opt into eager execution, 4,096 batched tokens, 128 sequences, "
            "and disabled prefix caching; otherwise retain vLLM defaults."
        ),
    )
    parser.add_argument(
        "--dual-vllm-max-num-batched-tokens",
        type=int,
        default=4096,
        help="Token limit used only with --vllm-conservative-settings.",
    )
    parser.add_argument(
        "--dual-vllm-max-num-seqs",
        type=int,
        default=128,
        help="Sequence limit used only with --vllm-conservative-settings.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Maximum vLLM context length; defaults to the model configuration.",
    )
    parser.add_argument(
        "--vllm-disable-images",
        action="store_true",
        help="Disable image inputs and image-cache profiling for text-only inference.",
    )
    parser.add_argument(
        "--optimizer-device",
        default=None,
        help="Optimizer model device map; defaults to --device.",
    )
    parser.add_argument(
        "--keep-models-loaded", #aunabil3rd: what this is actually doing?
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
        help="The 1,500-example JSONL with three folds used to select prompts.",
    )
    parser.add_argument(
        "--validation-fold-size",
        type=int,
        default=None,
        help=(
            "Use only the first N records from each of the three existing "
            "validation folds; by default use every record."
        ),
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
        help="Defaults to 8192 for math, 4096 for other reasoning, and 10 otherwise.", #aunabil: should we keep max reasoning tokens to 4096 , because that can happen is that most of the time the LLM is hallucinating, as a results a PO can be instruct no too think too or overthink. This creates more space to improve
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
        "--optimizer-feedback-max-tokens",
        type=int,
        default=2000, 
        help="Maximum tokens retained from one math reasoning trace in feedback.", #aunabil: if the reasoning itself is longer by the target model, will this 2000tokens enough for optimizer?
    )
    parser.add_argument(
        "--validation-std-penalty",
        type=float,
        default=1.0,
        help="Lambda in validation mean accuracy minus lambda times fold std.",
    )
    parser.add_argument(
        "--source-validation-cache-root",
        default=str(DEFAULT_SOURCE_VALIDATION_CACHE_ROOT),
        help="Shared JSON cache for repeated source-prompt validation results.",
    )
    parser.add_argument(
        "--refresh-source-validation-cache",
        action="store_true",
        help="Regenerate and replace a matching source validation result.",
    )
    parser.add_argument(
        "--disable-source-validation-cache",
        action="store_true",
        help="Evaluate source prompts without reading or writing the shared cache.",
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


def add_gradient_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add objective scoring, caches, and validation routing for gradient refiners."""
    parser.add_argument(
        "--objective-scoring-backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="Candidate loss/fluency backend; vllm requires dual mode and vLLM 0.11.0.",
    )
    parser.add_argument(
        "--objective-scoring-batch-size",
        type=int,
        default=128,
        help=(
            "Maximum sequences submitted per vLLM scoring call (not GPU batch size); "
            "HF scoring still uses --selection-batch-size."
        ),
    )
    parser.add_argument(
        "--final-evaluation-backend",
        choices=("vllm", "transformers"),
        default="vllm",
        help=(
            "Backend for source/candidate validation; regular mode unloads HF and "
            "loads vLLM, while dual mode reuses its resident vLLM engine."
        ),
    )
    add_training_cache_arguments(parser)


def add_training_cache_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared greedy training-response cache options for every refiner."""
    parser.add_argument(
        "--gradient-cache-root",
        default=str(DEFAULT_GRADIENT_CACHE_ROOT),
        help="Shared JSON training-response cache for LPO and gradient refiners.",
    )
    parser.add_argument(
        "--refresh-gradient-cache",
        action="store_true",
        help="Regenerate the requested training pool rather than reuse a cache.",
    )
    parser.add_argument(
        "--disable-gradient-cache",
        action="store_true",
        help="Run training-pool inference without reading or writing the shared cache.",
    )


def build_context(
    args: argparse.Namespace,
    optimizer_name: str,
) -> QAOptimizationContext:
    """Load one QA experiment and initialize its shared runtime services."""
    started_at = time.monotonic()
    objective_backend = getattr(args, "objective_scoring_backend", "transformers")
    if getattr(args, "objective_scoring_batch_size", 128) <= 0:
        raise ValueError("--objective-scoring-batch-size must be positive.")
    if objective_backend == "vllm":
        if args.backend != "dual":
            raise ValueError("--objective-scoring-backend vllm requires --backend dual.")
        from prompt_optimization.vllm_scoring import validate_vllm_scoring_version

        validate_vllm_scoring_version()
    dual_optimizers = {
        "greater",
        "greater_tg",
        "gradpo_gen",
        "gradpo_prob",
        "gradpo_gen_random",
    }
    if args.backend == "dual" and optimizer_name not in dual_optimizers:
        raise ValueError(
            "--backend dual is only for GreaTer and GradPO gradient runs; "
            "use --backend vllm for reasoning-based optimizers and LPO."
        )
    mode = resolve_mode(args.qa_mode, args.qa_task)
    max_new_tokens = args.target_max_new_tokens or mode.default_max_new_tokens
    if max_new_tokens <= 0:
        raise ValueError("--target-max-new-tokens must be positive.")
    if args.target_batch_size <= 0 or args.optimizer_batch_size <= 0:
        raise ValueError("Target and optimizer batch sizes must be positive.")
    if args.optimizer_max_new_tokens <= 0 or args.optimizer_feedback_max_tokens <= 0:
        raise ValueError("Optimizer token limits must be positive.")
    if args.validation_std_penalty < 0:
        raise ValueError("--validation-std-penalty must be non-negative.")
    if args.validation_fold_size is not None and args.validation_fold_size <= 0:
        raise ValueError("--validation-fold-size must be positive when provided.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be greater than 0 and at most 1.")
    if not 0.0 < args.dual_vllm_gpu_memory_utilization <= 1.0:
        raise ValueError(
            "--dual-vllm-gpu-memory-utilization must be greater than 0 and at most 1."
        )
    if min(
        args.dual_vllm_max_num_batched_tokens,
        args.dual_vllm_max_num_seqs,
    ) <= 0:
        raise ValueError("Dual vLLM scheduler limits must be positive.")
    if args.vllm_max_model_len is not None and args.vllm_max_model_len <= 0:
        raise ValueError("--vllm-max-model-len must be positive when provided.")
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
    if (
        getattr(args, "disable_gradient_cache", False)
        and getattr(args, "refresh_gradient_cache", False)
    ):
        raise ValueError(
            "--disable-gradient-cache and --refresh-gradient-cache cannot be combined."
        )
    if (
        args.disable_source_validation_cache
        and args.refresh_source_validation_cache
    ):
        raise ValueError(
            "--disable-source-validation-cache and "
            "--refresh-source-validation-cache cannot be combined."
        )
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    train_path = args.train_path
    validation_path = args.validation_path
    math_grading = None
    if args.qa_task == "anli":
        if train_path == str(DEFAULT_TRAIN_PATH):
            train_path = str(ANLI_TRAIN_PATH)
        if validation_path == str(DEFAULT_VALIDATION_PATH):
            validation_path = str(ANLI_VALIDATION_PATH)
    elif args.qa_task == "hotpotqa":
        if train_path == str(DEFAULT_TRAIN_PATH):
            train_path = str(HOTPOTQA_TRAIN_PATH)
        if validation_path == str(DEFAULT_VALIDATION_PATH):
            validation_path = str(HOTPOTQA_VALIDATION_PATH)
    elif args.qa_task == "math500":
        if train_path == str(DEFAULT_TRAIN_PATH):
            train_path = str(MATH500_TRAIN_PATH)
        if validation_path == str(DEFAULT_VALIDATION_PATH):
            validation_path = str(MATH500_VALIDATION_PATH)
        from math_grading.graders import grader_metadata, validate_grading_dependencies

        validate_grading_dependencies()
        math_grading = grader_metadata()
    train_records = load_qa_records(train_path, args.qa_task)
    all_validation_records = load_qa_records(validation_path, args.qa_task)
    validation_records = select_validation_fold_subset(
        all_validation_records,
        args.validation_fold_size,
    )
    initial_prompt = load_initial_prompt(
        mode,
        args.initial_prompt,
        args.initial_prompt_file,
    )
    output_root = args.output_root
    if args.qa_task == "math500" and output_root == str(DEFAULT_OUTPUT_ROOT):
        output_root = str(DEFAULT_MATH_OUTPUT_ROOT)
    run_dir = create_run_directory(
        output_root,
        optimizer_name,
        args.qa_mode,
        args.code,
        args.overwrite,
    )
    logger = RunLogger(run_dir)
    resolved_vllm_max_model_len = args.vllm_max_model_len
    if args.backend == "dual" and resolved_vllm_max_model_len is None:
        resolved_vllm_max_model_len = DEFAULT_DUAL_VLLM_MAX_MODEL_LEN
    resolved_hf_device = args.hf_device or args.device
    resolved_gradient_cache_root = None
    if hasattr(args, "gradient_cache_root"):
        resolved_gradient_cache_root = str(
            resolve_gradient_cache_root(args.gradient_cache_root)
        )
    resolved_source_validation_cache_root = str(
        resolve_source_validation_cache_root(args.source_validation_cache_root)
    )
    model_pool = ModelPool(
        target_model_id=args.model,
        optimizer_model_id=args.optimizer_model,
        target_device=args.device,
        hf_target_device=resolved_hf_device,
        optimizer_device=args.optimizer_device,
        keep_models_loaded=args.keep_models_loaded,
        seed=args.seed,
        backend=args.backend,
        gpu_memory_utilization=(
            args.dual_vllm_gpu_memory_utilization
            if args.backend == "dual"
            else args.gpu_memory_utilization
        ),
        vllm_max_model_len=resolved_vllm_max_model_len,
        vllm_disable_images=args.vllm_disable_images,
        vllm_conservative_settings=args.vllm_conservative_settings,
        dual_vllm_max_num_batched_tokens=args.dual_vllm_max_num_batched_tokens,
        dual_vllm_max_num_seqs=args.dual_vllm_max_num_seqs,
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
            "resolved_train_path": train_path,
            "resolved_validation_path": validation_path,
            "resolved_output_root": output_root,
            "resolved_vllm_target_device": args.device,
            "resolved_hf_target_device": resolved_hf_device,
            "resolved_gradient_cache_root": resolved_gradient_cache_root,
            "resolved_source_validation_cache_root": (
                resolved_source_validation_cache_root
            ),
            "optimizer_name": optimizer_name,
            "qa_mode_config": asdict(mode),
            "resolved_target_max_new_tokens": max_new_tokens,
            "resolved_vllm_gpu_memory_utilization": (
                args.dual_vllm_gpu_memory_utilization
                if args.backend == "dual"
                else args.gpu_memory_utilization
            ),
            "resolved_vllm_max_model_len": resolved_vllm_max_model_len,
            "resolved_vllm_engine_settings": {
                "conservative_settings": args.vllm_conservative_settings,
                "enforce_eager": args.vllm_conservative_settings,
                "max_num_batched_tokens": (
                    args.dual_vllm_max_num_batched_tokens
                    if args.vllm_conservative_settings
                    else None
                ),
                "max_num_seqs": (
                    args.dual_vllm_max_num_seqs
                    if args.vllm_conservative_settings
                    else None
                ),
                "enable_prefix_caching": (
                    False if args.vllm_conservative_settings else None
                ),
            },
            "dataset_sizes": {
                "train": len(train_records),
                "validation": len(validation_records),
                "validation_available": len(all_validation_records),
            },
            "math_graders": math_grading,
        },
    )
    logger.event(
        "run_started",
        optimizer=optimizer_name,
        qa_task=args.qa_task,
        qa_mode=args.qa_mode,
        backend=args.backend,
        vllm_target_device=args.device,
        hf_target_device=resolved_hf_device,
        objective_scoring_backend=objective_backend,
        initial_prompt=initial_prompt,
    )
    return context
