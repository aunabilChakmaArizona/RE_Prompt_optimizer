"""Generate the OpenBookQA non-reasoning second-stage command matrix."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any, Sequence


DEFAULT_OUTPUT_ROOT = Path("outputs/qa_prompt_optimization")
DEFAULT_COMMAND_DIR = Path("codes")

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen": {
        "target": "Qwen/Qwen3-4B",
        "optimizer": "Qwen/Qwen3-14B",
        "gpu": "2",
        "stage_one_codes": {
            "rpo": "openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1",
            "evoprompt": (
                "openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1"
            ),
            "etgpo": (
                "openbookqa_non_reasoning_qwen_etgpo_qwen14opt_"
                "lambda1_fixed_vs1500"
            ),
        },
        "gradpo_regions": 5,
        "gradpo_region_tokens": 2,
        "gradient_pool_size": 800,
        "vllm_extra": [],
    },
    "gemma": {
        "target": "google/gemma-3-4b-it",
        "optimizer": "google/gemma-3-12b-it",
        "gpu": "3",
        "stage_one_codes": {
            "rpo": "openbookqa_non_reasoning_gemma_rpo_gemma12opt_lambda1",
            "evoprompt": (
                "openbookqa_non_reasoning_gemma_evoprompt_gemma12opt_lambda1"
            ),
            "etgpo": (
                "openbookqa_non_reasoning_gemma_etgpo_gemma12opt_"
                "lambda1_fixed_vs1500"
            ),
        },
        "gradpo_regions": 3,
        "gradpo_region_tokens": 3,
        "gradient_pool_size": 600,
        "vllm_extra": ["--vllm-max-model-len", "32768", "--vllm-disable-images"],
    },
}

SOURCE_SPECS = (
    ("rpo5", "rpo", "rpo", "prompt_iteration_5.txt"),
    ("rpo10", "rpo", "rpo", "prompt_iteration_10.txt"),
    ("evoprompt5", "evoprompt_de", "evoprompt", "prompt_iteration_5.txt"),
    ("evoprompt10", "evoprompt_de", "evoprompt", "prompt_iteration_10.txt"),
    ("etgpo1", "etgpo", "etgpo", "final_prompt.txt"),
)

METHOD_SPECS = (
    ("lpo", "codes/run_qa_promptopt_lpo.py", None),
    ("greater", "codes/run_qa_promptopt_greater.py", "greater"),
    ("greater_tg", "codes/run_qa_promptopt_greater.py", "greater_tg"),
    ("gradpo_gen", "codes/run_qa_promptopt_gradpo.py", "gen"),
    ("gradpo_prob", "codes/run_qa_promptopt_gradpo.py", "prob"),
    ("gradpo_gen_random", "codes/run_qa_promptopt_gradpo.py", "gen_random"),
)


def parse_args() -> argparse.Namespace:
    """Read model-family, GPU, output-root, and command-directory overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-family",
        choices=("qwen", "gemma", "all"),
        default="all",
        help="Generate commands for Qwen, Gemma, or both.",
    )
    parser.add_argument("--qwen-gpu", default="2", help="Physical GPU for Qwen.")
    parser.add_argument("--gemma-gpu", default="3", help="Physical GPU for Gemma.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="First- and second-stage OpenBookQA output root.",
    )
    parser.add_argument(
        "--command-dir",
        type=Path,
        default=DEFAULT_COMMAND_DIR,
        help="Directory receiving the generated shell scripts.",
    )
    return parser.parse_args()


def source_prompt_path(
    output_root: Path,
    family: str,
    source_spec: Sequence[str],
) -> Path:
    """Resolve one saved OpenBookQA first-stage prompt snapshot."""
    _source_name, optimizer_dir, code_key, filename = source_spec
    stage_one_code = MODEL_CONFIGS[family]["stage_one_codes"][code_key]
    return (
        output_root
        / "non_reasoning"
        / optimizer_dir
        / stage_one_code
        / filename
    )


def common_arguments(
    *,
    family: str,
    source_name: str,
    method_name: str,
    prompt_path: Path,
    output_root: Path,
) -> list[str]:
    """Build arguments shared by every OpenBookQA second-stage method."""
    config = MODEL_CONFIGS[family]
    code = (
        f"openbookqa_non_reasoning_{family}_{source_name}_{method_name}_"
        "lambda1_vs1500"
    )
    return [
        "--code", code,
        "--qa-task", "openbookqa",
        "--qa-mode", "non_reasoning",
        "--train-path", "data/processed/openbookqa/train.jsonl",
        "--validation-path", "data/processed/openbookqa/validation.jsonl",
        "--initial-prompt-file", str(prompt_path),
        "--model", str(config["target"]),
        "--device", "cuda:0",
        "--target-max-new-tokens", "10",
        "--validation-std-penalty", "1.0",
        "--output-root", str(output_root),
        "--overwrite",
    ]


def method_arguments(family: str, method_name: str, variant: str | None) -> list[str]:
    """Build the fixed hyperparameters for one second-stage method."""
    config = MODEL_CONFIGS[family]
    if method_name == "lpo":
        return [
            "--optimizer-model", str(config["optimizer"]),
            "--optimizer-device", "cuda:0",
            "--backend", "vllm",
            "--gpu-memory-utilization", "0.90",
            *config["vllm_extra"],
            "--optimizer-max-new-tokens", "10000",
            "--train-sample-size", "512",
            "--feedback-examples", "3",
            "--max-locations", "5",
            "--max-words-per-location", "3",
            "--num-candidates", "5",
            "--top-z", "5",
        ]
    if method_name in {"greater", "greater_tg"}:
        return [
            "--backend", "transformers",
            "--variant", str(variant),
            "--train-sample-size", str(config["gradient_pool_size"]),
            "--gradient-sample-size", "200",
            "--gradient-batch-size", "4",
            "--selection-batch-size", "8",
            "--proposal-top-k", "25",
            "--proposal-example-size", "50",
            "--proposal-min-candidates", "10",
            "--selection-top-mu", "10",
            "--top-u", "5",
            "--fluency-lambda", "0.2",
            "--region-expansion-threshold", "0.6",
        ]
    return [
        "--backend", "transformers",
        "--variant", str(variant),
        "--train-sample-size", str(config["gradient_pool_size"]),
        "--gradient-sample-size", "200",
        "--gradient-batch-size", "2",
        "--selection-batch-size", "4",
        "--num-edit-regions", str(config["gradpo_regions"]),
        "--max-region-tokens", str(config["gradpo_region_tokens"]),
        "--region-expansion-threshold", "0.6",
        "--num-region-candidates", "5",
        "--beam-width", "5",
        "--beam-replacement-mode", "llm_synthesis",
        "--fluency-lambda", "0.5",
        "--candidate-max-new-tokens", "10000",
        "--synthesis-max-new-tokens", "10000",
        "--synthesis-batch-size", "4",
    ]


def split_argument_groups(arguments: Sequence[str]) -> list[list[str]]:
    """Group each option with its value for readable multiline commands."""
    groups: list[list[str]] = []
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item.startswith("--") and index + 1 < len(arguments):
            next_item = arguments[index + 1]
            if not next_item.startswith("--"):
                groups.append([item, next_item])
                index += 2
                continue
        groups.append([item])
        index += 1
    return groups


def format_command(gpu: str, runner: str, arguments: Sequence[str]) -> str:
    """Render one command with one option group per shell line."""
    lines = [f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} python -u {runner} \\"]
    groups = split_argument_groups(arguments)
    for index, group in enumerate(groups):
        suffix = " \\" if index < len(groups) - 1 else ""
        lines.append("  " + " ".join(shlex.quote(value) for value in group) + suffix)
    return "\n".join(lines)


def commands_for_family(
    family: str,
    output_root: Path,
    gpu: str,
) -> tuple[list[Path], list[str]]:
    """Create source paths and thirty commands for one target-model family."""
    source_paths: list[Path] = []
    commands: list[str] = []
    for source_spec in SOURCE_SPECS:
        source_name = source_spec[0]
        prompt_path = source_prompt_path(output_root, family, source_spec)
        source_paths.append(prompt_path)
        for method_name, runner, variant in METHOD_SPECS:
            arguments = common_arguments(
                family=family,
                source_name=source_name,
                method_name=method_name,
                prompt_path=prompt_path,
                output_root=output_root,
            )
            arguments.extend(method_arguments(family, method_name, variant))
            commands.append(format_command(gpu, runner, arguments))
    return source_paths, commands


def script_text(
    family: str,
    source_paths: Sequence[Path],
    commands: Sequence[str],
) -> str:
    """Wrap one model-family command matrix in a fail-fast shell script."""
    filename = f"codes/run_openbookqa_second_stage_{family}.sh"
    log_path = f"codes/nohup_outs/openbookqa_second_stage_{family}.log"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# OpenBookQA non-reasoning: five source prompts x six refiners.",
        "# Test evaluation remains disabled; selection uses 3 x 500 validation.",
        f"# nohup bash {filename} > {log_path} 2>&1 &",
        "",
    ]
    for path in source_paths:
        quoted = shlex.quote(str(path))
        lines.append(
            f"[[ -f {quoted} ]] || {{ echo 'Missing first-stage prompt: "
            f"{quoted}'; exit 1; }}"
        )
    lines.append("")
    lines.append("\n\n".join(commands))
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write Qwen and/or Gemma OpenBookQA second-stage shell scripts."""
    args = parse_args()
    families = ("qwen", "gemma") if args.model_family == "all" else (args.model_family,)
    gpu_overrides = {"qwen": args.qwen_gpu, "gemma": args.gemma_gpu}
    args.command_dir.mkdir(parents=True, exist_ok=True)
    for family in families:
        source_paths, commands = commands_for_family(
            family,
            args.output_root,
            gpu_overrides[family],
        )
        output_path = args.command_dir / f"run_openbookqa_second_stage_{family}.sh"
        output_path.write_text(
            script_text(family, source_paths, commands),
            encoding="utf-8",
        )
        output_path.chmod(0o755)
        print(f"Saved {len(commands)} commands to: {output_path}")
        available = [path for path in source_paths if path.is_file()]
        print(f"First-stage sources ready for {family}: {len(available)}/5")
        for path in source_paths:
            state = "READY" if path.is_file() else "MISSING"
            print(f"  [{state}] {path}")


if __name__ == "__main__":
    main()
