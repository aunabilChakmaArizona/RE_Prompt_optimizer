"""Generate the complete OpenBookQA first- and second-stage command matrix."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

from prompt_optimization.run_io import DEFAULT_OUTPUT_ROOT


MODEL_CONFIGS = {
    "qwen": {
        "target": "Qwen/Qwen3-4B",
        "optimizer": "Qwen/Qwen3-14B",
    },
    "gemma": {
        "target": "google/gemma-3-4b-it",
        "optimizer": "google/gemma-3-12b-it",
    },
}


def parse_args() -> argparse.Namespace:
    """Read matrix filters, devices, output root, and optional command file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("first_stage", "second_stage", "all"),
        default="all",
    )
    parser.add_argument(
        "--qa-mode",
        choices=("reasoning", "non_reasoning", "all"),
        default="all",
    )
    parser.add_argument(
        "--model-family",
        choices=("qwen", "gemma", "all"),
        default="all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--optimizer-device", default=None)
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="Use vLLM where the optimizer does not require gradients.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of selected GPU memory reserved by each vLLM engine.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--output-file", type=Path, default=None)
    args = parser.parse_args()
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be greater than 0 and at most 1.")
    return args


def selected_values(value: str, choices: Sequence[str]) -> list[str]:
    """Expand an all-valued matrix filter into its concrete choices."""
    return list(choices) if value == "all" else [value]


def shell_command(parts: Sequence[str]) -> str:
    """Quote one argv sequence as a copy-pasteable shell command."""
    return shlex.join(list(parts))


def shared_parts(
    *,
    code: str,
    mode: str,
    model_config: dict[str, str],
    args: argparse.Namespace,
    include_optimizer: bool,
    backend: str,
) -> list[str]:
    """Construct command arguments common to every optimizer runner."""
    parts = [
        "--code",
        code,
        "--qa-mode",
        mode,
        "--model",
        model_config["target"],
        "--device",
        args.device,
        "--backend",
        backend,
        "--output-root",
        str(Path(args.output_root).expanduser()),
    ]
    if backend == "vllm":
        parts.extend(
            ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        )
    if include_optimizer:
        parts.extend(["--optimizer-model", model_config["optimizer"]])
        if args.optimizer_device:
            parts.extend(["--optimizer-device", args.optimizer_device])
    if args.include_test:
        parts.append("--evaluate-test")
    return parts


def first_stage_commands(args: argparse.Namespace) -> list[str]:
    """Generate three first-stage jobs for each model and QA mode."""
    commands = []
    modes = selected_values(args.qa_mode, ("reasoning", "non_reasoning"))
    families = selected_values(args.model_family, ("qwen", "gemma"))
    runners = (
        ("rpo", "codes/run_qa_promptopt_rpo.py"),
        ("evoprompt", "codes/run_qa_promptopt_evoprompt.py"),
        ("etgpo", "codes/run_qa_promptopt_etgpo.py"),
    )
    for mode in modes:
        for family in families:
            model_config = MODEL_CONFIGS[family]
            for method, runner in runners:
                code = f"openbookqa_{mode}_{family}_{method}"
                parts = ["python", "-u", runner]
                parts.extend(
                    shared_parts(
                        code=code,
                        mode=mode,
                        model_config=model_config,
                        args=args,
                        include_optimizer=True,
                        backend=args.backend,
                    )
                )
                commands.append(shell_command(parts))
    return commands


def stage_one_sources(
    output_root: Path,
    mode: str,
    family: str,
) -> list[dict[str, str]]:
    """Return deterministic paths for the five first-stage starting prompts."""
    rpo_code = f"openbookqa_{mode}_{family}_rpo"
    evo_code = f"openbookqa_{mode}_{family}_evoprompt"
    etgpo_code = f"openbookqa_{mode}_{family}_etgpo"
    return [
        {
            "name": "rpo5",
            "path": str(output_root / mode / "rpo" / rpo_code / "prompt_iteration_5.txt"),
        },
        {
            "name": "rpo10",
            "path": str(output_root / mode / "rpo" / rpo_code / "prompt_iteration_10.txt"),
        },
        {
            "name": "evoprompt5",
            "path": str(
                output_root
                / mode
                / "evoprompt_de"
                / evo_code
                / "prompt_iteration_5.txt"
            ),
        },
        {
            "name": "evoprompt10",
            "path": str(
                output_root
                / mode
                / "evoprompt_de"
                / evo_code
                / "prompt_iteration_10.txt"
            ),
        },
        {
            "name": "etgpo1",
            "path": str(output_root / mode / "etgpo" / etgpo_code / "final_prompt.txt"),
        },
    ]


def second_stage_specs() -> list[dict[str, Any]]:
    """Describe the six second-stage methods and their variant flags."""
    return [
        {
            "name": "lpo",
            "runner": "codes/run_qa_promptopt_lpo.py",
            "optimizer": True,
            "supports_vllm": True,
            "extra": [],
        },
        {
            "name": "greater",
            "runner": "codes/run_qa_promptopt_greater.py",
            "optimizer": False,
            "supports_vllm": False,
            "extra": ["--variant", "greater"],
        },
        {
            "name": "greater_tg",
            "runner": "codes/run_qa_promptopt_greater.py",
            "optimizer": False,
            "supports_vllm": False,
            "extra": ["--variant", "greater_tg"],
        },
        {
            "name": "gradpo_gen",
            "runner": "codes/run_qa_promptopt_gradpo.py",
            "optimizer": False,
            "supports_vllm": False,
            "extra": ["--variant", "gen"],
        },
        {
            "name": "gradpo_prob",
            "runner": "codes/run_qa_promptopt_gradpo.py",
            "optimizer": False,
            "supports_vllm": False,
            "extra": ["--variant", "prob"],
        },
        {
            "name": "gradpo_gen_random",
            "runner": "codes/run_qa_promptopt_gradpo.py",
            "optimizer": False,
            "supports_vllm": False,
            "extra": ["--variant", "gen_random"],
        },
    ]


def second_stage_commands(args: argparse.Namespace) -> list[str]:
    """Generate six refinements for each of five first-stage prompt snapshots."""
    commands = []
    modes = selected_values(args.qa_mode, ("reasoning", "non_reasoning"))
    families = selected_values(args.model_family, ("qwen", "gemma"))
    output_root = Path(args.output_root).expanduser().resolve()
    for mode in modes:
        for family in families:
            model_config = MODEL_CONFIGS[family]
            for source in stage_one_sources(output_root, mode, family):
                for method in second_stage_specs():
                    backend = (
                        args.backend if method["supports_vllm"] else "transformers"
                    )
                    code = (
                        f"openbookqa_{mode}_{family}_{source['name']}_{method['name']}"
                    )
                    parts = ["python", "-u", method["runner"]]
                    parts.extend(
                        shared_parts(
                            code=code,
                            mode=mode,
                            model_config=model_config,
                            args=args,
                            include_optimizer=bool(method["optimizer"]),
                            backend=backend,
                        )
                    )
                    parts.extend(["--initial-prompt-file", source["path"]])
                    parts.extend(method["extra"])
                    commands.append(shell_command(parts))
    return commands


def main() -> None:
    """Generate, optionally save, and report the command matrix size."""
    args = parse_args()
    commands = []
    if args.phase in {"first_stage", "all"}:
        commands.extend(first_stage_commands(args))
    if args.phase in {"second_stage", "all"}:
        commands.extend(second_stage_commands(args))
    text = "\n".join(commands) + "\n"
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(text, encoding="utf-8")
        print(f"Saved {len(commands)} commands to {args.output_file}", file=sys.stderr)
    else:
        print(text, end="")
        print(f"Generated {len(commands)} commands.", file=sys.stderr)


if __name__ == "__main__":
    main()
