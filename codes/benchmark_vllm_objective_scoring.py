"""Compare HF and vLLM objectives on a small, fixed set of cached responses."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from prompt_optimization.models import ModelPool, TARGET_ROLE
from prompt_optimization.qa_task import load_qa_records, resolve_mode
from prompt_optimization.run_io import save_json
from prompt_optimization.sequence_gradients import score_combined_objectives
from prompt_optimization.vllm_scoring import (
    score_combined_objectives_vllm,
    validate_vllm_scoring_version,
)


def parse_args() -> argparse.Namespace:
    """Read benchmark inputs without starting generation or an optimizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Target model used by both scorers.")
    parser.add_argument("--qa-task", choices=("openbookqa", "hotpotqa", "math500"), required=True, help="Task of the saved responses.")
    parser.add_argument("--qa-mode", choices=("reasoning", "non_reasoning"), required=True, help="Whether cached reasoning conditions the loss.")
    parser.add_argument("--train-path", required=True, help="Processed training JSONL containing cached record IDs.")
    parser.add_argument("--cache-file", required=True, help="Existing shared training-response cache JSON; never modified.")
    parser.add_argument("--prompt-files", nargs="+", required=True, help="Candidate instruction TXT files to compare.")
    parser.add_argument("--sample-size", type=int, default=4, help="Maximum cached examples, spread across response lengths.")
    parser.add_argument("--selection-batch-size", type=int, default=1, help="HF scoring batch size.")
    parser.add_argument("--objective-scoring-batch-size", type=int, default=128, help="Maximum sequences per vLLM submission.")
    parser.add_argument("--device", default="cuda:0", help="Logical GPU for vLLM.")
    parser.add_argument("--hf-device", default=None, help="Optional separate HF GPU; otherwise --device.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4, help="GPU fraction reserved by vLLM.")
    parser.add_argument("--vllm-max-model-len", type=int, default=32768, help="vLLM context limit, including supplied reasoning.")
    parser.add_argument("--vllm-disable-images", action="store_true", help="Disable image inputs for text-only Gemma runs.")
    parser.add_argument("--vllm-conservative-settings", action="store_true", help="Opt into the existing four engine safeguards.")
    parser.add_argument("--fluency-lambda", type=float, default=0.5, help="Same objective weight for both scorers.")
    parser.add_argument("--output", default="outputs/benchmarks/vllm_objective_scoring.json", help="Comparison report JSON, not an experiment score.")
    args = parser.parse_args()
    if min(args.sample_size, args.selection_batch_size, args.objective_scoring_batch_size) <= 0:
        parser.error("Sample and batch sizes must be positive.")
    if not 0 < args.gpu_memory_utilization <= 1 or args.fluency_lambda < 0:
        parser.error("GPU fraction must be in (0, 1] and fluency weight non-negative.")
    return args


def load_benchmark_examples(args) -> tuple[list[dict], list[str] | None]:
    """Restore unchanged cached records, sampling short through long responses."""
    payload = json.loads(Path(args.cache_file).read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    for field, expected in (("model_id", args.model), ("task_name", args.qa_task), ("mode_name", args.qa_mode)):
        if metadata[field] != expected:
            raise ValueError(f"Benchmark cache {field} does not match {expected!r}.")
    records = {str(record["id"]): record for record in load_qa_records(args.train_path, expected_task=args.qa_task)}
    rows = sorted(payload["pool_results"], key=lambda row: len(row["raw_response"]))
    count = min(args.sample_size, len(rows))
    if not count:
        raise ValueError("Benchmark cache has no saved responses.")
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)] if count > 1 else [0]
    selected = [rows[index] for index in indices]
    selected_records = [records[str(row["record_id"])] for row in selected]
    for record, row in zip(selected_records, selected):
        if str(record["answer"]).strip() != str(row["gold_answer"]).strip():
            raise ValueError("Benchmark record answer differs from its cached gold answer.")
    traces = [row["raw_response"] for row in selected] if args.qa_mode == "reasoning" else None
    return selected_records, traces


def main() -> None:
    """Benchmark identical objectives and save differences, rankings, and timing."""
    args = parse_args()
    validate_vllm_scoring_version()
    mode = resolve_mode(args.qa_mode, args.qa_task)
    records, traces = load_benchmark_examples(args)
    prompts = [Path(path).read_text(encoding="utf-8").strip() for path in args.prompt_files]
    pool = ModelPool(
        target_model_id=args.model, optimizer_model_id=None,
        target_device=args.device, hf_target_device=args.hf_device,
        optimizer_device=None, keep_models_loaded=True, seed=42,
        backend="dual", gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_disable_images=args.vllm_disable_images,
        vllm_conservative_settings=args.vllm_conservative_settings,
    )
    try:
        hf_model, tokenizer = pool.ensure(TARGET_ROLE)
        vllm_model, _ = pool.ensure_vllm(TARGET_ROLE)
        common = dict(mode=mode, tokenizer=tokenizer, fluency_lambda=args.fluency_lambda, reasoning_traces=traces)
        # Small warmups separate startup/JIT cost from the measured scoring calls.
        warmup = {**common, "reasoning_traces": traces[:1] if traces is not None else None}
        score_combined_objectives(prompts[:1], records[:1], model=hf_model, batch_size=1, **warmup)
        score_combined_objectives_vllm(prompts[:1], records[:1], model=vllm_model, batch_size=1, **warmup)
        started = time.monotonic()
        hf = score_combined_objectives(prompts, records, model=hf_model, batch_size=args.selection_batch_size, **common)
        hf_seconds = time.monotonic() - started
        started = time.monotonic()
        vllm = score_combined_objectives_vllm(prompts, records, model=vllm_model, batch_size=args.objective_scoring_batch_size, **common)
        vllm_seconds = time.monotonic() - started
        hf_order = sorted(range(len(prompts)), key=lambda index: hf[index]["combined_score"])
        vllm_order = sorted(range(len(prompts)), key=lambda index: vllm[index]["combined_score"])
        report = {
            "settings": vars(args), "record_ids": [record["id"] for record in records],
            "hf_seconds": hf_seconds, "vllm_seconds": vllm_seconds,
            "speedup": hf_seconds / vllm_seconds, "hf": hf, "vllm": vllm,
            "absolute_differences": [{field: abs(first[field] - second[field]) for field in first} for first, second in zip(hf, vllm)],
            "hf_ranking": hf_order, "vllm_ranking": vllm_order,
            "ranking_matches": hf_order == vllm_order,
            "note": "Same supplied reasoning; no new answers generated. Small numerical differences are expected. This tiny benchmark is not a full-beam speed estimate.",
        }
        save_json(Path(args.output), report)
        print(f"HF={hf_seconds:.2f}s | vLLM={vllm_seconds:.2f}s | speedup={report['speedup']:.2f}x | ranking_matches={report['ranking_matches']}")
        print(f"Saved comparison to: {args.output}")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
