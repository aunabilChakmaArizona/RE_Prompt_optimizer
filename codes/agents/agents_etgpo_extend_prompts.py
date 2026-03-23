#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
CODES_DIR = CURRENT_DIR.parent
ETGPO_DIR = CODES_DIR / "etgpo"
if str(CODES_DIR) not in sys.path:
    sys.path.append(str(CODES_DIR))
if str(ETGPO_DIR) not in sys.path:
    sys.path.append(str(ETGPO_DIR))

from etgpo.etgpo_full_adjusted import (  # noqa: E402
    DEFAULT_MAIN_MODEL,
    DEFAULT_TAXONOMY_MODEL,
    FailureRecord,
    INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    IssueCategory,
    NumpyEncoder,
    TraceAssignment,
    UnifiedPromptOptimizer,
    compute_per_run_stats,
    run_re_episode_set_evaluation,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a previous ETGPO taxonomy/taxonomy_cluster run and generate "
            "additional prompt candidates from the saved taxonomy."
        )
    )
    parser.add_argument(
        "--source_run_dir",
        type=Path,
        required=True,
        help="Previous ETGPO output directory containing final_results.json and taxonomy.json",
    )
    parser.add_argument(
        "--num_additional_prompts",
        type=int,
        required=True,
        help="How many new prompt candidates to generate",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Optional device_map override for local HF model loading",
    )
    parser.add_argument(
        "--main_model",
        type=str,
        default=None,
        help="Optional main model override if the saved config is incomplete",
    )
    parser.add_argument(
        "--taxonomy_model",
        type=str,
        default=None,
        help="Optional taxonomy model override if the saved config is incomplete",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=NumpyEncoder)


def _load_run_config(source_run_dir: Path) -> Dict[str, Any]:
    final_results_path = source_run_dir / "final_results.json"
    if not final_results_path.exists():
        raise FileNotFoundError(f"Missing final_results.json in {source_run_dir}")
    final_results = _load_json(final_results_path)
    config = final_results.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Saved config missing in {final_results_path}")
    return config


def _load_existing_prompt_bundle(
    source_run_dir: Path,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    bundle_path = source_run_dir / "generated_prompt_candidates.json"
    if not bundle_path.exists():
        return [], [], []
    bundle = _load_json(bundle_path)
    return (
        list(bundle.get("candidate_prompts", [])),
        list(bundle.get("candidate_metadata", [])),
        list(bundle.get("clusters", [])),
    )


def _load_taxonomy_state(
    source_run_dir: Path,
) -> Tuple[List[IssueCategory], List[TraceAssignment], List[FailureRecord]]:
    taxonomy_path = source_run_dir / "taxonomy.json"
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Missing taxonomy.json in {source_run_dir}")

    payload = _load_json(taxonomy_path)

    issue_categories = [
        IssueCategory(**item) for item in payload.get("categories", [])
    ]
    trace_key = "trace_assignments" if "trace_assignments" in payload else "assignments"
    trace_assignments = [
        TraceAssignment(**item) for item in payload.get(trace_key, [])
    ]
    failure_records = [
        FailureRecord(**item) for item in payload.get("failure_records", [])
    ]

    if not issue_categories:
        raise ValueError(f"No taxonomy categories found in {taxonomy_path}")
    if not trace_assignments:
        raise ValueError(f"No taxonomy assignments found in {taxonomy_path}")

    return issue_categories, trace_assignments, failure_records


def _resolve_method(config: Dict[str, Any]) -> str:
    methods = list(config.get("methods", []))
    taxonomy_methods = [name for name in methods if name in {"taxonomy", "taxonomy_cluster"}]
    if len(taxonomy_methods) != 1:
        raise ValueError(
            "Expected exactly one taxonomy-style method in saved config, "
            f"found: {taxonomy_methods}"
        )
    return taxonomy_methods[0]


def _build_optimizer(
    source_run_dir: Path,
    config: Dict[str, Any],
    *,
    num_additional_prompts: int,
    device_map_override: Optional[str],
    main_model_override: Optional[str],
    taxonomy_model_override: Optional[str],
) -> UnifiedPromptOptimizer:
    methods = list(config.get("methods", []))
    if not methods:
        raise ValueError("Saved config is missing methods")

    resolved_main_model = main_model_override or config.get("main_model", DEFAULT_MAIN_MODEL)
    # For prompt extension we only need guidance generation, so reuse the main
    # model backend and avoid loading a separate taxonomy model.
    resolved_taxonomy_model = resolved_main_model

    return UnifiedPromptOptimizer(
        dataset=config.get("dataset", "aimo"),
        task_mode=config.get("task_mode", ""),
        valid_size=int(config.get("valid_size", 90)),
        methods=methods,
        main_model=resolved_main_model,
        device_map=device_map_override if device_map_override is not None else config.get("device_map"),
        taxonomy_model=resolved_taxonomy_model,
        taxonomy_runs=int(config.get("taxonomy_runs", 5)),
        coverage_threshold=float(config.get("coverage_threshold", 0.7)),
        max_guidances=int(config.get("max_guidances", 7)),
        min_problems=int(config.get("min_problems", 2)),
        prompt_style=config.get("prompt_style", "detailed"),
        eval_runs=int(config.get("eval_runs", 30)),
        test_runs=int(config.get("test_runs", 30)),
        optimizer_auto=config.get("optimizer_auto", "heavy"),
        seed=int(config.get("seed", 42)),
        skip_validation=bool(config.get("skip_validation", False)),
        ablation_mode=bool(config.get("ablation_mode", False)),
        track_category_level=not bool(config.get("no_category_tracking", False)),
        num_guidance_prompts=num_additional_prompts,
        taxonomy_num_clusters=int(config.get("taxonomy_num_clusters", 5)),
        taxonomy_cluster_coverage_ratio=float(config.get("taxonomy_cluster_coverage_ratio", 0.75)),
        output_dir=str(source_run_dir.parent),
        experiment_name=source_run_dir.name,
        data_dir=config.get("data_dir"),
        train_samples_file=config.get("train_samples_file", "fs_tacred_train_samples.pkl"),
        re_inference_mode=config.get("re_inference_mode", INFERENCE_MODE_SEPARATE_NO_EXAMPLES),
        re_feedback_prompt_key=config.get("re_feedback_prompt_key", "mistakes_v1"),
        inference_batch_size=int(config.get("inference_batch_size", 8)),
        feedback_batch_size=int(config.get("feedback_batch_size", 4)),
        feedback_max_new_tokens=int(config.get("feedback_max_new_tokens", 10000)),
        log_every=int(config.get("log_every", 20)),
        re_eval_n_chunks=int(config.get("re_eval_n_chunks", 3)),
        dev_split=config.get("dev_split", "dev"),
        test_split=config.get("test_split", "test"),
        query_index=int(config.get("query_index", 0)),
        eval_batch_size=int(config.get("eval_batch_size", 8)),
        max_new_tokens=config.get("max_new_tokens"),
    )


def _backup_existing_prompt_bundle(source_run_dir: Path) -> None:
    bundle_path = source_run_dir / "generated_prompt_candidates.json"
    backup_path = source_run_dir / "generated_prompt_candidates.before_extension.json"
    if bundle_path.exists() and not backup_path.exists():
        shutil.copy2(bundle_path, backup_path)


def _merge_candidate_metadata(
    existing_prompts: Sequence[str],
    existing_metadata: Sequence[Dict[str, Any]],
    new_metadata: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged_metadata = [dict(item) for item in existing_metadata]
    next_candidate_index = len(existing_prompts) + 1

    next_prompt_index_by_cluster: Dict[Any, int] = {}
    for item in existing_metadata:
        cluster_id = item.get("cluster_id")
        cluster_prompt_index = item.get("cluster_prompt_index")
        if cluster_prompt_index is None:
            continue
        current_next = next_prompt_index_by_cluster.get(cluster_id, 0)
        next_prompt_index_by_cluster[cluster_id] = max(current_next, int(cluster_prompt_index) + 1)

    for item in new_metadata:
        updated = dict(item)
        cluster_id = updated.get("cluster_id")
        next_prompt_index = next_prompt_index_by_cluster.get(cluster_id, 0)
        updated["candidate_index"] = next_candidate_index
        updated["cluster_prompt_index"] = next_prompt_index
        next_candidate_index += 1
        next_prompt_index_by_cluster[cluster_id] = next_prompt_index + 1
        merged_metadata.append(updated)

    return merged_metadata


def _write_generated_prompt_text(source_run_dir: Path, prompt_text: str) -> None:
    prompt_path = source_run_dir / "generated_prompt.txt"
    with prompt_path.open("w", encoding="utf-8") as handle:
        handle.write(prompt_text)


def _new_candidate_names(
    method_name: str,
    existing_prompts: Sequence[str],
    new_prompts: Sequence[str],
    merged_metadata: Sequence[Dict[str, Any]],
) -> List[str]:
    start_idx = len(existing_prompts)
    names: List[str] = []
    for idx in range(start_idx, start_idx + len(new_prompts)):
        if method_name == "taxonomy":
            names.append(f"taxonomy_{idx + 1}")
            continue

        metadata = merged_metadata[idx] if idx < len(merged_metadata) else {}
        cluster_id = metadata.get("cluster_id")
        cluster_prompt_index = metadata.get("cluster_prompt_index")
        if cluster_id is None or cluster_prompt_index is None:
            raise ValueError(
                "Missing cluster metadata for newly generated taxonomy_cluster prompt "
                f"at merged index {idx}"
            )
        names.append(f"taxonomy_cluster_c{cluster_id}_p{cluster_prompt_index}")
    return names


def _evaluate_new_candidates(
    optimizer: UnifiedPromptOptimizer,
    *,
    candidate_names: Sequence[str],
) -> Dict[str, Any]:
    evaluation_results: Dict[str, Any] = {}
    for candidate_name in candidate_names:
        if candidate_name not in optimizer.programs:
            raise ValueError(f"Program not registered for candidate {candidate_name}")

        print(f"\nEvaluating new candidate: {candidate_name}")
        program = optimizer.programs[candidate_name]
        val_scores, _ = run_re_episode_set_evaluation(
            program=program,
            method_name=candidate_name,
            split_name="validation",
            dataset_type=optimizer.dataset,
            episodes_data=optimizer.re_dev_data,
            num_runs=optimizer.eval_runs,
            model=optimizer.local_model,
            tokenizer=optimizer.local_tokenizer,
            batch_size=optimizer.eval_batch_size,
            yes_token_id=optimizer.yes_token_id,
            no_token_id=optimizer.no_token_id,
            log_every=optimizer.log_every,
            n_chunks=optimizer.re_eval_n_chunks,
            query_index=optimizer.query_index,
            output_dir=optimizer.output_dir,
        )
        val_stats = compute_per_run_stats(val_scores) if val_scores else None
        evaluation_results[candidate_name] = {
            "validation": {
                "mean_accuracy": val_stats["mean"],
                "ci_lower": val_stats["ci_lower"],
                "ci_upper": val_stats["ci_upper"],
                "std": val_stats["std"],
                "min": val_stats["min"],
                "max": val_stats["max"],
                "num_runs": val_stats["num_runs"],
                "per_run_accuracies": val_scores,
            } if val_stats else None
        }
    return evaluation_results


def _update_final_results(
    source_run_dir: Path,
    *,
    merged_metadata: Sequence[Dict[str, Any]],
    guidance_clusters: Sequence[Dict[str, Any]],
    new_evaluation_results: Dict[str, Any],
    extension_summary: Dict[str, Any],
) -> None:
    final_results_path = source_run_dir / "final_results.json"
    final_results = _load_json(final_results_path)

    results = final_results.setdefault("results", {})
    for candidate_name, payload in new_evaluation_results.items():
        results[candidate_name] = payload

    final_results["generated_prompt_candidate_metadata"] = list(merged_metadata)
    final_results["guidance_clusters"] = list(guidance_clusters)
    extension_history = final_results.setdefault("prompt_extensions", [])
    extension_history.append(extension_summary)

    _write_json(final_results_path, final_results)


def main() -> None:
    args = _parse_args()
    if args.num_additional_prompts <= 0:
        raise ValueError("--num_additional_prompts must be >= 1")
    if not args.source_run_dir.exists():
        raise FileNotFoundError(f"Source run directory not found: {args.source_run_dir}")

    config = _load_run_config(args.source_run_dir)
    method_name = _resolve_method(config)
    issue_categories, trace_assignments, failure_records = _load_taxonomy_state(args.source_run_dir)
    existing_prompts, existing_metadata, existing_clusters = _load_existing_prompt_bundle(args.source_run_dir)

    optimizer = _build_optimizer(
        args.source_run_dir,
        config,
        num_additional_prompts=args.num_additional_prompts,
        device_map_override=args.device_map,
        main_model_override=args.main_model,
        taxonomy_model_override=args.taxonomy_model,
    )

    print("Prompt extension configuration")
    print(f"  source_run_dir: {args.source_run_dir}")
    print(f"  method: {method_name}")
    print(f"  existing_prompt_count: {len(existing_prompts)}")
    print(f"  requested_new_prompt_count: {args.num_additional_prompts}")
    print(f"  resolved_main_model: {optimizer.main_model}")
    print(f"  resolved_taxonomy_model: {optimizer.taxonomy_model}")
    print(f"  resolved_device_map: {optimizer.device_map}")
    print("  taxonomy_model_load: skipped_separate_load")

    optimizer.setup()
    optimizer.issue_categories = issue_categories
    optimizer.trace_assignments = trace_assignments
    optimizer.failure_records = failure_records
    optimizer.select_categories()

    _backup_existing_prompt_bundle(args.source_run_dir)
    optimizer.generate_guidance()

    new_prompts = list(optimizer.generated_prompt_candidates)
    new_metadata = list(optimizer.generated_prompt_candidate_metadata)
    guidance_clusters = list(optimizer.guidance_clusters) or existing_clusters

    merged_prompts = list(existing_prompts) + new_prompts
    merged_metadata = _merge_candidate_metadata(existing_prompts, existing_metadata, new_metadata)

    optimizer.generated_prompt_candidates = merged_prompts
    optimizer.generated_prompt_candidate_metadata = merged_metadata
    optimizer.guidance_clusters = guidance_clusters
    optimizer.generated_prompt = merged_prompts[0] if merged_prompts else ""
    optimizer._save_prompt_candidates("generated_prompt_candidates.json")
    if optimizer.generated_prompt:
        _write_generated_prompt_text(args.source_run_dir, optimizer.generated_prompt)

    added_bundle_path = args.source_run_dir / "generated_prompt_candidates_added.json"
    added_payload: Dict[str, Any] = {
        "source_run_dir": str(args.source_run_dir),
        "method": method_name,
        "num_existing_prompts": len(existing_prompts),
        "num_added_prompts": len(new_prompts),
        "candidate_prompts": new_prompts,
    }
    if new_metadata:
        added_payload["candidate_metadata"] = new_metadata
    if guidance_clusters:
        added_payload["clusters"] = guidance_clusters
    _write_json(added_bundle_path, added_payload)

    optimizer._register_taxonomy_candidate_programs()
    new_candidate_names = _new_candidate_names(
        method_name,
        existing_prompts,
        new_prompts,
        merged_metadata,
    )
    new_evaluation_results = _evaluate_new_candidates(
        optimizer,
        candidate_names=new_candidate_names,
    )

    summary_path = args.source_run_dir / "prompt_extension_summary.json"
    summary_payload = {
        "source_run_dir": str(args.source_run_dir),
        "method": method_name,
        "num_existing_prompts": len(existing_prompts),
        "num_added_prompts": len(new_prompts),
        "num_total_prompts": len(merged_prompts),
        "device_map": optimizer.device_map,
        "main_model": optimizer.main_model,
        "taxonomy_model": optimizer.taxonomy_model,
        "new_candidate_names": new_candidate_names,
        "selected_categories": [cat.category_name for cat in optimizer.selected_categories],
        "output_files": {
            "generated_prompt_candidates": str(args.source_run_dir / "generated_prompt_candidates.json"),
            "generated_prompt_candidates_added": str(added_bundle_path),
            "prompt_extension_summary": str(summary_path),
        },
    }
    _write_json(summary_path, summary_payload)
    _update_final_results(
        args.source_run_dir,
        merged_metadata=merged_metadata,
        guidance_clusters=guidance_clusters,
        new_evaluation_results=new_evaluation_results,
        extension_summary=summary_payload,
    )

    print("\nGenerated additional prompts")
    for idx, prompt in enumerate(new_prompts, start=1):
        metadata_suffix = ""
        if idx - 1 < len(new_metadata):
            metadata_suffix = f" | metadata={json.dumps(new_metadata[idx - 1], sort_keys=True)}"
        print(f"  {idx}. chars={len(prompt)}{metadata_suffix}")

    print("\nEvaluated new prompts")
    for candidate_name in new_candidate_names:
        validation = new_evaluation_results[candidate_name]["validation"]
        print(
            f"  {candidate_name} | "
            f"mean_f1={validation['mean_accuracy']:.4f} | "
            f"runs={validation['num_runs']}"
        )

    print("\nSaved files")
    print(f"  merged prompts: {args.source_run_dir / 'generated_prompt_candidates.json'}")
    print(f"  added prompts: {added_bundle_path}")
    print(f"  summary: {summary_path}")
    print(f"  final results updated: {args.source_run_dir / 'final_results.json'}")
    print(f"  total prompts now: {len(merged_prompts)}")


if __name__ == "__main__":
    main()
