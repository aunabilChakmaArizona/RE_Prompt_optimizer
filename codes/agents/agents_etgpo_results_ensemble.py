#!/usr/bin/env python3

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
CODES_DIR = CURRENT_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.append(str(CODES_DIR))

from agents.agent_metrics import compute_prf_stats


@dataclass
class CandidatePrediction:
    run_dir: Path
    candidate_name: str
    method_name: str
    prediction_file: Path
    f1: float
    precision: float
    recall: float
    labels: List[str]
    predictions: List[str]
    cluster_id: Optional[int]
    cluster_prompt_index: Optional[int]
    candidate_index: Optional[int]
    prompt_text: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class RunEnsembleResult:
    run_dir: Path
    run_config: Dict[str, Any]
    selected_candidates: List[CandidatePrediction]
    output_payload: Dict[str, Any]
    output_path: Path


def _json_dumps_pretty(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensemble saved ETGPO validation predictions across taxonomy or "
            "taxonomy_cluster runs."
        )
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        required=True,
        help="Root directory containing ETGPO run folders with final_results.json",
    )
    parser.add_argument(
        "--method",
        choices=["taxonomy", "taxonomy_cluster"],
        required=True,
        help="Which ETGPO method to ensemble",
    )
    parser.add_argument(
        "--q",
        type=int,
        required=True,
        help="Number of prompts to keep for the ensemble",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,
        help=(
            "Optional prefilter. For taxonomy: consider only the first W prompts before "
            "taking the top Q by score. For taxonomy_cluster: select the best prompt from "
            "the first W prompts within each cluster before taking the top Q cluster winners"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Saved split to read from eval_outputs (default: validation)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to <root_dir>/ensemble_<method>_q<Q>.json",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "top_q_scores_only"],
        default="full",
        help=(
            "Output mode. 'full' keeps the current verbose report. "
            "'top_q_scores_only' prints only the selected top-Q prompt scores and their average."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_run_dirs(root_dir: Path) -> List[Path]:
    return sorted(
        path.parent for path in root_dir.rglob("final_results.json")
    )


def _candidate_index_from_name(candidate_name: str, method_name: str) -> Optional[int]:
    if method_name == "taxonomy":
        match = re.fullmatch(r"taxonomy_(\d+)", candidate_name)
        return int(match.group(1)) if match else None
    return None


def _cluster_parts_from_name(candidate_name: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.fullmatch(r"taxonomy_cluster_c(\d+)_p(\d+)", candidate_name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _load_generated_prompt_bundle(
    run_dir: Path,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    bundle_path = run_dir / "generated_prompt_candidates.json"
    if not bundle_path.exists():
        return [], []
    bundle = _load_json(bundle_path)
    prompts = bundle.get("candidate_prompts", [])
    metadata = bundle.get("candidate_metadata", [])
    return prompts, metadata


def _resolve_prompt_and_metadata(
    candidate_name: str,
    method_name: str,
    prompts: Sequence[str],
    metadata_list: Sequence[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[str], Dict[str, Any], Optional[int], Optional[int]]:
    cluster_id = None
    cluster_prompt_index = None
    candidate_index = _candidate_index_from_name(candidate_name, method_name)

    if method_name == "taxonomy_cluster":
        cluster_id, cluster_prompt_index = _cluster_parts_from_name(candidate_name)

    metadata: Dict[str, Any] = {}
    if method_name == "taxonomy_cluster" and cluster_id is not None and cluster_prompt_index is not None:
        for item in metadata_list:
            if (
                item.get("cluster_id") == cluster_id
                and item.get("cluster_prompt_index") == cluster_prompt_index
            ):
                metadata = item
                candidate_index = item.get("candidate_index")
                break
    elif candidate_index is not None:
        for item in metadata_list:
            if item.get("candidate_index") == candidate_index:
                metadata = item
                cluster_id = item.get("cluster_id")
                cluster_prompt_index = item.get("cluster_prompt_index")
                break

    prompt_text = None
    if candidate_index is not None and 1 <= candidate_index <= len(prompts):
        prompt_text = prompts[candidate_index - 1]

    return candidate_index, prompt_text, metadata, cluster_id, cluster_prompt_index


def _load_candidate_prediction(
    run_dir: Path,
    candidate_name: str,
    method_name: str,
    split: str,
    prompts: Sequence[str],
    metadata_list: Sequence[Dict[str, Any]],
) -> Optional[CandidatePrediction]:
    prediction_file = (
        run_dir
        / "eval_outputs"
        / f"EVALID_{candidate_name}_{split}_0_labels_predictions.json"
    )
    if not prediction_file.exists():
        return None

    payload = _load_json(prediction_file)
    labels = payload.get("labels", [])
    predictions = payload.get("predictions", [])
    metrics = payload.get("metrics", {})
    if not labels or not predictions or len(labels) != len(predictions):
        return None

    candidate_index, prompt_text, metadata, cluster_id, cluster_prompt_index = (
        _resolve_prompt_and_metadata(candidate_name, method_name, prompts, metadata_list)
    )

    return CandidatePrediction(
        run_dir=run_dir,
        candidate_name=candidate_name,
        method_name=method_name,
        prediction_file=prediction_file,
        f1=float(metrics.get("f1_mean", 0.0)),
        precision=float(metrics.get("precision_mean", 0.0)),
        recall=float(metrics.get("recall_mean", 0.0)),
        labels=list(labels),
        predictions=list(predictions),
        cluster_id=cluster_id,
        cluster_prompt_index=cluster_prompt_index,
        candidate_index=candidate_index,
        prompt_text=prompt_text,
        metadata=metadata,
    )


def _collect_candidates(root_dir: Path, method_name: str, split: str) -> List[CandidatePrediction]:
    candidates: List[CandidatePrediction] = []
    for run_dir in _iter_run_dirs(root_dir):
        final_results_path = run_dir / "final_results.json"
        final_results = _load_json(final_results_path)
        results = final_results.get("results", {})
        prompts, metadata_list = _load_generated_prompt_bundle(run_dir)

        for candidate_name in sorted(results.keys()):
            if method_name == "taxonomy":
                if re.fullmatch(r"taxonomy_\d+", candidate_name) is None:
                    continue
            elif method_name == "taxonomy_cluster":
                if re.fullmatch(r"taxonomy_cluster_c\d+_p\d+", candidate_name) is None:
                    continue
            else:
                continue
            candidate = _load_candidate_prediction(
                run_dir=run_dir,
                candidate_name=candidate_name,
                method_name=method_name,
                split=split,
                prompts=prompts,
                metadata_list=metadata_list,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _group_candidates_by_run(
    candidates: Sequence[CandidatePrediction],
) -> Dict[Path, List[CandidatePrediction]]:
    grouped: Dict[Path, List[CandidatePrediction]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.run_dir].append(candidate)
    return dict(grouped)


def _sort_candidates(candidates: Sequence[CandidatePrediction]) -> List[CandidatePrediction]:
    return sorted(
        candidates,
        key=lambda item: (
            -item.f1,
            -item.precision,
            -item.recall,
            str(item.run_dir),
            item.candidate_name,
        ),
    )


def _select_candidates(
    candidates: Sequence[CandidatePrediction],
    method_name: str,
    q: int,
    w: Optional[int] = None,
) -> List[CandidatePrediction]:
    ranked = _sort_candidates(candidates)
    if method_name == "taxonomy":
        if w is not None:
            ranked = [
                candidate
                for candidate in ranked
                if candidate.candidate_index is not None and candidate.candidate_index <= w
            ]
        return ranked[: min(q, len(ranked))]

    best_by_cluster: Dict[int, CandidatePrediction] = {}
    for candidate in ranked:
        if candidate.cluster_id is None:
            continue
        if w is not None:
            if candidate.cluster_prompt_index is None:
                continue
            if candidate.cluster_prompt_index >= w:
                continue
        if candidate.cluster_id not in best_by_cluster:
            best_by_cluster[candidate.cluster_id] = candidate

    cluster_winners = _sort_candidates(list(best_by_cluster.values()))
    return cluster_winners[: min(q, len(cluster_winners))]


def _validate_label_alignment(candidates: Sequence[CandidatePrediction]) -> List[str]:
    if not candidates:
        raise ValueError("No candidates selected for ensembling.")
    reference = candidates[0].labels
    for candidate in candidates[1:]:
        if candidate.labels != reference:
            raise ValueError(
                "Selected candidates do not share the same label ordering. "
                f"Mismatch found in {candidate.prediction_file}."
            )
    return reference


def _majority_vote_predictions(
    candidates: Sequence[CandidatePrediction],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    labels = _validate_label_alignment(candidates)
    ensemble_predictions: List[str] = []
    vote_traces: List[Dict[str, Any]] = []

    for idx in range(len(labels)):
        votes = [candidate.predictions[idx] for candidate in candidates]
        counts = Counter(votes)
        best_count = max(counts.values())
        tied_labels = {label for label, count in counts.items() if count == best_count}

        winner = None
        for candidate in candidates:
            candidate_vote = candidate.predictions[idx]
            if candidate_vote in tied_labels:
                winner = candidate_vote
                break

        assert winner is not None
        ensemble_predictions.append(winner)
        vote_traces.append(
            {
                "example_index": idx,
                "votes": votes,
                "vote_counts": dict(sorted(counts.items())),
                "prediction": winner,
            }
        )

    return ensemble_predictions, vote_traces


def _build_output_payload(
    method_name: str,
    q: int,
    w: Optional[int],
    split: str,
    selected_candidates: Sequence[CandidatePrediction],
    ensemble_labels: Sequence[str],
    ensemble_predictions: Sequence[str],
    vote_traces: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    ensemble_metrics = compute_prf_stats(ensemble_labels, ensemble_predictions, n_chunks=1)
    return {
        "method": method_name,
        "requested_q": q,
        "requested_w": w,
        "selected_q": len(selected_candidates),
        "split": split,
        "selection_strategy": (
            (
                "top_q_by_f1_from_first_w" if w is not None else "top_q_by_f1"
            ) if method_name == "taxonomy"
            else (
                "best_per_cluster_from_first_w_then_top_q_cluster_winners"
                if w is not None else
                "best_per_cluster_then_top_q_cluster_winners"
            )
        ),
        "ensemble_metrics": ensemble_metrics,
        "selected_candidates": [
            {
                "rank": rank,
                "run_dir": str(candidate.run_dir),
                "candidate_name": candidate.candidate_name,
                "f1": candidate.f1,
                "precision": candidate.precision,
                "recall": candidate.recall,
                "cluster_id": candidate.cluster_id,
                "cluster_prompt_index": candidate.cluster_prompt_index,
                "candidate_index": candidate.candidate_index,
                "prediction_file": str(candidate.prediction_file),
                "metadata": candidate.metadata,
                "prompt_text": candidate.prompt_text,
            }
            for rank, candidate in enumerate(selected_candidates, start=1)
        ],
        "labels": list(ensemble_labels),
        "predictions": list(ensemble_predictions),
        "vote_traces": list(vote_traces),
    }


def _default_output_paths(
    method_name: str,
    selected_candidates: Sequence[CandidatePrediction],
    w: Optional[int] = None,
) -> Path:
    if not selected_candidates:
        raise ValueError("No selected candidates available to determine output path.")
    w_suffix = f"_w{w}" if w is not None else ""
    filename = f"ensemble_{method_name}_q{len(selected_candidates)}{w_suffix}.json"
    return selected_candidates[0].run_dir / filename


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_candidate_score_and_prompt(candidate: CandidatePrediction) -> None:
    print(f"{candidate.precision*100:.2f}\t{candidate.recall*100:.2f}\t{candidate.f1*100:.2f}")
    print(candidate.prompt_text if candidate.prompt_text is not None else "[prompt unavailable]")


def _print_top_q_scores_only(run_results: Sequence[RunEnsembleResult]) -> None:
    for run_idx, run_result in enumerate(run_results):
        for candidate in run_result.selected_candidates:
            _print_candidate_score_and_prompt(candidate)

        avg_precision = _mean([candidate.precision for candidate in run_result.selected_candidates])
        avg_recall = _mean([candidate.recall for candidate in run_result.selected_candidates])
        avg_f1 = _mean([candidate.f1 for candidate in run_result.selected_candidates])
        print("AVG "
            f"{avg_precision*100:.2f}\t{avg_recall*100:.2f}\t{avg_f1*100:.2f}")

        if run_idx < len(run_results) - 1:
            print()


def main() -> None:
    args = _parse_args()
    if args.q <= 0:
        raise ValueError("--q must be >= 1")
    if args.w is not None and args.w <= 0:
        raise ValueError("--w must be >= 1 when provided")
    if not args.root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {args.root_dir}")

    candidates = _collect_candidates(args.root_dir, args.method, args.split)
    if not candidates:
        raise ValueError(
            f"No saved {args.method} candidates with split={args.split!r} were found under {args.root_dir}"
        )
    run_results: List[RunEnsembleResult] = []
    grouped_candidates = _group_candidates_by_run(candidates)

    for run_dir in sorted(grouped_candidates.keys(), key=lambda path: str(path)):
        run_candidates = grouped_candidates[run_dir]
        selected_candidates = _select_candidates(run_candidates, args.method, args.q, args.w)
        if not selected_candidates:
            continue

        ensemble_predictions, vote_traces = _majority_vote_predictions(selected_candidates)
        ensemble_labels = selected_candidates[0].labels
        output_payload = _build_output_payload(
            method_name=args.method,
            q=args.q,
            w=args.w,
            split=args.split,
            selected_candidates=selected_candidates,
            ensemble_labels=ensemble_labels,
            ensemble_predictions=ensemble_predictions,
            vote_traces=vote_traces,
        )
        output_payload["run_dir"] = str(run_dir)
        output_payload["run_config"] = _load_json(run_dir / "final_results.json").get("config", {})

        output_path = args.output if args.output else _default_output_paths(
            args.method,
            selected_candidates,
            args.w,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2)

        run_results.append(
            RunEnsembleResult(
                run_dir=run_dir,
                run_config=output_payload["run_config"],
                selected_candidates=selected_candidates,
                output_payload=output_payload,
                output_path=output_path,
            )
        )

    if not run_results:
        raise ValueError(
            f"No per-run ensemble results were produced for method={args.method!r} under {args.root_dir}"
        )

    if args.mode == "top_q_scores_only":
        _print_top_q_scores_only(run_results)
        return

    print("Ensemble run configuration")
    print(f"  root_dir: {args.root_dir}")
    print(f"  method: {args.method}")
    print(f"  split: {args.split}")
    print(f"  requested_q: {args.q}")
    print(f"  requested_w: {args.w}")
    print(f"  matched_run_count: {len(run_results)}")

    for run_result in run_results:
        metrics = run_result.output_payload["ensemble_metrics"]
        print(f"\nRUN_OUTPUT_DIR: {run_result.run_dir}")
        print(_json_dumps_pretty(run_result.run_config))
        print("")
        print(
            f"Selected {len(run_result.selected_candidates)} candidates for method={args.method}"
        )
        print("Selected candidate scores:")
        for candidate in run_result.selected_candidates:
            _print_candidate_score_and_prompt(candidate)
        print("Ensemble scores:")
        print(f"  precision={metrics['precision_mean']:.4f}")
        print(f"  recall={metrics['recall_mean']:.4f}")
        print(f"  f1={metrics['f1_mean']:.4f}")
        print(f"Saved ensemble output to: {run_result.output_path}")


if __name__ == "__main__":
    main()
