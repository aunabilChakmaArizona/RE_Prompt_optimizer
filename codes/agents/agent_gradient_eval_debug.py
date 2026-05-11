#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
CURRENT_DIR = Path(__file__).resolve().parent
CODES_DIR = CURRENT_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.append(str(CODES_DIR))

from agents.agent_data_loader import DEFAULT_DATA_DIR, load_split_episodes, load_train_samples
from agents.agent_binary_inference import run_binary_inference
from agents.agent_gradient_token_analysis import (
    TOKEN_CANDIDATE_MODE_CHOICES,
    TOKEN_CANDIDATE_MODE_NEAREST_UPDATED,
    analyze_relation_extraction_binary_pairs,
    build_relation_prompt,
    score_instruction_prompt_fluency_perplexity,
    score_binary_prompts_with_ce_and_perplexity,
)
from agents.agent_llm_prompting import run_prompts
from agents.agent_memory import clear_cuda_cache, clear_model_memory
from agents.agent_metrics import compute_prf_stats
from agents.agent_models import load_model_and_tokenizer
from agents.agent_prompts import (
    GRADIENT_REGION_CANDIDATE_COMBINATION_PROMPT_V1,
    GRADIENT_REGION_CANDIDATE_SUGGESTION_PROMPT_V1,
    GRADIENT_REGION_CANDIDATE_SYNTHESIS_PROMPT_V1,
    GRADIENT_REGION_MUTATION_PROMPT_V1,
    INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
)
from agents.agent_relation_utils import get_relation_description
from agents.agent_sample_feedback import sample_feedback_fn
from agents.agent_scorer import NO_RELATION
from agents.agent_token_cluster import select_centroid_token_ids
from agents.agent_utils import (
    DEFAULT_F1_STABILITY_STD_MULTIPLIER,
    extract_json_object,
    extract_tagged_text,
    get_sentence_with_tags,
    load_json_file,
    stable_prf_score_or_neg_inf,
)

MODE_DIRECT_CANDIDATE_GENERATION = "DIRECT_CANDIDATE_GENERATION"
MODE_LLM_CANDIDATE_SUGGESTION = "LLM_CANDIDATE_SUGGESTION"
MODE_LM_PROBABILITY_CANDIDATE_SUGGESTION = "LM_PROBABILITY_CANDIDATE_SUGGESTION"
LM_PROBABILITY_SUBMODE_NO_CONTEXT = "NO_CONTEXT"
LM_PROBABILITY_SUBMODE_FULL_PROMPT_AS_CONTEXT = "FULL_PROMPT_AS_CONTEXT"
LM_PROBABILITY_SUBMODE_CHOICES = [
    LM_PROBABILITY_SUBMODE_NO_CONTEXT,
    LM_PROBABILITY_SUBMODE_FULL_PROMPT_AS_CONTEXT,
]
BEAM_REPLACEMENT_MODE_LLM_SYNTHESIS = "LLM_SYNTHESIS"
BEAM_REPLACEMENT_MODE_DIRECT_REPLACEMENT = "DIRECT_REPLACEMENT"
BEAM_REPLACEMENT_MODE_CHOICES = [
    BEAM_REPLACEMENT_MODE_LLM_SYNTHESIS,
    BEAM_REPLACEMENT_MODE_DIRECT_REPLACEMENT,
]
MODE_CHOICES = [
    MODE_DIRECT_CANDIDATE_GENERATION,
    MODE_LLM_CANDIDATE_SUGGESTION,
    MODE_LM_PROBABILITY_CANDIDATE_SUGGESTION,
]


def _create_output_run_dir(
    *,
    output_root_dir: Path,
    output_substring: str,
) -> Path:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cleaned_substring = output_substring.strip().replace("/", "-")
    run_name = f"{run_stamp}_{cleaned_substring}" if cleaned_substring else run_stamp
    run_dir = output_root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _namespace_to_json_dict(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample train feedback-style TACRED binary pairs, run prompt-token "
            "gradient analysis on a balanced subset, generate prompt edits, and "
            "evaluate the edited prompts on the balanced train subset and full dev set."
        )
    )
    parser.add_argument("--model", required=True, help="HF model id to load")
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default=MODE_DIRECT_CANDIDATE_GENERATION,
        help="Prompt generation mode after gradient analysis.",
    )
    parser.add_argument(
        "--meta-prompt-model",
        default=None,
        help=(
            "Optional HF model id used only by DIRECT_CANDIDATE_GENERATION for "
            "generating prompt variants from the meta prompt. If omitted, --model "
            "is reused."
        ),
    )
    parser.add_argument(
        "--device-map",
        default="cuda:0",
        help="Device map override (e.g. cuda:0, cuda:1, cpu, auto)",
    )
    parser.add_argument(
        "--prompt-source-path",
        type=Path,
        required=True,
        help="Path to generated_prompt_candidates.json or population.json",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=None,
        help="0-based prompt index for generated_prompt_candidates.json",
    )
    parser.add_argument(
        "--prompt-node-id",
        type=int,
        default=None,
        help="node_id for population.json",
    )
    parser.add_argument("--dataset-type", default="fs_tacred")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--train-samples",
        default="fs_tacred_train_samples.pkl",
        help="Train samples pickle used by the trainer-style feedback sampler.",
    )
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-gradient-sample-size",
        type=int,
        default=None,
        help=(
            "New mode only. Sample D binary pairs from the train split, bucket them "
            "by fresh predictions from the original prompt, and use a balanced subset "
            "for gradient collection."
        ),
    )
    parser.add_argument(
        "--full-eval-split",
        default="dev",
        help=(
            "Split used for full evaluation of generated prompts in the new mode. "
            "Defaults to dev."
        ),
    )
    parser.add_argument(
        "--gradient-batch-size",
        type=int,
        default=8,
        help="Optional batch size used while accumulating gradients over the sampled pairs.",
    )
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument(
        "--candidate-mode",
        choices=TOKEN_CANDIDATE_MODE_CHOICES,
        default=TOKEN_CANDIDATE_MODE_NEAREST_UPDATED,
    )
    parser.add_argument("--max-regions", type=int, default=5)
    parser.add_argument("--max-total-region-tokens", type=int, default=15)
    parser.add_argument(
        "--max-region-tokens",
        type=int,
        default=None,
        help=(
            "Optional cap on tokens selected for each gradient region. Region expansion "
            "still uses --region-expansion-threshold-ratio, but adds one boundary token "
            "at a time from the left or right side with the higher gradient."
        ),
    )
    parser.add_argument(
        "--region-expansion-threshold-ratio",
        type=float,
        default=0.6,
        help="Threshold ratio used when expanding a peak token into a larger gradient region.",
    )
    parser.add_argument("--embedding-step-size", type=float, default=1.0)
    parser.add_argument(
        "--num-edit-regions",
        type=int,
        default=1,
        help="Use the top X ranked gradient regions for localized prompt editing.",
    )
    parser.add_argument(
        "--num-generated-prompts",
        type=int,
        default=0,
        help="DIRECT_CANDIDATE_GENERATION only. How many prompt variants to generate.",
    )
    parser.add_argument(
        "--num-region-candidates",
        type=int,
        default=5,
        help=(
            "LLM_CANDIDATE_SUGGESTION and LM_PROBABILITY_CANDIDATE_SUGGESTION only. "
            "Number of candidate rewrites to generate per region."
        ),
    )
    parser.add_argument(
        "--lm-probability-submode",
        choices=LM_PROBABILITY_SUBMODE_CHOICES,
        default=LM_PROBABILITY_SUBMODE_NO_CONTEXT,
        help=(
            "LM_PROBABILITY_CANDIDATE_SUGGESTION only. NO_CONTEXT uses the raw prompt "
            "prefix for next-token selection. FULL_PROMPT_AS_CONTEXT wraps the full "
            "prompt in a paraphrase instruction and predicts the next token after the "
            "rewritten prefix."
        ),
    )
    parser.add_argument(
        "--top-k-prompts",
        type=int,
        default=5,
        help=(
            "Legacy candidate-suggestion option. Beam-search candidate suggestion now "
            "uses --beam-width as the final top-k size."
        ),
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=3,
        help=(
            "Candidate-suggestion modes only. Beam width used while expanding regions "
            "one at a time. The final beam is also the final top-k prompt set."
        ),
    )
    parser.add_argument(
        "--beam-replacement-mode",
        choices=BEAM_REPLACEMENT_MODE_CHOICES,
        default=BEAM_REPLACEMENT_MODE_LLM_SYNTHESIS,
        help=(
            "Candidate-suggestion modes only. LLM_SYNTHESIS applies each selected "
            "replacement set through the multi-region synthesis meta prompt. "
            "DIRECT_REPLACEMENT preserves the legacy exact span splice."
        ),
    )
    parser.add_argument(
        "--Q",
        type=int,
        default=1,
        help=(
            "Candidate-suggestion modes only. Number of outer gradient-based prompt "
            "optimization iterations. Q=1 preserves the current single-pass behavior."
        ),
    )
    parser.add_argument(
        "--selection-perplexity-lambda",
        type=float,
        default=0.2,
        help=(
            "Candidate-suggestion modes only. Combined score = task cross_entropy + "
            "lambda * prompt fluency perplexity."
        ),
    )
    parser.add_argument(
        "--use-log-fluency-score",
        action="store_true",
        help=(
            "Candidate-suggestion modes only. Use the raw mean token negative "
            "log-likelihood for prompt fluency instead of exp(NLL) perplexity."
        ),
    )
    parser.add_argument(
        "--selection-f1-std-penalty",
        type=float,
        default=DEFAULT_F1_STABILITY_STD_MULTIPLIER,
        help=(
            "Candidate-suggestion modes only. Retained beam prompts are ranked by "
            "full-evaluation F1 mean minus this multiplier times F1 std. "
            "Defaults to 2.5."
        ),
    )
    parser.add_argument(
        "--meta-prompt-max-new-tokens",
        type=int,
        default=10000,
        help="Generation budget for each region-edit meta prompt run.",
    )
    parser.add_argument(
        "--meta-prompt-batch-size",
        type=int,
        default=1,
        help="Batch size for running the region-edit meta prompt repeatedly.",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=8,
        help="Batch size for validating generated prompts on the sampled pairs.",
    )
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument(
        "--output-root-dir",
        type=Path,
        default=Path("gradients_experiments"),
        help="Root directory under which a timestamped run folder will be created.",
    )
    parser.add_argument(
        "--output-substring",
        default="",
        help="Optional suffix appended to the timestamped output directory name.",
    )
    return parser.parse_args()


def _resolve_prompt_from_generated_candidates(
    payload: Dict[str, Any],
    prompt_index: int | None,
) -> Tuple[str, Dict[str, Any]]:
    if prompt_index is None:
        raise ValueError("--prompt-index is required for generated_prompt_candidates.json")

    prompts = payload.get("candidate_prompts", [])
    metadata_list = payload.get("candidate_metadata", [])
    if not 0 <= prompt_index < len(prompts):
        raise IndexError(
            f"prompt_index={prompt_index} is out of range for {len(prompts)} prompts."
        )

    metadata: Dict[str, Any] = {}
    candidate_index = prompt_index + 1
    for item in metadata_list:
        if item.get("candidate_index") == candidate_index:
            metadata = item
            break
    
    # todo: load prf scores here too form the json

    return prompts[prompt_index], {
        "prompt_source_type": "generated_prompt_candidates",
        "prompt_index": prompt_index,
        "candidate_index": candidate_index,
        "candidate_metadata": metadata,
    }


def _resolve_prompt_from_population(
    payload: Dict[str, Any],
    prompt_node_id: int | None,
) -> Tuple[str, Dict[str, Any]]:
    if prompt_node_id is None:
        raise ValueError("--prompt-node-id is required for population.json")

    population = payload.get("population", [])
    for node in population:
        if node.get("node_id") == prompt_node_id:
            return node.get("inference_instruction_prompt", ""), {
                "prompt_source_type": "population",
                "prompt_node_id": prompt_node_id,
                "best_node_id": payload.get("best_node_id"),
                "node_summary": {
                    "node_id": node.get("node_id"),
                    "parent_id": node.get("parent_id"),
                    "children_ids": node.get("children_ids", []),
                    "inference_mode": node.get("inference_mode"),
                    "val_score": node.get("val_score"),
                },
            }
    raise KeyError(f"Could not find node_id={prompt_node_id} in population.json")


def resolve_instruction_prompt(
    prompt_source_path: Path,
    *,
    prompt_index: int | None,
    prompt_node_id: int | None,
) -> Tuple[str, Dict[str, Any]]:
    payload = load_json_file(prompt_source_path)
    if prompt_source_path.name == "generated_prompt_candidates.json":
        return _resolve_prompt_from_generated_candidates(payload, prompt_index)
    if prompt_source_path.name == "population.json":
        return _resolve_prompt_from_population(payload, prompt_node_id)
    raise ValueError(
        "Unsupported prompt source file. Expected generated_prompt_candidates.json "
        "or population.json."
    )

def _resolve_num_ways(episodes: Sequence[Dict[str, Any]]) -> int:
    if not episodes:
        raise ValueError("Dataset episodes are empty.")
    num_ways = len(episodes[0]["meta_train"])
    if num_ways <= 0:
        raise ValueError("Episodes contain no candidate ways.")
    return num_ways


def _pair_index_to_episode_way(pair_index: int, num_ways: int) -> Tuple[int, int]:
    return divmod(pair_index, num_ways)


def _pair_confusion_bucket(label: str, prediction: str) -> str:
    if label == "yes" and prediction == "yes":
        return "tp"
    if label == "no" and prediction == "no":
        return "tn"
    if label == "no" and prediction == "yes":
        return "fp"
    if label == "yes" and prediction == "no":
        return "fn"
    raise ValueError(f"Unsupported label/prediction pair: label={label}, prediction={prediction}")


def _build_prf_scores(
    *,
    predicted_labels: Sequence[str],
    target_labels: Sequence[str],
) -> Dict[str, float]:
    mapped_target_labels = [
        "relation" if label == "yes" else NO_RELATION for label in target_labels
    ]
    mapped_predicted_labels = [
        "relation" if label == "yes" else NO_RELATION for label in predicted_labels
    ]
    stats = compute_prf_stats(mapped_target_labels, mapped_predicted_labels, n_chunks=3)
    return {
        "precision": float(stats["precision_mean"] * 100.0),
        "precision_std": float(stats["precision_std"] * 100.0),
        "recall": float(stats["recall_mean"] * 100.0),
        "recall_std": float(stats["recall_std"] * 100.0),
        "f1": float(stats["f1_mean"] * 100.0),
        "f1_std": float(stats["f1_std"] * 100.0),
    }


def _build_episode_prf_scores(
    *,
    binary_pairs: Sequence[Dict[str, Any]],
    predicted_labels: Sequence[str],
) -> Dict[str, float]:
    if len(binary_pairs) != len(predicted_labels):
        raise ValueError("binary_pairs/predicted_labels length mismatch")

    episode_to_rows: Dict[int, List[Tuple[int, str, str, str]]] = {}
    for pair, predicted_label in zip(binary_pairs, predicted_labels):
        episode_index = pair.get("episode_index")
        if episode_index is None:
            raise ValueError("Episode-level PRF requires pairs with episode_index.")
        way_index = pair.get("way_index")
        relation = pair["support"]["relation"]
        episode_to_rows.setdefault(int(episode_index), []).append(
            (int(way_index), relation, pair["label"], predicted_label)
        )

    episode_labels: List[str] = []
    episode_predictions: List[str] = []
    for episode_index in sorted(episode_to_rows):
        rows = sorted(episode_to_rows[episode_index], key=lambda item: item[0])
        gold_relation = NO_RELATION
        predicted_relation = NO_RELATION
        for _, relation, target_label, predicted_label in rows:
            if target_label == "yes":
                gold_relation = relation
            if predicted_relation == NO_RELATION and predicted_label == "yes":
                predicted_relation = relation
        episode_labels.append(gold_relation)
        episode_predictions.append(predicted_relation)

    stats = compute_prf_stats(episode_labels, episode_predictions, n_chunks=3)
    return {
        "precision": float(stats["precision_mean"] * 100.0),
        "precision_std": float(stats["precision_std"] * 100.0),
        "recall": float(stats["recall_mean"] * 100.0),
        "recall_std": float(stats["recall_std"] * 100.0),
        "f1": float(stats["f1_mean"] * 100.0),
        "f1_std": float(stats["f1_std"] * 100.0),
    }


def _extract_prf_from_prompt_info(prompt_info: Dict[str, Any]) -> Dict[str, float] | None:
    score_sources: List[Any] = []
    node_summary = prompt_info.get("node_summary")
    if isinstance(node_summary, dict):
        score_sources.append(node_summary.get("val_score"))
    candidate_metadata = prompt_info.get("candidate_metadata")
    if isinstance(candidate_metadata, dict):
        score_sources.extend(
            [
                candidate_metadata.get("val_score"),
                candidate_metadata.get("dev_prf"),
                (candidate_metadata.get("full_evaluation") or {}).get("prf")
                if isinstance(candidate_metadata.get("full_evaluation"), dict)
                else None,
            ]
        )
    score_sources.extend(
        [
            prompt_info.get("dev_prf"),
            (prompt_info.get("full_evaluation") or {}).get("prf")
            if isinstance(prompt_info.get("full_evaluation"), dict)
            else None,
        ]
    )
    val_score = next((item for item in score_sources if isinstance(item, dict)), None)
    if not isinstance(val_score, dict):
        return None

    def _score_value(*keys: str) -> float:
        for key in keys:
            if key in val_score and val_score.get(key) is not None:
                return float(val_score[key])
        return 0.0

    prf = {
        "precision": _score_value("precision", "precision_mean"),
        "precision_std": _score_value("precision_std"),
        "recall": _score_value("recall", "recall_mean"),
        "recall_std": _score_value("recall_std"),
        "f1": _score_value("f1", "f1_mean"),
        "f1_std": _score_value("f1_std"),
    }
    if max(prf["precision"], prf["recall"], prf["f1"]) <= 1.0:
        prf = {key: value * 100.0 for key, value in prf.items()}
    return prf


def _format_prf_for_log(prf: Dict[str, float]) -> str:
    return (
        f"precision={prf.get('precision', 0.0):.2f} ± {prf.get('precision_std', 0.0):.2f}, "
        f"recall={prf.get('recall', 0.0):.2f} ± {prf.get('recall_std', 0.0):.2f}, "
        f"f1={prf.get('f1', 0.0):.2f} ± {prf.get('f1_std', 0.0):.2f}"
    )


def _format_stable_f1_for_log(
    prf: Dict[str, float],
    *,
    f1_std_penalty: float,
) -> str:
    return (
        f"stable_f1={stable_prf_score_or_neg_inf(prf, std_multiplier=f1_std_penalty):.2f} "
        f"(k={f1_std_penalty:g})"
    )


def _format_log_context(log_context: Dict[str, Any] | None) -> str:
    if not log_context:
        return ""
    parts = [
        f"{key}={value}"
        for key, value in log_context.items()
        if value is not None
    ]
    return " ".join(parts)


def _build_binary_pairs_from_feedback_samples(
    *,
    feedback_samples,
    shots_by_id: Dict[str, dict],
    dataset_type: str,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for pair_index, sample in enumerate(feedback_samples.all_samples):
        support = shots_by_id[sample.id_1shot]
        query = shots_by_id[sample.id_query]
        label = sample.label
        pairs.append(
            {
                "pair_index": pair_index,
                "episode_index": None,
                "way_index": None,
                "support": support,
                "query": query,
                "label": label,
                "prediction": None,
                "confusion_bucket": None,
                "support_sentence": get_sentence_with_tags(support).strip(),
                "query_sentence": get_sentence_with_tags(query).strip(),
                "relation_description": get_relation_description(
                    support["relation"],
                    dt=dataset_type,
                ),
            }
        )
    return pairs


def _build_train_sample_index(train_samples: Dict[str, Any]) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for instances in train_samples.values():
        if not isinstance(instances, list):
            continue
        for inst in instances:
            instance_id = inst.get("id")
            if instance_id is not None:
                index[instance_id] = inst
    return index
def _attach_predictions_to_pairs(
    *,
    pairs: Sequence[Dict[str, Any]],
    predicted_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    if len(pairs) != len(predicted_labels):
        raise ValueError("pairs/predicted_labels length mismatch")

    updated_pairs: List[Dict[str, Any]] = []
    for pair, predicted_label in zip(pairs, predicted_labels):
        updated_pair = dict(pair)
        updated_pair["prediction"] = predicted_label
        updated_pair["confusion_bucket"] = _pair_confusion_bucket(
            updated_pair["label"],
            predicted_label,
        )
        updated_pairs.append(updated_pair)
    return updated_pairs


def _sample_balanced_pairs_from_bucketed_pool(
    *,
    bucketed_pairs: Sequence[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "tp": [],
        "tn": [],
        "fp": [],
        "fn": [],
    }
    for pair in bucketed_pairs:
        buckets[pair["confusion_bucket"]].append(pair)

    selected_per_bucket = min(len(buckets["tp"]), len(buckets["tn"]), len(buckets["fp"]), len(buckets["fn"]))
    if selected_per_bucket <= 0:
        raise ValueError(
            "Unable to sample balanced tp/tn/fp/fn buckets from the train D set. "
            f"available_tp={len(buckets['tp'])}, available_tn={len(buckets['tn'])}, "
            f"available_fp={len(buckets['fp'])}, available_fn={len(buckets['fn'])}."
        )

    rng = random.Random(seed)
    selected_pairs: List[Dict[str, Any]] = []
    selected_counts: Dict[str, int] = {}
    for bucket_name in ("tp", "tn", "fp", "fn"):
        chosen = rng.sample(buckets[bucket_name], selected_per_bucket)
        selected_counts[bucket_name] = len(chosen)
        selected_pairs.extend(chosen)

    rng.shuffle(selected_pairs)
    bucket_stats = {
        "available": {bucket_name: len(bucket) for bucket_name, bucket in buckets.items()},
        "selected": selected_counts,
    }
    return selected_pairs, bucket_stats


def _build_evaluation_report(
    *,
    score_payload: Dict[str, Any],
    binary_pairs: Sequence[Dict[str, Any]] | None = None,
    prf_mode: str = "binary",
) -> Dict[str, Any]:
    confusion_matrix = _build_confusion_matrix_counts(
        predicted_labels=score_payload["predicted_labels"],
        target_labels=score_payload["target_labels"],
    )
    if prf_mode == "episode":
        if binary_pairs is None:
            raise ValueError("binary_pairs are required for episode-level PRF.")
        prf = _build_episode_prf_scores(
            binary_pairs=binary_pairs,
            predicted_labels=score_payload["predicted_labels"],
        )
    else:
        prf = _build_prf_scores(
            predicted_labels=score_payload["predicted_labels"],
            target_labels=score_payload["target_labels"],
        )
    return {
        "predicted_labels": score_payload["predicted_labels"],
        "target_labels": score_payload["target_labels"],
        "confusion_matrix": confusion_matrix,
        "prf": prf,
    }


def _build_score_payload_from_pairs(
    pairs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "predicted_labels": [pair["prediction"] for pair in pairs],
        "target_labels": [pair["label"] for pair in pairs],
    }


_LABEL_LITERAL_WORDS = {"yes", "no"}
_SMART_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
)
_LABEL_LITERAL_STRIP_CHARS = " \t\r\n\"'`.,:;()[]{}<>\u201c\u201d\u2018\u2019"
_LABEL_LITERAL_QUOTE_CHARS = {"\"", "'", "`", "\u201c", "\u201d", "\u2018", "\u2019"}


def _has_alphanumeric(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _normalize_quote_chars(text: Any) -> str:
    return str(text).translate(_SMART_QUOTE_TRANSLATION)


def _normalized_label_literal(text: str) -> str:
    return _normalize_quote_chars(text).strip().lower().strip(_LABEL_LITERAL_STRIP_CHARS)


def _is_quote_only_span(text: str) -> bool:
    stripped = _normalize_quote_chars(text).strip()
    return bool(stripped) and all(char in _LABEL_LITERAL_QUOTE_CHARS for char in stripped)


def _is_protected_yes_no_span(
    *,
    instruction_prompt: str,
    start_char: int,
    end_char: int,
) -> bool:
    token_text = instruction_prompt[start_char:end_char]
    core = _normalized_label_literal(token_text)
    if core not in _LABEL_LITERAL_WORDS:
        return False

    left = _normalize_quote_chars(instruction_prompt[max(0, start_char - 8):start_char])
    right = _normalize_quote_chars(instruction_prompt[end_char:end_char + 8])
    window = _normalize_quote_chars(
        instruction_prompt[max(0, start_char - 24):end_char + 24]
    ).lower()
    left_stripped = left.rstrip().lower()
    right_stripped = right.lstrip().lower()

    quoted = (
        left.rstrip().endswith(("\"", "'", "`"))
        and right.lstrip().startswith(("\"", "'", "`"))
    )
    slash_option = (
        right_stripped.startswith(("/yes", "/no"))
        or left_stripped.endswith(("yes/", "no/"))
    )
    punctuation_option = right_stripped.startswith((".", ",", ":", ";", ")", "]", "}"))
    local_label_pair = (
        "yes" in window
        and "no" in window
        and (" or " in window or "/" in window)
    )

    return quoted or slash_option or punctuation_option or local_label_pair


def _editable_token_groups_for_region(
    *,
    region: Dict[str, Any],
    offsets: Sequence[Sequence[int]],
    instruction_prompt: str,
) -> List[List[int]]:
    start_token = min(region["start"], len(offsets) - 1)
    end_token = min(region["end"], len(offsets) - 1)
    token_indices = [
        idx for idx in range(start_token, end_token + 1)
        if int(offsets[idx][1]) > int(offsets[idx][0])
    ]

    editable_groups: List[List[int]] = []
    current_group: List[int] = []
    remove_token_indices = set()
    for token_index in token_indices:
        start_char = int(offsets[token_index][0])
        end_char = int(offsets[token_index][1])
        if _is_protected_yes_no_span(
            instruction_prompt=instruction_prompt,
            start_char=start_char,
            end_char=end_char,
        ):
            remove_token_indices.add(token_index)
            for neighbor_index in (token_index - 1, token_index + 1):
                if neighbor_index not in token_indices:
                    continue
                neighbor_text = instruction_prompt[
                    int(offsets[neighbor_index][0]):int(offsets[neighbor_index][1])
                ]
                if _is_quote_only_span(neighbor_text):
                    remove_token_indices.add(neighbor_index)
            while current_group and current_group[-1] in remove_token_indices:
                current_group.pop()
            if current_group:
                editable_groups.append(current_group)
                current_group = []
            continue
        if token_index in remove_token_indices:
            if current_group:
                editable_groups.append(current_group)
                current_group = []
            continue
        current_group.append(token_index)

    if current_group:
        editable_groups.append(current_group)

    return [
        group for group in editable_groups
        if _has_alphanumeric(
            instruction_prompt[
                int(offsets[group[0]][0]):int(offsets[group[-1]][1])
            ]
        )
    ]


def _token_gradient_score_by_index(gradient_results: Dict[str, Any]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for record in gradient_results.get("token_gradients", []):
        if not isinstance(record, dict):
            continue
        token_index = record.get("token_index")
        gradient_norm = record.get("gradient_norm")
        if token_index is None or gradient_norm is None:
            continue
        scores[int(token_index)] = float(gradient_norm)
    return scores


def _trim_region_to_alphanumeric_edges(
    *,
    instruction_prompt: str,
    start_char: int,
    end_char: int,
) -> Dict[str, Any] | None:
    """Trim punctuation/space edges while leaving the surrounding prompt intact."""
    trimmed_start = start_char
    trimmed_end = end_char
    while trimmed_start < trimmed_end and not instruction_prompt[trimmed_start].isalnum():
        trimmed_start += 1
    while trimmed_end > trimmed_start and not instruction_prompt[trimmed_end - 1].isalnum():
        trimmed_end -= 1
    if trimmed_start >= trimmed_end:
        return None
    trimmed_text = instruction_prompt[trimmed_start:trimmed_end]
    if not _has_alphanumeric(trimmed_text):
        return None
    return {
        "start_char": trimmed_start,
        "end_char": trimmed_end,
        "region_text": trimmed_text,
        "trimmed_left_chars": instruction_prompt[start_char:trimmed_start],
        "trimmed_right_chars": instruction_prompt[trimmed_end:end_char],
    }


def _resolve_region_details(
    *,
    instruction_prompt: str,
    tokenizer,
    gradient_results: Dict[str, Any],
    num_edit_regions: int,
) -> Dict[str, Any]:
    top_regions = gradient_results.get("top_regions", [])
    if not top_regions:
        raise ValueError("No top_regions were returned by the gradient analysis.")
    if num_edit_regions <= 0:
        raise ValueError("--num-edit-regions must be positive.")

    encoded = tokenizer(
        instruction_prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoded["offset_mapping"]
    input_ids = encoded["input_ids"]
    if not offsets:
        raise ValueError("Instruction prompt produced no tokenizer offsets.")

    gradient_scores_by_token = _token_gradient_score_by_index(gradient_results)
    candidate_regions: List[Dict[str, Any]] = []
    for source_region_rank, region in enumerate(top_regions, start=1):
        editable_groups = _editable_token_groups_for_region(
            region=region,
            offsets=offsets,
            instruction_prompt=instruction_prompt,
        )
        for token_group in editable_groups:
            start_token = token_group[0]
            end_token = token_group[-1]
            original_start_char = int(offsets[start_token][0])
            original_end_char = int(offsets[end_token][1])
            trimmed_region = _trim_region_to_alphanumeric_edges(
                instruction_prompt=instruction_prompt,
                start_char=original_start_char,
                end_char=original_end_char,
            )
            if trimmed_region is None:
                continue
            kept_gradient_scores = [
                gradient_scores_by_token.get(token_index, 0.0)
                for token_index in token_group
            ]
            candidate_regions.append(
                {
                    "source_region_rank": source_region_rank,
                    "start_token": start_token,
                    "end_token": end_token,
                    "original_start_char": original_start_char,
                    "original_end_char": original_end_char,
                    "original_region_text": instruction_prompt[
                        original_start_char:original_end_char
                    ],
                    "start_char": trimmed_region["start_char"],
                    "end_char": trimmed_region["end_char"],
                    "region_text": trimmed_region["region_text"],
                    "region_position_mode": "character",
                    "trimmed_left_chars": trimmed_region["trimmed_left_chars"],
                    "trimmed_right_chars": trimmed_region["trimmed_right_chars"],
                    "region_score": max(kept_gradient_scores) if kept_gradient_scores else 0.0,
                    "region_score_method": "max_kept_token_gradient_norm",
                    "source_region_score": region["score"],
                    "region_tokens": tokenizer.convert_ids_to_tokens(
                        [input_ids[token_index] for token_index in token_group]
                    ),
                    "region_token_indices": token_group,
                    "protected_yes_no_filter_applied": True,
                }
            )

    if not candidate_regions:
        raise ValueError(
            "No editable top_regions remained after protecting yes/no label tokens."
        )
    candidate_regions.sort(
        key=lambda item: (
            item["region_score"],
            -int(item["source_region_rank"]),
            -int(item["start_token"]),
        ),
        reverse=True,
    )
    selected_regions = candidate_regions[:num_edit_regions]
    for region_rank, region in enumerate(selected_regions, start=1):
        region["region_rank"] = region_rank

    marked_prompt = instruction_prompt
    for region in sorted(selected_regions, key=lambda item: item["start_char"], reverse=True):
        marked_prompt = (
            marked_prompt[: region["start_char"]]
            + f"<edit_start_{region['region_rank']}>"
            + marked_prompt[region["start_char"] : region["end_char"]]
            + f"<edit_end_{region['region_rank']}>"
            + marked_prompt[region["end_char"] :]
        )

    return {
        "num_edit_regions": len(selected_regions),
        "selected_regions": selected_regions,
        "marked_prompt": marked_prompt,
    }


def _build_gradient_region_meta_prompt(
    *,
    instruction_prompt: str,
    region_details: Dict[str, Any],
) -> str:
    prompt = GRADIENT_REGION_MUTATION_PROMPT_V1
    prompt = prompt.replace("#INFERENCE_PROMPT#", instruction_prompt)
    prompt = prompt.replace("#MARKED_PROMPT#", region_details["marked_prompt"])
    return prompt


def _build_multi_region_marked_prompt(
    *,
    instruction_prompt: str,
    selected_regions: Sequence[Dict[str, Any]],
) -> str:
    marked_prompt = instruction_prompt
    for region in sorted(selected_regions, key=lambda item: item["start_char"], reverse=True):
        region_rank = int(region["region_rank"])
        marked_prompt = (
            marked_prompt[: region["start_char"]]
            + f"<span_{region_rank}>"
            + marked_prompt[region["start_char"] : region["end_char"]]
            + f"</span_{region_rank}>"
            + marked_prompt[region["end_char"] :]
        )
    return marked_prompt


def _build_region_candidate_request_blocks(
    *,
    selected_regions: Sequence[Dict[str, Any]],
) -> str:
    blocks: List[str] = []
    for region in selected_regions:
        blocks.append(
            "\n".join(
                [
                    f"Span {region['region_rank']}",
                    f"Text: ```{region['region_text']}```",
                ]
            )
        )
    return "\n\n".join(blocks)


def _build_region_candidate_meta_prompt(
    *,
    instruction_prompt: str,
    region_details: Dict[str, Any],
    num_region_candidates: int,
) -> str:
    prompt = GRADIENT_REGION_CANDIDATE_SUGGESTION_PROMPT_V1
    prompt = prompt.replace(
        "#MARKED_PROMPT#",
        _build_multi_region_marked_prompt(
            instruction_prompt=instruction_prompt,
            selected_regions=region_details["selected_regions"],
        ),
    )
    prompt = prompt.replace(
        "#REGION_CANDIDATE_REQUEST_BLOCKS#",
        _build_region_candidate_request_blocks(
            selected_regions=region_details["selected_regions"],
        ),
    )
    prompt = prompt.replace("#NUM_CANDIDATES#", str(num_region_candidates))
    prompt = prompt.replace("#NUM_REGIONS#", str(region_details["num_edit_regions"]))
    # prompt += (
    #     "\n\nJSON formatting requirement: use valid JSON only, with ASCII double quotes "
    #     '(") for every key and string delimiter. Do not use curly quotes as JSON delimiters.'
    # )
    return prompt


def _build_region_candidate_blocks(
    *,
    selected_regions: Sequence[Dict[str, Any]],
    region_candidates: Sequence[Dict[str, Any]],
) -> str:
    candidate_index = {
        int(item["region_rank"]): item for item in region_candidates
    }
    blocks: List[str] = []
    for region in selected_regions:
        candidates = candidate_index.get(int(region["region_rank"]), {}).get("candidates", [])
        candidate_list_text = json.dumps(candidates if candidates else [], ensure_ascii=False)
        blocks.append(
            "\n".join(
                [
                    f"Span {region['region_rank']}",
                    f"Text: ```{region['region_text']}```",
                    f"Candidates: {candidate_list_text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _build_region_candidate_combination_prompt(
    *,
    instruction_prompt: str,
    region_details: Dict[str, Any],
    region_candidates: Sequence[Dict[str, Any]],
    num_generated_prompts: int,
) -> str:
    prompt = GRADIENT_REGION_CANDIDATE_COMBINATION_PROMPT_V1
    prompt = prompt.replace(
        "#ALL_MARKED_PROMPT#",
        _build_multi_region_marked_prompt(
            instruction_prompt=instruction_prompt,
            selected_regions=region_details["selected_regions"],
        ),
    )
    prompt = prompt.replace(
        "#REGION_CANDIDATE_BLOCKS#",
        _build_region_candidate_blocks(
            selected_regions=region_details["selected_regions"],
            region_candidates=region_candidates,
        ),
    )
    prompt = prompt.replace("#NUM_PROMPT#", str(num_generated_prompts))
    prompt = prompt.replace("#NUM_REGIONS#", str(region_details["num_edit_regions"]))
    return prompt


def _build_selected_replacement_blocks(
    *,
    selected_regions: Sequence[Dict[str, Any]],
    selected_replacements: Dict[int, str],
) -> str:
    blocks: List[str] = []
    for region in selected_regions:
        region_rank = int(region["region_rank"])
        blocks.append(
            "\n".join(
                [
                    f"Span {region_rank}",
                    f"Text: ```{region['region_text']}```",
                    "Replace with: "
                    f"```{selected_replacements.get(region_rank, region['region_text'])}```",
                ]
            )
        )
    return "\n\n".join(blocks)


def _build_region_candidate_synthesis_prompt(
    *,
    instruction_prompt: str,
    region_details: Dict[str, Any],
    selected_replacements: Dict[int, str],
) -> str:
    prompt = GRADIENT_REGION_CANDIDATE_SYNTHESIS_PROMPT_V1
    prompt = prompt.replace(
        "#ALL_MARKED_PROMPT#",
        _build_multi_region_marked_prompt(
            instruction_prompt=instruction_prompt,
            selected_regions=region_details["selected_regions"],
        ),
    )
    prompt = prompt.replace(
        "#SELECTED_REPLACEMENTS#",
        _build_selected_replacement_blocks(
            selected_regions=region_details["selected_regions"],
            selected_replacements=selected_replacements,
        ),
    )
    return prompt


def _prepare_region_candidates_for_beam_search(
    *,
    region_details: Dict[str, Any],
    region_candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidate_index = {
        int(item["region_rank"]): item for item in region_candidates
    }
    prepared_candidates: List[Dict[str, Any]] = []
    for region in region_details["selected_regions"]:
        region_rank = int(region["region_rank"])
        original_text = region["region_text"]
        raw_candidates = candidate_index.get(region_rank, {}).get("candidates", [])
        original_candidate = _align_replacement_whitespace_to_region(
            region_text=original_text,
            replacement_text=original_text,
        )
        candidates: List[str] = [original_candidate]
        seen_normalized: set[str] = {original_candidate.strip().casefold()}
        for candidate_text in raw_candidates:
            normalized = _normalize_span_replacement_text(candidate_text)
            if not normalized:
                continue
            if _normalized_label_literal(normalized) in _LABEL_LITERAL_WORDS:
                continue
            aligned_candidate = _align_replacement_whitespace_to_region(
                region_text=original_text,
                replacement_text=normalized,
            )
            normalized_candidate = aligned_candidate.strip().casefold()
            if normalized_candidate in seen_normalized:
                continue
            seen_normalized.add(normalized_candidate)
            candidates.append(aligned_candidate)
        prepared_candidates.append(
            {
                **copy.deepcopy(candidate_index.get(region_rank, {})),
                "region_rank": region_rank,
                "region_text": original_text,
                "candidates": candidates,
            }
        )
    return prepared_candidates


def _align_replacement_whitespace_to_region(
    *,
    region_text: str,
    replacement_text: str,
) -> str:
    leading_whitespace_length = len(region_text) - len(region_text.lstrip())
    trailing_whitespace_length = len(region_text) - len(region_text.rstrip())
    leading_whitespace = region_text[:leading_whitespace_length]
    trailing_whitespace = (
        region_text[len(region_text) - trailing_whitespace_length :]
        if trailing_whitespace_length > 0
        else ""
    )
    return f"{leading_whitespace}{replacement_text.strip()}{trailing_whitespace}"


def _build_prompt_from_region_replacements( #todo: we should prompt to build the prompt?
    *,
    instruction_prompt: str,
    selected_regions: Sequence[Dict[str, Any]],
    selected_replacements: Dict[int, str],
) -> str:
    revised_prompt = instruction_prompt
    for region in sorted(selected_regions, key=lambda item: item["start_char"], reverse=True):
        region_rank = int(region["region_rank"])
        replacement_text = selected_replacements.get(region_rank)
        if replacement_text is None:
            continue
        replacement_text = _align_replacement_whitespace_to_region(
            region_text=region["region_text"],
            replacement_text=replacement_text,
        )
        revised_prompt = (
            revised_prompt[: region["start_char"]]
            + replacement_text
            + revised_prompt[region["end_char"] :]
        )
    return revised_prompt


def _normalize_synthesized_prompt_output(raw_output: str) -> str:
    text = raw_output.strip()
    fenced_match = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()
    return _strip_region_markup(text)


def _build_complete_region_replacements(
    *,
    selected_regions: Sequence[Dict[str, Any]],
    selected_replacements: Dict[int, str],
) -> Dict[int, str]:
    return {
        int(region["region_rank"]): selected_replacements.get(
            int(region["region_rank"]),
            region["region_text"],
        )
        for region in selected_regions
    }


def _run_multi_region_candidate_synthesis(
    *,
    instruction_prompt: str,
    region_details: Dict[str, Any],
    selected_replacements: Dict[int, str],
    model,
    tokenizer,
    max_new_tokens: int,
    batch_size: int,
    use_chat_template: bool,
) -> Dict[str, Any]:
    complete_replacements = _build_complete_region_replacements(
        selected_regions=region_details["selected_regions"],
        selected_replacements=selected_replacements,
    )
    meta_prompt = _build_region_candidate_synthesis_prompt(
        instruction_prompt=instruction_prompt,
        region_details=region_details,
        selected_replacements=complete_replacements,
    )
    raw_output = run_prompts(
        [meta_prompt],
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        do_sample=True,
        do_log=True,
        log_label="agent_gradient_eval_debug_region_synthesis",
    )[0]
    revised_prompt = _normalize_synthesized_prompt_output(raw_output)
    fallback_reason = None
    if not revised_prompt:
        fallback_reason = "empty_synthesis_output"
        revised_prompt = _build_prompt_from_region_replacements(
            instruction_prompt=instruction_prompt,
            selected_regions=region_details["selected_regions"],
            selected_replacements=selected_replacements,
        )

    return {
        "revised_prompt": revised_prompt,
        "meta_prompt": meta_prompt,
        "raw_output": raw_output,
        "used_fallback": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "complete_replacements": complete_replacements,
    }


def _build_beam_candidate_prompt(
    *,
    replacement_mode: str,
    instruction_prompt: str,
    region_details: Dict[str, Any],
    selected_replacements: Dict[int, str],
    region_rank: int,
    model,
    tokenizer,
    max_new_tokens: int,
    batch_size: int,
    use_chat_template: bool,
    synthesis_cache: Dict[Tuple[Tuple[int, str], ...], Dict[str, Any]],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "replacement_mode": replacement_mode,
        "region_rank": region_rank,
        "used_fallback": False,
        "fallback_reason": None,
        "meta_prompt": None,
        "raw_output": None,
        "complete_replacements": _build_complete_region_replacements(
            selected_regions=region_details["selected_regions"],
            selected_replacements=selected_replacements,
        ),
    }

    if not selected_replacements:
        return {
            "revised_prompt": instruction_prompt,
            "replacement_metadata": {
                **metadata,
                "no_op": True,
            },
        }

    if replacement_mode == BEAM_REPLACEMENT_MODE_DIRECT_REPLACEMENT:
        revised_prompt = _build_prompt_from_region_replacements(
            instruction_prompt=instruction_prompt,
            selected_regions=region_details["selected_regions"],
            selected_replacements=selected_replacements,
        )
        return {
            "revised_prompt": revised_prompt,
            "replacement_metadata": {
                **metadata,
                "no_op": False,
            },
        }

    if replacement_mode != BEAM_REPLACEMENT_MODE_LLM_SYNTHESIS:
        raise ValueError(f"Unsupported beam replacement mode: {replacement_mode}")

    cache_key = tuple(sorted((int(rank), text) for rank, text in selected_replacements.items()))
    cached_synthesis = synthesis_cache.get(cache_key)
    if cached_synthesis is None:
        cached_synthesis = _run_multi_region_candidate_synthesis(
            instruction_prompt=instruction_prompt,
            region_details=region_details,
            selected_replacements=selected_replacements,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            use_chat_template=use_chat_template,
        )
        synthesis_cache[cache_key] = copy.deepcopy(cached_synthesis)

    revised_prompt = cached_synthesis["revised_prompt"]
    return {
        "revised_prompt": revised_prompt,
        "replacement_metadata": {
            **metadata,
            "no_op": False,
            "complete_replacements": cached_synthesis.get("complete_replacements"),
            "meta_prompt": cached_synthesis.get("meta_prompt"),
            "raw_output": cached_synthesis.get("raw_output"),
            "used_fallback": bool(cached_synthesis.get("used_fallback")),
            "fallback_reason": cached_synthesis.get("fallback_reason"),
        },
    }


def _build_selected_replacements_payload(
    *,
    selected_regions: Sequence[Dict[str, Any]],
    selected_replacements: Dict[int, str],
) -> List[Dict[str, Any]]:
    return [
        {
            "region_rank": int(region["region_rank"]),
            "region_text": region["region_text"],
            "replacement_text": _align_replacement_whitespace_to_region(
                region_text=region["region_text"],
                replacement_text=selected_replacements.get(
                    int(region["region_rank"]),
                    region["region_text"],
                ),
            ),
        }
        for region in selected_regions
    ]


def _normalize_span_replacement_text(value: Any) -> str:
    text = _normalize_quote_chars(value).strip()
    if text.startswith("```") and text.endswith("```") and len(text) >= 6:
        text = text[3:-3].strip()
    elif text.startswith("`") and text.endswith("`") and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def _sort_key_with_numeric_suffix(value: Any) -> Tuple[str, int]:
    text = str(value)
    match = re.search(r"(\d+)$", text)
    if match:
        return text[: match.start()], int(match.group(1))
    return text, 0


def _extract_region_candidate_values(
    *,
    parsed_output: Dict[str, Any],
    region_rank: int,
) -> List[Any]:
    region_payload = parsed_output.get(f"span_{region_rank}")
    if not isinstance(region_payload, dict):
        return []

    candidate_values = region_payload.get("candidates")
    if not isinstance(candidate_values, list):
        return []

    return list(candidate_values)


def _extract_quoted_candidate_items(array_text: str) -> List[str]:
    quote_chars = {'"', "\u201c", "\u201d"}
    line_candidates: List[str] = []
    for line in array_text.splitlines():
        item = line.strip()
        if not item:
            continue
        if item.endswith(","):
            item = item[:-1].rstrip()
        while item and item[0] in quote_chars:
            item = item[1:].lstrip()
        while item and item[-1] in quote_chars:
            item = item[:-1].rstrip()
        if item:
            line_candidates.append(item)
    if line_candidates:
        return line_candidates

    candidates: List[str] = []
    index = 0
    while index < len(array_text):
        while index < len(array_text) and array_text[index] in " \t\r\n,":
            index += 1
        if index >= len(array_text):
            break

        quote_char = array_text[index]
        if quote_char not in {'"', "\u201c", "\u201d"}:
            next_comma = array_text.find(",", index)
            if next_comma < 0:
                break
            index = next_comma + 1
            continue

        closing_quote = '"' if quote_char == '"' else "\u201d"
        index += 1
        chars: List[str] = []
        escaped = False
        while index < len(array_text):
            char = array_text[index]
            if escaped:
                chars.append(char)
                escaped = False
                index += 1
                continue
            if char == "\\":
                escaped = True
                index += 1
                continue
            if char == closing_quote:
                index += 1
                break
            chars.append(char)
            index += 1
        candidates.append("".join(chars))

        while index < len(array_text) and array_text[index] not in ",":
            if array_text[index] not in " \t\r\n":
                break
            index += 1
        if index < len(array_text) and array_text[index] == ",":
            index += 1
    return candidates


def _extract_region_candidate_values_from_raw_output(
    *,
    raw_output: str,
    region_rank: int,
) -> List[str]:
    span_pattern = re.compile(
        rf'"span_{region_rank}"\s*:\s*\{{.*?"candidates"\s*:\s*\[(.*?)\]',
        flags=re.DOTALL,
    )
    match = span_pattern.search(raw_output)
    if not match:
        return []
    return _extract_quoted_candidate_items(match.group(1))


def _strip_region_markup(text: str) -> str:
    cleaned = re.sub(r"</?target>", "", text)
    cleaned = re.sub(r"</?span_\d+>", "", cleaned)
    cleaned = re.sub(r"</?span>", "", cleaned)
    return cleaned.strip()


def _model_device(model) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("Model has no parameters.")
    return parameter.device


def _decode_token_text(
    *,
    tokenizer,
    token_ids: Sequence[int],
) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _collect_top_token_ids_from_prefix(
    *,
    prefix_token_ids: Sequence[int],
    model,
    tokenizer,
    num_candidate_pool_tokens: int,
) -> List[int]:
    if not prefix_token_ids:
        raise ValueError("prefix_token_ids must not be empty for probability-based token selection.")
    if num_candidate_pool_tokens <= 0:
        return []

    device = _model_device(model)
    input_ids = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False) # todo: clear cuda cache
    next_token_logits = outputs.logits[0, -1, :]
    vocab_size = int(next_token_logits.size(0))
    special_token_ids = set(getattr(tokenizer, "all_special_ids", []))

    search_budget = min(vocab_size, max(num_candidate_pool_tokens * 8, num_candidate_pool_tokens))
    collected_ids: List[int] = []
    while True:
        topk_indices = torch.topk(next_token_logits, k=search_budget).indices.tolist()
        collected_ids = []
        seen_ids: set[int] = set()
        for token_id in topk_indices:
            token_id = int(token_id)
            if token_id in seen_ids or token_id in special_token_ids:
                continue
            token_text = _decode_token_text(tokenizer=tokenizer, token_ids=[token_id])
            if token_text == "":
                continue
            seen_ids.add(token_id)
            collected_ids.append(token_id)
            if len(collected_ids) == num_candidate_pool_tokens:
                break
        if len(collected_ids) >= num_candidate_pool_tokens or search_budget == vocab_size:
            break
        search_budget = min(vocab_size, search_budget * 2)
    del input_ids, outputs, next_token_logits
    clear_cuda_cache()
    return collected_ids


def _build_probability_region_prompt_token_ids(
    *,
    instruction_prompt: str,
    tokenizer,
) -> List[int]:
    encoded = tokenizer(
        instruction_prompt,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    return list(encoded["input_ids"])


def _build_lm_probability_context_prompt(
    *,
    full_instruction_prompt: str,
    prompt_prefix_text: str,
) -> str:
    return (
        "Original Prompt:\n"
        f"{full_instruction_prompt}\n\n"
        "Task:\n"
        "Write a natural revised version of the original prompt while preserving meaning, structure, and tone. Prefer paraphrase or other clear, robust, and effective wording.\n\n"
        "Revised prompt:\n"
        f"{prompt_prefix_text}"
    )


def _build_probability_context_prefix_token_ids(
    *,
    instruction_prompt: str,
    prompt_prefix_text: str,
    tokenizer,
    lm_probability_submode: str,
) -> List[int]:
    if lm_probability_submode == LM_PROBABILITY_SUBMODE_FULL_PROMPT_AS_CONTEXT:
        encoded = tokenizer(
            _build_lm_probability_context_prompt(
                full_instruction_prompt=instruction_prompt,
                prompt_prefix_text=prompt_prefix_text,
            ),
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return list(encoded["input_ids"])
    if lm_probability_submode == LM_PROBABILITY_SUBMODE_NO_CONTEXT:
        encoded = tokenizer(
            prompt_prefix_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return list(encoded["input_ids"])
    raise ValueError(f"Unsupported lm_probability_submode: {lm_probability_submode}")


def _generate_region_candidates_from_lm_probability(
    *,
    instruction_prompt: str,
    region_details: Dict[str, Any],
    num_region_candidates: int,
    model,
    tokenizer,
    max_new_tokens: int,
    batch_size: int,
    lm_probability_submode: str,
) -> List[Dict[str, Any]]:
    if num_region_candidates <= 0:
        return []

    prompt_token_ids = _build_probability_region_prompt_token_ids(
        instruction_prompt=instruction_prompt,
        tokenizer=tokenizer,
    )
    region_candidates: List[Dict[str, Any]] = []
    for region in region_details["selected_regions"]:
        region_rank = int(region["region_rank"])
        start_token = int(region["start_token"])
        end_token = int(region["end_token"])
        region_token_ids = prompt_token_ids[start_token : end_token + 1]
        if not region_token_ids:
            region_candidates.append(
                {
                    "region_rank": region_rank,
                    "region_text": region["region_text"],
                    "generation_method": "lm_probability",
                    "lm_probability_submode": lm_probability_submode,
                    "meta_prompt": None,
                    "raw_output": None,
                    "candidates": [],
                }
            )
            continue

        region_token_length = len(region_token_ids)
        original_first_token_id = int(region_token_ids[0])
        prefix_token_ids = prompt_token_ids[:start_token]
        prompt_prefix_text = instruction_prompt[: region["start_char"]]
        candidate_pool_size = max(num_region_candidates, num_region_candidates * num_region_candidates)
        if start_token == 0:
            initial_top_token_ids = [original_first_token_id]
            clustered_first_token_ids = [original_first_token_id]
            selected_first_token_ids = [original_first_token_id] * num_region_candidates
            print(
                "[agent_gradient_eval_debug] LM probability first-token selection:",
                f"region_rank={region_rank}",
                "start_token=0 using original first token only",
            )
        else:
            probability_context_prefix_token_ids = _build_probability_context_prefix_token_ids(
                instruction_prompt=instruction_prompt,
                prompt_prefix_text=prompt_prefix_text,
                tokenizer=tokenizer,
                lm_probability_submode=lm_probability_submode,
            )
            initial_top_token_ids = _collect_top_token_ids_from_prefix(
                prefix_token_ids=probability_context_prefix_token_ids,
                model=model,
                tokenizer=tokenizer,
                num_candidate_pool_tokens=candidate_pool_size,
            )
            # clustered_first_token_ids = select_centroid_token_ids(
            #     token_ids=initial_top_token_ids,
            #     embedding_weight=embedding_weight,
            #     num_clusters=num_region_candidates,
            # )
            clustered_first_token_ids = []
            selected_first_token_ids = list(initial_top_token_ids[:num_region_candidates])
            print(
                "[agent_gradient_eval_debug] LM probability first-token selection:",
                f"region_rank={region_rank}",
                f"submode={lm_probability_submode}",
                f"candidate_pool_size={len(initial_top_token_ids)}",
                f"selected_first_tokens={json.dumps([_decode_token_text(tokenizer=tokenizer, token_ids=[token_id]) for token_id in selected_first_token_ids], ensure_ascii=False)}",
            )

        first_token_texts = [
            _decode_token_text(tokenizer=tokenizer, token_ids=[token_id])
            for token_id in selected_first_token_ids
        ]
        continuation_prompts = [
            _decode_token_text(
                tokenizer=tokenizer,
                token_ids=prefix_token_ids + [token_id],
            )
            for token_id in selected_first_token_ids
        ]
        remaining_region_tokens = max(0, region_token_length - 1)
        if remaining_region_tokens > 0:
            continuation_outputs = run_prompts(
                continuation_prompts,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=min(max_new_tokens, remaining_region_tokens),
                batch_size=batch_size,
                use_chat_template=False,
                add_generation_prompt=False,
                enable_thinking=False,
                do_sample=True,
                do_log=True,
                log_label=f"agent_gradient_eval_debug_lm_region_candidates_r{region_rank}",
            )
        else:
            continuation_outputs = [""] * len(continuation_prompts)

        candidates: List[str] = []
        seen_candidates: set[str] = set()
        raw_generations: List[Dict[str, Any]] = []
        for generation_index, (token_id, first_token_text, continuation_output) in enumerate(
            zip(selected_first_token_ids, first_token_texts, continuation_outputs)
        ):
            candidate_text = first_token_text + continuation_output
            normalized_candidate = candidate_text.strip()
            raw_generations.append(
                {
                    "generation_index": generation_index,
                    "first_token_id": int(token_id),
                    "first_token_text": first_token_text,
                    "continuation_output": continuation_output,
                    "candidate_text": candidate_text,
                }
            )
            print(
                "[agent_gradient_eval_debug] LM probability candidate span:",
                f"region_rank={region_rank}",
                f"generation_index={generation_index}",
                f"candidate_text={candidate_text!r}",
            )
            if not normalized_candidate or normalized_candidate in seen_candidates:
                continue
            seen_candidates.add(normalized_candidate)
            candidates.append(normalized_candidate)
            if len(candidates) == num_region_candidates:
                break

        region_candidates.append(
            {
                "region_rank": region_rank,
                "region_text": region["region_text"],
                "generation_method": "lm_probability",
                "lm_probability_submode": lm_probability_submode,
                "meta_prompt": None,
                "raw_output": None,
                "initial_top_token_ids": initial_top_token_ids,
                "clustered_first_token_ids": clustered_first_token_ids,
                "raw_generations": raw_generations,
                "candidates": candidates,
            }
        )
        print(
            "[agent_gradient_eval_debug] parsed LM probability region candidates:",
            f"region_rank={region_rank}",
            f"candidates={json.dumps(candidates, ensure_ascii=False)}",
        )
    return region_candidates


def _build_binary_prompts_for_instruction(
    *,
    instruction_prompt: str,
    binary_pairs: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    prompts: List[str] = []
    target_labels: List[str] = []
    for pair in binary_pairs:
        support = pair["support"]
        relation = support["relation"]
        target_label = pair["label"]
        prompts.append(
            build_relation_prompt(
                instruction_prompt=instruction_prompt,
                relation=relation,
                relation_description=pair["relation_description"],
                support_sentence=pair["support_sentence"],
                query_sentence=pair["query_sentence"],
                inference_mode=INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
            )
        )
        target_labels.append(target_label)
    return prompts, target_labels


def _predict_binary_prompts(
    *,
    prompts: Sequence[str],
    target_labels: Sequence[str],
    model,
    tokenizer,
    batch_size: int,
    use_chat_template: bool,
) -> Dict[str, Any]:
    if len(prompts) != len(target_labels):
        raise ValueError("prompts/target_labels length mismatch")
    predicted_labels = run_binary_inference(
        prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        log_label="gradient_eval_debug",
    )

    return {
        "predicted_labels": predicted_labels,
        "target_labels": list(target_labels),
    }


def _summarize_validation_by_bucket(
    *,
    sampled_pairs: Sequence[Dict[str, Any]],
    score_payload: Dict[str, Any],
) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, Any]] = {
        "overall": {"count": 0, "correct": 0},
        "tp": {"count": 0, "correct": 0},
        "tn": {"count": 0, "correct": 0},
        "fp": {"count": 0, "correct": 0},
        "fn": {"count": 0, "correct": 0},
    }
    fixes_from_mistakes = 0
    regressions_from_correct = 0

    for pair, predicted_label, target_label in zip(
        sampled_pairs,
        score_payload["predicted_labels"],
        score_payload["target_labels"],
    ):
        confusion_bucket = pair["confusion_bucket"]
        is_correct = predicted_label == target_label
        for bucket_name in ("overall", confusion_bucket):
            buckets[bucket_name]["count"] += 1
            buckets[bucket_name]["correct"] += int(is_correct)

        if confusion_bucket in {"fp", "fn"} and is_correct:
            fixes_from_mistakes += 1
        if confusion_bucket in {"tp", "tn"} and not is_correct:
            regressions_from_correct += 1

    summary: Dict[str, Any] = {}
    for bucket_name, stats in buckets.items():
        count = stats["count"]
        summary[bucket_name] = {
            "total": count,
            "count": stats["correct"],
            "accuracy": (stats["correct"] / count) if count else 0.0,
        }
    summary["fixes_from_mistakes"] = fixes_from_mistakes
    summary["regressions_from_correct"] = regressions_from_correct
    return summary


def _compute_summary_delta(
    *,
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    for bucket_name in ("overall", "tp", "tn", "fp", "fn"):
        delta[bucket_name] = {
            "accuracy": (
                candidate_summary[bucket_name]["accuracy"]
                - baseline_summary[bucket_name]["accuracy"]
            ),
        }
    delta["fixes_from_mistakes"] = (
        candidate_summary["fixes_from_mistakes"] - baseline_summary["fixes_from_mistakes"]
    )
    delta["regressions_from_correct"] = (
        candidate_summary["regressions_from_correct"]
        - baseline_summary["regressions_from_correct"]
    )
    return delta


def _build_confusion_matrix_counts(
    *,
    predicted_labels: Sequence[str],
    target_labels: Sequence[str],
) -> Dict[str, int]:
    if len(predicted_labels) != len(target_labels):
        raise ValueError("predicted_labels/target_labels length mismatch")

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for prediction, target in zip(predicted_labels, target_labels):
        if target == "yes" and prediction == "yes":
            counts["tp"] += 1
        elif target == "no" and prediction == "no":
            counts["tn"] += 1
        elif target == "no" and prediction == "yes":
            counts["fp"] += 1
        elif target == "yes" and prediction == "no":
            counts["fn"] += 1
        else:
            raise ValueError(
                f"Unsupported target/prediction pair: target={target}, prediction={prediction}"
            )
    return counts


def _build_summary_variants(
    *,
    prompt_editing_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    generated_variants = prompt_editing_payload.get("generated_prompt_variants", [])
    summary_variants: List[Dict[str, Any]] = []
    for variant in generated_variants:
        validation = variant.get("validation")
        full_evaluation = variant.get("full_evaluation")
        summary_variants.append(
            {
                "prompt": variant.get("revised_prompt"),
                "selection_metrics": variant.get("selection_metrics"),
                "bucket_scores": (
                    validation["confusion_matrix"]["updated"]
                    if validation
                    else None
                ),
                "balanced_train_prf": (
                    validation["prf"]
                    if validation
                    else None
                ),
                "dev_prf": (
                    full_evaluation["prf"]
                    if full_evaluation
                    else None
                ),
            }
        )
    return summary_variants


def _select_best_variant(
    *,
    prompt_editing_payload: Dict[str, Any],
    f1_std_penalty: float = DEFAULT_F1_STABILITY_STD_MULTIPLIER,
) -> Dict[str, Any] | None:
    """Select among retained beam prompts after full evaluation.

    Beam search itself is pruned by sampled train scoring before this point.
    Once retained prompts have full dev scores, choose the prompt with the best
    conservative full-evaluation F1 score:
        f1 - f1_std_penalty * f1_std
    sampled train scores are used only for fallback/tie-breaks.
    """
    generated_variants = prompt_editing_payload.get("generated_prompt_variants", [])
    ranked_variants = [
        variant
        for variant in generated_variants
        if variant.get("revised_prompt")
        and variant.get("validation") is not None
    ]
    if not ranked_variants:
        return None
    return max(
        ranked_variants,
        key=lambda variant: (
            1 if variant.get("full_evaluation") is not None else 0,
            stable_prf_score_or_neg_inf(
                ((variant.get("full_evaluation") or {}).get("prf") or {}),
                std_multiplier=f1_std_penalty,
            ),
            float(((variant.get("full_evaluation") or {}).get("prf") or {}).get("f1", float("-inf"))),
            -float(((variant.get("full_evaluation") or {}).get("prf") or {}).get("f1_std", 0.0) or 0.0),
            stable_prf_score_or_neg_inf(
                variant["validation"]["prf"],
                std_multiplier=f1_std_penalty,
            ),
            float(variant["validation"]["prf"].get("f1", float("-inf"))),
            -float(variant["validation"]["prf"].get("f1_std", 0.0) or 0.0),
            float(variant["validation"]["summary"]["overall"].get("accuracy", float("-inf"))),
            int(variant["validation"]["summary"].get("fixes_from_mistakes", 0)),
            -int(variant["validation"]["summary"].get("regressions_from_correct", 0)),
            -float(variant.get("selection_metrics", {}).get("combined_score", float("inf"))),
            -float(variant.get("selection_metrics", {}).get("mean_cross_entropy", float("inf"))),
            -float(variant.get("selection_metrics", {}).get("mean_perplexity", float("inf"))),
        ),
    )


def _variant_beats_baseline_full_evaluation(
    *,
    selected_variant: Dict[str, Any] | None,
    baseline_full_evaluation: Dict[str, Any] | None,
    f1_std_penalty: float = DEFAULT_F1_STABILITY_STD_MULTIPLIER,
) -> bool:
    if selected_variant is None:
        return False
    baseline_prf = (baseline_full_evaluation or {}).get("prf")
    if not baseline_prf:
        return False
    full_evaluation = selected_variant.get("full_evaluation")
    if not full_evaluation:
        return False
    variant_prf = full_evaluation.get("prf")
    variant_stable_f1 = stable_prf_score_or_neg_inf(
        variant_prf,
        std_multiplier=f1_std_penalty,
    )
    baseline_stable_f1 = stable_prf_score_or_neg_inf(
        baseline_prf,
        std_multiplier=f1_std_penalty,
    )
    if variant_stable_f1 != baseline_stable_f1:
        return variant_stable_f1 > baseline_stable_f1
    return float((variant_prf or {}).get("f1", float("-inf"))) > float(
        baseline_prf.get("f1", float("-inf"))
    )


def _build_selected_prompt_summary(
    *,
    iteration_index: int,
    input_prompt: str,
    selected_variant: Dict[str, Any] | None,
    baseline_prf: Dict[str, float] | None = None,
    baseline_full_evaluation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if selected_variant is None:
        return {
            "iteration_index": iteration_index,
            "selection_strategy": "fallback_to_input_prompt",
            "prompt": input_prompt,
            "changed_vs_input": False,
            "generation_index": None,
            "beam_index": None,
            "selection_metrics": None,
            "bucket_scores": None,
            "balanced_train_prf": copy.deepcopy(baseline_prf),
            "dev_prf": copy.deepcopy((baseline_full_evaluation or {}).get("prf")),
        }

    validation = selected_variant.get("validation")
    full_evaluation = selected_variant.get("full_evaluation")
    revised_prompt = selected_variant.get("revised_prompt") or input_prompt
    return {
        "iteration_index": iteration_index,
        "selection_strategy": (
            "best_dev_stable_f1"
            if full_evaluation is not None
            else "best_train_stable_f1"
        ),
        "prompt": revised_prompt,
        "changed_vs_input": revised_prompt != input_prompt,
        "generation_index": selected_variant.get("generation_index"),
        "beam_index": selected_variant.get("beam_index"),
        "selection_metrics": copy.deepcopy(selected_variant.get("selection_metrics")),
        "bucket_scores": (
            copy.deepcopy(validation["confusion_matrix"]["updated"])
            if validation
            else None
        ),
        "balanced_train_prf": (
            copy.deepcopy(validation["prf"])
            if validation
            else None
        ),
        "dev_prf": (
            copy.deepcopy(full_evaluation["prf"])
            if full_evaluation
            else None
        ),
    }


def _build_iteration_summary(
    *,
    iteration_result: Dict[str, Any],
) -> Dict[str, Any]:
    prompt_editing_payload = iteration_result["prompt_editing_payload"]
    gradient_results = iteration_result["gradient_analysis"]
    return {
        "iteration_index": iteration_result["iteration_index"],
        "input_prompt": {
            "prompt": iteration_result["input_prompt"],
            "bucket_scores": copy.deepcopy(iteration_result["baseline_confusion_matrix"]),
            "balanced_train_prf": copy.deepcopy(iteration_result["baseline_prf"]),
            "dev_prf": copy.deepcopy(
                (iteration_result.get("baseline_full_evaluation") or {}).get("prf")
            ),
        },
        "sampling_summary": copy.deepcopy(iteration_result["sampling_summary"]),
        "train_gradient_collection": copy.deepcopy(iteration_result["train_gradient_collection"]),
        "gradient_analysis_summary": {
            "num_instances": gradient_results.get("num_instances"),
            "token_gradients": len(gradient_results.get("token_gradients", [])),
            "top_regions_count": len(gradient_results.get("top_regions", [])),
            "selected_regions": copy.deepcopy(prompt_editing_payload.get("selected_regions", [])),
        },
        "region_candidate_meta_prompt": prompt_editing_payload.get(
            "region_candidate_meta_prompt"
        ),
        "region_candidate_raw_output": prompt_editing_payload.get(
            "region_candidate_raw_output"
        ),
        "region_candidate_meta_prompts": copy.deepcopy(
            prompt_editing_payload.get("region_candidate_meta_prompts", [])
        ),
        "region_candidates": copy.deepcopy(prompt_editing_payload.get("region_candidates", [])),
        "beam_width": prompt_editing_payload.get("beam_width"),
        "beam_search_steps": copy.deepcopy(prompt_editing_payload.get("beam_search_steps", [])),
        "generated_prompts": _build_summary_variants(
            prompt_editing_payload=prompt_editing_payload,
        ),
        "selected_prompt": copy.deepcopy(iteration_result["selected_prompt"]),
    }


def _build_summary_payload(
    *,
    instruction_prompt: str,
    baseline_prf: Dict[str, float],
    prompt_editing_payload: Dict[str, Any],
    baseline_confusion_matrix: Dict[str, int],
    original_dev_prf: Dict[str, float] | None,
    sampling_summary: Dict[str, Any],
    iteration_results: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    summary_variants = _build_summary_variants(
        prompt_editing_payload=prompt_editing_payload,
    )
    iteration_summaries = [
        _build_iteration_summary(iteration_result=iteration_result)
        for iteration_result in (iteration_results or [])
    ]
    final_selected_prompt = (
        copy.deepcopy(iteration_summaries[-1]["selected_prompt"])
        if iteration_summaries
        else None
    )

    return {
        "num_iterations": max(len(iteration_summaries), 1),
        "original_prompt": {
            "prompt": instruction_prompt,
            "bucket_scores": baseline_confusion_matrix,
            "balanced_train_prf": baseline_prf,
            "dev_prf": original_dev_prf,
        },
        "sampling_summary": copy.deepcopy(sampling_summary),
        "iterations": iteration_summaries,
        "final_selected_prompt": final_selected_prompt,
        "meta_prompt": prompt_editing_payload.get("meta_prompt"),
        "region_candidate_meta_prompt": prompt_editing_payload.get(
            "region_candidate_meta_prompt"
        ),
        "region_candidate_raw_output": prompt_editing_payload.get(
            "region_candidate_raw_output"
        ),
        "region_candidate_meta_prompts": prompt_editing_payload.get(
            "region_candidate_meta_prompts", []
        ),
        "beam_width": prompt_editing_payload.get("beam_width"),
        "beam_replacement_mode": prompt_editing_payload.get("beam_replacement_mode"),
        "beam_search_steps": prompt_editing_payload.get("beam_search_steps", []),
        "combination_meta_prompt": prompt_editing_payload.get("combination_meta_prompt"),
        "synthesis_meta_prompt": prompt_editing_payload.get("synthesis_meta_prompt"),
        "synthesis_meta_prompts": prompt_editing_payload.get("synthesis_meta_prompts", []),
        "generated_prompts": summary_variants,
    }


def _build_combination_generated_prompts_payload(
    *,
    prompt_editing_payload: Dict[str, Any],
) -> Dict[str, Any]:
    generated_variants = prompt_editing_payload.get("generated_prompt_variants", [])
    prompts_with_scores: List[Dict[str, Any]] = []
    for variant in generated_variants:
        selection_metrics = variant.get("selection_metrics") or {}
        prompts_with_scores.append(
            {
                "generation_index": variant.get("generation_index"),
                "beam_index": variant.get("beam_index"),
                "combination_key": variant.get("combination_key"),
                "selected_replacements": variant.get("selected_replacements"),
                "num_changed_spans": variant.get("num_changed_spans"),
                "prompt": variant.get("revised_prompt"),
                "changed": variant.get("changed"),
                "duplicate_prompt": variant.get("duplicate_prompt"),
                "retained_for_full_validation": variant.get("retained_for_full_validation"),
                "loss": selection_metrics.get("mean_cross_entropy"),
                "perplexity": selection_metrics.get("mean_perplexity"),
                "combined_score": selection_metrics.get("combined_score"),
            }
        )

    return {
        "mode": prompt_editing_payload.get("mode"),
        "beam_width": prompt_editing_payload.get("beam_width"),
        "beam_search_steps": prompt_editing_payload.get("beam_search_steps", []),
        "combination_meta_prompt": prompt_editing_payload.get("combination_meta_prompt"),
        "candidate_combinations": prompt_editing_payload.get("candidate_combinations", []),
        "generated_prompts_with_scores": prompts_with_scores,
    }


def _strip_prediction_arrays_from_prompt_editing_payload(
    *,
    prompt_editing_payload: Dict[str, Any],
) -> Dict[str, Any]:
    compact_payload = copy.deepcopy(prompt_editing_payload)
    compact_variants: List[Dict[str, Any]] = []
    for variant in compact_payload.get("generated_prompt_variants", []):
        compact_variant = copy.deepcopy(variant)
        validation = compact_variant.get("validation")
        if isinstance(validation, dict):
            validation.pop("predicted_labels", None)
            validation.pop("target_labels", None)
        full_evaluation = compact_variant.get("full_evaluation")
        if isinstance(full_evaluation, dict):
            full_evaluation.pop("predicted_labels", None)
            full_evaluation.pop("target_labels", None)
        compact_variants.append(compact_variant)
    compact_payload["generated_prompt_variants"] = compact_variants
    return compact_payload


def _compact_selection_metrics(selection_metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mean_cross_entropy": selection_metrics.get("mean_cross_entropy"),
        "mean_perplexity": selection_metrics.get("mean_perplexity"),
        "prompt_fluency_perplexity": selection_metrics.get("prompt_fluency_perplexity"),
        "combined_score": selection_metrics.get("combined_score"),
    }


def _build_sampled_data_and_predictions_payload(
    *,
    sampling: Dict[str, Any],
    sampled_examples: Sequence[Dict[str, Any]],
    prompt_editing_payload: Dict[str, Any],
) -> Dict[str, Any]:
    generated_variants: List[Dict[str, Any]] = []
    for variant in prompt_editing_payload.get("generated_prompt_variants", []):
        validation = variant.get("validation")
        full_evaluation = variant.get("full_evaluation")
        generated_variants.append(
            {
                "generation_index": variant.get("generation_index"),
                "combination_key": variant.get("combination_key"),
                "prompt": variant.get("revised_prompt"),
                "selected_replacements": variant.get("selected_replacements"),
                "validation": (
                    {
                        "predicted_labels": validation.get("predicted_labels", []),
                        "target_labels": validation.get("target_labels", []),
                    }
                    if validation
                    else None
                ),
                "full_evaluation": (
                    {
                        "predicted_labels": full_evaluation.get("predicted_labels", []),
                        "target_labels": full_evaluation.get("target_labels", []),
                    }
                    if full_evaluation
                    else None
                ),
            }
        )

    return {
        "sampling": copy.deepcopy(sampling),
        "sampled_examples": copy.deepcopy(list(sampled_examples)),
        "generated_prompt_predictions": generated_variants,
    }


def _build_iterative_sampled_data_and_predictions_payload(
    *,
    iteration_results: Sequence[Dict[str, Any]],
    final_payload: Dict[str, Any],
) -> Dict[str, Any]:
    if not iteration_results:
        return final_payload
    return {
        **copy.deepcopy(final_payload),
        "num_iterations": len(iteration_results),
        "iterations": [
            {
                "iteration_index": iteration_result["iteration_index"],
                **_build_sampled_data_and_predictions_payload(
                    sampling=iteration_result["sampling_payload"],
                    sampled_examples=iteration_result["sampled_examples"],
                    prompt_editing_payload=iteration_result["prompt_editing_payload"],
                ),
            }
            for iteration_result in iteration_results
        ],
    }


def _generate_region_prompt_variants(
    *,
    meta_prompt: str,
    instruction_prompt: str,
    num_generated_prompts: int,
    model,
    tokenizer,
    max_new_tokens: int,
    batch_size: int,
    use_chat_template: bool,
) -> List[Dict[str, Any]]:
    if num_generated_prompts <= 0:
        return []

    raw_outputs = run_prompts(
        [meta_prompt] * num_generated_prompts,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        do_sample=True,
        do_log=True,
        log_label="agent_gradient_eval_debug_meta_prompt",
    )
    variants: List[Dict[str, Any]] = []
    for generation_index, raw_output in enumerate(raw_outputs):
        revised_prompt = extract_tagged_text(raw_output, "<p>", "</p>")
        variants.append(
            {
                "generation_index": generation_index,
                "raw_output": raw_output,
                "revised_prompt": revised_prompt,
                "changed": bool(revised_prompt) and revised_prompt != instruction_prompt,
            }
        )
    return variants


def _generate_region_candidates(
    *,
    instruction_prompt: str,
    region_details: Dict[str, Any],
    num_region_candidates: int,
    model,
    tokenizer,
    max_new_tokens: int,
    batch_size: int,
    use_chat_template: bool,
) -> Dict[str, Any]:
    if num_region_candidates <= 0:
        return {
            "meta_prompt": None,
            "raw_output": "",
            "parsed_output": {},
            "selected_json_length": None,
            "region_candidates": [],
        }

    meta_prompt = _build_region_candidate_meta_prompt(
        instruction_prompt=instruction_prompt,
        region_details=region_details,
        num_region_candidates=num_region_candidates,
    )
    print("[agent_gradient_eval_debug] all-region candidate meta prompt:")
    print(meta_prompt)
    raw_output = run_prompts(
        [meta_prompt],
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        do_sample=True,
        do_log=True,
        log_label="agent_gradient_eval_debug_region_candidates",
    )[0]
    print("[agent_gradient_eval_debug] all-region candidate raw output:")
    print(raw_output)

    try:
        parsed_output = extract_json_object(raw_output)
    except (ValueError, TypeError):
        parsed_output = {}
    if not isinstance(parsed_output, dict):
        parsed_output = {}
    used_raw_candidate_fallback = False

    json_length = None
    if parsed_output:
        json_length = len(json.dumps(parsed_output, ensure_ascii=False, sort_keys=True))
        print(
            "[agent_gradient_eval_debug] all-region candidate parsed json length:",
            f"json_length={json_length}",
        )

    region_candidates: List[Dict[str, Any]] = []
    for region in region_details["selected_regions"]:
        ordered_candidates = _extract_region_candidate_values(
            parsed_output=parsed_output,
            region_rank=int(region["region_rank"]),
        )
        if not ordered_candidates:
            ordered_candidates = _extract_region_candidate_values_from_raw_output(
                raw_output=raw_output,
                region_rank=int(region["region_rank"]),
            )
            if ordered_candidates:
                used_raw_candidate_fallback = True
        candidates: List[str] = []
        seen: set[str] = set()
        for candidate_text in ordered_candidates:
            normalized = _normalize_span_replacement_text(candidate_text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(normalized)
            if len(candidates) == num_region_candidates:
                break
        region_candidates.append(
            {
                "region_rank": region["region_rank"],
                "region_text": region["region_text"],
                "meta_prompt": meta_prompt,
                "raw_output": raw_output,
                "raw_outputs": [raw_output],
                "selected_generation_index": 0,
                "selected_json_length": json_length,
                "candidates": candidates,
            }
        )
        print(
            "[agent_gradient_eval_debug] parsed region candidates:",
            f"region_rank={region['region_rank']}",
            f"candidates={json.dumps(candidates, ensure_ascii=False)}",
        )
    if used_raw_candidate_fallback:
        print(
            "[agent_gradient_eval_debug] used raw candidate fallback parser "
            "for malformed all-region JSON"
        )

    return {
        "meta_prompt": meta_prompt,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "selected_json_length": json_length,
        "used_raw_candidate_fallback": used_raw_candidate_fallback,
        "region_candidates": region_candidates,
    }


def _score_instruction_prompt_for_candidate_selection(
    *,
    instruction_prompt: str,
    sampled_pairs: Sequence[Dict[str, Any]],
    selection_perplexity_lambda: float,
    use_log_fluency_score: bool,
    validation_batch_size: int,
    use_chat_template: bool,
    prompt_result_cache: Dict[str, Dict[str, Any]],
    model,
    tokenizer,
) -> Dict[str, Any]:
    cached_result = prompt_result_cache.get(instruction_prompt)
    if cached_result is not None:
        return copy.deepcopy(cached_result)

    prompts, target_labels = _build_binary_prompts_for_instruction(
        instruction_prompt=instruction_prompt,
        binary_pairs=sampled_pairs,
    )
    raw_selection_metrics = score_binary_prompts_with_ce_and_perplexity(
        prompts=prompts,
        target_labels=target_labels,
        model=model,
        tokenizer=tokenizer,
        batch_size=validation_batch_size,
        use_chat_template=use_chat_template,
    )
    prompt_perplexity = score_instruction_prompt_fluency_perplexity(
        instruction_prompt=instruction_prompt,
        model=model,
        tokenizer=tokenizer,
        return_log_score=use_log_fluency_score,
    )
    fluency_metric_name = (
        "prompt_fluency_log_score"
        if use_log_fluency_score
        else "prompt_fluency_perplexity"
    )
    selection_metrics = {
        "mean_cross_entropy": raw_selection_metrics["mean_cross_entropy"],
        "mean_perplexity": prompt_perplexity,
        "prompt_fluency_perplexity": prompt_perplexity,
        fluency_metric_name: prompt_perplexity,
        "prompt_fluency_metric": fluency_metric_name,
    }
    combined_score = (
        selection_metrics["mean_cross_entropy"]
        + selection_perplexity_lambda * selection_metrics["mean_perplexity"]
    )
    cached_result = {
        "selection_metrics": {
            **selection_metrics,
            "combined_score": combined_score,
        },
    }
    prompt_result_cache[instruction_prompt] = copy.deepcopy(cached_result)
    return copy.deepcopy(cached_result)


def _run_region_candidate_beam_search(
    *,
    args: argparse.Namespace,
    instruction_prompt: str,
    sampled_pairs: Sequence[Dict[str, Any]],
    region_details: Dict[str, Any],
    region_candidates: Sequence[Dict[str, Any]],
    model,
    tokenizer,
) -> Dict[str, Any]:
    prepared_region_candidates = _prepare_region_candidates_for_beam_search(
        region_details=region_details,
        region_candidates=region_candidates,
    )
    candidate_index = {
        int(item["region_rank"]): item for item in prepared_region_candidates
    }
    prompt_result_cache: Dict[str, Dict[str, Any]] = {}
    synthesis_cache: Dict[Tuple[Tuple[int, str], ...], Dict[str, Any]] = {}
    beam_nodes: List[Dict[str, Any]] = [
        {
            "beam_index": 0,
            "parent_beam_index": None,
            "selected_replacements": {},
            "revised_prompt": instruction_prompt,
            "replacement_steps": [],
            "selection_metrics": None,
            "changed": False,
            "duplicate_prompt": False,
            "num_changed_spans": 0,
        }
    ]
    beam_search_steps: List[Dict[str, Any]] = []

    for step_index, region in enumerate(region_details["selected_regions"], start=1):
        region_rank = int(region["region_rank"])
        region_text = region["region_text"]
        region_candidate_texts = candidate_index.get(region_rank, {}).get("candidates", [])
        parent_count = len(beam_nodes)
        expanded_by_prompt: Dict[str, Dict[str, Any]] = {}

        print(
            "[agent_gradient_eval_debug] beam search expanding region:",
            f"step_index={step_index}",
            f"region_rank={region_rank}",
            f"beam_size={len(beam_nodes)}",
            f"num_candidates={len(region_candidate_texts)}",
        )

        for parent_beam_index, parent_node in enumerate(beam_nodes):
            for candidate_position, candidate_text in enumerate(region_candidate_texts):
                selected_replacements = copy.deepcopy(parent_node["selected_replacements"])
                if candidate_text == region_text:
                    selected_replacements.pop(region_rank, None)
                else:
                    selected_replacements[region_rank] = candidate_text

                prompt_build = _build_beam_candidate_prompt(
                    replacement_mode=args.beam_replacement_mode,
                    instruction_prompt=instruction_prompt,
                    region_details=region_details,
                    selected_replacements=selected_replacements,
                    region_rank=region_rank,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=args.meta_prompt_max_new_tokens,
                    batch_size=args.meta_prompt_batch_size,
                    use_chat_template=not args.disable_chat_template,
                    synthesis_cache=synthesis_cache,
                )
                revised_prompt = prompt_build["revised_prompt"]
                cached_result = _score_instruction_prompt_for_candidate_selection(
                    instruction_prompt=revised_prompt,
                    sampled_pairs=sampled_pairs,
                    selection_perplexity_lambda=args.selection_perplexity_lambda,
                    use_log_fluency_score=args.use_log_fluency_score,
                    validation_batch_size=args.validation_batch_size,
                    use_chat_template=not args.disable_chat_template,
                    prompt_result_cache=prompt_result_cache,
                    model=model,
                    tokenizer=tokenizer,
                )
                expanded_node = {
                    "beam_index": None,
                    "parent_beam_index": parent_beam_index,
                    "expanded_region_rank": region_rank,
                    "candidate_position": candidate_position,
                    "selected_replacements": selected_replacements,
                    "revised_prompt": revised_prompt,
                    "replacement_steps": parent_node.get("replacement_steps", [])
                    + [prompt_build["replacement_metadata"]],
                    "selection_metrics": cached_result["selection_metrics"],
                    "changed": revised_prompt != instruction_prompt,
                    "duplicate_prompt": False,
                    "num_changed_spans": sum(
                        1
                        for selected_region in region_details["selected_regions"]
                        if selected_replacements.get(
                            int(selected_region["region_rank"]),
                            selected_region["region_text"],
                        )
                        != selected_region["region_text"]
                    ),
                }
                existing_node = expanded_by_prompt.get(revised_prompt)
                if existing_node is None:
                    expanded_by_prompt[revised_prompt] = expanded_node
                    continue
                existing_metrics = existing_node["selection_metrics"]
                candidate_metrics = expanded_node["selection_metrics"]
                existing_key = (
                    existing_metrics["combined_score"],
                    existing_metrics["mean_cross_entropy"],
                    existing_metrics["mean_perplexity"],
                    existing_node["num_changed_spans"],
                )
                candidate_key = (
                    candidate_metrics["combined_score"],
                    candidate_metrics["mean_cross_entropy"],
                    candidate_metrics["mean_perplexity"],
                    expanded_node["num_changed_spans"],
                )
                if candidate_key < existing_key:
                    expanded_by_prompt[revised_prompt] = expanded_node

        ranked_nodes = sorted(
            expanded_by_prompt.values(),
            key=lambda item: (
                item["selection_metrics"]["combined_score"],
                item["selection_metrics"]["mean_cross_entropy"],
                item["selection_metrics"]["mean_perplexity"],
            ),
        )
        retained_prompt_to_beam_index = {
            node["revised_prompt"]: beam_index
            for beam_index, node in enumerate(ranked_nodes[: args.beam_width])
        }
        beam_nodes = []
        for beam_index, node in enumerate(ranked_nodes[: args.beam_width]):
            node_copy = copy.deepcopy(node)
            node_copy["beam_index"] = beam_index
            beam_nodes.append(node_copy)

        print("[agent_gradient_eval_debug] beam step retained prompts:")
        if beam_nodes:
            for node in beam_nodes:
                print(
                    "[beam_node]",
                    f"step_index={step_index}",
                    f"beam_index={node['beam_index']}",
                    f"parent_beam_index={node['parent_beam_index']}",
                    f"region_rank={region_rank}",
                    f"combined_score={node['selection_metrics']['combined_score']:.6f}",
                    f"num_changed_spans={node['num_changed_spans']}",
                )
                print(node["revised_prompt"])
        else:
            print("[]")

        beam_search_steps.append(
            {
                "step_index": step_index,
                "region_rank": region_rank,
                "region_text": region_text,
                "num_parent_nodes": parent_count,
                "num_candidates_considered_per_parent": len(region_candidate_texts),
                "num_expanded_unique_prompts": len(expanded_by_prompt),
                "expanded_unique_nodes": [
                    {
                        "beam_index": retained_prompt_to_beam_index.get(node["revised_prompt"]),
                        "parent_beam_index": node["parent_beam_index"],
                        "candidate_position": node["candidate_position"],
                        "selected_replacements": _build_selected_replacements_payload(
                            selected_regions=region_details["selected_regions"],
                            selected_replacements=node["selected_replacements"],
                        ),
                        "num_changed_spans": node["num_changed_spans"],
                        "prompt": node["revised_prompt"],
                        "replacement_metadata": copy.deepcopy(
                            node["replacement_steps"][-1]
                            if node.get("replacement_steps")
                            else None
                        ),
                        "selection_metrics": copy.deepcopy(node["selection_metrics"]),
                        "retained_in_beam": node["revised_prompt"] in retained_prompt_to_beam_index,
                    }
                    for node in ranked_nodes
                ],
                "retained_beam_nodes": [
                    {
                        "beam_index": node["beam_index"],
                        "parent_beam_index": node["parent_beam_index"],
                        "selected_replacements": _build_selected_replacements_payload(
                            selected_regions=region_details["selected_regions"],
                            selected_replacements=node["selected_replacements"],
                        ),
                        "num_changed_spans": node["num_changed_spans"],
                        "prompt": node["revised_prompt"],
                        "replacement_steps": copy.deepcopy(node.get("replacement_steps", [])),
                        "selection_metrics": copy.deepcopy(node["selection_metrics"]),
                    }
                    for node in beam_nodes
                ],
            }
        )

    final_variants: List[Dict[str, Any]] = []
    for generation_index, node in enumerate(beam_nodes):
        final_variants.append(
            {
                "generation_index": generation_index,
                "beam_index": node["beam_index"],
                "selected_replacements": _build_selected_replacements_payload(
                    selected_regions=region_details["selected_regions"],
                    selected_replacements=node["selected_replacements"],
                ),
                "num_changed_spans": node["num_changed_spans"],
                "meta_prompt": None,
                "raw_output": None,
                "revised_prompt": node["revised_prompt"],
                "replacement_mode": args.beam_replacement_mode,
                "replacement_steps": copy.deepcopy(node.get("replacement_steps", [])),
                "changed": node["changed"],
                "duplicate_prompt": False,
                "selection_metrics": copy.deepcopy(node["selection_metrics"]),
                "validation": None,
                "full_evaluation": None,
                "delta_vs_baseline": None,
                "retained_for_full_validation": True,
            }
        )

    return {
        "beam_width": args.beam_width,
        "beam_replacement_mode": args.beam_replacement_mode,
        "region_candidates": prepared_region_candidates,
        "beam_search_steps": beam_search_steps,
        "synthesis_meta_prompts": [
            {
                "selected_replacements": [
                    {"region_rank": region_rank, "replacement_text": replacement_text}
                    for region_rank, replacement_text in cache_key
                ],
                "complete_replacements": synthesis_payload.get("complete_replacements"),
                "meta_prompt": synthesis_payload["meta_prompt"],
                "raw_output": synthesis_payload["raw_output"],
                "used_fallback": synthesis_payload["used_fallback"],
                "fallback_reason": synthesis_payload["fallback_reason"],
            }
            for cache_key, synthesis_payload in synthesis_cache.items()
        ],
        "generated_prompt_variants": final_variants,
    }

def build_full_binary_pairs(
    *,
    dataset: Dict[str, Any],
    dataset_type: str,
    query_index: int,
) -> List[Dict[str, Any]]:
    episodes = dataset["episodes"]
    shots = dataset["shots"]
    queries = dataset["queries"]
    num_ways = _resolve_num_ways(episodes)

    full_pairs: List[Dict[str, Any]] = []
    for pair_index in range(len(episodes) * num_ways):
        episode_index, way_index = _pair_index_to_episode_way(pair_index, num_ways)
        episode = episodes[episode_index]
        support_id = episode["meta_train"][way_index][0]
        query_id = episode["meta_test"][query_index]
        support = shots[support_id]
        query = queries[query_id]
        label = "yes" if support["relation"] == query["relation"] else "no"
        prediction = None
        full_pairs.append(
            {
                "pair_index": pair_index,
                "episode_index": episode_index,
                "way_index": way_index,
                "support": support,
                "query": query,
                "label": label,
                "prediction": prediction,
                "confusion_bucket": None,
                "support_sentence": get_sentence_with_tags(support).strip(),
                "query_sentence": get_sentence_with_tags(query).strip(),
                "relation_description": get_relation_description(
                    support["relation"],
                    dt=dataset_type,
                ),
            }
        )
    return full_pairs


def _evaluate_prompt_variant(
    *,
    revised_prompt: str,
    sampled_pairs: Sequence[Dict[str, Any]],
    full_eval_pairs: Sequence[Dict[str, Any]],
    full_eval_split: str,
    baseline_validation: Dict[str, Any],
    baseline_confusion_matrix: Dict[str, int],
    model,
    tokenizer,
    validation_batch_size: int,
    use_chat_template: bool,
    f1_std_penalty: float = DEFAULT_F1_STABILITY_STD_MULTIPLIER,
    log_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    eval_start = time.perf_counter()
    prompts, target_labels = _build_binary_prompts_for_instruction(
        instruction_prompt=revised_prompt,
        binary_pairs=sampled_pairs,
    )
    score_payload = _predict_binary_prompts(
        prompts=prompts,
        target_labels=target_labels,
        model=model,
        tokenizer=tokenizer,
        batch_size=validation_batch_size,
        use_chat_template=use_chat_template,
    )
    validation_summary = _summarize_validation_by_bucket(
        sampled_pairs=sampled_pairs,
        score_payload=score_payload,
    )
    updated_confusion_matrix = _build_confusion_matrix_counts(
        predicted_labels=score_payload["predicted_labels"],
        target_labels=score_payload["target_labels"],
    )
    delta_vs_baseline = _compute_summary_delta(
        baseline_summary=baseline_validation,
        candidate_summary=validation_summary,
    )
    validation_prf = _build_prf_scores(
        predicted_labels=score_payload["predicted_labels"],
        target_labels=score_payload["target_labels"],
    )
    validation_elapsed = time.perf_counter() - eval_start
    context_text = _format_log_context(log_context)
    context_prefix = f"{context_text} " if context_text else ""
    print(
        f"[agent_gradient_eval_debug] prompt validation: {context_prefix}"
        f"done in {validation_elapsed:.2f}s, "
        f"{_format_prf_for_log(validation_prf)}, "
        f"overall_accuracy={validation_summary['overall']['accuracy']:.4f}, "
        f"fixes_from_mistakes={validation_summary['fixes_from_mistakes']}, "
        f"regressions_from_correct={validation_summary['regressions_from_correct']}"
    )
    print(
        "[agent_gradient_eval_debug] prompt validation:",
        f"{context_prefix}{_format_stable_f1_for_log(validation_prf, f1_std_penalty=f1_std_penalty)}",
    )

    full_eval_start = time.perf_counter()
    full_variant_prompts, full_variant_targets = _build_binary_prompts_for_instruction(
        instruction_prompt=revised_prompt,
        binary_pairs=full_eval_pairs,
    )
    full_variant_score_payload = _predict_binary_prompts(
        prompts=full_variant_prompts,
        target_labels=full_variant_targets,
        model=model,
        tokenizer=tokenizer,
        batch_size=validation_batch_size,
        use_chat_template=use_chat_template,
    )
    full_evaluation = _build_evaluation_report(
        score_payload=full_variant_score_payload,
        binary_pairs=full_eval_pairs,
        prf_mode="episode",
    )
    full_eval_elapsed = time.perf_counter() - full_eval_start
    print(
        f"[agent_gradient_eval_debug] prompt full evaluation: {context_prefix}"
        f"split={full_eval_split} done in {full_eval_elapsed:.2f}s, "
        f"{_format_prf_for_log(full_evaluation['prf'])}"
    )
    print(
        "[agent_gradient_eval_debug] prompt full evaluation:",
        f"{context_prefix}{_format_stable_f1_for_log(full_evaluation['prf'], f1_std_penalty=f1_std_penalty)}",
    )

    validation_payload = {
        "predicted_labels": score_payload["predicted_labels"],
        "target_labels": score_payload["target_labels"],
        "summary": validation_summary,
        "prf": validation_prf,
        "confusion_matrix": {
            "original": baseline_confusion_matrix,
            "updated": updated_confusion_matrix,
        },
    }
    return {
        "validation": validation_payload,
        "full_evaluation": full_evaluation,
        "delta_vs_baseline": delta_vs_baseline,
    }


def _evaluate_prompt_full_evaluation(
    *,
    instruction_prompt: str,
    full_eval_pairs: Sequence[Dict[str, Any]],
    full_eval_split: str,
    model,
    tokenizer,
    validation_batch_size: int,
    use_chat_template: bool,
    f1_std_penalty: float = DEFAULT_F1_STABILITY_STD_MULTIPLIER,
    log_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    full_eval_start = time.perf_counter()
    full_variant_prompts, full_variant_targets = _build_binary_prompts_for_instruction(
        instruction_prompt=instruction_prompt,
        binary_pairs=full_eval_pairs,
    )
    full_variant_score_payload = _predict_binary_prompts(
        prompts=full_variant_prompts,
        target_labels=full_variant_targets,
        model=model,
        tokenizer=tokenizer,
        batch_size=validation_batch_size,
        use_chat_template=use_chat_template,
    )
    full_evaluation = _build_evaluation_report(
        score_payload=full_variant_score_payload,
        binary_pairs=full_eval_pairs,
        prf_mode="episode",
    )
    full_eval_elapsed = time.perf_counter() - full_eval_start
    context_text = _format_log_context(log_context)
    context_prefix = f"{context_text} " if context_text else ""
    print(
        f"[agent_gradient_eval_debug] prompt full evaluation: {context_prefix}"
        f"split={full_eval_split} done in {full_eval_elapsed:.2f}s, "
        f"{_format_prf_for_log(full_evaluation['prf'])}"
    )
    print(
        "[agent_gradient_eval_debug] prompt full evaluation:",
        f"{context_prefix}{_format_stable_f1_for_log(full_evaluation['prf'], f1_std_penalty=f1_std_penalty)}",
    )
    return full_evaluation


def _save_prompt_eval_outputs(
    *,
    eval_outputs_dir: Path,
    full_eval_split: str,
    generated_prompt_variants: Sequence[Dict[str, Any]],
) -> None:
    for variant in generated_prompt_variants:
        full_evaluation = variant.get("full_evaluation")
        revised_prompt = variant.get("revised_prompt")
        if not full_evaluation or not revised_prompt:
            continue
        generation_index = variant.get("generation_index")
        output_path = eval_outputs_dir / f"EVALID_prompt_{generation_index}_labels_predictions.json"
        _save_json(
            output_path,
            {
                "eval_id": f"prompt_{generation_index}",
                "split": full_eval_split,
                "prompt": revised_prompt,
                "generation_index": generation_index,
                "pair_labels": full_evaluation.get("target_labels", []),
                "pair_predictions": full_evaluation.get("predicted_labels", []),
                "confusion_matrix": full_evaluation.get("confusion_matrix"),
                "metrics": full_evaluation.get("prf"),
            },
        )


def _run_direct_candidate_generation(
    *,
    args: argparse.Namespace,
    instruction_prompt: str,
    sampled_pairs: Sequence[Dict[str, Any]],
    full_eval_pairs: Sequence[Dict[str, Any]],
    gradient_results: Dict[str, Any],
    baseline_validation: Dict[str, Any],
    baseline_confusion_matrix: Dict[str, int],
    model,
    tokenizer,
) -> Dict[str, Any]:
    prompt_editing_payload: Dict[str, Any] = {
        "mode": MODE_DIRECT_CANDIDATE_GENERATION,
        "baseline_validation": baseline_validation,
        "selected_regions": [],
        "meta_prompt": None,
        "generated_prompt_variants": [],
    }
    if args.num_generated_prompts <= 0:
        print("[agent_gradient_eval_debug] prompt variant generation skipped")
        return prompt_editing_payload

    region_details = _resolve_region_details(
        instruction_prompt=instruction_prompt,
        tokenizer=tokenizer,
        gradient_results=gradient_results,
        num_edit_regions=args.num_edit_regions,
    )
    meta_prompt = _build_gradient_region_meta_prompt(
        instruction_prompt=instruction_prompt,
        region_details=region_details,
    )
    print("[agent_gradient_eval_debug] direct meta prompt:\n" f"{meta_prompt}")
    print(
        "[agent_gradient_eval_debug] generating direct prompt variants:",
        f"num_edit_regions={region_details['num_edit_regions']}",
        f"region_ranks={[region['region_rank'] for region in region_details['selected_regions']]}",
        f"num_generated_prompts={args.num_generated_prompts}",
    )

    if args.meta_prompt_model:
        model = None
        tokenizer = None
        clear_model_memory()
        print("[agent_gradient_eval_debug] base model cleared from memory before direct generation")
        generation_model, generation_tokenizer = load_model_and_tokenizer(
            model_id=args.meta_prompt_model,
            device_map=args.device_map,
        )
        print(
            "[agent_gradient_eval_debug] direct generation model loaded:",
            args.meta_prompt_model,
        )
    else:
        generation_model = model
        generation_tokenizer = tokenizer

    generated_variants = _generate_region_prompt_variants(
        meta_prompt=meta_prompt,
        instruction_prompt=instruction_prompt,
        num_generated_prompts=args.num_generated_prompts,
        model=generation_model,
        tokenizer=generation_tokenizer,
        max_new_tokens=args.meta_prompt_max_new_tokens,
        batch_size=args.meta_prompt_batch_size,
        use_chat_template=not args.disable_chat_template,
    )

    if args.meta_prompt_model:
        generation_model = None
        generation_tokenizer = None
        clear_model_memory()
        print("[agent_gradient_eval_debug] direct generation model cleared from memory")
        model, tokenizer = load_model_and_tokenizer(
            model_id=args.model,
            device_map=args.device_map,
        )
        print("[agent_gradient_eval_debug] base model and tokenizer reloaded")

    validated_variants: List[Dict[str, Any]] = []
    prompt_result_cache: Dict[str, Dict[str, Any]] = {}
    for variant in generated_variants:
        revised_prompt = variant["revised_prompt"]
        if not revised_prompt:
            validated_variants.append(
                {
                    **variant,
                    "validation": None,
                    "full_evaluation": None,
                    "delta_vs_baseline": None,
                }
            )
            continue

        cached_result = prompt_result_cache.get(revised_prompt)
        if cached_result is None:
            cached_result = _evaluate_prompt_variant(
                revised_prompt=revised_prompt,
                sampled_pairs=sampled_pairs,
                full_eval_pairs=full_eval_pairs,
                full_eval_split=args.full_eval_split,
                baseline_validation=baseline_validation,
                baseline_confusion_matrix=baseline_confusion_matrix,
                model=model,
                tokenizer=tokenizer,
                validation_batch_size=args.validation_batch_size,
                use_chat_template=not args.disable_chat_template,
                f1_std_penalty=args.selection_f1_std_penalty,
                log_context={
                    "generation_index": variant.get("generation_index"),
                },
            )
            prompt_result_cache[revised_prompt] = copy.deepcopy(cached_result)
        validated_variants.append(
            {
                **variant,
                "validation": copy.deepcopy(cached_result["validation"]),
                "full_evaluation": copy.deepcopy(cached_result["full_evaluation"]),
                "delta_vs_baseline": copy.deepcopy(cached_result["delta_vs_baseline"]),
            }
        )

    prompt_editing_payload["selected_regions"] = region_details["selected_regions"]
    prompt_editing_payload["meta_prompt"] = meta_prompt
    prompt_editing_payload["generated_prompt_variants"] = validated_variants
    return prompt_editing_payload


def _build_region_candidate_prompt_editing_payload(
    *,
    mode: str,
    baseline_validation: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "mode": mode,
        "baseline_validation": baseline_validation,
        "selected_regions": [],
        "region_candidate_meta_prompt": None,
        "region_candidate_raw_output": "",
        "region_candidate_meta_prompts": [],
        "region_candidates": [],
        "beam_width": None,
        "beam_replacement_mode": None,
        "beam_search_steps": [],
        "combination_meta_prompt": None,
        "combination_raw_output": "",
        "candidate_combinations": [],
        "synthesis_meta_prompt": None,
        "synthesis_meta_prompts": [],
        "generated_prompt_variants": [],
        "retained_top_k_prompts": [],
    }


def _build_and_evaluate_region_candidate_prompts(
    *,
    args: argparse.Namespace,
    instruction_prompt: str,
    sampled_pairs: Sequence[Dict[str, Any]],
    full_eval_pairs: Sequence[Dict[str, Any]],
    baseline_validation: Dict[str, Any],
    baseline_confusion_matrix: Dict[str, int],
    region_details: Dict[str, Any],
    region_candidates: Sequence[Dict[str, Any]],
    region_candidate_generation_payload: Dict[str, Any] | None,
    prompt_editing_payload: Dict[str, Any],
    model,
    tokenizer,
) -> Dict[str, Any]:
    beam_payload = _run_region_candidate_beam_search(
        args=args,
        instruction_prompt=instruction_prompt,
        sampled_pairs=sampled_pairs,
        region_details=region_details,
        region_candidates=region_candidates,
        model=model,
        tokenizer=tokenizer,
    )
    scored_variants = beam_payload["generated_prompt_variants"]
    retained_prompt_texts = [variant["revised_prompt"] for variant in scored_variants]
    retained_prompt_text_set = set(retained_prompt_texts)
    print("[agent_gradient_eval_debug] final beam prompts:")
    if retained_prompt_texts:
        for retained_index, prompt_text in enumerate(retained_prompt_texts):
            print(f"[beam_prompt {retained_index}]")
            print(prompt_text)
    else:
        print("[]")
    evaluation_cache: Dict[str, Dict[str, Any]] = {}
    selection_metrics_by_prompt = {
        variant["revised_prompt"]: variant["selection_metrics"]
        for variant in scored_variants
        if variant["selection_metrics"] is not None
    }

    for variant in scored_variants:
        revised_prompt = variant["revised_prompt"]
        if revised_prompt not in retained_prompt_text_set:
            continue
        evaluation = evaluation_cache.get(revised_prompt)
        if evaluation is None:
            evaluation = _evaluate_prompt_variant(
                revised_prompt=revised_prompt,
                sampled_pairs=sampled_pairs,
                full_eval_pairs=full_eval_pairs,
                full_eval_split=args.full_eval_split,
                baseline_validation=baseline_validation,
                baseline_confusion_matrix=baseline_confusion_matrix,
                model=model,
                tokenizer=tokenizer,
                validation_batch_size=args.validation_batch_size,
                use_chat_template=not args.disable_chat_template,
                f1_std_penalty=args.selection_f1_std_penalty,
                log_context={
                    "generation_index": variant.get("generation_index"),
                    "beam_index": variant.get("beam_index"),
                },
            )
            evaluation_cache[revised_prompt] = copy.deepcopy(evaluation)
        variant["validation"] = evaluation["validation"]
        variant["full_evaluation"] = evaluation["full_evaluation"]
        variant["delta_vs_baseline"] = evaluation["delta_vs_baseline"]
        variant["retained_for_full_validation"] = True

    prompt_editing_payload["selected_regions"] = region_details["selected_regions"]
    prompt_editing_payload["region_candidate_meta_prompt"] = (
        region_candidate_generation_payload.get("meta_prompt")
        if region_candidate_generation_payload
        else None
    )
    prompt_editing_payload["region_candidate_raw_output"] = (
        region_candidate_generation_payload.get("raw_output", "")
        if region_candidate_generation_payload
        else ""
    )
    prompt_editing_payload["region_candidate_meta_prompts"] = [
        {
            "region_rank": item["region_rank"],
            "meta_prompt": item.get("meta_prompt"),
            "raw_output": item.get("raw_output"),
        }
        for item in beam_payload["region_candidates"]
    ]
    prompt_editing_payload["region_candidates"] = list(beam_payload["region_candidates"])
    prompt_editing_payload["beam_width"] = beam_payload["beam_width"]
    prompt_editing_payload["beam_replacement_mode"] = beam_payload["beam_replacement_mode"]
    prompt_editing_payload["beam_search_steps"] = beam_payload["beam_search_steps"]
    prompt_editing_payload["combination_meta_prompt"] = None
    prompt_editing_payload["combination_raw_output"] = ""
    prompt_editing_payload["candidate_combinations"] = []
    prompt_editing_payload["synthesis_meta_prompt"] = None
    prompt_editing_payload["synthesis_meta_prompts"] = beam_payload.get(
        "synthesis_meta_prompts",
        [],
    )
    prompt_editing_payload["generated_prompt_variants"] = scored_variants
    prompt_editing_payload["retained_top_k_prompts"] = [
        {
            "revised_prompt": prompt_text,
            "combined_score": selection_metrics_by_prompt[prompt_text]["combined_score"],
            "mean_cross_entropy": selection_metrics_by_prompt[prompt_text]["mean_cross_entropy"],
            "mean_perplexity": selection_metrics_by_prompt[prompt_text]["mean_perplexity"],
        }
        for prompt_text in retained_prompt_texts
    ]
    return prompt_editing_payload


def _run_llm_candidate_suggestion(
    *,
    args: argparse.Namespace,
    instruction_prompt: str,
    sampled_pairs: Sequence[Dict[str, Any]],
    full_eval_pairs: Sequence[Dict[str, Any]],
    gradient_results: Dict[str, Any],
    baseline_validation: Dict[str, Any],
    baseline_confusion_matrix: Dict[str, int],
    model,
    tokenizer,
) -> Dict[str, Any]:
    prompt_editing_payload = _build_region_candidate_prompt_editing_payload(
        mode=MODE_LLM_CANDIDATE_SUGGESTION,
        baseline_validation=baseline_validation,
    )
    if args.num_region_candidates <= 0:
        raise ValueError("--num-region-candidates must be positive in LLM_CANDIDATE_SUGGESTION.")
    if args.beam_width <= 0:
        raise ValueError("--beam-width must be positive in LLM_CANDIDATE_SUGGESTION.")

    region_details = _resolve_region_details(
        instruction_prompt=instruction_prompt,
        tokenizer=tokenizer,
        gradient_results=gradient_results,
        num_edit_regions=args.num_edit_regions,
    )
    print(
        "[agent_gradient_eval_debug] generating all-region candidates in a single LLM run:",
        f"regions={[region['region_rank'] for region in region_details['selected_regions']]}",
        f"num_region_candidates={args.num_region_candidates}",
        f"beam_width={args.beam_width}",
    )
    region_candidate_generation_payload = _generate_region_candidates(
        instruction_prompt=instruction_prompt,
        region_details=region_details,
        num_region_candidates=args.num_region_candidates,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.meta_prompt_max_new_tokens,
        batch_size=args.meta_prompt_batch_size,
        use_chat_template=not args.disable_chat_template,
    )
    region_candidates = region_candidate_generation_payload["region_candidates"]
    return _build_and_evaluate_region_candidate_prompts(
        args=args,
        instruction_prompt=instruction_prompt,
        sampled_pairs=sampled_pairs,
        full_eval_pairs=full_eval_pairs,
        baseline_validation=baseline_validation,
        baseline_confusion_matrix=baseline_confusion_matrix,
        region_details=region_details,
        region_candidates=region_candidates,
        region_candidate_generation_payload=region_candidate_generation_payload,
        prompt_editing_payload=prompt_editing_payload,
        model=model,
        tokenizer=tokenizer,
    )

def _run_lm_probability_candidate_suggestion(
    *,
    args: argparse.Namespace,
    instruction_prompt: str,
    sampled_pairs: Sequence[Dict[str, Any]],
    full_eval_pairs: Sequence[Dict[str, Any]],
    gradient_results: Dict[str, Any],
    baseline_validation: Dict[str, Any],
    baseline_confusion_matrix: Dict[str, int],
    model,
    tokenizer,
) -> Dict[str, Any]:
    prompt_editing_payload = _build_region_candidate_prompt_editing_payload(
        mode=MODE_LM_PROBABILITY_CANDIDATE_SUGGESTION,
        baseline_validation=baseline_validation,
    )
    if args.num_region_candidates <= 0:
        raise ValueError(
            "--num-region-candidates must be positive in LM_PROBABILITY_CANDIDATE_SUGGESTION."
        )
    if args.beam_width <= 0:
        raise ValueError("--beam-width must be positive in LM_PROBABILITY_CANDIDATE_SUGGESTION.")

    region_details = _resolve_region_details(
        instruction_prompt=instruction_prompt,
        tokenizer=tokenizer,
        gradient_results=gradient_results,
        num_edit_regions=args.num_edit_regions,
    )
    print(
        "[agent_gradient_eval_debug] generating LM probability region candidates:",
        f"regions={[region['region_rank'] for region in region_details['selected_regions']]}",
        f"num_region_candidates={args.num_region_candidates}",
        f"beam_width={args.beam_width}",
        f"lm_probability_submode={args.lm_probability_submode}",
    )
    region_candidates = _generate_region_candidates_from_lm_probability(
        instruction_prompt=instruction_prompt,
        region_details=region_details,
        num_region_candidates=args.num_region_candidates,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.meta_prompt_max_new_tokens,
        batch_size=args.meta_prompt_batch_size,
        lm_probability_submode=args.lm_probability_submode,
    )
    return _build_and_evaluate_region_candidate_prompts(
        args=args,
        instruction_prompt=instruction_prompt,
        sampled_pairs=sampled_pairs,
        full_eval_pairs=full_eval_pairs,
        baseline_validation=baseline_validation,
        baseline_confusion_matrix=baseline_confusion_matrix,
        region_details=region_details,
        region_candidates=region_candidates,
        region_candidate_generation_payload=None,
        prompt_editing_payload=prompt_editing_payload,
        model=model,
        tokenizer=tokenizer,
    )


def _run_candidate_suggestion_iteration(
    *,
    args: argparse.Namespace,
    iteration_index: int,
    instruction_prompt: str,
    input_full_evaluation: Dict[str, Any] | None,
    train_samples: Dict[str, Any],
    train_sample_index: Dict[str, dict],
    full_eval_pairs: Sequence[Dict[str, Any]],
    model,
    tokenizer,
) -> Dict[str, Any]:
    print(
        "[agent_gradient_eval_debug] starting optimization iteration:",
        f"iteration={iteration_index}",
        f"Q={args.Q}",
    )
    print("[agent_gradient_eval_debug] iteration input prompt:")
    print(instruction_prompt)

    iteration_rng = random.Random(args.seed + iteration_index - 1)
    train_feedback_samples = sample_feedback_fn(
        train_samples,
        k=args.train_gradient_sample_size,
        rng=iteration_rng,
    )
    train_d_pair_pool = _build_binary_pairs_from_feedback_samples(
        feedback_samples=train_feedback_samples,
        shots_by_id=train_sample_index,
        dataset_type=args.dataset_type,
    )
    print(
        "[agent_gradient_eval_debug] train D sample built:",
        f"iteration={iteration_index}",
        f"D={len(train_d_pair_pool)}",
    )

    train_d_prompts, train_d_target_labels = _build_binary_prompts_for_instruction(
        instruction_prompt=instruction_prompt,
        binary_pairs=train_d_pair_pool,
    )
    train_d_predicted_labels = run_binary_inference(
        train_d_prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.validation_batch_size,
        use_chat_template=not args.disable_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        log_label=f"train_gradient_collection_iter_{iteration_index}",
    )
    train_d_pairs = _attach_predictions_to_pairs(
        pairs=train_d_pair_pool,
        predicted_labels=train_d_predicted_labels,
    )
    train_d_report = {
        "confusion_matrix": _build_confusion_matrix_counts(
            predicted_labels=train_d_predicted_labels,
            target_labels=train_d_target_labels,
        ),
        "prf": _build_prf_scores(
            predicted_labels=train_d_predicted_labels,
            target_labels=train_d_target_labels,
        ),
    }
    sampled_pairs, bucket_stats = _sample_balanced_pairs_from_bucketed_pool(
        bucketed_pairs=train_d_pairs,
        seed=args.seed,
    )
    sampled_examples = [dict(pair) for pair in sampled_pairs]
    sampled_indices = {
        "correct_indices": [
            pair["pair_index"] for pair in sampled_pairs if pair["confusion_bucket"] in {"tp", "tn"}
        ],
        "mistake_indices": [
            pair["pair_index"] for pair in sampled_pairs if pair["confusion_bucket"] in {"fp", "fn"}
        ],
        "bucket_stats": bucket_stats,
    }
    train_gradient_collection = {
        "iteration_index": iteration_index,
        "gradient_split": "train",
        "evaluation_split": args.full_eval_split,
        "train_gradient_sample_size": args.train_gradient_sample_size,
        "train_d_pair_indices": [pair["pair_index"] for pair in train_d_pairs],
        "train_d_bucket_stats": {
            "available": bucket_stats["available"],
            "selected_for_gradients": bucket_stats["selected"],
        },
        "train_d_evaluation_original_prompt": train_d_report,
    }
    print(
        "[agent_gradient_eval_debug] train D pair pool:",
        f"iteration={iteration_index}",
        f"available_tp={bucket_stats['available']['tp']}",
        f"available_tn={bucket_stats['available']['tn']}",
        f"available_fp={bucket_stats['available']['fp']}",
        f"available_fn={bucket_stats['available']['fn']}",
        f"selected_tp={bucket_stats['selected']['tp']}",
        f"selected_tn={bucket_stats['selected']['tn']}",
        f"selected_fp={bucket_stats['selected']['fp']}",
        f"selected_fn={bucket_stats['selected']['fn']}",
    )
    print(
        "[agent_gradient_eval_debug] train gradient batch built:",
        f"iteration={iteration_index}",
        f"pairs={len(sampled_pairs)}",
    )

    print(
        "[agent_gradient_eval_debug] running gradient analysis",
        f"iteration={iteration_index}",
    )
    gradient_results = analyze_relation_extraction_binary_pairs(
        instruction_prompt=instruction_prompt,
        binary_pairs=sampled_pairs,
        model=model,
        tokenizer=tokenizer,
        dataset_type=args.dataset_type,
        gradient_batch_size=args.gradient_batch_size,
        num_candidates=args.num_candidates,
        candidate_mode=args.candidate_mode,
        max_regions=args.max_regions,
        max_total_region_tokens=args.max_total_region_tokens,
        max_region_tokens=args.max_region_tokens,
        region_expansion_threshold_ratio=args.region_expansion_threshold_ratio,
        embedding_step_size=args.embedding_step_size,
        use_chat_template=not args.disable_chat_template,
    )
    print(
        "[agent_gradient_eval_debug] gradient analysis done:",
        f"iteration={iteration_index}",
        f"instances={gradient_results['num_instances']}",
        f"candidate_mode={args.candidate_mode}",
        f"token_gradients={len(gradient_results['token_gradients'])}",
        f"top_regions={len(gradient_results['top_regions'])}",
    )

    baseline_score_payload = _build_score_payload_from_pairs(sampled_pairs)
    baseline_confusion_matrix = _build_confusion_matrix_counts(
        predicted_labels=baseline_score_payload["predicted_labels"],
        target_labels=baseline_score_payload["target_labels"],
    )
    baseline_prf = _build_prf_scores(
        predicted_labels=baseline_score_payload["predicted_labels"],
        target_labels=baseline_score_payload["target_labels"],
    )
    baseline_validation = _summarize_validation_by_bucket(
        sampled_pairs=sampled_pairs,
        score_payload=baseline_score_payload,
    )
    print(
        "[agent_gradient_eval_debug] baseline validation:",
        f"iteration={iteration_index}",
        _format_prf_for_log(baseline_prf),
        f"overall_accuracy={baseline_validation['overall']['accuracy']:.4f}",
        f"fixes_from_mistakes={baseline_validation['fixes_from_mistakes']}",
        f"regressions_from_correct={baseline_validation['regressions_from_correct']}",
    )
    print(
        "[agent_gradient_eval_debug] baseline validation:",
        f"iteration={iteration_index}",
        _format_stable_f1_for_log(
            baseline_prf,
            f1_std_penalty=args.selection_f1_std_penalty,
        ),
    )

    if args.mode == MODE_LLM_CANDIDATE_SUGGESTION:
        prompt_editing_payload = _run_llm_candidate_suggestion(
            args=args,
            instruction_prompt=instruction_prompt,
            sampled_pairs=sampled_pairs,
            full_eval_pairs=full_eval_pairs,
            gradient_results=gradient_results,
            baseline_validation=baseline_validation,
            baseline_confusion_matrix=baseline_confusion_matrix,
            model=model,
            tokenizer=tokenizer,
        )
    else:
        prompt_editing_payload = _run_lm_probability_candidate_suggestion(
            args=args,
            instruction_prompt=instruction_prompt,
            sampled_pairs=sampled_pairs,
            full_eval_pairs=full_eval_pairs,
            gradient_results=gradient_results,
            baseline_validation=baseline_validation,
            baseline_confusion_matrix=baseline_confusion_matrix,
            model=model,
            tokenizer=tokenizer,
        )

    baseline_full_evaluation = copy.deepcopy(input_full_evaluation)
    for variant in prompt_editing_payload.get("generated_prompt_variants", []):
        if baseline_full_evaluation is not None:
            break
        if variant.get("revised_prompt") != instruction_prompt:
            continue
        full_evaluation = variant.get("full_evaluation")
        if full_evaluation:
            baseline_full_evaluation = copy.deepcopy(full_evaluation)
            print(
                "[agent_gradient_eval_debug] input prompt full evaluation reused:",
                f"iteration={iteration_index}",
            )
            break
    if baseline_full_evaluation is None:
        baseline_full_evaluation = _evaluate_prompt_full_evaluation(
            instruction_prompt=instruction_prompt,
            full_eval_pairs=full_eval_pairs,
            full_eval_split=args.full_eval_split,
            model=model,
            tokenizer=tokenizer,
            validation_batch_size=args.validation_batch_size,
            use_chat_template=not args.disable_chat_template,
            f1_std_penalty=args.selection_f1_std_penalty,
            log_context={
                "generation_index": "input",
                "beam_index": "input",
            },
        )
    selected_variant = _select_best_variant(
        prompt_editing_payload=prompt_editing_payload,
        f1_std_penalty=args.selection_f1_std_penalty,
    )
    selected_variant_beats_input = _variant_beats_baseline_full_evaluation(
        selected_variant=selected_variant,
        baseline_full_evaluation=baseline_full_evaluation,
        f1_std_penalty=args.selection_f1_std_penalty,
    )
    if selected_variant is not None and not selected_variant_beats_input:
        candidate_full_prf = (selected_variant.get("full_evaluation") or {}).get("prf") or {}
        input_full_prf = baseline_full_evaluation.get("prf") or {}
        print(
            "[agent_gradient_eval_debug] retaining input prompt:",
            f"iteration={iteration_index}",
            "reason=no_generated_prompt_beats_input_on_full_eval",
            (
                "candidate_stable_f1="
                f"{stable_prf_score_or_neg_inf(candidate_full_prf, std_multiplier=args.selection_f1_std_penalty):.2f}"
            ),
            (
                "input_stable_f1="
                f"{stable_prf_score_or_neg_inf(input_full_prf, std_multiplier=args.selection_f1_std_penalty):.2f}"
            ),
        )
        selected_variant = None
    selected_prompt = _build_selected_prompt_summary(
        iteration_index=iteration_index,
        input_prompt=instruction_prompt,
        selected_variant=selected_variant,
        baseline_prf=baseline_prf,
        baseline_full_evaluation=baseline_full_evaluation,
    )
    next_prompt = selected_prompt["prompt"]
    next_full_evaluation = (
        copy.deepcopy(selected_variant.get("full_evaluation"))
        if selected_variant is not None
        else copy.deepcopy(baseline_full_evaluation)
    )
    print(
        "[agent_gradient_eval_debug] selected prompt for next iteration:",
        f"iteration={iteration_index}",
        f"strategy={selected_prompt['selection_strategy']}",
        f"generation_index={selected_prompt['generation_index']}",
        f"beam_index={selected_prompt['beam_index']}",
        (
            f"dev_f1={selected_prompt['dev_prf']['f1']:.2f}"
            if selected_prompt.get("dev_prf")
            else "dev_f1=None"
        ),
        (
            f"balanced_train_f1={selected_prompt['balanced_train_prf']['f1']:.2f}"
            if selected_prompt.get("balanced_train_prf")
            else "balanced_train_f1=None"
        ),
        f"changed={selected_prompt['changed_vs_input']}",
    )
    print(next_prompt)

    sampling_summary = {
        "iteration_index": iteration_index,
        "gradient_split": "train",
        "evaluation_split": args.full_eval_split,
        "train_gradient_sample_size": args.train_gradient_sample_size,
        "train_d_pair_pool_size": len(train_d_pairs),
        "sampled_pair_count": len(sampled_pairs),
        "full_eval_pair_count": len(full_eval_pairs),
        "bucket_stats": copy.deepcopy(bucket_stats),
        "baseline_bucket_validation": copy.deepcopy(baseline_validation),
        "baseline_confusion_matrix": copy.deepcopy(baseline_confusion_matrix),
    }
    sampling_payload = {
        "seed": args.seed,
        "iteration_index": iteration_index,
        "correct_pair_indices": sampled_indices["correct_indices"],
        "mistake_pair_indices": sampled_indices["mistake_indices"],
        "bucket_stats": sampled_indices.get("bucket_stats", {}),
    }
    return {
        "iteration_index": iteration_index,
        "input_prompt": instruction_prompt,
        "gradient_analysis": gradient_results,
        "train_gradient_collection": train_gradient_collection,
        "baseline_prf": baseline_prf,
        "baseline_full_evaluation": baseline_full_evaluation,
        "baseline_confusion_matrix": baseline_confusion_matrix,
        "baseline_validation": baseline_validation,
        "sampling_summary": sampling_summary,
        "sampling_payload": sampling_payload,
        "sampled_examples": sampled_examples,
        "prompt_editing_payload": prompt_editing_payload,
        "selected_prompt": selected_prompt,
        "next_prompt": next_prompt,
        "next_full_evaluation": next_full_evaluation,
    }


# todo: optimize GPU usage
def main() -> None:
    args = _parse_args()
    print("[agent_gradient_eval_debug] starting")
    if args.selection_f1_std_penalty < 0:
        raise ValueError("--selection-f1-std-penalty must be non-negative.")

    instruction_prompt, prompt_info = resolve_instruction_prompt(
        args.prompt_source_path,
        prompt_index=args.prompt_index,
        prompt_node_id=args.prompt_node_id,
    )
    print(
        "[agent_gradient_eval_debug] prompt resolved:",
        prompt_info.get("prompt_source_type"),
    )
    print(
        "[agent_gradient_eval_debug] initial instruction prompt:\n"
        f"{instruction_prompt}"
    )

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model,
        device_map=args.device_map,
    )
    print("[agent_gradient_eval_debug] model and tokenizer loaded")

    meta_prompt_model_id = (
        args.meta_prompt_model
        if args.mode == MODE_DIRECT_CANDIDATE_GENERATION
        else args.model
    )
    if args.mode == MODE_DIRECT_CANDIDATE_GENERATION and args.Q != 1:
        raise ValueError("--Q is currently supported only for candidate-suggestion modes.")
    if args.mode in {
        MODE_LLM_CANDIDATE_SUGGESTION,
        MODE_LM_PROBABILITY_CANDIDATE_SUGGESTION,
    } and args.Q <= 0:
        raise ValueError("--Q must be positive.")

    full_eval_pairs: List[Dict[str, Any]] = []
    if args.train_gradient_sample_size is None or args.train_gradient_sample_size <= 0:
        raise ValueError("--train-gradient-sample-size must be set to a positive integer.")

    train_samples = load_train_samples(
        data_dir=args.data_dir,
        filename=args.train_samples,
    )
    train_sample_index = _build_train_sample_index(train_samples)
    print(
        "[agent_gradient_eval_debug] train samples loaded:",
        f"relations={len(train_samples)}",
        f"dataset_type={args.dataset_type}",
    )

    full_eval_dataset = load_split_episodes(
        split=args.full_eval_split,
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
    )
    print(
        "[agent_gradient_eval_debug] full evaluation dataset loaded:",
        f"split={args.full_eval_split}",
        f"episodes={len(full_eval_dataset['episodes'])}",
    )
    full_eval_pairs = build_full_binary_pairs(
        dataset=full_eval_dataset,
        dataset_type=args.dataset_type,
        query_index=args.query_index,
    )
    original_dev_prf = _extract_prf_from_prompt_info(prompt_info)
    current_full_evaluation = (
        {"prf": copy.deepcopy(original_dev_prf)}
        if original_dev_prf is not None
        else None
    )
    if current_full_evaluation is None and args.mode in {
        MODE_LLM_CANDIDATE_SUGGESTION,
        MODE_LM_PROBABILITY_CANDIDATE_SUGGESTION,
    }:
        current_full_evaluation = _evaluate_prompt_full_evaluation(
            instruction_prompt=instruction_prompt,
            full_eval_pairs=full_eval_pairs,
            full_eval_split=args.full_eval_split,
            model=model,
            tokenizer=tokenizer,
            validation_batch_size=args.validation_batch_size,
            use_chat_template=not args.disable_chat_template,
            f1_std_penalty=args.selection_f1_std_penalty,
            log_context={
                "generation_index": "initial",
                "beam_index": "initial",
            },
        )
        original_dev_prf = copy.deepcopy(current_full_evaluation.get("prf"))

    iteration_results: List[Dict[str, Any]] = []
    sampled_examples: List[Dict[str, Any]] = []
    sampling_payload: Dict[str, Any] = {}
    sampling_summary: Dict[str, Any] = {}
    baseline_prf: Dict[str, float] = {}
    baseline_confusion_matrix: Dict[str, int] = {}
    gradient_results: Dict[str, Any] = {}
    train_gradient_collection: Dict[str, Any] = {}

    if args.mode == MODE_DIRECT_CANDIDATE_GENERATION:
        rng = random.Random(args.seed)
        train_feedback_samples = sample_feedback_fn(
            train_samples,
            k=args.train_gradient_sample_size,
            rng=rng,
        )
        train_d_pair_pool = _build_binary_pairs_from_feedback_samples(
            feedback_samples=train_feedback_samples,
            shots_by_id=train_sample_index,
            dataset_type=args.dataset_type,
        )
        print(
            "[agent_gradient_eval_debug] train D sample built:",
            f"D={len(train_d_pair_pool)}",
        )
        train_d_prompts, train_d_target_labels = _build_binary_prompts_for_instruction(
            instruction_prompt=instruction_prompt,
            binary_pairs=train_d_pair_pool,
        )
        train_d_predicted_labels = run_binary_inference(
            train_d_prompts,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.validation_batch_size,
            use_chat_template=not args.disable_chat_template,
            add_generation_prompt=True,
            enable_thinking=False,
            log_label="train_gradient_collection",
        )

        train_d_pairs = _attach_predictions_to_pairs(
            pairs=train_d_pair_pool,
            predicted_labels=train_d_predicted_labels,
        )
        train_d_report = {
            "confusion_matrix": _build_confusion_matrix_counts(
                predicted_labels=train_d_predicted_labels,
                target_labels=train_d_target_labels,
            ),
            "prf": _build_prf_scores(
                predicted_labels=train_d_predicted_labels,
                target_labels=train_d_target_labels,
            ),
        }
        sampled_pairs, bucket_stats = _sample_balanced_pairs_from_bucketed_pool(
            bucketed_pairs=train_d_pairs,
            seed=args.seed,
        )
        sampled_examples = [dict(pair) for pair in sampled_pairs]
        sampled_indices = {
            "correct_indices": [
                pair["pair_index"] for pair in sampled_pairs if pair["confusion_bucket"] in {"tp", "tn"}
            ],
            "mistake_indices": [
                pair["pair_index"] for pair in sampled_pairs if pair["confusion_bucket"] in {"fp", "fn"}
            ],
            "bucket_stats": bucket_stats,
        }
        train_gradient_collection = {
            "gradient_split": "train",
            "evaluation_split": args.full_eval_split,
            "train_gradient_sample_size": args.train_gradient_sample_size,
            "train_d_pair_indices": [pair["pair_index"] for pair in train_d_pairs],
            "train_d_bucket_stats": {
                "available": bucket_stats["available"],
                "selected_for_gradients": bucket_stats["selected"],
            },
            "train_d_evaluation_original_prompt": train_d_report,
        }
        print(
            "[agent_gradient_eval_debug] train D pair pool:",
            f"available_tp={bucket_stats['available']['tp']}",
            f"available_tn={bucket_stats['available']['tn']}",
            f"available_fp={bucket_stats['available']['fp']}",
            f"available_fn={bucket_stats['available']['fn']}",
            f"selected_tp={bucket_stats['selected']['tp']}",
            f"selected_tn={bucket_stats['selected']['tn']}",
            f"selected_fp={bucket_stats['selected']['fp']}",
            f"selected_fn={bucket_stats['selected']['fn']}",
        )
        print(
            "[agent_gradient_eval_debug] train gradient batch built:",
            f"pairs={len(sampled_pairs)}",
        )

        print("[agent_gradient_eval_debug] running gradient analysis")
        gradient_results = analyze_relation_extraction_binary_pairs(
            instruction_prompt=instruction_prompt,
            binary_pairs=sampled_pairs,
            model=model,
            tokenizer=tokenizer,
            dataset_type=args.dataset_type,
            gradient_batch_size=args.gradient_batch_size,
            num_candidates=args.num_candidates,
            candidate_mode=args.candidate_mode,
            max_regions=args.max_regions,
            max_total_region_tokens=args.max_total_region_tokens,
            max_region_tokens=args.max_region_tokens,
            region_expansion_threshold_ratio=args.region_expansion_threshold_ratio,
            embedding_step_size=args.embedding_step_size,
            use_chat_template=not args.disable_chat_template,
        )
        print(
            "[agent_gradient_eval_debug] gradient analysis done:",
            f"instances={gradient_results['num_instances']}",
            f"candidate_mode={args.candidate_mode}",
            f"token_gradients={len(gradient_results['token_gradients'])}",
            f"top_regions={len(gradient_results['top_regions'])}",
        )

        baseline_score_payload = _build_score_payload_from_pairs(sampled_pairs)
        baseline_confusion_matrix = _build_confusion_matrix_counts(
            predicted_labels=baseline_score_payload["predicted_labels"],
            target_labels=baseline_score_payload["target_labels"],
        )
        baseline_prf = _build_prf_scores(
            predicted_labels=baseline_score_payload["predicted_labels"],
            target_labels=baseline_score_payload["target_labels"],
        )
        baseline_validation = _summarize_validation_by_bucket(
            sampled_pairs=sampled_pairs,
            score_payload=baseline_score_payload,
        )
        print(
            "[agent_gradient_eval_debug] baseline validation:",
            _format_prf_for_log(baseline_prf),
            f"overall_accuracy={baseline_validation['overall']['accuracy']:.4f}",
            f"fixes_from_mistakes={baseline_validation['fixes_from_mistakes']}",
            f"regressions_from_correct={baseline_validation['regressions_from_correct']}",
        )
        print(
            "[agent_gradient_eval_debug] baseline validation:",
            _format_stable_f1_for_log(
                baseline_prf,
                f1_std_penalty=args.selection_f1_std_penalty,
            ),
        )
        sampling_summary = {
            "gradient_split": "train",
            "evaluation_split": args.full_eval_split,
            "train_gradient_sample_size": args.train_gradient_sample_size,
            "train_d_pair_pool_size": len(train_d_pairs),
            "sampled_pair_count": len(sampled_pairs),
            "full_eval_pair_count": len(full_eval_pairs),
            "bucket_stats": copy.deepcopy(bucket_stats),
            "baseline_bucket_validation": copy.deepcopy(baseline_validation),
            "baseline_confusion_matrix": copy.deepcopy(baseline_confusion_matrix),
        }
        prompt_editing_payload = _run_direct_candidate_generation(
            args=args,
            instruction_prompt=instruction_prompt,
            sampled_pairs=sampled_pairs,
            full_eval_pairs=full_eval_pairs,
            gradient_results=gradient_results,
            baseline_validation=baseline_validation,
            baseline_confusion_matrix=baseline_confusion_matrix,
            model=model,
            tokenizer=tokenizer,
        )
        sampling_payload = {
            "seed": args.seed,
            "correct_pair_indices": sampled_indices["correct_indices"],
            "mistake_pair_indices": sampled_indices["mistake_indices"],
            "bucket_stats": sampled_indices.get("bucket_stats", {}),
        }
    else:
        current_prompt = instruction_prompt
        for iteration_index in range(1, args.Q + 1):
            iteration_result = _run_candidate_suggestion_iteration(
                args=args,
                iteration_index=iteration_index,
                instruction_prompt=current_prompt,
                input_full_evaluation=current_full_evaluation,
                train_samples=train_samples,
                train_sample_index=train_sample_index,
                full_eval_pairs=full_eval_pairs,
                model=model,
                tokenizer=tokenizer,
            )
            iteration_results.append(iteration_result)
            current_prompt = iteration_result["next_prompt"]
            current_full_evaluation = iteration_result.get("next_full_evaluation")

        final_iteration_result = iteration_results[-1]
        initial_iteration_result = iteration_results[0]
        prompt_editing_payload = final_iteration_result["prompt_editing_payload"]
        sampled_examples = final_iteration_result["sampled_examples"]
        sampling_payload = final_iteration_result["sampling_payload"]
        sampling_summary = initial_iteration_result["sampling_summary"]
        baseline_prf = initial_iteration_result["baseline_prf"]
        baseline_confusion_matrix = initial_iteration_result["baseline_confusion_matrix"]
        gradient_results = final_iteration_result["gradient_analysis"]
        train_gradient_collection = final_iteration_result["train_gradient_collection"]
        prompt_editing_payload["iterations"] = [
            {
                "iteration_index": iteration_result["iteration_index"],
                "input_prompt": iteration_result["input_prompt"],
                "selected_prompt": copy.deepcopy(iteration_result["selected_prompt"]),
                "train_gradient_collection": copy.deepcopy(
                    iteration_result["train_gradient_collection"]
                ),
                "sampling_summary": copy.deepcopy(iteration_result["sampling_summary"]),
                "gradient_analysis": copy.deepcopy(iteration_result["gradient_analysis"]),
                "prompt_region_editing": _strip_prediction_arrays_from_prompt_editing_payload(
                    prompt_editing_payload=iteration_result["prompt_editing_payload"],
                ),
            }
            for iteration_result in iteration_results
        ]

    final_sampled_data_and_predictions_payload = _build_sampled_data_and_predictions_payload(
        sampling=sampling_payload,
        sampled_examples=sampled_examples,
        prompt_editing_payload=prompt_editing_payload,
    )
    sampled_data_and_predictions_payload = _build_iterative_sampled_data_and_predictions_payload(
        iteration_results=iteration_results,
        final_payload=final_sampled_data_and_predictions_payload,
    )
    payload = {
        "args": _namespace_to_json_dict(args),
        "model": args.model,
        "meta_prompt_model": meta_prompt_model_id,
        "device_map": args.device_map,
        "dataset_type": args.dataset_type,
        "candidate_mode": args.candidate_mode,
        "mode": args.mode,
        "dataset_split": "train",
        "prompt_source_path": str(args.prompt_source_path),
        "prompt_info": prompt_info,
        "num_iterations": max(len(iteration_results), 1),
        "gradient_analysis": gradient_results,
        "train_gradient_collection": train_gradient_collection,
        "prompt_region_editing": _strip_prediction_arrays_from_prompt_editing_payload(
            prompt_editing_payload=prompt_editing_payload,
        ),
        "iterations": [
            {
                "iteration_index": iteration_result["iteration_index"],
                "input_prompt": iteration_result["input_prompt"],
                "selected_prompt": copy.deepcopy(iteration_result["selected_prompt"]),
                "train_gradient_collection": copy.deepcopy(
                    iteration_result["train_gradient_collection"]
                ),
                "sampling_summary": copy.deepcopy(iteration_result["sampling_summary"]),
                "gradient_analysis": copy.deepcopy(iteration_result["gradient_analysis"]),
                "prompt_region_editing": _strip_prediction_arrays_from_prompt_editing_payload(
                    prompt_editing_payload=iteration_result["prompt_editing_payload"],
                ),
            }
            for iteration_result in iteration_results
        ],
        "external_data_files": {
            "sampled_data_and_predictions": "sampled_data_and_predictions.json",
        },
    }
    summary_payload = _build_summary_payload(
        instruction_prompt=instruction_prompt,
        baseline_prf=baseline_prf,
        prompt_editing_payload=prompt_editing_payload,
        baseline_confusion_matrix=baseline_confusion_matrix,
        original_dev_prf=original_dev_prf,
        sampling_summary=sampling_summary,
        iteration_results=iteration_results,
    )
    run_dir = _create_output_run_dir(
        output_root_dir=args.output_root_dir,
        output_substring=args.output_substring,
    )
    eval_outputs_dir = run_dir / "eval_outputs"
    _save_json(run_dir / "args.json", _namespace_to_json_dict(args))
    _save_json(run_dir / "all_data.json", payload)
    _save_json(run_dir / "summary.json", summary_payload)
    _save_json(
        run_dir / "sampled_data_and_predictions.json",
        sampled_data_and_predictions_payload,
    )
    if args.mode in {
        MODE_LLM_CANDIDATE_SUGGESTION,
        MODE_LM_PROBABILITY_CANDIDATE_SUGGESTION,
    }:
        combination_payload = _build_combination_generated_prompts_payload(
            prompt_editing_payload=prompt_editing_payload,
        )
        if iteration_results:
            combination_payload["num_iterations"] = len(iteration_results)
            combination_payload["iterations"] = [
                {
                    "iteration_index": iteration_result["iteration_index"],
                    **_build_combination_generated_prompts_payload(
                        prompt_editing_payload=iteration_result["prompt_editing_payload"],
                    ),
                }
                for iteration_result in iteration_results
            ]
        _save_json(
            run_dir / "combination_generated_prompts.json",
            combination_payload,
        )
    _save_prompt_eval_outputs(
        eval_outputs_dir=eval_outputs_dir,
        full_eval_split=args.full_eval_split,
        generated_prompt_variants=prompt_editing_payload.get("generated_prompt_variants", []),
    )
    print(f"saved run directory to {run_dir}")
    print(f"saved args to {run_dir / 'args.json'}")
    print(f"saved full payload to {run_dir / 'all_data.json'}")
    print(f"saved summary to {run_dir / 'summary.json'}")
    print(f"saved sampled data and predictions to {run_dir / 'sampled_data_and_predictions.json'}")
    if args.mode in {
        MODE_LLM_CANDIDATE_SUGGESTION,
        MODE_LM_PROBABILITY_CANDIDATE_SUGGESTION,
    }:
        print(f"saved combination prompts to {run_dir / 'combination_generated_prompts.json'}")
    print(f"saved eval outputs to {eval_outputs_dir}")


if __name__ == "__main__":
    main()



# python3 codes/agents/agent_gradient_eval_debug.py \
#   --model "Qwen/Qwen3-4B" \
#   --device-map "cuda:0" \
#   --eval-output-path "trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
#   --prompt-source-path "trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
#   --prompt-node-id 12 \
#   --output-file "gradients_experiments/gradient_debug_node12_f1-37.json"

# python3 codes/agents/agent_gradient_eval_debug.py \
#   --model "google/gemma-3-4b-it" \
#   --device-map "cuda:0" \
#   --eval-output-path "trainings/20260128_151114_Qwen-Qwen3-4B/eval_outputs/EVALID_0_labels_predictions.json" \
#   --prompt-source-path "unified_optimization_results/unified_opt_fs_tacred_20260325_215139/generated_prompt_candidates.json" \
#   --prompt-index 0 \
# --output-file "gradients_experiments/gradient_debug_node9.json"
