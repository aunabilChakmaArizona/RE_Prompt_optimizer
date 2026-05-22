#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
CODES_DIR = REPO_ROOT / "codes"
for path in (CODES_DIR,):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from agents.agent_binary_inference import run_binary_inference
from agents.agent_data_loader import DEFAULT_DATA_DIR, load_split_episodes, load_train_samples
from agents.agent_gradient_eval_debug import (
    _attach_predictions_to_pairs,
    _build_binary_pairs_from_feedback_samples,
    _build_binary_prompts_for_instruction,
    _build_confusion_matrix_counts,
    _build_evaluation_report,
    _build_prf_scores,
    _build_score_payload_from_pairs,
    _build_train_sample_index,
    _sample_balanced_pairs_from_bucketed_pool,
    build_full_binary_pairs,
    resolve_instruction_prompt,
)
from agents.agent_llm_prompting import run_prompts
from agents.agent_memory import clear_model_memory
from agents.agent_models import load_model_and_tokenizer
from agents.agent_prompts import (
    INFERENCE_INSTRUCTION_PROMPT_V1,
    LPO_LOCAL_REWRITE_PROMPT_V1,
    LPO_LOCATION_TAGGING_PROMPT_V1,
)
from agents.agent_sample_feedback import sample_feedback_fn
from agents.agent_utils import extract_json_object, load_json_file, stable_prf_score_or_neg_inf

PROMPT_SOURCE_SCRATCH = "scratch"
PROMPT_SOURCE_POPULATION = "population"
TARGET_ROLE = "target"
OPTIMIZER_ROLE = "optimizer"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run gradient-free Local Prompt Optimization for binary relation "
            "extraction prompts using optimizer-model reasoning to select edit locations."
        )
    )
    parser.add_argument("--model", required=True, help="HF task model id.")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--optimizer-model",
        default=None,
        help="Optional larger HF model id used for LPO location tagging and rewriting.",
    )
    parser.add_argument("--optimizer-device-map", default=None)
    parser.add_argument(
        "--prompt-source",
        choices=[PROMPT_SOURCE_SCRATCH, PROMPT_SOURCE_POPULATION],
        default=PROMPT_SOURCE_SCRATCH,
    )
    parser.add_argument(
        "--population-path",
        type=Path,
        default=None,
        help=(
            "Path to population.json or generated_prompt_candidates.json. With a "
            "generated candidates file, --prompt-node-id is treated as the 0-based "
            "candidate prompt index."
        ),
    )
    parser.add_argument(
        "--prompt-node-id",
        type=int,
        default=None,
        help=(
            "node_id for population.json, or 0-based prompt index for "
            "generated_prompt_candidates.json."
        ),
    )
    parser.add_argument("--dataset-type", default="fs_tacred")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-samples", default="fs_tacred_train_samples.pkl")
    parser.add_argument("--full-eval-split", default="dev")
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument(
        "--train-feedback-sample-size",
        "--train-gradient-sample-size",
        dest="train_feedback_sample_size",
        type=int,
        default=128,
        help="Train feedback sample size before balanced TP/TN/FP/FN selection.",
    )
    parser.add_argument("--selection-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-locations", type=int, default=5)
    parser.add_argument("--max-edit-tags", type=int, default=5)
    parser.add_argument("--max-words-per-edit-tag", type=int, default=5)
    parser.add_argument("--feedback-example-size", type=int, default=16)
    parser.add_argument("--mistake-example-size", type=int, default=8)
    parser.add_argument("--top-z", type=int, default=3)
    parser.add_argument("--dev-f1-std-penalty", type=float, default=2.0)
    parser.add_argument("--optimizer-max-new-tokens", type=int, default=5000)
    parser.add_argument("--optimizer-batch-size", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument("--disable-optimizer-chat-template", action="store_true")
    parser.add_argument("--output-root-dir", type=Path, default=Path("codes/lpo_experiments"))
    parser.add_argument("--output-substring", default="")
    return parser.parse_args()


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _create_run_dir(output_root_dir: Path, output_substring: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = output_substring.strip().replace("/", "-")
    run_dir = output_root_dir / (f"{stamp}_{suffix}" if suffix else stamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _namespace_to_json_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _format_prf(prf: Dict[str, float]) -> str:
    return (
        f"P={float(prf.get('precision', 0.0)):.2f} +/- {float(prf.get('precision_std', 0.0)):.2f}, "
        f"R={float(prf.get('recall', 0.0)):.2f} +/- {float(prf.get('recall_std', 0.0)):.2f}, "
        f"F1={float(prf.get('f1', 0.0)):.2f} +/- {float(prf.get('f1_std', 0.0)):.2f}"
    )


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class LpoRoleModelSwitcher:
    def __init__(
        self,
        *,
        target_model_id: str,
        target_device_map: str,
        optimizer_model_id: str,
        optimizer_device_map: str,
    ) -> None:
        self.target_model_id = target_model_id
        self.target_device_map = target_device_map
        self.optimizer_model_id = optimizer_model_id
        self.optimizer_device_map = optimizer_device_map
        self.current_role: str | None = None
        self.current_model_id: str | None = None
        self.current_device_map: str | None = None
        self.model = None
        self.tokenizer = None

    def _role_spec(self, role: str) -> Tuple[str, str]:
        if role == TARGET_ROLE:
            return self.target_model_id, self.target_device_map
        if role == OPTIMIZER_ROLE:
            return self.optimizer_model_id, self.optimizer_device_map
        raise ValueError(f"Unsupported model role: {role}")

    def unload(self) -> None:
        if self.model is None and self.tokenizer is None:
            return
        print(
            "[relation_extraction_lpo] unloading model:",
            self.current_model_id,
            f"role={self.current_role}",
            f"device_map={self.current_device_map}",
        )
        self.model = None
        self.tokenizer = None
        self.current_role = None
        self.current_model_id = None
        self.current_device_map = None
        clear_model_memory()

    def ensure(self, role: str):
        model_id, device_map = self._role_spec(role)
        if (
            self.model is not None
            and self.current_model_id == model_id
            and self.current_device_map == device_map
        ):
            self.current_role = role
            return self.model, self.tokenizer

        self.unload()
        print(
            "[relation_extraction_lpo] loading",
            role,
            "model:",
            model_id,
            f"device_map={device_map}",
        )
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_id=model_id,
            device_map=device_map,
        )
        self.current_role = role
        self.current_model_id = model_id
        self.current_device_map = device_map
        return self.model, self.tokenizer


def _resolve_initial_instruction_prompt(args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    if args.prompt_source == PROMPT_SOURCE_SCRATCH:
        return INFERENCE_INSTRUCTION_PROMPT_V1, {
            "prompt_source": PROMPT_SOURCE_SCRATCH,
            "source_symbol": "INFERENCE_INSTRUCTION_PROMPT_V1",
            "source_file": "codes/agents/agent_prompts.py",
        }
    if args.population_path is None:
        raise ValueError("--population-path is required with --prompt-source=population.")
    if args.prompt_node_id is None:
        raise ValueError("--prompt-node-id is required with --prompt-source=population.")

    prompt, prompt_info = resolve_instruction_prompt(
        args.population_path,
        prompt_index=None,
        prompt_node_id=args.prompt_node_id,
    )
    prompt_info = {
        "prompt_source": PROMPT_SOURCE_POPULATION,
        "population_path": str(args.population_path),
        "prompt_node_id": args.prompt_node_id,
        **prompt_info,
    }
    return prompt, prompt_info


def _sample_balanced_train_pairs_for_step(
    *,
    instruction_prompt: str,
    train_samples: Dict[str, Any],
    train_sample_index: Dict[str, dict],
    dataset_type: str,
    sample_size: int,
    seed: int,
    model,
    tokenizer,
    batch_size: int,
    use_chat_template: bool,
    log_every: int,
    log_label: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, float]]:
    rng = random.Random(seed)
    train_feedback_samples = sample_feedback_fn(
        train_samples,
        k=sample_size,
        rng=rng,
    )
    train_pair_pool = _build_binary_pairs_from_feedback_samples(
        feedback_samples=train_feedback_samples,
        shots_by_id=train_sample_index,
        dataset_type=dataset_type,
    )
    prompts, target_labels = _build_binary_prompts_for_instruction(
        instruction_prompt=instruction_prompt,
        binary_pairs=train_pair_pool,
    )
    predicted_labels = run_binary_inference(
        prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        log_every=log_every,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        log_label=log_label,
    )
    bucketed_pairs = _attach_predictions_to_pairs(
        pairs=train_pair_pool,
        predicted_labels=predicted_labels,
    )
    sampled_pairs, bucket_stats = _sample_balanced_pairs_from_bucketed_pool(
        bucketed_pairs=bucketed_pairs,
        seed=seed,
    )
    baseline_payload = _build_score_payload_from_pairs(sampled_pairs)
    baseline_prf = _build_prf_scores(
        predicted_labels=baseline_payload["predicted_labels"],
        target_labels=baseline_payload["target_labels"],
    )
    bucket_stats = {
        **bucket_stats,
        "train_pair_pool_size": len(train_pair_pool),
        "balanced_pair_count": len(sampled_pairs),
    }
    return sampled_pairs, bucket_stats, baseline_prf


def _evaluate_on_pairs(
    *,
    instruction_prompt: str,
    binary_pairs: Sequence[Dict[str, Any]],
    model,
    tokenizer,
    batch_size: int,
    use_chat_template: bool,
    log_every: int,
    prf_mode: str = "binary",
) -> Dict[str, Any]:
    prompts, target_labels = _build_binary_prompts_for_instruction(
        instruction_prompt=instruction_prompt,
        binary_pairs=binary_pairs,
    )
    predicted_labels = run_binary_inference(
        prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        log_every=log_every,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
        enable_thinking=False,
        log_label="lpo_evaluation",
    )
    return _build_evaluation_report(
        score_payload={"predicted_labels": predicted_labels, "target_labels": target_labels},
        binary_pairs=binary_pairs,
        prf_mode=prf_mode,
    )


def _stable_f1(prf: Dict[str, float], *, penalty: float) -> float:
    return stable_prf_score_or_neg_inf(prf, std_multiplier=penalty)


def _example_line(pair: Dict[str, Any], index: int) -> str:
    return (
        f"Example {index} [{pair['confusion_bucket']}]: gold={pair['label']}, "
        f"prediction={pair['prediction']}\n"
        f"Relation: {pair['support']['relation']} ({pair['relation_description']})\n"
        f"Support Sentence: {pair['support_sentence']}\n"
        f"Query Sentence: {pair['query_sentence']}"
    )


def _build_feedback_examples(
    *,
    sampled_pairs: Sequence[Dict[str, Any]],
    feedback_example_size: int,
    mistake_example_size: int,
    seed: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    mistakes = [pair for pair in sampled_pairs if pair.get("confusion_bucket") in {"fp", "fn"}]
    corrects = [pair for pair in sampled_pairs if pair.get("confusion_bucket") in {"tp", "tn"}]
    rng.shuffle(mistakes)
    rng.shuffle(corrects)
    selected = mistakes[: max(0, mistake_example_size)]
    remaining = max(0, feedback_example_size - len(selected))
    selected.extend(corrects[:remaining])
    if len(selected) < feedback_example_size:
        seen = {id(pair) for pair in selected}
        rest = [pair for pair in sampled_pairs if id(pair) not in seen]
        rng.shuffle(rest)
        selected.extend(rest[: feedback_example_size - len(selected)])
    text = "\n\n".join(_example_line(pair, idx) for idx, pair in enumerate(selected, start=1))
    return text or "(No feedback examples available.)", [dict(pair) for pair in selected]


def _build_location_prompt(
    *,
    instruction_prompt: str,
    feedback_examples: str,
    max_edit_tags: int,
    max_words_per_edit_tag: int,
) -> str:
    return (
        LPO_LOCATION_TAGGING_PROMPT_V1
        .replace("#INFERENCE_PROMPT#", instruction_prompt)
        .replace("#FEEDBACK_EXAMPLES#", feedback_examples)
        .replace("#MAX_EDIT_TAGS#", str(max_edit_tags))
        .replace("#MAX_WORDS_PER_EDIT_TAG#", str(max_words_per_edit_tag))
    )


def _build_rewrite_prompt(
    *,
    tagged_prompt: str,
    feedback_examples: str,
) -> str:
    return (
        LPO_LOCAL_REWRITE_PROMPT_V1
        .replace("#TAGGED_PROMPT#", tagged_prompt)
        .replace("#FEEDBACK_EXAMPLES#", feedback_examples)
    )


def _parse_tagged_prompt(
    *,
    original_prompt: str,
    raw_tagged_prompt: str,
    max_edit_tags: int,
    max_words_per_edit_tag: int,
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    tagged_prompt = raw_tagged_prompt.strip()
    fenced = re.search(r"```(?:text)?\s*(.*?)\s*```", tagged_prompt, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        tagged_prompt = fenced.group(1).strip()
    prompt_wrapped = re.search(r"\[P\](.*?)\[/P\]", tagged_prompt, flags=re.DOTALL | re.IGNORECASE)
    if prompt_wrapped:
        tagged_prompt = prompt_wrapped.group(1).strip()
    prompt_wrapped = re.search(r"<p>(.*?)</p>", tagged_prompt, flags=re.DOTALL | re.IGNORECASE)
    if prompt_wrapped:
        tagged_prompt = prompt_wrapped.group(1).strip()

    warnings: List[str] = []
    spans: List[Dict[str, Any]] = []
    plain_parts: List[str] = []
    tagged_parts: List[str] = []
    cursor = 0
    tag_count = 0
    for match in re.finditer(r"<edit>(.*?)</edit>", tagged_prompt, flags=re.DOTALL | re.IGNORECASE):
        span_text = match.group(1)
        prefix = tagged_prompt[cursor : match.start()]
        plain_parts.append(prefix)
        tagged_parts.append(prefix)
        tag_count += 1
        if tag_count > max_edit_tags:
            warnings.append("extra_edit_tags_ignored")
            plain_parts.append(span_text)
            tagged_parts.append(span_text)
            cursor = match.end()
            continue

        start_char = sum(len(part) for part in plain_parts)
        word_count = len(re.findall(r"\S+", span_text))
        if word_count > max_words_per_edit_tag:
            warnings.append(f"edit_tag_{len(spans) + 1}_exceeds_word_limit")
        plain_parts.append(span_text)
        tagged_parts.append(f"<edit>{span_text}</edit>")
        end_char = sum(len(part) for part in plain_parts)
        spans.append(
            {
                "location_rank": len(spans) + 1,
                "start_char": start_char,
                "end_char": end_char,
                "text": span_text,
                "word_count": word_count,
            }
        )
        cursor = match.end()
    suffix = tagged_prompt[cursor:]
    plain_parts.append(suffix)
    tagged_parts.append(suffix)
    plain_prompt = "".join(plain_parts)
    sanitized_tagged_prompt = "".join(tagged_parts)

    if not spans:
        warnings.append("no_edit_tags_found")
    if plain_prompt.strip() != original_prompt.strip():
        warnings.append("tagged_prompt_plain_text_differs_from_input")
        warnings.append("using_tagged_prompt_as_returned")
    return sanitized_tagged_prompt, spans, warnings


def _clean_candidate_prompt(text: str) -> str:
    cleaned = text.strip()
    fenced = re.search(r"```(?:text)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    cleaned = re.sub(r"\[/?p\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?p>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?edit>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _candidate_prompts_from_raw_outputs(
    raw_outputs: Sequence[str],
    fallback_prompt: str,
) -> List[str]:
    candidates: List[str] = []
    for raw_output in raw_outputs:
        found_tagged_prompt = False
        for match in re.finditer(r"\[P\](.*?)\[/P\]", raw_output, flags=re.DOTALL | re.IGNORECASE):
            text = match.group(1).strip()
            if text:
                candidates.append(text)
                found_tagged_prompt = True
        for match in re.finditer(r"<p>(.*?)</p>", raw_output, flags=re.DOTALL | re.IGNORECASE):
            text = match.group(1).strip()
            if text:
                candidates.append(text)
                found_tagged_prompt = True
        if found_tagged_prompt:
            continue
        try:
            parsed = extract_json_object(raw_output)
        except (ValueError, TypeError, json.JSONDecodeError):
            parsed = {}
        raw_candidates = parsed.get("candidate_prompts") if isinstance(parsed, dict) else None
        if isinstance(raw_candidates, list):
            candidates.extend(str(item).strip() for item in raw_candidates if str(item).strip())
            continue
        if raw_output.strip():
            candidates.append(raw_output.strip())
    deduped: List[str] = []
    for candidate in [fallback_prompt] + candidates:
        cleaned = _clean_candidate_prompt(candidate)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped


def _score_candidates_on_train(
    *,
    candidate_prompts: Sequence[str],
    sampled_pairs: Sequence[Dict[str, Any]],
    model,
    tokenizer,
    batch_size: int,
    use_chat_template: bool,
    log_every: int,
    f1_std_penalty: float,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for index, prompt in enumerate(candidate_prompts):
        report = _evaluate_on_pairs(
            instruction_prompt=prompt,
            binary_pairs=sampled_pairs,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            use_chat_template=use_chat_template,
            log_every=log_every,
            prf_mode="binary",
        )
        scored.append(
            {
                "candidate_index": index,
                "prompt": prompt,
                "train_evaluation": report,
                "train_stable_f1": _stable_f1(report["prf"], penalty=f1_std_penalty),
            }
        )
        print(
            "[relation_extraction_lpo] train candidate:",
            f"index={index}",
            _format_prf(report["prf"]),
            f"stable_f1={scored[-1]['train_stable_f1']:.2f}",
        )
    return scored


def _select_candidate(
    *,
    scored_candidates: Sequence[Dict[str, Any]],
    full_eval_pairs: Sequence[Dict[str, Any]],
    model,
    tokenizer,
    batch_size: int,
    use_chat_template: bool,
    log_every: int,
    full_eval_split: str,
    top_z: int,
    f1_std_penalty: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ranked = sorted(
        scored_candidates,
        key=lambda item: (
            float(item["train_stable_f1"]),
            float(item["train_evaluation"]["prf"].get("f1", float("-inf"))),
            -float(item["train_evaluation"]["prf"].get("f1_std", 0.0) or 0.0),
            -int(item["candidate_index"]),
        ),
        reverse=True,
    )
    if top_z <= 0:
        return ranked[0], []

    dev_evaluations: List[Dict[str, Any]] = []
    for rank, candidate in enumerate(ranked[: min(top_z, len(ranked))], start=1):
        print(
            "[relation_extraction_lpo] full eval candidate:",
            f"rank={rank}",
            f"candidate_index={candidate['candidate_index']}",
            f"split={full_eval_split}",
            f"is_source_candidate={int(candidate['candidate_index']) == 0}",
        )
        print("[relation_extraction_lpo] full eval candidate prompt BEGIN")
        print(candidate["prompt"])
        print("[relation_extraction_lpo] full eval candidate prompt END")
        report = _evaluate_on_pairs(
            instruction_prompt=candidate["prompt"],
            binary_pairs=full_eval_pairs,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            use_chat_template=use_chat_template,
            log_every=log_every,
            prf_mode="episode",
        )
        record = {
            **candidate,
            "dev_rank": rank,
            "full_evaluation": report,
            "dev_stable_f1": _stable_f1(report["prf"], penalty=f1_std_penalty),
        }
        dev_evaluations.append(record)
        print(
            "[relation_extraction_lpo] dev candidate:",
            f"rank={rank}",
            _format_prf(report["prf"]),
            f"stable_f1={record['dev_stable_f1']:.2f}",
        )
    selected = max(
        dev_evaluations,
        key=lambda item: (
            float(item["dev_stable_f1"]),
            float(item["full_evaluation"]["prf"].get("f1", float("-inf"))),
            -float(item["full_evaluation"]["prf"].get("f1_std", 0.0) or 0.0),
            float(item["train_stable_f1"]),
        ),
    )
    return selected, dev_evaluations


def main() -> None:
    args = _parse_args()
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive.")
    if args.train_feedback_sample_size <= 0:
        raise ValueError("--train-feedback-sample-size must be positive.")
    if args.max_locations <= 0 or args.max_locations > 5:
        raise ValueError("--max-locations must be in [1, 5].")
    if args.max_edit_tags <= 0:
        raise ValueError("--max-edit-tags must be positive.")
    if args.max_edit_tags > args.max_locations:
        args.max_edit_tags = args.max_locations
    if args.num_candidates <= 0:
        raise ValueError("--num-candidates must be positive.")
    if args.dev_f1_std_penalty < 0:
        raise ValueError("--dev-f1-std-penalty must be non-negative.")

    start_time = time.monotonic()
    run_dir = _create_run_dir(args.output_root_dir, args.output_substring)
    steps_path = run_dir / "steps.jsonl"
    _save_json(run_dir / "args.json", _namespace_to_json_dict(args))

    instruction_prompt, prompt_source_info = _resolve_initial_instruction_prompt(args)
    (run_dir / "initial_prompt.txt").write_text(instruction_prompt, encoding="utf-8")
    _save_json(run_dir / "prompt_source.json", prompt_source_info)
    print("[relation_extraction_lpo] run_dir:", run_dir)
    print("[relation_extraction_lpo] prompt_source:", prompt_source_info)
    print("[relation_extraction_lpo] initial prompt:\n" + instruction_prompt)

    optimizer_model_id = args.optimizer_model or args.model
    optimizer_device_map = args.optimizer_device_map or args.device_map
    print("[relation_extraction_lpo] target model:", args.model)
    print("[relation_extraction_lpo] optimizer model:", optimizer_model_id)
    print("[relation_extraction_lpo] model switching enabled; one model is loaded at a time")
    switcher = LpoRoleModelSwitcher(
        target_model_id=args.model,
        target_device_map=args.device_map,
        optimizer_model_id=optimizer_model_id,
        optimizer_device_map=optimizer_device_map,
    )
    use_chat_template = not args.disable_chat_template
    optimizer_use_chat_template = not args.disable_optimizer_chat_template

    print("[relation_extraction_lpo] loading train samples")
    train_samples = load_train_samples(data_dir=args.data_dir, filename=args.train_samples)
    train_sample_index = _build_train_sample_index(train_samples)

    print("[relation_extraction_lpo] loading full eval split:", args.full_eval_split)
    full_eval_dataset = load_split_episodes(
        split=args.full_eval_split,
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
    )
    full_eval_pairs = build_full_binary_pairs(
        dataset=full_eval_dataset,
        dataset_type=args.dataset_type,
        query_index=args.query_index,
    )

    current_prompt = instruction_prompt
    best_prompt = current_prompt
    best_train_stable_f1 = float("-inf")
    best_dev_stable_f1 = float("-inf")
    best_full_eval: Dict[str, Any] | None = None
    final_bucket_stats: Dict[str, Any] = {}
    final_baseline_prf: Dict[str, float] = {}

    for step_index in range(1, args.n_steps + 1):
        step_start = time.monotonic()
        step_input_prompt = current_prompt
        print(f"[relation_extraction_lpo] step {step_index}/{args.n_steps}")
        print(f"[Prompt to optimize] prompt = {step_input_prompt}")
        model, tokenizer = switcher.ensure(TARGET_ROLE)
        sampled_pairs, bucket_stats, baseline_prf = _sample_balanced_train_pairs_for_step(
            instruction_prompt=step_input_prompt,
            train_samples=train_samples,
            train_sample_index=train_sample_index,
            dataset_type=args.dataset_type,
            sample_size=args.train_feedback_sample_size,
            seed=args.seed + step_index,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.selection_batch_size,
            use_chat_template=use_chat_template,
            log_every=args.log_every,
            log_label=f"lpo_train_bucket_step_{step_index}",
        )
        final_bucket_stats = bucket_stats
        final_baseline_prf = baseline_prf
        current_train_stable_f1 = _stable_f1(baseline_prf, penalty=args.dev_f1_std_penalty)
        if current_train_stable_f1 > best_train_stable_f1:
            best_train_stable_f1 = current_train_stable_f1
            best_prompt = current_prompt
        print(
            "[relation_extraction_lpo] balanced train current:",
            _format_prf(baseline_prf),
            f"stable_f1={current_train_stable_f1:.2f}",
            f"bucket_stats={bucket_stats}",
        )

        feedback_examples, selected_feedback_pairs = _build_feedback_examples(
            sampled_pairs=sampled_pairs,
            feedback_example_size=args.feedback_example_size,
            mistake_example_size=args.mistake_example_size,
            seed=args.seed + 10_000 + step_index,
        )
        location_prompt = _build_location_prompt(
            instruction_prompt=step_input_prompt,
            feedback_examples=feedback_examples,
            max_edit_tags=args.max_edit_tags,
            max_words_per_edit_tag=args.max_words_per_edit_tag,
        )
        model = None
        tokenizer = None
        optimizer_model, optimizer_tokenizer = switcher.ensure(OPTIMIZER_ROLE)
        raw_tagged_output = run_prompts(
            [location_prompt],
            model=optimizer_model,
            tokenizer=optimizer_tokenizer,
            max_new_tokens=args.optimizer_max_new_tokens,
            batch_size=args.optimizer_batch_size,
            use_chat_template=optimizer_use_chat_template,
            add_generation_prompt=True,
            enable_thinking=True,
            do_sample=True,
            do_log=True,
            log_label="relation_extraction_lpo_location",
        )[0]
        tagged_prompt, edit_spans, tag_warnings = _parse_tagged_prompt(
            original_prompt=step_input_prompt,
            raw_tagged_prompt=raw_tagged_output,
            max_edit_tags=args.max_edit_tags,
            max_words_per_edit_tag=args.max_words_per_edit_tag,
        )
        print(
            "[relation_extraction_lpo] location tagging:",
            f"locations={len(edit_spans)}",
            f"warnings={tag_warnings}",
        )
        print(tagged_prompt)

        if not edit_spans:
            accepted = False
            selected_candidate: Dict[str, Any] | None = None
            scored_candidates: List[Dict[str, Any]] = []
            dev_evaluations: List[Dict[str, Any]] = []
            next_prompt = current_prompt
            raw_rewrite_outputs = []
            rewrite_prompt = ""
            optimizer_model = None
            optimizer_tokenizer = None
        else:
            rewrite_prompt = _build_rewrite_prompt(
                tagged_prompt=tagged_prompt,
                feedback_examples=feedback_examples,
            )
            raw_rewrite_outputs = run_prompts(
                [rewrite_prompt] * args.num_candidates,
                model=optimizer_model,
                tokenizer=optimizer_tokenizer,
                max_new_tokens=args.optimizer_max_new_tokens,
                batch_size=args.optimizer_batch_size,
                use_chat_template=optimizer_use_chat_template,
                add_generation_prompt=True,
                enable_thinking=True,
                do_sample=True,
                do_log=True,
                log_label="relation_extraction_lpo_rewrite",
            )
            candidate_prompts = _candidate_prompts_from_raw_outputs(
                raw_rewrite_outputs,
                fallback_prompt=step_input_prompt,
            )
            print(
                "[relation_extraction_lpo] candidates generated:",
                f"count={len(candidate_prompts)}",
            )
            optimizer_model = None
            optimizer_tokenizer = None
            model, tokenizer = switcher.ensure(TARGET_ROLE)
            scored_candidates = _score_candidates_on_train(
                candidate_prompts=candidate_prompts,
                sampled_pairs=sampled_pairs,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.selection_batch_size,
                use_chat_template=use_chat_template,
                log_every=args.log_every,
                f1_std_penalty=args.dev_f1_std_penalty,
            )
            selected_candidate, dev_evaluations = _select_candidate(
                scored_candidates=scored_candidates,
                full_eval_pairs=full_eval_pairs,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.eval_batch_size,
                use_chat_template=use_chat_template,
                log_every=args.log_every,
                full_eval_split=args.full_eval_split,
                top_z=args.top_z,
                f1_std_penalty=args.dev_f1_std_penalty,
            )
            if args.top_z > 0 and selected_candidate.get("full_evaluation"):
                selected_score = float(selected_candidate["dev_stable_f1"])
                accepted = selected_score > best_dev_stable_f1
                if accepted:
                    best_dev_stable_f1 = selected_score
                    best_full_eval = selected_candidate["full_evaluation"]
                    best_prompt = selected_candidate["prompt"]
            else:
                selected_score = float(selected_candidate["train_stable_f1"])
                accepted = selected_score >= current_train_stable_f1
                if accepted and selected_score > best_train_stable_f1:
                    best_train_stable_f1 = selected_score
                    best_prompt = selected_candidate["prompt"]
            next_prompt = selected_candidate["prompt"] if accepted else current_prompt

        current_prompt = next_prompt
        if args.eval_every > 0 and (step_index % args.eval_every == 0 or step_index == args.n_steps):
            print("[relation_extraction_lpo] current prompt after selection:")
            print(current_prompt)

        step_payload = {
            "step": step_index,
            "input_prompt": step_input_prompt,
            "output_prompt": current_prompt,
            "bucket_stats": bucket_stats,
            "baseline_prf": baseline_prf,
            "feedback_examples": selected_feedback_pairs,
            "location_prompt": location_prompt,
            "raw_tagged_output": raw_tagged_output,
            "tagged_prompt": tagged_prompt,
            "edit_spans": edit_spans,
            "tag_warnings": tag_warnings,
            "rewrite_prompt": rewrite_prompt,
            "raw_rewrite_outputs": raw_rewrite_outputs,
            "candidate_scores": scored_candidates,
            "dev_evaluations": dev_evaluations,
            "selected_candidate": selected_candidate,
            "accepted": accepted,
            "elapsed_seconds": time.monotonic() - step_start,
        }
        _append_jsonl(steps_path, step_payload)
        (run_dir / "latest_prompt.txt").write_text(current_prompt, encoding="utf-8")
        print(
            "[relation_extraction_lpo] step result:",
            f"accepted={accepted}",
            f"elapsed={step_payload['elapsed_seconds']:.2f}s",
        )

    (run_dir / "final_prompt.txt").write_text(current_prompt, encoding="utf-8")
    (run_dir / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")
    summary = {
        "run_dir": str(run_dir),
        "prompt_source": prompt_source_info,
        "initial_prompt": instruction_prompt,
        "final_prompt": current_prompt,
        "best_prompt": best_prompt,
        "best_train_stable_f1": best_train_stable_f1,
        "best_dev_stable_f1": best_dev_stable_f1,
        "dev_f1_std_penalty": args.dev_f1_std_penalty,
        "last_full_evaluation": best_full_eval,
        "bucket_stats": final_bucket_stats,
        "baseline_prf": final_baseline_prf,
        "elapsed_seconds": time.monotonic() - start_time,
    }
    _save_json(run_dir / "summary.json", summary)
    model = None
    tokenizer = None
    optimizer_model = None
    optimizer_tokenizer = None
    switcher.unload()
    print("[relation_extraction_lpo] done:", run_dir)


if __name__ == "__main__":
    main()
