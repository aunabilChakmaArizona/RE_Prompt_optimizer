from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from agents.agent_llm_prompting import run_prompt
from agents.agent_prompts import DIFFERENTIATE_PROMPT

DEFAULT_F1_STABILITY_STD_MULTIPLIER = 2.5


def stable_f1_score_or_neg_inf(
    score: Mapping[str, Any] | None,
    *,
    mean_key: str = "f1_mean",
    std_key: str = "f1_std",
    std_multiplier: float = DEFAULT_F1_STABILITY_STD_MULTIPLIER,
) -> float:
    if not isinstance(score, Mapping):
        return float("-inf")
    f1_mean = score.get(mean_key)
    if f1_mean is None:
        return float("-inf")
    f1_std = score.get(std_key, 0.0) or 0.0
    return float(f1_mean) - std_multiplier * float(f1_std)


def stable_prf_score_or_neg_inf(
    prf: Mapping[str, Any] | None,
    *,
    std_multiplier: float = DEFAULT_F1_STABILITY_STD_MULTIPLIER,
) -> float:
    return stable_f1_score_or_neg_inf(
        prf,
        mean_key="f1",
        std_key="f1_std",
        std_multiplier=std_multiplier,
    )


def stable_node_score_or_neg_inf(
    node,
    *,
    std_multiplier: float = DEFAULT_F1_STABILITY_STD_MULTIPLIER,
) -> float:
    return stable_f1_score_or_neg_inf(
        getattr(node, "val_score", None),
        std_multiplier=std_multiplier,
    )


def score_node_or_neg_inf(node) -> float:
    score = getattr(node, "val_score", None)
    if not isinstance(score, dict):
        return float("-inf")
    value = score.get("f1_mean")
    if value is None:
        return float("-inf")
    return float(value)


def extract_tagged_text(text: str, open_tag: str, close_tag: str) -> str:
    pattern = rf"{re.escape(open_tag)}(.*?){re.escape(close_tag)}"
    matches = list(re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE))
    if not matches:
        return text.strip()
    return matches[-1].group(1).strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    candidate_text = fenced_match.group(1) if fenced_match else text.strip()
    start = candidate_text.find("{")
    end = candidate_text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in model output.")
    json_text = candidate_text[start : end + 1]
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        normalized_json_text = json_text.translate(
            str.maketrans(
                {
                    "\u201c": '"',
                    "\u201d": '"',
                    "\u2018": "'",
                    "\u2019": "'",
                }
            )
        )
        return json.loads(normalized_json_text)


def load_json_file(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def has_tagged_text(text: str, open_tag: str, close_tag: str) -> bool:
    pattern = rf"{re.escape(open_tag)}(.*?){re.escape(close_tag)}"
    return re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE) is not None


def get_sentence_with_tags(meta_data: Dict) -> str:
    token_with_tags = list(meta_data["token"])
    token_with_tags[meta_data["subj_start"]] = (
        "<subject>" + token_with_tags[meta_data["subj_start"]]
    )
    token_with_tags[meta_data["subj_end"]] = (
        token_with_tags[meta_data["subj_end"]] + "</subject>"
    )
    token_with_tags[meta_data["obj_start"]] = (
        "<object>" + token_with_tags[meta_data["obj_start"]]
    )
    token_with_tags[meta_data["obj_end"]] = (
        token_with_tags[meta_data["obj_end"]] + "</object>"
    )
    return " ".join(token_with_tags)


def build_support_block(support_sentences: Sequence[str]) -> str:
    if not support_sentences:
        return ""
    if len(support_sentences) == 1:
        return f"Support Sentence: {support_sentences[0]}"
    lines = []
    for idx, sent in enumerate(support_sentences, start=1):
        lines.append(f"Support Sentence {idx}: {sent}")
    return "\n".join(lines)


def resolve_way_shots(way_ids: Sequence[int], shots: Dict) -> List[Dict]:
    resolved = []
    if not way_ids:
        return resolved
    resolved.append(shots["shots"][way_ids[0]])

    # may come from LLM generation / UMBC index
    # for shot_id in way_ids[1:]:
    #     resolved.append(shots["umbc_shots"][shot_id])

    return resolved


def differentiate_prompts(
    prompt_1: str,
    prompt_2: str,
    *,
    model,
    tokenizer,
    max_new_tokens: int,
    do_sample: bool,
) -> tuple[str, str, str]:
    differentiate_prompt = DIFFERENTIATE_PROMPT.replace("#PROMPT1#", prompt_1).replace(
        "#PROMPT2#", prompt_2
    )
    raw_response = run_prompt(
        differentiate_prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    return (
        raw_response,
        extract_tagged_text(raw_response, "<d>", "</d>"),
        differentiate_prompt,
    )
