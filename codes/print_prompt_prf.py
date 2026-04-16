#!/usr/bin/env python3
"""Print precision, recall, and F1 for original and generated prompts."""

import json
from pathlib import Path


# Set this to the summary.json file you want to read.
SUMMARY_JSON = (
    "gradients_experiments/"
    "20260408_204909_llm_cand_sugg_region3s_qwen4/"
    "summary.json"
)

# Use "dev_prf" for dev scores or "balanced_train_prf" for balanced train scores.
SCORE_KEY = "dev_prf"


def get_scores(prompt_info):
    scores = prompt_info.get(SCORE_KEY)
    if scores is None:
        return None

    required_keys = {"precision", "recall", "f1"}
    if not required_keys.issubset(scores):
        missing = ", ".join(sorted(required_keys - set(scores)))
        raise KeyError(f"Missing {missing} in {SCORE_KEY}")

    return scores


def print_prf(scores):
    print(
        f"{scores['precision']:.1f}\t"
        f"{scores['recall']:.1f}\t"
        f"{scores['f1']:.1f}\t"
    )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    summary_path = Path(SUMMARY_JSON)
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    print_prf(summary["original_prompt"][SCORE_KEY])

    for prompt_info in summary.get("generated_prompts", []):
        scores = get_scores(prompt_info)
        if scores is None:
            continue
        print_prf(scores)


if __name__ == "__main__":
    main()
