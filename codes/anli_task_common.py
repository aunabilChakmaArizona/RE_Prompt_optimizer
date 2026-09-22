"""Shared ANLI labels and prompts for preparation, testing, and optimization."""

from __future__ import annotations

from typing import Any


ANLI_INITIAL_PROMPT = (
    "You are given a Natural Language Inference (NLI) task. Given a premise and a "
    "hypothesis, determine whether the hypothesis is Entailment, Neutral, or "
    "Contradiction with respect to the premise."
)
ANLI_REASONING_INITIAL_PROMPT = (
    "You are given a Natural Language Inference (NLI) task. Given a premise and a "
    "hypothesis, reason carefully and determine whether the hypothesis is Entailment, "
    "Neutral, or Contradiction with respect to the premise."
)
ANLI_ANSWER_INSTRUCTION = (
    "Do not think or provide any reasoning. Output exactly one relationship name "
    "between the tags <answer> and </answer>: entailment, neutral, or contradiction. "
    "For example: <answer>entailment</answer>. Do not output anything else."
)
ANLI_REASONING_ANSWER_INSTRUCTION = (
    "After you finish reasoning, output exactly one relationship name between the "
    "tags <answer> and </answer>: entailment, neutral, or contradiction. For example: "
    "<answer>entailment</answer>."
)
ANLI_LABEL_NAMES = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}
ANLI_OPTION_LABELS = {
    0: "A",
    1: "B",
    2: "C",
}
ANLI_RELATION_BY_OPTION = {
    "A": "entailment",
    "B": "neutral",
    "C": "contradiction",
}
ANLI_CHOICES = (
    {
        "label": "A",
        "text": "Entailment: the premise makes the hypothesis true.",
    },
    {
        "label": "B",
        "text": (
            "Neutral: the premise does not determine whether the hypothesis is true "
            "or false."
        ),
    },
    {
        "label": "C",
        "text": "Contradiction: the premise makes the hypothesis false.",
    },
)


def build_anli_prompt(
    instruction_prompt: str,
    answer_instruction: str,
    premise: str,
    hypothesis: str,
) -> str:
    """Render an ANLI instance without artificial multiple-choice options."""
    return (
        f"{instruction_prompt.strip()}\n\n"
        f"{answer_instruction.strip()}\n\n"
        f"Premise:\n{premise.strip()}\n\n"
        f"Hypothesis:\n{hypothesis.strip()}"
    )


def normalize_anli_relation(value: str | None) -> str | None:
    """Normalize one generated ANLI relationship name for exact scoring."""
    if value is None:
        return None
    relation = " ".join(value.strip().casefold().split())
    if relation in ANLI_RELATION_BY_OPTION.values():
        return relation
    return None


def anli_gold_relation(record: dict[str, Any]) -> str:
    """Return the textual gold relationship from one processed ANLI record."""
    answer_text = normalize_anli_relation(str(record.get("answer_text", "")))
    if answer_text is not None:
        return answer_text
    option = str(record.get("answer", "")).strip().upper()
    try:
        return ANLI_RELATION_BY_OPTION[option]
    except KeyError as error:
        raise ValueError(f"ANLI record {record.get('id')} has no valid gold relation.") from error
