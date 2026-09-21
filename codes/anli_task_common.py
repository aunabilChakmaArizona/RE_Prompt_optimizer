"""Shared ANLI labels and prompts for preparation, testing, and optimization."""


ANLI_INITIAL_PROMPT = (
    "You are given a Natural Language Inference (NLI) task. Given a premise and a hypothesis, determine whether the hypothesis is Entailment, Contradiction, or Neutral with respect to the premise."
)
ANLI_ANSWER_INSTRUCTION = (
    "Do not think or provide any reasoning. Output only the option label exactly once "
    "between the tags <answer> and </answer>, for example: <answer>A</answer>. "
    "Do not output anything else."
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
