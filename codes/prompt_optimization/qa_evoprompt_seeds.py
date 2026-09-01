"""Fixed QA seed prompts for the five-member EvoPrompt-DE population."""

from __future__ import annotations


# The labels make the provisional source of each seed explicit. Replace these
# prompts with the final curated human and automatically generated prompts while
# preserving the same five-member experimental population.

QA_EVOPROMPT_SEEDS = {
    "reasoning": [
        {
            "label": "human_1",
            "prompt": (
                "Read the multiple-choice question carefully and think before choosing "
                "the best answer. Do not overthink it."
            ),
        },
        {
            "label": "human_2",
            "prompt": (
                "Read the multiple-choice question and all of the answer choices carefully. "
                "Make sure you understand what the question is asking, think through the "
                "relevant facts, and compare the choices before selecting the answer that "
                "is best supported. Take your time, but do not overthink the question."
            ),
        },
        {
            "label": "automatic_1",
            "prompt": (
                "Review the question and choices carefully. Compare the options and select "
                "the one that best answers the question."
            ),
        },
        {
            "label": "automatic_2",
            "prompt": (
                "Carefully review the question and each available choice before answering. "
                "Think through the relevant information, rule out choices that do not fit, "
                "and pay attention to important details in the question. Select the option "
                "that most clearly and accurately answers what is being asked."
            ),
        },
    ],
    "non_reasoning": [
        {
            "label": "human_1",
            "prompt": (
                "Read the multiple-choice question and select the best answer directly. "
                "Do not overthink it."
            ),
        },
        {
            "label": "human_2",
            "prompt": (
                "Read the multiple-choice question and all of the answer choices carefully. "
                "Make sure you understand what is being asked, then select the answer that "
                "best fits the question. Keep it straightforward and do not overthink. "
                "Give only the answer without explaining your reasoning."
            ),
        },
        {
            "label": "automatic_1",
            "prompt": (
                "Review the question and choices, then select the option that best answers "
                "the question."
            ),
        },
        {
            "label": "automatic_2",
            "prompt": (
                "Carefully review the question and each available choice. Pay attention to "
                "the important details and select the option that most accurately answers "
                "what is being asked. Choose the best answer directly without providing "
                "reasoning or additional explanation."
            ),
        },
    ],
}
